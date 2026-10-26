from __future__ import annotations

import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from mrixfields_unsb.constants import FIELDS, MODALITIES, VOLUME_SHAPE
from mrixfields_unsb.models import Generator, generator_from_checkpoint_state
from mrixfields_unsb.training.bridge import deterministic_rollout
from mrixfields_unsb.training.checkpoint import load_checkpoint


def load_generator(checkpoint: str | Path, device: torch.device) -> Generator:
    state = load_checkpoint(checkpoint, device)
    model = generator_from_checkpoint_state(state).to(device)
    if "ema" not in state or "module" not in state["ema"]:
        raise ValueError("inference requires an EMA generator checkpoint")
    model.load_state_dict(state["ema"]["module"])
    return model.eval().requires_grad_(False)


def _save_like_source(data: np.ndarray, source: nib.Nifti1Image, output: str | Path) -> None:
    header = source.header.copy()
    header.set_data_dtype(np.float32)
    image = nib.Nifti1Image(data.astype(np.float32, copy=False), source.affine.copy(), header)
    qform, qcode = source.get_qform(coded=True)
    sform, scode = source.get_sform(coded=True)
    affine = source.affine.copy()
    image.set_qform(qform if qform is not None else affine, int(qcode or 1))
    image.set_sform(sform if sform is not None else affine, int(scode or 1))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.nii.gz")
    nib.save(image, temporary)
    os.replace(temporary, output)


@torch.inference_mode()
def infer_volume(
    model: Generator,
    source_path: str | Path,
    output_path: str | Path,
    modality: str,
    source_field: str,
    target_field: str,
    batch_size: int,
    device: torch.device,
    precision: str = "bf16",
) -> dict:
    source_image = nib.load(str(source_path))
    if source_image.shape != VOLUME_SHAPE:
        raise ValueError(f"expected source shape {VOLUME_SHAPE}, got {source_image.shape}: {source_path}")
    source = np.asarray(source_image.dataobj, dtype=np.float32)
    if not np.isfinite(source).all() or source.min() < -1e-6 or source.max() > 1.0 + 1e-6:
        raise ValueError(f"source must be finite in [0,1]: {source_path}")
    slices = np.moveaxis(source, 2, 0)
    slices = np.pad(slices, ((0, 0), (2, 2), (6, 6)), mode="constant")
    output_slices = []
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    for start in range(0, VOLUME_SHAPE[2], batch_size):
        end = min(start + batch_size, VOLUME_SHAPE[2])
        tensor = torch.from_numpy(slices[start:end]).unsqueeze(1).to(device)
        tensor = tensor * 2.0 - 1.0
        count = end - start
        values = {
            "modality": torch.full((count,), MODALITIES.index(modality), dtype=torch.long, device=device),
            "source_field": torch.full((count,), FIELDS.index(source_field), dtype=torch.long, device=device),
            "target_field": torch.full((count,), FIELDS.index(target_field), dtype=torch.long, device=device),
            "position": torch.arange(start, end, device=device, dtype=torch.float32) * (2.0 / 363.0) - 1.0,
        }
        with torch.autocast(device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            prediction = deterministic_rollout(model, tensor, values)
        prediction = ((prediction.float().clamp(-1.0, 1.0) + 1.0) * 0.5).cpu().numpy()[:, 0]
        output_slices.append(prediction[:, 2:-2, 6:-6])
    output = np.moveaxis(np.concatenate(output_slices, axis=0), 0, 2).astype(np.float32)
    if output.shape != VOLUME_SHAPE or not np.isfinite(output).all():
        raise RuntimeError(f"invalid generated volume shape or values: {output.shape}")
    _save_like_source(output, source_image, output_path)
    return {
        "source": str(Path(source_path).resolve()),
        "output": str(Path(output_path).resolve()),
        "modality": modality,
        "source_field": source_field,
        "target_field": target_field,
        "shape": list(output.shape),
        "dtype": str(output.dtype),
        "minimum": float(output.min()),
        "maximum": float(output.max()),
    }
