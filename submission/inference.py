#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from runtime.constants import FIELDS, MODALITIES, VOLUME_SHAPE
from runtime.volume import infer_volume, load_generator


EXPECTED_TASK3_OUTPUTS = 120
NIFTI_PATTERN = re.compile(r"P_(T1W|T2W|T2FLAIR)_(0\.1T|1\.5T|3T|5T|7T)_(\d{4})\.nii\.gz")


def safe_relative(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"manifest sample has invalid {label}: {value!r}")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"manifest {label} must be a safe relative path: {path}")
    return path


def validate_sample(sample: object, input_root: Path, output_root: Path) -> tuple[Path, Path, str, str, str]:
    if not isinstance(sample, dict):
        raise ValueError(f"manifest sample must be an object: {sample!r}")
    required = ("input", "output", "modality", "source_field", "target_field")
    missing = [name for name in required if name not in sample]
    if missing:
        raise ValueError(f"manifest sample is missing {missing}: {sample}")

    input_relative = safe_relative(sample["input"], "input")
    output_relative = safe_relative(sample["output"], "output")
    modality = str(sample["modality"])
    source_field = str(sample["source_field"])
    target_field = str(sample["target_field"])
    pair = f"{source_field}_to_{target_field}"
    if modality not in MODALITIES or source_field not in FIELDS or target_field not in FIELDS:
        raise ValueError(f"unsupported Task 3 condition: {sample}")
    if source_field == target_field:
        raise ValueError(f"source and target fields must differ: {sample}")
    if sample.get("task", "task3") != "task3":
        raise ValueError(f"sample is not assigned to task3: {sample}")

    expected_input_parent = Path(modality) / pair / "src"
    expected_output_parent = Path(modality) / pair / "pred"
    input_match = NIFTI_PATTERN.fullmatch(input_relative.name)
    output_match = NIFTI_PATTERN.fullmatch(output_relative.name)
    if input_relative.parent != expected_input_parent or not input_match:
        raise ValueError(f"input path disagrees with its Task 3 condition: {input_relative}")
    if output_relative.parent != expected_output_parent or not output_match:
        raise ValueError(f"output path disagrees with its Task 3 condition: {output_relative}")
    if input_match.group(1) != modality or input_match.group(2) != source_field:
        raise ValueError(f"input filename disagrees with manifest fields: {input_relative}")
    if output_match.group(1) != modality or output_match.group(2) != target_field:
        raise ValueError(f"output filename disagrees with manifest fields: {output_relative}")
    if input_match.group(3) != output_match.group(3):
        raise ValueError(f"input/output case IDs differ: {input_relative} -> {output_relative}")

    source = input_root / input_relative
    output = output_root / output_relative
    if not source.is_file():
        raise FileNotFoundError(source)
    return source, output, modality, source_field, target_field


def validate_prediction(path: Path) -> None:
    image = nib.load(str(path))
    if image.shape != VOLUME_SHAPE:
        raise RuntimeError(f"output shape must be {VOLUME_SHAPE}, got {image.shape}: {path}")
    if np.dtype(image.get_data_dtype()) != np.dtype(np.float32):
        raise RuntimeError(f"output dtype must be float32, got {image.get_data_dtype()}: {path}")
    data = np.asarray(image.dataobj, dtype=np.float32)
    if not np.isfinite(data).all() or data.min() < -1e-6 or data.max() > 1.0 + 1e-6:
        raise RuntimeError(f"output must contain finite values in [0,1]: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MRIxFields2026 Task 3 inference")
    parser.add_argument("--input", type=Path, default=Path("/input"))
    parser.add_argument("--output", type=Path, default=Path("/output"))
    parser.add_argument("--checkpoint", type=Path, default=Path(os.environ.get("MRIXFIELDS_CHECKPOINT", "/app/model/final.pt")))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--precision", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--expected-count", type=int, default=EXPECTED_TASK3_OUTPUTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.input / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("task") != "task3":
        raise ValueError(f"expected a task3 manifest, got {payload.get('task')!r}")
    samples = payload.get("samples")
    if not isinstance(samples, list) or len(samples) != args.expected_count:
        raise ValueError(f"expected {args.expected_count} Task 3 samples, got {len(samples) if isinstance(samples, list) else 'invalid'}")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    jobs = [validate_sample(sample, args.input, args.output) for sample in samples]
    outputs = [job[1].resolve() for job in jobs]
    if len(set(outputs)) != len(outputs):
        raise ValueError("manifest contains duplicate output paths")
    existing = sorted(path.resolve() for path in args.output.rglob("*.nii.gz"))
    if existing:
        raise FileExistsError(f"output directory must not contain pre-existing NIfTI files: {existing[0]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Task 3 production inference requires an NVIDIA GPU")
    print(f"loading checkpoint={args.checkpoint} device={torch.cuda.get_device_name(device)} samples={len(jobs)}", flush=True)
    model = load_generator(args.checkpoint, device)

    for index, (source, output, modality, source_field, target_field) in enumerate(jobs, start=1):
        infer_volume(
            model=model,
            source_path=source,
            output_path=output,
            modality=modality,
            source_field=source_field,
            target_field=target_field,
            batch_size=args.batch_size,
            device=device,
            precision=args.precision,
        )
        validate_prediction(output)
        print(f"inference progress {index}/{len(jobs)} {modality} {source_field}->{target_field} {source.name}", flush=True)

    actual = sorted(path.resolve() for path in args.output.rglob("*.nii.gz"))
    if sorted(outputs) != actual:
        raise RuntimeError("output NIfTI set differs from the exact manifest-required set")
    print(f"completed exactly {len(actual)} Task 3 full-volume predictions", flush=True)


if __name__ == "__main__":
    main()

