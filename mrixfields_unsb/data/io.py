from audit_utils import audit_file_loading, audit_dataset_wrapper
audit_dataset_wrapper(__file__)
# PLEASE DO NOT EDIT THE FIRST THREE LINES

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from .index import SliceDescriptor


SLICE_SHAPE = (368, 448)
EXPECTED_KEYS = {"image_u16", "z_index", "foreground_ratio"}


def load_npz_slice(item: SliceDescriptor | str | Path) -> dict:
    path = Path(item.path if isinstance(item, SliceDescriptor) else item)
    audit_file_loading(path)
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != EXPECTED_KEYS:
            raise ValueError(f"expected only {sorted(EXPECTED_KEYS)}, got {payload.files}: {path}")
        image = payload["image_u16"].astype(np.float32) / 65535.0
        if image.shape != SLICE_SHAPE:
            raise ValueError(f"unexpected image shape {image.shape}: {path}")
        if not np.isfinite(image).all():
            raise ValueError(f"non-finite image: {path}")
        return {
            "image": torch.from_numpy(image * 2.0 - 1.0).unsqueeze(0),
            "z": int(payload["z_index"]),
            "foreground_ratio": float(payload["foreground_ratio"]),
            "path": str(path),
        }


def load_npz_batch(
    descriptors: list[SliceDescriptor], workers: int = 4, pin_memory: bool = False,
) -> dict:
    if workers <= 1:
        values = [load_npz_slice(item) for item in descriptors]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            values = list(pool.map(load_npz_slice, descriptors))
    images = torch.stack([item["image"] for item in values])
    if pin_memory and torch.cuda.is_available():
        images = images.pin_memory()
    return {
        "image": images,
        "z": torch.tensor([item["z"] for item in values], dtype=torch.long),
        "position": torch.tensor(
            [2.0 * item["z"] / 363.0 - 1.0 for item in values], dtype=torch.float32
        ),
        "foreground_ratio": torch.tensor(
            [item["foreground_ratio"] for item in values], dtype=torch.float32
        ),
        "path": [item["path"] for item in values],
    }
