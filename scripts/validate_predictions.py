#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from mrixfields_unsb.config import load_config
from mrixfields_unsb.constants import VOLUME_SHAPE


def validate(path: Path) -> None:
    image = nib.load(str(path))
    if image.shape != VOLUME_SHAPE:
        raise ValueError(f"expected {VOLUME_SHAPE}, got {image.shape}: {path}")
    if np.dtype(image.get_data_dtype()) != np.dtype(np.float32):
        raise ValueError(f"expected float32, got {image.get_data_dtype()}: {path}")
    data = np.asarray(image.dataobj, dtype=np.float32)
    if not np.isfinite(data).all() or data.min() < -1e-6 or data.max() > 1.0 + 1e-6:
        raise ValueError(f"values must be finite in [0,1]: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate native full-volume inference results")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[], help="OmegaConf key=value override")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.override)
    manifest_path = Path(cfg.paths.inference_manifest)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not args.allow_partial and (payload.get("partial") or len(records) != 180):
        raise RuntimeError(f"complete result requires 180 records, got {len(records)}")
    expected = set()
    for record in records:
        path = Path(record["output"]).resolve()
        if path in expected:
            raise RuntimeError(f"duplicate output: {path}")
        expected.add(path)
        validate(path)
    actual = {path.resolve() for path in (Path(cfg.paths.output_root) / "full" / "task3").rglob("*.nii.gz")}
    if actual != expected:
        raise RuntimeError(f"output set mismatch: missing={len(expected - actual)} extra={len(actual - expected)}")
    print(f"validated {len(expected)} native full-volume predictions")


if __name__ == "__main__":
    main()
