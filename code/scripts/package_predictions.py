#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np

from mrixfields_unsb.config import load_config
from mrixfields_unsb.constants import VOLUME_SHAPE


def main() -> None:
    parser = argparse.ArgumentParser(description="Build validated Task 3 prediction ZIP")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[], help="OmegaConf key=value override")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config, args.override)
    payload = json.loads(Path(cfg.paths.inference_manifest).read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if payload.get("partial") or len(records) != 180:
        raise RuntimeError(f"refusing to package incomplete inference: {len(records)} records")
    package_root = Path(cfg.paths.package_root).resolve()
    staging = package_root / "staging" / "task3"
    if staging.exists():
        if not args.clean:
            raise FileExistsError(f"staging exists; pass --clean: {staging}")
        expected_parent = package_root / "staging"
        if staging.parent != expected_parent:
            raise RuntimeError("unsafe staging path")
        shutil.rmtree(staging)
    start, end = (int(value) for value in cfg.package.z_crop)
    if not 0 <= start < end <= VOLUME_SHAPE[2]:
        raise ValueError(f"invalid z crop: {(start, end)}")
    written = set()
    for record in records:
        source = Path(record["output"])
        relative = Path(record["relative_output"])
        if relative.parts[0] != "task3" or ".." in relative.parts:
            raise ValueError(f"unsafe relative output: {relative}")
        destination = staging.parent / relative
        image = nib.load(str(source))
        if image.shape != VOLUME_SHAPE or np.dtype(image.get_data_dtype()) != np.dtype(np.float32):
            raise ValueError(f"invalid full prediction: {source}")
        slab = image.slicer[:, :, start:end]
        data = np.asarray(slab.dataobj, dtype=np.float32)
        if data.shape != (VOLUME_SHAPE[0], VOLUME_SHAPE[1], end - start):
            raise RuntimeError(f"unexpected slab shape {data.shape}: {source}")
        if not np.isfinite(data).all() or data.min() < -1e-6 or data.max() > 1.0 + 1e-6:
            raise ValueError(f"invalid prediction values: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        nib.save(slab, str(destination))
        written.add(destination.resolve())
    if len(written) != 180:
        raise RuntimeError(f"expected 180 packaged predictions, got {len(written)}")
    zip_path = package_root / str(cfg.package.zip_name)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = zip_path.with_suffix(zip_path.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(staging.rglob("*.nii.gz")):
            archive.write(path, path.relative_to(staging.parent))
    temporary.replace(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        roots = {Path(name).parts[0] for name in names}
        if roots != {"task3"} or len(names) != 180:
            raise RuntimeError(f"invalid ZIP structure roots={roots} files={len(names)}")
    print(f"packaged {len(written)} predictions to {zip_path}")


if __name__ == "__main__":
    main()
