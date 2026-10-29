#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


EXPECTED_KEYS = {"image_u16", "z_index", "foreground_ratio"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate plain MRI slice NPZ files")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    count = 0
    for path in args.paths:
        candidates = sorted(path.rglob("*.npz")) if path.is_dir() else [path]
        for candidate in candidates:
            with np.load(candidate, allow_pickle=False) as payload:
                if set(payload.files) != EXPECTED_KEYS:
                    raise ValueError(f"unexpected keys {payload.files}: {candidate}")
                image = payload["image_u16"]
                z = np.asarray(payload["z_index"])
                ratio = np.asarray(payload["foreground_ratio"])
                if image.shape != (368, 448) or image.dtype != np.uint16:
                    raise ValueError(f"invalid image array: {candidate}")
                if z.shape != () or z.dtype != np.int16 or not 0 <= int(z) < 364:
                    raise ValueError(f"invalid z_index: {candidate}")
                if ratio.shape != () or ratio.dtype != np.float32 or not 0.0 <= float(ratio) <= 1.0:
                    raise ValueError(f"invalid foreground_ratio: {candidate}")
                if np.any(image[:2]) or np.any(image[-2:]) or np.any(image[:, :6]) or np.any(image[:, -6:]):
                    raise ValueError(f"non-zero padding: {candidate}")
            count += 1
    print(f"validated {count} plain slice NPZ files")


if __name__ == "__main__":
    main()

