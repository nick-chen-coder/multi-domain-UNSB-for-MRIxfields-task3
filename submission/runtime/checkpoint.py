from __future__ import annotations

from pathlib import Path

import torch


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if "ema" not in payload or "module" not in payload["ema"] or "resolved_config" not in payload:
        raise ValueError(f"not an UNSB inference checkpoint: {path}")
    return payload

