from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    cpu_state = state["torch_cpu"].detach().to(device="cpu", dtype=torch.uint8)
    torch.set_rng_state(cpu_state)
    if torch.cuda.is_available() and state["torch_cuda"]:
        cuda_states = [value.detach().to(device="cpu", dtype=torch.uint8) for value in state["torch_cuda"]]
        torch.cuda.set_rng_state_all(cuda_states)


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if "step" not in payload or "rng" not in payload or "resolved_config" not in payload:
        raise ValueError(f"not a complete training checkpoint: {path}")
    return payload


def resolve_resume(value: str, checkpoint_dir: str | Path) -> Path | None:
    if value == "none":
        return None
    if value == "auto":
        candidate = Path(checkpoint_dir) / "latest.pt"
        return candidate if candidate.is_file() else None
    candidate = Path(value).resolve()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def permanent_checkpoint_path(
    checkpoint_dir: str | Path,
    completed_steps: int,
    total_steps: int,
    milestone_interval: int,
) -> Path | None:
    """Return one non-duplicated permanent checkpoint destination."""
    checkpoint_dir = Path(checkpoint_dir)
    if completed_steps == total_steps:
        return checkpoint_dir / "final.pt"
    if completed_steps % milestone_interval == 0:
        return checkpoint_dir / f"step_{completed_steps:07d}.pt"
    return None
