#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export paired EMA generator for native inference")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    state = torch.load(args.source, map_location="cpu", weights_only=False)
    for key in ("step", "rng", "resolved_config", "ema"):
        if key not in state:
            raise ValueError(f"source checkpoint lacks {key}: {args.source}")
    exported = {
        "format_version": 1,
        "stage": "paired-inference",
        "step": state["step"],
        "rng": state["rng"],
        "resolved_config": state["resolved_config"],
        "ema": {"module": state["ema"]["module"]},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(exported, temporary)
    temporary.replace(args.output)
    print(f"exported={args.output.resolve()} step={exported['step']} sha256={sha256(args.output)}")


if __name__ == "__main__":
    main()

