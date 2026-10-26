from __future__ import annotations

import math
import signal
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

from mrixfields_unsb.config import config_container, save_resolved_config
from mrixfields_unsb.data import SliceIndex


class StopController:
    def __init__(self):
        self.requested = False
        self.signal = None
        signal.signal(signal.SIGTERM, self._request)
        signal.signal(signal.SIGINT, self._request)

    def _request(self, signum, _frame):
        self.requested = True
        self.signal = signum


def build_index(roots, cache_path: str | Path | None = None) -> SliceIndex:
    if cache_path and Path(cache_path).is_file():
        return SliceIndex.from_json(cache_path)
    index = SliceIndex.scan(list(roots))
    if cache_path:
        index.to_json(cache_path)
    return index


def initialize_run(cfg) -> None:
    checkpoint_dir = Path(cfg.paths.checkpoint_dir)
    log_dir = Path(cfg.paths.log_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(cfg, log_dir / "resolved_config.yaml")


def resolved_payload(cfg) -> dict:
    return config_container(cfg)


def cosine_warmup_lambda(step: int, total: int, warmup: int, min_ratio: float) -> float:
    if step < warmup:
        return max((step + 1) / max(warmup, 1), 1e-8)
    progress = min(max((step - warmup) / max(total - warmup, 1), 0.0), 1.0)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def constant_cosine_lambda(step: int, constant: int, cosine: int, min_ratio: float) -> float:
    if step < constant:
        return 1.0
    progress = min((step - constant) / max(cosine, 1), 1.0)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


@contextmanager
def frozen(module: nn.Module):
    states = [parameter.requires_grad for parameter in module.parameters()]
    module.requires_grad_(False)
    try:
        yield module
    finally:
        for parameter, state in zip(module.parameters(), states):
            parameter.requires_grad_(state)


@contextmanager
def module_mode(module: nn.Module, training: bool):
    previous = module.training
    module.train(training)
    try:
        yield module
    finally:
        module.train(previous)


def autocast_context(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision in {"bf16", "fp16"}
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)
