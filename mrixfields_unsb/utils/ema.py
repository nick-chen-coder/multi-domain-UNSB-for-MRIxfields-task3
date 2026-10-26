from __future__ import annotations

import copy

import torch
from torch import nn


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        self.module.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        source = dict(model.named_parameters())
        for name, target in self.module.named_parameters():
            target.lerp_(source[name].detach(), 1.0 - self.decay)
        source_buffers = dict(model.named_buffers())
        for name, target in self.module.named_buffers():
            target.copy_(source_buffers[name])

    def state_dict(self) -> dict:
        return {"decay": self.decay, "module": self.module.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state["decay"])
        self.module.load_state_dict(state["module"])

