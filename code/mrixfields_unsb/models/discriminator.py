from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .common import ResidualDownBlock, pad_model, position_fourier


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(1, 32, 3, padding=1)
        channels = (32, 64, 128, 256, 512, 512, 512, 512)
        self.blocks = nn.ModuleList(
            [ResidualDownBlock(left, right, normalized=False) for left, right in zip(channels, channels[1:])]
        )
        self.domain = nn.Linear(512, 15)
        self.step = nn.Embedding(5, 256)
        self.position = nn.Sequential(nn.Linear(16, 128), nn.SiLU(), nn.Linear(128, 256))
        self.projection = nn.Sequential(nn.Linear(512, 512), nn.SiLU(), nn.Linear(512, 512))

    def forward(
        self,
        x: torch.Tensor,
        step,
        target_domain: torch.Tensor,
        position: torch.Tensor,
    ) -> torch.Tensor:
        h = self.stem(pad_model(x))
        for block in self.blocks:
            h = block(h)
        if h.shape[-2:] != (3, 3):
            raise RuntimeError(f"discriminator map must be 3x3, got {h.shape[-2:]}")
        h = F.leaky_relu(h, 0.2).mean(dim=(2, 3))
        if not torch.is_tensor(step):
            step = torch.full_like(target_domain, int(step))
        condition = torch.cat((self.step(step), self.position(position_fourier(position))), dim=1)
        embedding = self.projection(condition)
        domain_logit = self.domain(h).gather(1, target_domain[:, None]).squeeze(1)
        return domain_logit + (h * embedding).sum(dim=1) / math.sqrt(512.0)

