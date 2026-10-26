from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .common import position_fourier


class MineCondition(nn.Module):
    def __init__(self):
        super().__init__()
        self.step = nn.Embedding(5, 256)
        self.target_domain = nn.Embedding(15, 256)
        self.position = nn.Sequential(nn.Linear(16, 128), nn.SiLU(), nn.Linear(128, 256))
        self.fusion = nn.Sequential(nn.Linear(768, 512), nn.SiLU(), nn.Linear(512, 256))

    def forward(self, step, target_domain, position):
        if not torch.is_tensor(step):
            step = torch.full_like(target_domain, int(step))
        return self.fusion(
            torch.cat(
                (self.step(step), self.target_domain(target_domain), self.position(position_fourier(position))),
                dim=1,
            )
        )


class FiLM(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.affine = nn.Linear(256, 2 * channels)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.affine(condition).chunk(2, dim=1)
        return (1.0 + gamma[..., None, None]) * x + beta[..., None, None]


class MineEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.condition = MineCondition()
        self.conv1 = nn.Conv2d(4, 64, 4, padding=1)
        self.add1 = nn.Linear(256, 64)
        self.conv2 = nn.Conv2d(64, 128, 4, padding=1)
        self.norm2 = nn.InstanceNorm2d(128, affine=False)
        self.film2 = FiLM(128)
        self.conv3 = nn.Conv2d(128, 256, 4, padding=1)
        self.norm3 = nn.InstanceNorm2d(256, affine=False)
        self.film3 = FiLM(256)
        self.conv4 = nn.Conv2d(256, 512, 4, padding=1)
        self.norm4 = nn.InstanceNorm2d(512, affine=False)
        self.film4 = FiLM(512)
        self.head = nn.Conv2d(512, 1, 4, padding=1)

    def forward(self, pairs: torch.Tensor, step, target_domain, position) -> torch.Tensor:
        if pairs.shape[1] != 4:
            raise ValueError("MINE input must concatenate two 2-channel pairs")
        condition = self.condition(step, target_domain, position)
        x = F.interpolate(
            F.pad(pairs, (0, 0, 8, 8), value=-1.0),
            size=(192, 224),
            mode="bilinear",
            align_corners=False,
        )
        x = self.conv1(x) + self.add1(condition)[..., None, None]
        x = F.avg_pool2d(F.leaky_relu(x, 0.2), 2)
        x = F.avg_pool2d(F.leaky_relu(self.film2(self.norm2(self.conv2(x)), condition), 0.2), 2)
        x = F.avg_pool2d(F.leaky_relu(self.film3(self.norm3(self.conv3(x)), condition), 0.2), 2)
        x = F.leaky_relu(self.film4(self.norm4(self.conv4(x)), condition), 0.2)
        score = self.head(x)
        if score.shape[-2:] != (21, 25):
            raise RuntimeError(f"MINE score map must be 21x25, got {score.shape[-2:]}")
        return score
