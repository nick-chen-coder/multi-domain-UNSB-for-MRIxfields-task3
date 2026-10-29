from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def position_fourier(position: torch.Tensor, bands: int = 8) -> torch.Tensor:
    position = position.reshape(-1, 1)
    frequencies = (2.0 ** torch.arange(bands, device=position.device, dtype=position.dtype)).reshape(1, -1)
    angles = math.pi * position * frequencies
    return torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)


def pad_model(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] != (368, 448):
        raise ValueError(f"expected 368x448 input, got {tuple(x.shape[-2:])}")
    return F.pad(x, (0, 0, 8, 8), value=-1.0)


def crop_model(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] != (384, 448):
        raise ValueError(f"expected 384x448 model output, got {tuple(x.shape[-2:])}")
    return x[..., 8:-8, :]


class ResidualDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalized: bool = True):
        super().__init__()
        norm = lambda channels: nn.InstanceNorm2d(channels, affine=True) if normalized else nn.Identity()
        self.norm1 = norm(in_channels)
        self.norm2 = norm(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = F.leaky_relu(self.norm1(x), 0.2)
        residual = self.conv1(residual)
        residual = F.avg_pool2d(self.conv2(F.leaky_relu(self.norm2(residual), 0.2)), 2)
        shortcut = F.avg_pool2d(self.shortcut(x), 2)
        return (residual + shortcut) / math.sqrt(2.0)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.leaky_relu(self.norm1(x), 0.2))
        h = self.conv2(F.leaky_relu(self.norm2(h), 0.2))
        return (x + h) / math.sqrt(2.0)


class AdaIN2d(nn.Module):
    def __init__(self, channels: int, condition_dim: int = 64):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.affine = nn.Linear(condition_dim, 2 * channels)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.affine(condition).chunk(2, dim=1)
        return (1.0 + gamma[..., None, None]) * self.norm(x) + beta[..., None, None]


class GatedSkipFusion(nn.Module):
    def __init__(
        self,
        skip_channels: int,
        out_channels: int,
        condition_dim: int = 64,
        initial_gate_bias: float = -2.0,
    ):
        super().__init__()
        self.projection = nn.Conv2d(skip_channels, out_channels, 1, bias=False)
        self.norm = nn.InstanceNorm2d(out_channels, affine=False)
        self.gate = nn.Linear(condition_dim, out_channels)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, initial_gate_bias)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        skip = self.norm(self.projection(skip))
        if skip.shape != x.shape:
            raise ValueError(
                f"skip/output shape mismatch: {tuple(skip.shape)} != {tuple(x.shape)}"
            )
        gate = torch.sigmoid(self.gate(condition))[..., None, None]
        return x + gate * skip


class ConditionalUpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample: bool,
        skip_channels: int | None = None,
    ):
        super().__init__()
        self.upsample = upsample
        self.norm1 = AdaIN2d(in_channels)
        self.norm2 = AdaIN2d(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.skip_fusion = (
            GatedSkipFusion(skip_channels, out_channels)
            if skip_channels is not None
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        skip: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = F.leaky_relu(self.norm1(x, condition), 0.2)
        shortcut = x
        if self.upsample:
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            shortcut = F.interpolate(shortcut, scale_factor=2, mode="nearest")
        h = self.conv1(h)
        if self.skip_fusion is not None:
            if skip is None:
                raise ValueError("configured gated skip is missing")
            h = self.skip_fusion(h, skip, condition)
        elif skip is not None:
            raise ValueError("skip supplied to a block without skip fusion")
        h = self.conv2(F.leaky_relu(self.norm2(h, condition), 0.2))
        return (h + self.shortcut(shortcut)) / math.sqrt(2.0)
