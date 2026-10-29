from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, distance_transform_edt


def soft_foreground(x: torch.Tensor, threshold: float = 0.05, scale: float = 0.02) -> torch.Tensor:
    x01 = (x.clamp(-1.0, 1.0) + 1.0) * 0.5
    return torch.sigmoid((x01 - threshold) / scale)


def foreground_logits(x: torch.Tensor, threshold: float = 0.05, scale: float = 0.02) -> torch.Tensor:
    x01 = (x.clamp(-1.0, 1.0) + 1.0) * 0.5
    return (x01 - threshold) / scale


def _source_distance_gates(source_mask: torch.Tensor, tolerance: float, gamma: float):
    gates = []
    present = []
    for sample in source_mask.detach().cpu().numpy().astype(bool):
        mask = sample[0]
        boundary = mask & ~binary_erosion(mask, structure=np.ones((3, 3), dtype=bool))
        has_boundary = bool(boundary.any())
        present.append(has_boundary)
        if has_boundary:
            distance = distance_transform_edt(~boundary).astype(np.float32)
            gate = 1.0 / (1.0 + np.exp(np.clip((distance - tolerance) / gamma, -80.0, 80.0)))
        else:
            gate = np.zeros_like(mask, dtype=np.float32)
        gates.append(torch.from_numpy(gate))
    return torch.stack(gates)[:, None].to(source_mask.device), present


def anatomical_structure_loss(
    source: torch.Tensor,
    generated: torch.Tensor,
    mask_threshold: float = 0.05,
    mask_scale: float = 0.02,
    foreground_threshold: float = 0.9,
    background_threshold: float = 0.1,
    tolerance: float = 2.0,
    gamma: float = 1.0,
    presence_epsilon: float = 1.0,
    output_scale: float | None = None,
    mode: str = "legacy",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if mode != "legacy":
        raise ValueError("final repository supports legacy ASP only")
    if source.shape != generated.shape or source.shape[-2:] != (368, 448):
        raise ValueError("ASP expects equal [B,1,368,448] source/generated tensors")

    output_logits = foreground_logits(
        generated, mask_threshold, mask_scale if output_scale is None else output_scale
    )
    output_soft = torch.sigmoid(output_logits)
    fg_bce = F.binary_cross_entropy_with_logits(
        output_logits, torch.ones_like(output_logits), reduction="none"
    )
    bg_bce = F.binary_cross_entropy_with_logits(
        output_logits, torch.zeros_like(output_logits), reduction="none"
    )
    maximum = F.max_pool2d(output_soft, 3, stride=1, padding=1)
    minimum = 1.0 - F.max_pool2d(1.0 - output_soft, 3, stride=1, padding=1)
    output_boundary = maximum - minimum

    source_soft = soft_foreground(source, mask_threshold, mask_scale).detach()
    core = (source_soft > foreground_threshold).to(output_soft.dtype)
    background = (source_soft < background_threshold).to(output_soft.dtype)
    dimensions = (1, 2, 3)
    foreground_loss = (fg_bce * core).sum(dim=dimensions) / (core.sum(dim=dimensions) + 1e-8)
    background_loss = (bg_bce * background).sum(dim=dimensions) / (
        background.sum(dim=dimensions) + 1e-8
    )
    trimap = (foreground_loss + background_loss).mean()

    gates, present = _source_distance_gates(source_soft > 0.5, tolerance, gamma)
    boundary_values = []
    for index, has_boundary in enumerate(present):
        if not has_boundary:
            boundary_values.append(output_boundary[index].sum() * 0.0)
            continue
        mass = output_boundary[index].sum()
        recall = (output_boundary[index] * gates[index]).sum() / (mass + presence_epsilon)
        boundary_values.append(1.0 - recall)
    boundary = torch.stack(boundary_values).mean()
    zero = output_boundary.sum() * 0.0
    return trimap + boundary, {
        "trimap": trimap,
        "boundary": boundary,
        "inner": zero,
        "outer": zero,
    }
