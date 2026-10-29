from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PatchProjection(nn.Module):
    def __init__(self, channels: tuple[int, ...] = (64, 128, 256, 512, 512)):
        super().__init__()
        self.heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(value, 256), nn.ReLU(), nn.Linear(256, 256)) for value in channels]
        )

    def forward(self, level: int, patches: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.heads[level](patches), dim=-1)


def sample_patch_ids(
    features: list[torch.Tensor], max_patches: int = 256, generator: torch.Generator | None = None
) -> list[torch.Tensor]:
    ids = []
    for feature in features:
        count = feature.shape[-2] * feature.shape[-1]
        ids.append(
            torch.stack(
                [torch.randperm(count, device=feature.device, generator=generator)[: min(count, max_patches)] for _ in range(feature.shape[0])]
            )
        )
    return ids


def gather_patches(feature: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    flattened = feature.flatten(2).transpose(1, 2)
    if ids.ndim == 1:
        ids = ids[None].expand(feature.shape[0], -1)
    if ids.shape[0] != feature.shape[0]:
        raise ValueError("PatchNCE IDs must have one row per image")
    return torch.gather(flattened, 1, ids[..., None].expand(-1, -1, flattened.shape[-1]))


class PatchNCELoss(nn.Module):
    def __init__(self, projection: PatchProjection, temperature: float = 0.07, max_patches: int = 256):
        super().__init__()
        self.projection = projection
        self.temperature = temperature
        self.max_patches = max_patches

    def forward(
        self,
        source_features: list[torch.Tensor],
        generated_features: list[torch.Tensor],
        patch_ids: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if len(source_features) != 5 or len(generated_features) != 5:
            raise ValueError("PatchNCE requires five encoder feature levels")
        patch_ids = patch_ids or sample_patch_ids(source_features, self.max_patches)
        losses = []
        for level, (source, generated, ids) in enumerate(
            zip(source_features, generated_features, patch_ids)
        ):
            key = self.projection(level, gather_patches(source, ids)).detach()
            query = self.projection(level, gather_patches(generated, ids))
            logits = torch.bmm(query, key.transpose(1, 2)) / self.temperature
            labels = torch.arange(logits.shape[1], device=logits.device).repeat(logits.shape[0])
            losses.append(F.cross_entropy(logits.flatten(0, 1), labels))
        return torch.stack(losses).mean()
