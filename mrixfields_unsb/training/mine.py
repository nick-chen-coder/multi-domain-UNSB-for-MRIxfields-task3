from __future__ import annotations

import torch


def mine_scores(
    model,
    positive_pair: torch.Tensor,
    negative_pair: torch.Tensor,
    step,
    target_domain: torch.Tensor,
    position: torch.Tensor,
    force_fp32: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if force_fp32:
        with torch.autocast(device_type=positive_pair.device.type, enabled=False):
            positive = model(positive_pair.float(), step, target_domain, position)
            negative = model(negative_pair.float(), step, target_domain, position)
        return positive.float(), negative.float()
    return (
        model(positive_pair, step, target_domain, position),
        model(negative_pair, step, target_domain, position),
    )
