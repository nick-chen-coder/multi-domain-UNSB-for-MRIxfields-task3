from __future__ import annotations

import math

import torch


def mine_statistics(
    positive: torch.Tensor,
    negative: torch.Tensor,
    subtract_log_spatial: bool = False,
) -> dict[str, torch.Tensor]:
    if positive.shape != negative.shape:
        raise ValueError("positive and negative MINE maps must have equal shape")
    positive = positive.float()
    negative = negative.float()
    batch = positive.shape[0]
    if batch < 1:
        raise ValueError("empty MINE batch")
    positive_mean = positive.mean()
    negative_logsumexp = torch.logsumexp(negative.reshape(-1), dim=0)
    spatial = negative[0].numel() if subtract_log_spatial else 1
    normalization_log = math.log(batch * spatial)
    temp = negative_logsumexp - normalization_log
    entropy = positive_mean - temp
    return {
        "positive_mean": positive_mean,
        "negative_logsumexp": negative_logsumexp,
        "normalization_log": positive_mean.new_tensor(normalization_log),
        "temp": temp,
        "positive_mean_squared": positive_mean.square(),
        "temp_squared": temp.square(),
        "entropy": entropy,
    }


def mine_objective(
    positive: torch.Tensor,
    negative: torch.Tensor,
    subtract_log_spatial: bool = False,
    positive_quadratic_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    stats = mine_statistics(positive, negative, subtract_log_spatial)
    loss = (
        -stats["positive_mean"]
        + stats["temp"]
        + stats["temp_squared"]
        + float(positive_quadratic_weight) * stats["positive_mean_squared"]
    )
    return loss, stats


def mine_loss(
    positive: torch.Tensor,
    negative: torch.Tensor,
    subtract_log_spatial: bool = False,
    positive_quadratic_weight: float = 0.0,
) -> torch.Tensor:
    return mine_objective(
        positive, negative, subtract_log_spatial, positive_quadratic_weight
    )[0]


def entropy_estimate(
    positive: torch.Tensor,
    negative: torch.Tensor,
    subtract_log_spatial: bool = False,
) -> torch.Tensor:
    return mine_statistics(positive, negative, subtract_log_spatial)["entropy"]
