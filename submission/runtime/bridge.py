from __future__ import annotations

import math

import torch

from .constants import BRIDGE_TIMES, TAU_SB


def bridge_coefficients(
    times: tuple[float, ...] = BRIDGE_TIMES, tau: float = TAU_SB
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if len(times) != 6 or times[0] != 0.0 or times[-1] != 1.0:
        raise ValueError("five-step bridge requires six nodes from 0 to 1")
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("bridge times must be strictly increasing")
    a_values = []
    sigma_values = []
    for left, right in zip(times, times[1:]):
        delta = right - left
        a = delta / (1.0 - left)
        a_values.append(a)
        sigma_values.append(math.sqrt(max(delta * (1.0 - a) * tau, 0.0)))
    return tuple(a_values), tuple(sigma_values)


def transition(
    state: torch.Tensor,
    endpoint: torch.Tensor,
    step: int,
    noise: torch.Tensor | None = None,
    deterministic: bool = False,
) -> torch.Tensor:
    a_values, sigma_values = bridge_coefficients()
    if not 0 <= step < 5:
        raise ValueError(f"bridge step must be in [0, 4], got {step}")
    if state.shape != endpoint.shape:
        raise ValueError(f"state and endpoint shapes differ: {state.shape} vs {endpoint.shape}")
    if deterministic or sigma_values[step] == 0.0:
        noise_term = 0.0
    else:
        noise = torch.randn_like(state) if noise is None else noise
        noise_term = sigma_values[step] * noise
    return (1.0 - a_values[step]) * state + a_values[step] * endpoint + noise_term


def deterministic_rollout(generator, source: torch.Tensor, condition: dict) -> torch.Tensor:
    state = source
    endpoint = source
    for step in range(5):
        endpoint = generator(state, step=step, **condition)
        state = transition(state, endpoint, step, deterministic=True)
    return endpoint


