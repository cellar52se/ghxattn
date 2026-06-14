from __future__ import annotations

import math

import torch


def concentration_ratio(attention: torch.Tensor, prior: torch.Tensor) -> float:
    weights = attention
    if weights.dim() == 4:
        weights = weights.mean(dim=(0, 1))
    elif weights.dim() == 3:
        weights = weights.mean(dim=0)
    positive = (prior >= 1.0).to(weights.dtype)
    per_location = (weights * positive).sum(dim=-1)
    return float(per_location.mean())


def rademacher_excess(
    gh_prior: torch.Tensor,
    alternative_prior: torch.Tensor,
    n_samples: int,
    lipschitz: float = 1.0,
) -> float:
    dimension = gh_prior.shape[0] * gh_prior.shape[1]
    gap = lipschitz * (
        float(torch.linalg.norm(alternative_prior)) - float(torch.linalg.norm(gh_prior))
    )
    return gap * math.sqrt(math.log(dimension) / n_samples)
