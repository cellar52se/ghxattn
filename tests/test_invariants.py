from __future__ import annotations

import torch

from ghxattn.inspection.concentration import concentration_ratio, rademacher_excess
from ghxattn.reed.prior import garway_heath_prior
from ghxattn.shuttle.attention import CrossModalAttention


def test_concentration_monotonic_in_lambda() -> None:
    torch.manual_seed(0)
    attention = CrossModalAttention(16, 4, 0.0, 0.3, use_gh=True, use_residual=False, dropout=0.0)
    attention.eval()
    query = torch.randn(2, 52, 16)
    context = torch.randn(2, 28, 16)
    prior = garway_heath_prior(28, 0.1)
    values = []
    for lam in [0.0, 1.0, 5.0]:
        with torch.no_grad():
            attention.prior_weight.fill_(lam)
            _, alpha = attention(query, context, prior)
            values.append(concentration_ratio(alpha, prior))
    assert values[0] < values[1] < values[2]


def test_rademacher_excess_positive_for_dense_prior() -> None:
    gh = garway_heath_prior(28, 0.1)
    dense = torch.full_like(gh, 0.1)
    dense[gh == 1.0] = 1.0
    assert rademacher_excess(gh, dense, 800) > 0.0
