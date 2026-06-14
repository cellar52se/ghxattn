from __future__ import annotations

import torch
from torch import nn


class SiteAwareBatchNorm(nn.Module):
    def __init__(self, features: int, n_buckets: int) -> None:
        super().__init__()
        self.features = features
        self.n_buckets = n_buckets
        self.norms = nn.ModuleList([nn.BatchNorm1d(features) for _ in range(n_buckets)])

    def forward(self, x: torch.Tensor, vendor: torch.Tensor) -> torch.Tensor:
        batch, tokens, features = x.shape
        out = torch.zeros_like(x)
        for bucket in range(self.n_buckets):
            selector = vendor == bucket
            if not bool(selector.any()):
                continue
            subset = x[selector].reshape(-1, features)
            normed = self.norms[bucket](subset)
            out[selector] = normed.reshape(-1, tokens, features)
        return out
