from __future__ import annotations

import torch
from torch import nn


class GradientReversal(nn.Module):
    def __init__(self, beta: float = 0.0) -> None:
        super().__init__()
        self.beta = beta

    def set_beta(self, beta: float) -> None:
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.requires_grad:
            beta = self.beta
            x.register_hook(lambda grad: -beta * grad)
        return x


class SiteDiscriminator(nn.Module):
    def __init__(self, d_model: int, n_sites: int, hidden: int = 128) -> None:
        super().__init__()
        self.reversal = GradientReversal()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_sites),
        )

    def set_beta(self, beta: float) -> None:
        self.reversal.set_beta(beta)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(self.reversal(latent))
