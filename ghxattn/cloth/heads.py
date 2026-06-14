from __future__ import annotations

import torch
from torch import nn


class VisualFieldDecoder(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.net(fused).squeeze(-1)


class SlopeHead(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        pooled = fused.mean(dim=1)
        return self.net(pooled).squeeze(-1)
