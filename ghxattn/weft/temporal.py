from __future__ import annotations

import math

import torch
from torch import nn


class DeltaTimeEncoding(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even for sinusoidal time encoding")
        self.d_model = d_model
        exponent = torch.arange(0, d_model, 2, dtype=torch.float32) / d_model
        div_term = torch.exp(-math.log(10000.0) * exponent)
        self.register_buffer("div_term", div_term)
        self.div_term: torch.Tensor

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        angles = times.unsqueeze(-1) * self.div_term
        encoding = torch.zeros(*times.shape, self.d_model, device=times.device)
        encoding[..., 0::2] = torch.sin(angles)
        encoding[..., 1::2] = torch.cos(angles)
        return encoding
