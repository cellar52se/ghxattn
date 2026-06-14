from __future__ import annotations

import math

import torch
from torch import nn


class CrossModalAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        lambda_init: float,
        rho: float,
        use_gh: bool,
        use_residual: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rho = rho
        self.use_gh = use_gh
        self.use_residual = use_residual
        self.prior_weight = nn.Parameter(torch.tensor(float(lambda_init)))
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.reshape(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        prior: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self._split(self.w_q(query_tokens))
        k = self._split(self.w_k(context_tokens))
        v = self._split(self.w_v(context_tokens))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        bias = self.prior_weight * prior if self.use_gh else torch.zeros_like(prior)
        constrained = torch.softmax(scores + bias, dim=-1)
        constrained = self.attn_dropout(constrained)
        fused = torch.matmul(constrained, v)

        if self.use_residual:
            residual = torch.softmax(scores, dim=-1)
            residual = self.attn_dropout(residual)
            fused = fused + self.rho * torch.matmul(residual, v)

        batch, _, length, _ = fused.shape
        fused = fused.transpose(1, 2).reshape(batch, length, self.d_model)
        return self.w_o(fused), constrained
