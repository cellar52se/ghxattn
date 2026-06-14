from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ghxattn.creel.bundles import VENDORS
from ghxattn.warp.site_norm import SiteAwareBatchNorm


def _orthonormal(raw_dim: int, d_model: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(max(raw_dim, d_model), d_model, generator=generator)
    q, _ = torch.linalg.qr(base)
    return q[:raw_dim, :d_model].contiguous()


class FrozenTokeniser(nn.Module):
    def __init__(self, raw_dim: int, d_model: int, seed: int = 20240601) -> None:
        super().__init__()
        self.register_buffer("projection", _orthonormal(raw_dim, d_model, seed))
        self.projection: torch.Tensor

    def forward(self, oct_raw: torch.Tensor) -> torch.Tensor:
        return oct_raw @ self.projection


class StructuralEncoder(nn.Module):
    def __init__(
        self, raw_dim: int, d_model: int, dropout: float, use_site_bn: bool = True
    ) -> None:
        super().__init__()
        self.use_site_bn = use_site_bn
        self.tokeniser = FrozenTokeniser(raw_dim, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.site_bn = SiteAwareBatchNorm(d_model, len(VENDORS) + 1)
        self.global_bn = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(
        self,
        oct_raw: torch.Tensor,
        vendor: torch.Tensor,
        has_oct: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self.tokeniser(oct_raw)
        tokens = self.activation(self.proj(tokens))
        if self.use_site_bn:
            tokens = self.site_bn(tokens, vendor)
        else:
            batch, length, features = tokens.shape
            tokens = self.global_bn(tokens.reshape(-1, features)).reshape(batch, length, features)
        tokens = self.dropout(tokens)
        gate = has_oct.to(tokens.dtype).reshape(-1, 1, 1)
        return tokens * gate


class RetfoundAdapter(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as error:
            raise RuntimeError(
                "RetfoundAdapter requires the optional 'retfound' extra (timm) and the "
                "RETFound checkpoint; install it or use the frozen tokeniser path"
            ) from error
        self.backbone: Any = timm.create_model(
            "vit_large_patch16_224", pretrained=False, num_classes=0
        )
        self.project = nn.Linear(self.backbone.num_features, d_model)

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(volume)
        return self.project(features)


def build_structural_encoder(
    kind: str,
    raw_dim: int,
    d_model: int,
    dropout: float,
    use_site_bn: bool = True,
) -> nn.Module:
    if kind == "frozen_tokeniser":
        return StructuralEncoder(raw_dim, d_model, dropout, use_site_bn)
    if kind == "retfound":
        return RetfoundAdapter(d_model)
    raise ValueError(f"unknown structural encoder kind: {kind}")
