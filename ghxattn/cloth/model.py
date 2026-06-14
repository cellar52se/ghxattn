from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ghxattn.cloth.heads import SlopeHead, VisualFieldDecoder
from ghxattn.creel.batching import EyeBatch
from ghxattn.reed.prior import TopographicPrior
from ghxattn.shuttle.attention import CrossModalAttention
from ghxattn.warp.encoder import build_structural_encoder
from ghxattn.weft.encoder import FunctionalEncoder


@dataclass(frozen=True)
class ModelHParams:
    d_model: int = 768
    n_oct: int = 196
    raw_dim: int = 32
    structural_kind: str = "frozen_tokeniser"
    cross_heads: int = 12
    weft_heads: int = 8
    weft_layers: int = 4
    lambda_init: float = 1.0
    rho: float = 0.3
    gh_epsilon: float = 0.1
    dropout: float = 0.1
    attn_dropout: float = 0.2
    use_gh: bool = True
    use_residual: bool = True
    use_site_bn: bool = True
    use_temporal_dt: bool = True
    fusion_mode: str = "xattn"
    branch: str = "full"


@dataclass(frozen=True)
class ModelOutput:
    vf_hat: torch.Tensor
    slope_hat: torch.Tensor
    latent: torch.Tensor
    attention: torch.Tensor | None


class GHXAttnMCID(nn.Module):
    def __init__(self, hparams: ModelHParams) -> None:
        super().__init__()
        self.hparams = hparams
        self.functional = FunctionalEncoder(
            hparams.d_model,
            hparams.weft_heads,
            hparams.weft_layers,
            hparams.dropout,
            hparams.use_temporal_dt,
        )
        self.structural = build_structural_encoder(
            hparams.structural_kind,
            hparams.raw_dim,
            hparams.d_model,
            hparams.dropout,
            hparams.use_site_bn,
        )
        self.prior = TopographicPrior(hparams.n_oct, hparams.gh_epsilon)
        self.attention = CrossModalAttention(
            hparams.d_model,
            hparams.cross_heads,
            hparams.lambda_init,
            hparams.rho,
            hparams.use_gh,
            hparams.use_residual,
            hparams.attn_dropout,
        )
        self.concat_fusion = nn.Sequential(
            nn.Linear(2 * hparams.d_model, hparams.d_model),
            nn.Sigmoid(),
        )
        self.vf_head = VisualFieldDecoder(hparams.d_model, hparams.dropout)
        self.slope_head = SlopeHead(hparams.d_model, hparams.dropout)

    def encode_structural(self, batch: EyeBatch) -> torch.Tensor:
        return self.structural(batch.oct_raw, batch.vendor, batch.has_oct)

    def forward(self, batch: EyeBatch) -> ModelOutput:
        func_tokens = self.functional(batch.vf_seq, batch.dt, batch.visit_mask)
        attention: torch.Tensor | None = None

        if self.hparams.branch == "functional":
            fused = func_tokens
            latent = func_tokens.mean(dim=1)
        else:
            struct_tokens = self.encode_structural(batch)
            latent = struct_tokens.mean(dim=1)
            if self.hparams.branch == "structural":
                fused = latent.unsqueeze(1).expand(-1, func_tokens.shape[1], -1)
            elif self.hparams.fusion_mode == "concat":
                struct_broadcast = latent.unsqueeze(1).expand(-1, func_tokens.shape[1], -1)
                gate = self.concat_fusion(torch.cat([func_tokens, struct_broadcast], dim=-1))
                fused = gate * func_tokens + (1.0 - gate) * struct_broadcast
            else:
                attn_out, attention = self.attention(func_tokens, struct_tokens, self.prior())
                gate = batch.has_oct.reshape(-1, 1, 1)
                fused = torch.where(gate, attn_out, func_tokens)

        vf_hat = self.vf_head(fused)
        slope_hat = self.slope_head(fused)
        return ModelOutput(vf_hat=vf_hat, slope_hat=slope_hat, latent=latent, attention=attention)


def build_model(hparams: ModelHParams) -> GHXAttnMCID:
    return GHXAttnMCID(hparams)
