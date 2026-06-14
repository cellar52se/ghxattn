from __future__ import annotations

import torch
from torch import nn

from ghxattn.creel.bundles import VF_LOCATIONS
from ghxattn.weft.temporal import DeltaTimeEncoding


class FunctionalEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        use_temporal_dt: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.use_temporal_dt = use_temporal_dt
        self.value_embed = nn.Linear(1, d_model)
        self.loc_embed = nn.Embedding(VF_LOCATIONS, d_model)
        self.temporal = DeltaTimeEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)

    def forward(
        self,
        vf_seq: torch.Tensor,
        dt: torch.Tensor,
        visit_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, visits, locations = vf_seq.shape
        if self.use_temporal_dt:
            times = torch.cumsum(dt, dim=1)
        else:
            times = torch.arange(visits, device=vf_seq.device, dtype=vf_seq.dtype).expand(
                batch, visits
            )

        value = self.value_embed(vf_seq.unsqueeze(-1))
        temporal = self.temporal(times).unsqueeze(2)
        loc_ids = torch.arange(locations, device=vf_seq.device)
        location = self.loc_embed(loc_ids).reshape(1, 1, locations, self.d_model)
        tokens = value + temporal + location

        tokens = tokens.permute(0, 2, 1, 3).reshape(batch * locations, visits, self.d_model)
        pad = ~visit_mask.unsqueeze(1).expand(batch, locations, visits)
        pad = pad.reshape(batch * locations, visits)
        encoded = self.encoder(tokens, src_key_padding_mask=pad)

        weight = visit_mask.unsqueeze(1).expand(batch, locations, visits)
        weight = weight.reshape(batch * locations, visits, 1).to(encoded.dtype)
        pooled = (encoded * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)
        return pooled.reshape(batch, locations, self.d_model)
