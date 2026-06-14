from __future__ import annotations

import torch
from torch import nn

from ghxattn.creel.bundles import VF_LOCATIONS, is_adjacent, structural_bundle_ids, vf_bundle_ids


def garway_heath_prior(n_oct: int, epsilon: float) -> torch.Tensor:
    vf_ids = vf_bundle_ids()
    oct_ids = structural_bundle_ids(n_oct)
    mask = torch.zeros(VF_LOCATIONS, n_oct, dtype=torch.float32)
    for i, vf_bundle in enumerate(vf_ids):
        for j, oct_bundle in enumerate(oct_ids):
            if vf_bundle == oct_bundle:
                mask[i, j] = 1.0
            elif is_adjacent(vf_bundle, oct_bundle):
                mask[i, j] = epsilon
    return mask


class TopographicPrior(nn.Module):
    def __init__(self, n_oct: int, epsilon: float) -> None:
        super().__init__()
        self.n_oct = n_oct
        self.epsilon = epsilon
        self.register_buffer("mask", garway_heath_prior(n_oct, epsilon))
        self.mask: torch.Tensor

    def forward(self) -> torch.Tensor:
        return self.mask
