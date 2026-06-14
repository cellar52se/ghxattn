from __future__ import annotations

import torch

from ghxattn.cloth.model import build_model
from ghxattn.loom.seeding import set_seed
from ghxattn.warp.site_norm import SiteAwareBatchNorm
from tests.helpers import oct_batch, tiny_hparams


def _seeded_output() -> torch.Tensor:
    set_seed(123)
    model = build_model(tiny_hparams())
    model.eval()
    batch = oct_batch(6)
    with torch.no_grad():
        return model(batch).slope_hat


def test_seeded_output_is_reproducible() -> None:
    assert torch.allclose(_seeded_output(), _seeded_output())


def test_site_bn_keeps_vendor_stats_independent() -> None:
    torch.manual_seed(0)
    norm = SiteAwareBatchNorm(8, 3)
    norm.train()
    x = torch.randn(10, 4, 8)
    x[5:] += 10.0
    vendor = torch.tensor([0] * 5 + [1] * 5)
    norm(x, vendor)
    assert not torch.allclose(norm.norms[0].running_mean, norm.norms[1].running_mean)
