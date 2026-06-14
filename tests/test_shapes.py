from __future__ import annotations

import torch

from ghxattn.cloth.model import build_model
from ghxattn.reed.prior import garway_heath_prior
from ghxattn.weft.temporal import DeltaTimeEncoding
from tests.helpers import oct_batch, tiny_hparams


def test_prior_shape() -> None:
    assert garway_heath_prior(28, 0.1).shape == (52, 28)


def test_temporal_encoding_shape() -> None:
    encoding = DeltaTimeEncoding(16)
    out = encoding(torch.zeros(3, 5))
    assert out.shape == (3, 5, 16)


def test_forward_shapes() -> None:
    model = build_model(tiny_hparams())
    model.eval()
    batch = oct_batch(6)
    with torch.no_grad():
        output = model(batch)
    assert output.vf_hat.shape == (6, 52)
    assert output.slope_hat.shape == (6,)
    assert output.latent.shape == (6, 16)
    assert output.attention is not None
    assert output.attention.shape[-2:] == (52, 28)
