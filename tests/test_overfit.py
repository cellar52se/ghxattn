from __future__ import annotations

import torch

from ghxattn.cloth.model import build_model
from ghxattn.tension.objective import prediction_loss
from tests.helpers import oct_batch, tiny_hparams


def test_overfit_single_batch() -> None:
    torch.manual_seed(0)
    model = build_model(tiny_hparams(d_model=32))
    batch = oct_batch(8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()

    output = model(batch)
    first = float(
        prediction_loss(output.vf_hat, batch.vf_target, output.slope_hat, batch.slope).detach()
    )
    for _ in range(250):
        output = model(batch)
        loss = prediction_loss(output.vf_hat, batch.vf_target, output.slope_hat, batch.slope)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert float(loss.detach()) < 0.2 * first
