from __future__ import annotations

import torch

from ghxattn.cloth.model import build_model
from ghxattn.tension.adversary import GradientReversal, SiteDiscriminator
from ghxattn.tension.distill import RaceConditionedTeacher
from ghxattn.tension.objective import CompositeLoss
from tests.helpers import oct_batch, tiny_hparams


def test_gradient_reversal_flips_sign() -> None:
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    reversal = GradientReversal(1.0)
    reversal(x).sum().backward()
    assert x.grad is not None
    assert torch.allclose(x.grad, -torch.ones(3))


def test_all_core_params_receive_grad() -> None:
    torch.manual_seed(0)
    hparams = tiny_hparams()
    model = build_model(hparams)
    discriminator = SiteDiscriminator(hparams.d_model, 3)
    discriminator.set_beta(1.0)
    teacher = RaceConditionedTeacher(hparams.raw_dim, hparams.d_model)
    composite = CompositeLoss(0.1, 0.1)

    batch = oct_batch(6)
    output = model(batch)
    site_logits = discriminator(output.latent)
    teacher_latent = teacher(batch.oct_raw, batch.race)
    terms = composite(output, batch, site_logits, teacher_latent)
    terms.total.backward()

    for name, parameter in model.named_parameters():
        if any(skip in name for skip in ("concat_fusion", "site_bn", "global_bn")):
            continue
        assert parameter.grad is not None, name
