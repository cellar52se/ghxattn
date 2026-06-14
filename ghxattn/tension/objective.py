from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ghxattn.cloth.model import ModelOutput
from ghxattn.creel.batching import EyeBatch
from ghxattn.tension.distill import fairdist_kl


@dataclass(frozen=True)
class LossTerms:
    total: torch.Tensor
    prediction: torch.Tensor
    site: torch.Tensor
    distillation: torch.Tensor


def prediction_loss(
    vf_hat: torch.Tensor,
    vf_target: torch.Tensor,
    slope_hat: torch.Tensor,
    slope: torch.Tensor,
) -> torch.Tensor:
    return F.mse_loss(vf_hat, vf_target) + F.mse_loss(slope_hat, slope)


class CompositeLoss(nn.Module):
    def __init__(self, lambda_site: float, lambda_kd: float) -> None:
        super().__init__()
        self.lambda_site = lambda_site
        self.lambda_kd = lambda_kd

    def forward(
        self,
        output: ModelOutput,
        batch: EyeBatch,
        site_logits: torch.Tensor,
        teacher_latent: torch.Tensor,
    ) -> LossTerms:
        l_pred = prediction_loss(output.vf_hat, batch.vf_target, output.slope_hat, batch.slope)
        l_site = F.cross_entropy(site_logits, batch.site)
        l_kd = fairdist_kl(output.latent, teacher_latent)
        total = l_pred + self.lambda_site * l_site + self.lambda_kd * l_kd
        return LossTerms(total=total, prediction=l_pred, site=l_site, distillation=l_kd)
