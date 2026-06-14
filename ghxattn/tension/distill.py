from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ghxattn.creel.bundles import RACES


class RaceConditionedTeacher(nn.Module):
    def __init__(self, raw_dim: int, d_model: int, seed: int = 20240602) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        projection = torch.randn(raw_dim, d_model, generator=generator) / (raw_dim**0.5)
        race_bias = torch.randn(len(RACES), d_model, generator=generator) * 0.1
        self.register_buffer("projection", projection)
        self.register_buffer("race_bias", race_bias)
        self.projection: torch.Tensor
        self.race_bias: torch.Tensor

    def forward(self, oct_raw: torch.Tensor, race: torch.Tensor) -> torch.Tensor:
        pooled = oct_raw.mean(dim=1)
        projected = pooled @ self.projection
        return projected + self.race_bias[race]


def fairdist_kl(student_latent: torch.Tensor, teacher_latent: torch.Tensor) -> torch.Tensor:
    student = F.softmax(student_latent, dim=-1)
    teacher_log = F.log_softmax(teacher_latent, dim=-1)
    return F.kl_div(teacher_log, student, reduction="batchmean")
