from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from ghxattn.loom.schedule import cosine_warmup_factor


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    lr: float,
    weight_decay: float,
) -> AdamW:
    return AdamW(parameters, lr=lr, weight_decay=weight_decay)


def build_scheduler(optimizer: AdamW, warmup_steps: int, total_steps: int) -> LambdaLR:
    def factor(step: int) -> float:
        return cosine_warmup_factor(step, warmup_steps, total_steps)

    return LambdaLR(optimizer, lr_lambda=factor)
