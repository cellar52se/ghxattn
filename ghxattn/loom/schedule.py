from __future__ import annotations

import math


def grad_reversal_beta(epoch: int, ramp_epochs: int) -> float:
    if ramp_epochs <= 0:
        return 1.0
    return min(1.0, epoch / ramp_epochs)


def cosine_warmup_factor(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps and warmup_steps > 0:
        return step / warmup_steps
    if total_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
