from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from ghxattn.cloth.model import GHXAttnMCID
from ghxattn.creel.assembly import iter_batches
from ghxattn.creel.batching import EyeRecord
from ghxattn.loom.optim import build_optimizer, build_scheduler
from ghxattn.loom.schedule import grad_reversal_beta
from ghxattn.tension.adversary import SiteDiscriminator
from ghxattn.tension.distill import RaceConditionedTeacher, fairdist_kl
from ghxattn.tension.objective import CompositeLoss

logger = logging.getLogger("ghxattn.loom")


@dataclass(frozen=True)
class TrainHParams:
    lr: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 32
    max_epochs: int = 150
    warmup_steps: int = 1000
    lambda_site: float = 0.1
    lambda_kd: float = 0.1
    beta_ramp_epochs: int = 10
    pretrain_epochs: int = 5
    early_stop_patience: int = 10
    grad_clip: float = 1.0
    seed: int = 42


@dataclass(frozen=True)
class EpochLog:
    epoch: int
    total: float
    prediction: float
    site: float
    distillation: float
    val_mse: float
    beta: float


@dataclass(frozen=True)
class TrainResult:
    best_val_mse: float
    epochs_run: int
    history: list[EpochLog]


class Trainer:
    def __init__(
        self,
        model: GHXAttnMCID,
        hparams: TrainHParams,
        n_sites: int,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.hparams = hparams
        self.device = device
        self.composite = CompositeLoss(hparams.lambda_site, hparams.lambda_kd).to(device)
        self.discriminator = SiteDiscriminator(model.hparams.d_model, n_sites).to(device)
        self.teacher = RaceConditionedTeacher(model.hparams.raw_dim, model.hparams.d_model).to(
            device
        )

    def _clip(self, parameters: Sequence[torch.nn.Parameter]) -> None:
        if self.hparams.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(parameters, self.hparams.grad_clip)

    def pretrain(self, records: Sequence[EyeRecord]) -> None:
        if not records:
            return
        params = list(self.model.structural.parameters())
        optimizer = build_optimizer(params, self.hparams.lr, self.hparams.weight_decay)
        self.model.train()
        for epoch in range(self.hparams.pretrain_epochs):
            for batch in iter_batches(
                records, self.hparams.batch_size, True, self.hparams.seed + epoch
            ):
                batch = batch.to(self.device)
                latent = self.model.encode_structural(batch).mean(dim=1)
                teacher_latent = self.teacher(batch.oct_raw, batch.race)
                loss = fairdist_kl(latent, teacher_latent)
                optimizer.zero_grad()
                loss.backward()
                self._clip(params)
                optimizer.step()

    def _validate(self, records: Sequence[EyeRecord]) -> float:
        self.model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch in iter_batches(records, self.hparams.batch_size, False, 0):
                batch = batch.to(self.device)
                output = self.model(batch)
                total += float(F.mse_loss(output.slope_hat, batch.slope, reduction="sum"))
                count += batch.batch_size
        return total / count if count else float("nan")

    def _snapshot(self) -> dict[str, Any]:
        return {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}

    def fit(self, train: Sequence[EyeRecord], val: Sequence[EyeRecord]) -> TrainResult:
        params = list(self.model.parameters()) + list(self.discriminator.parameters())
        optimizer = build_optimizer(params, self.hparams.lr, self.hparams.weight_decay)
        steps_per_epoch = max(1, math.ceil(len(train) / self.hparams.batch_size))
        scheduler = build_scheduler(
            optimizer, self.hparams.warmup_steps, steps_per_epoch * self.hparams.max_epochs
        )

        best_val = float("inf")
        best_state = self._snapshot()
        patience = self.hparams.early_stop_patience
        history: list[EpochLog] = []
        epochs_run = 0

        for epoch in range(self.hparams.max_epochs):
            epochs_run = epoch + 1
            self.model.train()
            beta = grad_reversal_beta(epoch, self.hparams.beta_ramp_epochs)
            self.discriminator.set_beta(beta)
            sums = [0.0, 0.0, 0.0, 0.0]
            batches = 0
            for batch in iter_batches(
                train, self.hparams.batch_size, True, self.hparams.seed + epoch
            ):
                batch = batch.to(self.device)
                output = self.model(batch)
                site_logits = self.discriminator(output.latent)
                teacher_latent = self.teacher(batch.oct_raw, batch.race)
                terms = self.composite(output, batch, site_logits, teacher_latent)
                optimizer.zero_grad()
                terms.total.backward()
                self._clip(params)
                optimizer.step()
                scheduler.step()
                sums[0] += float(terms.total.detach())
                sums[1] += float(terms.prediction.detach())
                sums[2] += float(terms.site.detach())
                sums[3] += float(terms.distillation.detach())
                batches += 1

            val_mse = self._validate(val) if val else sums[1] / max(1, batches)
            history.append(
                EpochLog(
                    epoch=epoch,
                    total=sums[0] / max(1, batches),
                    prediction=sums[1] / max(1, batches),
                    site=sums[2] / max(1, batches),
                    distillation=sums[3] / max(1, batches),
                    val_mse=val_mse,
                    beta=beta,
                )
            )
            logger.info(
                "epoch=%d total=%.4f val_mse=%.4f beta=%.2f",
                epoch,
                sums[0] / max(1, batches),
                val_mse,
                beta,
            )

            if val_mse < best_val - 1e-6:
                best_val = val_mse
                best_state = self._snapshot()
                patience = self.hparams.early_stop_patience
            else:
                patience -= 1
                if patience <= 0:
                    break

        self.model.load_state_dict(best_state)
        return TrainResult(best_val_mse=best_val, epochs_run=epochs_run, history=history)
