from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

import torch

from ghxattn.cloth.model import GHXAttnMCID, build_model
from ghxattn.creel.assembly import assemble_splits
from ghxattn.creel.bundles import SITES, site_index
from ghxattn.dressing.schema import RootConfig
from ghxattn.fulling.conformal import calibrate_site, prediction_interval
from ghxattn.inspection.discrimination import progression_labels
from ghxattn.inspection.protocol import SiteReport, collect_predictions, evaluate_site
from ghxattn.loom.checkpoint import load_checkpoint, save_atomic
from ghxattn.loom.seeding import set_seed
from ghxattn.loom.trainer import Trainer, TrainResult

logger = logging.getLogger("ghxattn.mill")


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _checkpoint_path(cfg: RootConfig) -> Path:
    return Path(cfg.output_dir) / cfg.data.holdout / f"seed{cfg.seed}.ckpt"


def _load_model(cfg: RootConfig, device: torch.device) -> GHXAttnMCID:
    path = _checkpoint_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint at {path}; run weave first")
    state = load_checkpoint(path)
    model = build_model(cfg.model.to_hparams())
    model.load_state_dict(state["model"])
    model.to(device)
    return model


def weave(cfg: RootConfig) -> TrainResult:
    set_seed(cfg.seed)
    device = _device()
    spec = cfg.data.to_spec(cfg.model)
    splits = assemble_splits(
        spec,
        cfg.seed,
        cfg.data.n_per_site,
        cfg.data.holdout,
        cfg.data.cal_fraction,
        cfg.data.pretrain_eyes,
    )
    model = build_model(cfg.model.to_hparams())
    trainer = Trainer(model, cfg.train.to_hparams(cfg.seed), len(SITES), device)
    trainer.pretrain(splits.pretrain)

    holdout_size = max(1, len(splits.train) // 5)
    fit_train = splits.train[:-holdout_size]
    val = splits.train[-holdout_size:]
    result = trainer.fit(fit_train, val)

    state = {
        "model": model.state_dict(),
        "model_hparams": asdict(model.hparams),
        "seed": cfg.seed,
        "holdout": cfg.data.holdout,
    }
    save_atomic(_checkpoint_path(cfg), state)
    logger.info(
        "weave done holdout=%s best_val=%.4f epochs=%d",
        cfg.data.holdout,
        result.best_val_mse,
        result.epochs_run,
    )
    return result


def inspect(cfg: RootConfig) -> SiteReport:
    device = _device()
    spec = cfg.data.to_spec(cfg.model)
    splits = assemble_splits(
        spec, cfg.seed, cfg.data.n_per_site, cfg.data.holdout, cfg.data.cal_fraction
    )
    model = _load_model(cfg, device)
    report = evaluate_site(
        model,
        cfg.data.holdout,
        site_index(cfg.data.holdout),
        splits.calibration,
        splits.test,
        cfg.experiment.alpha,
        cfg.experiment.tau,
        device,
        cfg.experiment.eval_batch_size,
        cfg.experiment.use_conformal,
    )
    logger.info(
        "inspect %s auc=%.3f pmae=%.3f coverage=%.3f es_auc=%.3f concentration=%.3f",
        report.site,
        report.auc,
        report.pmae,
        report.coverage,
        report.es_auc,
        report.concentration,
    )
    return report


def forecast(cfg: RootConfig) -> list[tuple[float, float, float, int]]:
    device = _device()
    spec = cfg.data.to_spec(cfg.model)
    splits = assemble_splits(
        spec, cfg.seed, cfg.data.n_per_site, cfg.data.holdout, cfg.data.cal_fraction
    )
    model = _load_model(cfg, device)

    calibration = collect_predictions(
        model, splits.calibration, device, cfg.experiment.eval_batch_size
    )
    result = calibrate_site(
        site_index(cfg.data.holdout),
        calibration.slope_hat,
        calibration.slope_true,
        cfg.experiment.alpha,
        cfg.experiment.tau,
    )
    sample = splits.test[:5]
    predictions = collect_predictions(model, sample, device, cfg.experiment.eval_batch_size)
    lower, upper = prediction_interval(predictions.slope_hat, result.quantile)
    flags = progression_labels(predictions.slope_hat)

    rows: list[tuple[float, float, float, int]] = []
    for index in range(predictions.slope_hat.shape[0]):
        row = (
            float(predictions.slope_hat[index]),
            float(lower[index]),
            float(upper[index]),
            int(flags[index]),
        )
        rows.append(row)
        logger.info(
            "forecast eye=%d slope=%.3f interval=[%.3f, %.3f] escalate=%d",
            index,
            row[0],
            row[1],
            row[2],
            row[3],
        )
    return rows


def export(cfg: RootConfig) -> Path:
    device = _device()
    model = _load_model(cfg, device)
    out = Path(cfg.output_dir) / "export" / f"{cfg.data.holdout}_seed{cfg.seed}.pt"
    save_atomic(out, {"model": model.state_dict(), "model_hparams": asdict(model.hparams)})
    logger.info("export -> %s", out)
    return out
