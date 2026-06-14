from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from ghxattn.cloth.model import GHXAttnMCID
from ghxattn.creel.assembly import iter_batches
from ghxattn.creel.batching import EyeRecord
from ghxattn.fulling.conformal import calibrate_site, prediction_interval
from ghxattn.inspection.concentration import concentration_ratio
from ghxattn.inspection.coverage import empirical_coverage, mean_width
from ghxattn.inspection.discrimination import fast_progression_auc, progression_labels
from ghxattn.inspection.equity import equity_scaled_auc
from ghxattn.inspection.pointwise import pointwise_mae
from ghxattn.inspection.significance import delong_ci

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class Predictions:
    slope_hat: FloatArray
    slope_true: FloatArray
    vf_hat: FloatArray
    vf_target: FloatArray
    race: IntArray
    concentration: float


@dataclass(frozen=True)
class SiteReport:
    site: str
    auc: float
    auc_lo: float
    auc_hi: float
    pmae: float
    es_auc: float
    coverage: float
    width: float
    concentration: float
    n_test: int


def collect_predictions(
    model: GHXAttnMCID,
    records: Sequence[EyeRecord],
    device: torch.device,
    batch_size: int = 32,
) -> Predictions:
    model.eval()
    slope_hat: list[float] = []
    slope_true: list[float] = []
    vf_hat: list[FloatArray] = []
    vf_target: list[FloatArray] = []
    race: list[int] = []
    concentration_sum = 0.0
    concentration_count = 0

    prior = model.prior().to(device)
    with torch.no_grad():
        for batch in iter_batches(records, batch_size, shuffle=False, seed=0):
            batch = batch.to(device)
            output = model(batch)
            slope_hat.extend(output.slope_hat.cpu().tolist())
            slope_true.extend(batch.slope.cpu().tolist())
            vf_hat.append(output.vf_hat.cpu().numpy())
            vf_target.append(batch.vf_target.cpu().numpy())
            race.extend(batch.race.cpu().tolist())
            if output.attention is not None:
                concentration_sum += concentration_ratio(output.attention, prior) * batch.batch_size
                concentration_count += batch.batch_size

    concentration = concentration_sum / concentration_count if concentration_count else float("nan")
    return Predictions(
        slope_hat=np.asarray(slope_hat, dtype=np.float64),
        slope_true=np.asarray(slope_true, dtype=np.float64),
        vf_hat=np.concatenate(vf_hat, axis=0).astype(np.float64),
        vf_target=np.concatenate(vf_target, axis=0).astype(np.float64),
        race=np.asarray(race, dtype=np.int64),
        concentration=concentration,
    )


def evaluate_site(
    model: GHXAttnMCID,
    site: str,
    site_index: int,
    calibration: Sequence[EyeRecord],
    test: Sequence[EyeRecord],
    alpha: float,
    tau: float,
    device: torch.device,
    batch_size: int = 32,
    use_conformal: bool = True,
) -> SiteReport:
    pred = collect_predictions(model, test, device, batch_size)
    labels = progression_labels(pred.slope_true)
    auc, lo, hi = delong_ci(-pred.slope_hat, labels)
    if np.isnan(auc):
        auc = fast_progression_auc(pred.slope_hat, pred.slope_true)

    if use_conformal:
        cal = collect_predictions(model, calibration, device, batch_size)
        calibration_result = calibrate_site(site_index, cal.slope_hat, cal.slope_true, alpha, tau)
        lower, upper = prediction_interval(pred.slope_hat, calibration_result.quantile)
        coverage = empirical_coverage(pred.slope_true, lower, upper)
        width = mean_width(lower, upper)
    else:
        coverage = float("nan")
        width = float("nan")

    return SiteReport(
        site=site,
        auc=auc,
        auc_lo=lo,
        auc_hi=hi,
        pmae=pointwise_mae(pred.vf_hat, pred.vf_target),
        es_auc=equity_scaled_auc(pred.slope_hat, pred.slope_true, pred.race),
        coverage=coverage,
        width=width,
        concentration=pred.concentration,
        n_test=len(test),
    )


def leave_site_out(
    run_holdout: Callable[[str], SiteReport],
    sites: Sequence[str],
) -> dict[str, SiteReport]:
    return {site: run_holdout(site) for site in sites}


def cross_site_gap(reports: dict[str, SiteReport]) -> float:
    aucs = [report.auc for report in reports.values() if not np.isnan(report.auc)]
    if len(aucs) < 2:
        return float("nan")
    return max(aucs) - min(aucs)
