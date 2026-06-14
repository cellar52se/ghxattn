from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def nonconformity(slope_hat: FloatArray, slope_true: FloatArray) -> FloatArray:
    return np.abs(slope_hat - slope_true)


def quantile_index(n: int, alpha: float) -> int:
    return int(math.ceil((1.0 - alpha) * (n + 1)))


def conformal_quantile(residuals: FloatArray, alpha: float) -> float:
    ordered = np.sort(residuals)
    n = ordered.shape[0]
    if n == 0:
        return float("inf")
    rank = quantile_index(n, alpha)
    if rank > n:
        return float(ordered[-1])
    return float(ordered[rank - 1])


@dataclass(frozen=True)
class SiteCalibration:
    site: int
    quantile: float
    n_calibration: int


def calibrate_site(
    site: int,
    slope_hat: FloatArray,
    slope_true: FloatArray,
    alpha: float,
    tau: float,
) -> SiteCalibration:
    mask = np.abs(slope_true) >= tau
    residuals = nonconformity(slope_hat[mask], slope_true[mask])
    quantile = conformal_quantile(residuals, alpha)
    return SiteCalibration(site=site, quantile=quantile, n_calibration=int(mask.sum()))


def prediction_interval(slope_hat: FloatArray, quantile: float) -> tuple[FloatArray, FloatArray]:
    return slope_hat - quantile, slope_hat + quantile
