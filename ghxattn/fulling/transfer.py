from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def levy_prokhorov_distance(source: FloatArray, target: FloatArray, grid: int = 256) -> float:
    if source.shape[0] == 0 or target.shape[0] == 0:
        return float("nan")
    lo = float(min(source.min(), target.min()))
    hi = float(max(source.max(), target.max()))
    if hi <= lo:
        return 0.0
    points = np.linspace(lo, hi, grid)
    src_cdf = np.searchsorted(np.sort(source), points, side="right") / source.shape[0]
    tgt_cdf = np.searchsorted(np.sort(target), points, side="right") / target.shape[0]
    return float(np.max(np.abs(src_cdf - tgt_cdf)))


def coverage_transfer_bound(source_coverage: float, delta: float) -> float:
    return max(0.0, source_coverage - 2.0 * delta)


def equity_coverage_transfer(
    source_coverage: float,
    strata_deltas: FloatArray,
    marginal_delta: float,
    beta: float = 0.83,
) -> float:
    worst = float(np.max(strata_deltas)) if strata_deltas.shape[0] > 0 else 0.0
    return max(0.0, source_coverage - 2.0 * worst - beta * marginal_delta)
