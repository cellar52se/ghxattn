from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ghxattn.inspection.discrimination import fast_progression_auc

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def group_aucs(slope_hat: FloatArray, slope_true: FloatArray, group: IntArray) -> list[float]:
    values: list[float] = []
    for label in np.unique(group):
        mask = group == label
        auc = fast_progression_auc(slope_hat[mask], slope_true[mask])
        if not np.isnan(auc):
            values.append(auc)
    return values


def equity_scaled_auc(
    slope_hat: FloatArray,
    slope_true: FloatArray,
    group: IntArray,
    beta: float = 0.5,
) -> float:
    raw = fast_progression_auc(slope_hat, slope_true)
    if np.isnan(raw):
        return float("nan")
    aucs = group_aucs(slope_hat, slope_true, group)
    if not aucs:
        return float("nan")
    return (1.0 - beta) * min(aucs) + beta * raw
