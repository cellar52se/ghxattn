from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def empirical_coverage(slope_true: FloatArray, lower: FloatArray, upper: FloatArray) -> float:
    inside = (slope_true >= lower) & (slope_true <= upper)
    return float(np.mean(inside))


def mean_width(lower: FloatArray, upper: FloatArray) -> float:
    return float(np.mean(upper - lower))


def expected_calibration_error(
    probabilities: FloatArray, labels: FloatArray, bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    total = probabilities.shape[0]
    if total == 0:
        return float("nan")
    for index in range(bins):
        lo, hi = edges[index], edges[index + 1]
        selector = (probabilities >= lo) & (
            probabilities < hi if index < bins - 1 else probabilities <= hi
        )
        count = int(np.sum(selector))
        if count == 0:
            continue
        confidence = float(np.mean(probabilities[selector]))
        accuracy = float(np.mean(labels[selector]))
        error += (count / total) * abs(accuracy - confidence)
    return error
