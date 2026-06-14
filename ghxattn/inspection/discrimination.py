from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

from ghxattn.creel.bundles import MCID_SLOPE

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def progression_labels(slope_true: FloatArray) -> IntArray:
    return (slope_true <= MCID_SLOPE).astype(np.int64)


def safe_auc(labels: IntArray, scores: FloatArray) -> float:
    if np.unique(labels).shape[0] < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def fast_progression_auc(slope_hat: FloatArray, slope_true: FloatArray) -> float:
    labels = progression_labels(slope_true)
    return safe_auc(labels, -slope_hat)
