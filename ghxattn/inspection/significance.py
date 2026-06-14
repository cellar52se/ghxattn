from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.stats import binomtest, norm

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
MetricFn = Callable[[FloatArray, FloatArray], float]


def mcnemar_exact(correct_a: NDArray[np.bool_], correct_b: NDArray[np.bool_]) -> float:
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c
    if n == 0:
        return 1.0
    return float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)


def compute_midrank(x: FloatArray) -> FloatArray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = x.shape[0]
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def delong_variance(scores: FloatArray, labels: IntArray) -> tuple[float, float]:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    m = positive.shape[0]
    n = negative.shape[0]
    tx = compute_midrank(positive)
    ty = compute_midrank(negative)
    tz = compute_midrank(np.concatenate([positive, negative]))
    auc = (tz[:m].sum() / m - (m + 1) / 2.0) / n
    v01 = (tz[:m] - tx) / n
    v10 = 1.0 - (tz[m:] - ty) / m
    variance = float(np.var(v01, ddof=1) / m + np.var(v10, ddof=1) / n)
    return float(auc), variance


def delong_ci(
    scores: FloatArray, labels: IntArray, alpha: float = 0.05
) -> tuple[float, float, float]:
    if np.unique(labels).shape[0] < 2:
        return (float("nan"), float("nan"), float("nan"))
    auc, variance = delong_variance(scores, labels)
    se = float(np.sqrt(max(variance, 0.0)))
    z = float(norm.ppf(1.0 - alpha / 2.0))
    return (auc, auc - z * se, auc + z * se)


def stratified_bootstrap(
    metric_fn: MetricFn,
    slope_hat: FloatArray,
    slope_true: FloatArray,
    strata: IntArray,
    n_resamples: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    index_by_stratum = {label: np.where(strata == label)[0] for label in np.unique(strata)}
    values: list[float] = []
    for _ in range(n_resamples):
        chunks = [
            rng.choice(idx, size=idx.shape[0], replace=True) for idx in index_by_stratum.values()
        ]
        sample = np.concatenate(chunks)
        value = metric_fn(slope_hat[sample], slope_true[sample])
        if not np.isnan(value):
            values.append(value)
    if not values:
        return (float("nan"), float("nan"), float("nan"))
    arr = np.asarray(values)
    lo = float(np.percentile(arr, 100 * alpha / 2))
    hi = float(np.percentile(arr, 100 * (1 - alpha / 2)))
    return (float(np.mean(arr)), lo, hi)
