from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from ghxattn.inspection.discrimination import fast_progression_auc, progression_labels
from ghxattn.inspection.equity import equity_scaled_auc
from ghxattn.inspection.pointwise import per_bundle_mae, pointwise_mae
from ghxattn.inspection.significance import delong_ci


def test_auc_matches_sklearn() -> None:
    rng = np.random.default_rng(0)
    slope_true = rng.normal(-0.5, 0.5, 200)
    slope_hat = slope_true + rng.normal(0, 0.2, 200)
    labels = progression_labels(slope_true)
    expected = float(roc_auc_score(labels, -slope_hat))
    assert abs(fast_progression_auc(slope_hat, slope_true) - expected) < 1e-9


def test_es_auc_degenerate_under_perfect_equity() -> None:
    slope_true = np.array([-1.0, -1.0, 0.0, 0.0] * 3)
    slope_hat = np.array([-1.0, -1.0, 1.0, 1.0] * 3)
    group = np.repeat(np.arange(3), 4)
    raw = fast_progression_auc(slope_hat, slope_true)
    es = equity_scaled_auc(slope_hat, slope_true, group)
    assert abs(es - raw) < 1e-9


def test_delong_matches_point_auc() -> None:
    rng = np.random.default_rng(1)
    labels = (rng.random(300) > 0.5).astype(np.int64)
    scores = labels + rng.normal(0, 1, 300)
    auc, lo, hi = delong_ci(scores, labels)
    assert abs(auc - float(roc_auc_score(labels, scores))) < 1e-9
    assert lo < auc < hi


def test_pointwise_and_bundle_mae() -> None:
    vf_hat = np.zeros((4, 52))
    vf_target = np.ones((4, 52))
    assert abs(pointwise_mae(vf_hat, vf_target) - 1.0) < 1e-9
    bundles = per_bundle_mae(vf_hat, vf_target)
    assert all(abs(value - 1.0) < 1e-9 for value in bundles.values())
