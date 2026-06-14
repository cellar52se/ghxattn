from __future__ import annotations

import numpy as np

from ghxattn.fulling.conformal import conformal_quantile, quantile_index
from ghxattn.fulling.transfer import coverage_transfer_bound, levy_prokhorov_distance


def test_split_conformal_coverage() -> None:
    rng = np.random.default_rng(0)
    calibration = np.abs(rng.normal(0, 1, 2000))
    quantile = conformal_quantile(calibration, 0.1)
    test = np.abs(rng.normal(0, 1, 5000))
    coverage = float(np.mean(test <= quantile))
    assert 0.86 < coverage < 0.94


def test_quantile_index() -> None:
    assert quantile_index(99, 0.1) == 90


def test_levy_prokhorov_distance() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 1000)
    assert levy_prokhorov_distance(a, a.copy()) == 0.0
    b = rng.normal(3, 1, 1000)
    assert levy_prokhorov_distance(a, b) > 0.3


def test_coverage_transfer_bound() -> None:
    assert abs(coverage_transfer_bound(0.9, 0.02) - 0.86) < 1e-12
