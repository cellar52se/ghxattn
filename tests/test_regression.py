from __future__ import annotations

from ghxattn.creel.bundles import vf_bundle_ids
from ghxattn.creel.synthetic import SyntheticSpec, synthesize_eye
from ghxattn.reed.prior import garway_heath_prior


def test_prior_golden_counts() -> None:
    prior = garway_heath_prior(196, 0.1)
    assert int((prior == 1.0).sum()) == 1716
    assert abs(float(prior.sum()) - 2092.7998) < 1e-2


def test_bundle_partition_covers_52() -> None:
    ids = vf_bundle_ids()
    assert len(ids) == 52
    assert set(ids) == set(range(7))


def test_synthetic_eye_golden_slope() -> None:
    spec = SyntheticSpec(n_oct=196, raw_dim=32)
    eye = synthesize_eye(0, "GRAPE", 0, 123, spec)
    assert eye.vf_seq.shape[0] == 6
    assert abs(eye.slope - (-0.773088)) < 1e-4
