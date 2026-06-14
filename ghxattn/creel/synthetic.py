from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ghxattn.creel.batching import EyeRecord
from ghxattn.creel.bundles import (
    HORIZON_YEARS,
    SITE_TO_VENDORS,
    VF_LOCATIONS,
    site_index,
    structural_bundle_ids,
    vendor_index,
    vf_bundle_ids,
)

FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class SyntheticSpec:
    n_oct: int = 196
    raw_dim: int = 32
    min_visits: int = 3
    max_visits: int = 7
    healthy_db: float = 30.0
    damage_to_slope: float = 0.9
    base_decline: float = 0.15
    noise: float = 0.4


def _eye_seed(base_seed: int, eye_id: int) -> int:
    return (base_seed * 1_000_003 + eye_id * 97 + 17) % (2**31 - 1)


def synthesize_eye(
    eye_id: int,
    site: str,
    race: int,
    base_seed: int,
    spec: SyntheticSpec,
) -> EyeRecord:
    rng = np.random.default_rng(_eye_seed(base_seed, eye_id))
    vf_ids = vf_bundle_ids()
    oct_ids = structural_bundle_ids(spec.n_oct)
    n_bundles = max(vf_ids) + 1

    damage = rng.gamma(shape=1.4, scale=0.5, size=n_bundles).astype(np.float32)
    loc_damage = np.array([damage[b] for b in vf_ids], dtype=np.float32)

    baseline = spec.healthy_db - 2.2 * loc_damage + rng.normal(0.0, spec.noise, VF_LOCATIONS)
    baseline = baseline.astype(np.float32)
    loc_slope = -(spec.base_decline + spec.damage_to_slope * loc_damage)
    loc_slope = loc_slope.astype(np.float32) + rng.normal(0.0, 0.05, VF_LOCATIONS).astype(
        np.float32
    )

    vendors = SITE_TO_VENDORS[site]
    has_oct = len(vendors) > 0
    vendor = vendor_index(vendors[rng.integers(0, len(vendors))]) if has_oct else -1

    visits = int(rng.integers(spec.min_visits, spec.max_visits + 1))
    intervals = rng.uniform(0.5, 1.0, size=visits).astype(np.float32)
    intervals[0] = 0.0
    times = np.cumsum(intervals).astype(np.float32)

    vf_seq = np.zeros((visits, VF_LOCATIONS), dtype=np.float32)
    for v in range(visits):
        vf_seq[v] = baseline + loc_slope * times[v]
        vf_seq[v] += rng.normal(0.0, spec.noise, VF_LOCATIONS).astype(np.float32)

    last_time = float(times[-1])
    vf_target = baseline + loc_slope * (last_time + HORIZON_YEARS)
    vf_target = vf_target.astype(np.float32)
    slope = float(np.mean(loc_slope))

    oct_raw = rng.normal(0.0, 0.3, size=(spec.n_oct, spec.raw_dim)).astype(np.float32)
    if has_oct:
        for j, bundle in enumerate(oct_ids):
            oct_raw[j, bundle % spec.raw_dim] += damage[bundle]
    else:
        oct_raw[:] = 0.0

    return EyeRecord(
        eye_id=eye_id,
        site=site_index(site),
        vendor=vendor,
        race=race,
        has_oct=has_oct,
        vf_seq=vf_seq,
        dt=intervals,
        oct_raw=oct_raw,
        vf_target=vf_target,
        slope=slope,
    )


class SyntheticCohort:
    def __init__(self, spec: SyntheticSpec, seed: int) -> None:
        self.spec = spec
        self.seed = seed

    def generate(self, site: str, n_eyes: int, start_id: int = 0) -> list[EyeRecord]:
        rng = np.random.default_rng(_eye_seed(self.seed, site_index(site) * 7919))
        records: list[EyeRecord] = []
        for offset in range(n_eyes):
            race = int(rng.integers(0, 3))
            eye_id = start_id + offset
            records.append(synthesize_eye(eye_id, site, race, self.seed, self.spec))
        return records
