from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ghxattn.creel.bundles import BUNDLE_NAMES, vf_bundle_ids

FloatArray = NDArray[np.float64]

BUNDLE_MAP: dict[str, list[int]] = {
    name: [loc for loc, bundle in enumerate(vf_bundle_ids()) if bundle == index]
    for index, name in enumerate(BUNDLE_NAMES)
}


def pointwise_mae(vf_hat: FloatArray, vf_target: FloatArray) -> float:
    return float(np.mean(np.abs(vf_hat - vf_target)))


def per_bundle_mae(vf_hat: FloatArray, vf_target: FloatArray) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, locations in BUNDLE_MAP.items():
        diff = np.abs(vf_hat[:, locations] - vf_target[:, locations])
        result[name] = float(np.mean(diff))
    return result
