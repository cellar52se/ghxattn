from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ghxattn.creel.bundles import BUNDLE_NAMES, vf_bundle_ids

IntArray = NDArray[np.int64]

MACULAR_BUNDLE = BUNDLE_NAMES.index("MAC")
PERIPHERAL_BUNDLE = BUNDLE_NAMES.index("PER")


def discordance_rate(model_flag: IntArray, reference_flag: IntArray) -> float:
    if model_flag.shape[0] == 0:
        return float("nan")
    return float(np.mean(model_flag != reference_flag))


def discordance_ratio(early_macular_rate: float, late_peripheral_rate: float) -> float:
    if late_peripheral_rate == 0.0:
        return float("inf")
    return early_macular_rate / late_peripheral_rate


def regional_partition() -> tuple[list[int], list[int]]:
    ids = vf_bundle_ids()
    macular = [loc for loc, bundle in enumerate(ids) if bundle == MACULAR_BUNDLE]
    peripheral = [loc for loc, bundle in enumerate(ids) if bundle == PERIPHERAL_BUNDLE]
    return macular, peripheral
