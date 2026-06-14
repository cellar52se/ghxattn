from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import cohen_kappa_score

IntArray = NDArray[np.int64]


def cohens_kappa(flags_a: IntArray, flags_b: IntArray) -> float:
    if np.unique(np.concatenate([flags_a, flags_b])).shape[0] < 2:
        return float("nan")
    return float(cohen_kappa_score(flags_a, flags_b))
