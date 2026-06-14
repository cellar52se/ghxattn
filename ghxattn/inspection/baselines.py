from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from ghxattn.creel.batching import EyeRecord
from ghxattn.creel.bundles import MCID_SLOPE
from ghxattn.inspection.discordance import regional_partition

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

COMPARATOR_REGISTRY: dict[str, str] = {
    "MD-slope OLR": "clinical",
    "SVM rapid-progressor": "ML",
    "Wen-CascadeNet5": "CNN",
    "Park-RNN": "RNN",
    "Berchuck-VAE-RNN": "VAE-RNN",
    "Christopher-OCT-DL": "CNN",
    "Park-Inception-V3": "CNN",
    "Hou-GTN": "Transformer",
    "Sriwatana-DINO-ViT": "ViT",
    "Ashtari-Majlan-Spatial-Transformer-GRU": "Transformer-GRU",
    "FairDist-EfficientNet-KD": "CNN-KD",
    "Boudoumi-Diffusion-Conformal": "Diffusion",
    "ConvNeXt-V2-BiLSTM": "CNN-BiLSTM",
}


def _md_series(record: EyeRecord) -> tuple[FloatArray, FloatArray]:
    times = np.cumsum(record.dt).astype(np.float64)
    md = record.vf_seq.mean(axis=1).astype(np.float64)
    return times, md


def _ols_slope(times: FloatArray, values: FloatArray) -> float:
    variance = float(np.var(times))
    if variance == 0.0:
        return 0.0
    covariance = float(np.mean((times - times.mean()) * (values - values.mean())))
    return covariance / variance


class MdSlopeOLR:
    def predict_slope(self, records: Sequence[EyeRecord]) -> FloatArray:
        slopes = [_ols_slope(*_md_series(record)) for record in records]
        return np.asarray(slopes, dtype=np.float64)

    def predict_flag(self, records: Sequence[EyeRecord]) -> IntArray:
        return (self.predict_slope(records) <= MCID_SLOPE).astype(np.int64)


class GuidedProgressionProxy:
    def __init__(self, location_threshold: float = -1.0, min_locations: int = 3) -> None:
        self.location_threshold = location_threshold
        self.min_locations = min_locations

    def _location_slopes(self, record: EyeRecord) -> FloatArray:
        times = np.cumsum(record.dt).astype(np.float64)
        slopes = np.array(
            [
                _ols_slope(times, record.vf_seq[:, loc].astype(np.float64))
                for loc in range(record.vf_seq.shape[1])
            ]
        )
        return slopes

    def predict_score(self, records: Sequence[EyeRecord]) -> FloatArray:
        scores = []
        for record in records:
            slopes = self._location_slopes(record)
            scores.append(float(np.mean(slopes < self.location_threshold)))
        return np.asarray(scores, dtype=np.float64)

    def predict_flag(self, records: Sequence[EyeRecord]) -> IntArray:
        counts = []
        for record in records:
            slopes = self._location_slopes(record)
            counts.append(int(np.sum(slopes < self.location_threshold) >= self.min_locations))
        return np.asarray(counts, dtype=np.int64)


class ChartReviewProxy:
    def __init__(self, macular_weight: float = 2.0) -> None:
        self.macular_weight = macular_weight

    def predict_score(self, records: Sequence[EyeRecord]) -> FloatArray:
        macular, _ = regional_partition()
        scores = []
        for record in records:
            times = np.cumsum(record.dt).astype(np.float64)
            global_slope = _ols_slope(times, record.vf_seq.mean(axis=1).astype(np.float64))
            macular_slope = _ols_slope(
                times, record.vf_seq[:, macular].mean(axis=1).astype(np.float64)
            )
            scores.append(-(global_slope + self.macular_weight * macular_slope))
        return np.asarray(scores, dtype=np.float64)

    def predict_flag(self, records: Sequence[EyeRecord]) -> IntArray:
        return (self.predict_score(records) >= -self.macular_weight * MCID_SLOPE).astype(np.int64)
