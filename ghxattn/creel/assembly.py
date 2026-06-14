from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from ghxattn.creel.batching import EyeBatch, EyeRecord, collate_eyes
from ghxattn.creel.bundles import SITES
from ghxattn.creel.synthetic import SyntheticCohort, SyntheticSpec, synthesize_eye


@dataclass(frozen=True)
class Splits:
    holdout: str
    train: list[EyeRecord]
    calibration: list[EyeRecord]
    test: list[EyeRecord]
    pretrain: list[EyeRecord]


def reliability_filter(records: Sequence[EyeRecord], drop_rate: float = 0.0) -> list[EyeRecord]:
    if drop_rate <= 0.0:
        return list(records)
    kept: list[EyeRecord] = []
    for record in records:
        if (record.eye_id % 1000) / 1000.0 >= drop_rate:
            kept.append(record)
    return kept


def assemble_pretrain(spec: SyntheticSpec, seed: int, n_eyes: int) -> list[EyeRecord]:
    cross_spec = SyntheticSpec(
        n_oct=spec.n_oct,
        raw_dim=spec.raw_dim,
        min_visits=1,
        max_visits=1,
        healthy_db=spec.healthy_db,
        damage_to_slope=spec.damage_to_slope,
        base_decline=spec.base_decline,
        noise=spec.noise,
    )
    records: list[EyeRecord] = []
    for offset in range(n_eyes):
        race = offset % 3
        records.append(synthesize_eye(900_000 + offset, "HarvardGDP", race, seed, cross_spec))
    return records


def assemble_splits(
    spec: SyntheticSpec,
    seed: int,
    n_per_site: int,
    holdout: str,
    cal_fraction: float = 0.2,
    pretrain_eyes: int = 0,
) -> Splits:
    cohort = SyntheticCohort(spec, seed)
    train: list[EyeRecord] = []
    calibration: list[EyeRecord] = []
    test: list[EyeRecord] = []

    for index, site in enumerate(SITES):
        records = cohort.generate(site, n_per_site, start_id=index * n_per_site)
        records = reliability_filter(records)
        if site != holdout:
            train.extend(records)
            continue
        cut = max(1, int(round(len(records) * cal_fraction)))
        calibration.extend(records[:cut])
        train.extend(records[:cut])
        test.extend(records[cut:])

    pretrain = assemble_pretrain(spec, seed, pretrain_eyes) if pretrain_eyes > 0 else []
    return Splits(
        holdout=holdout,
        train=train,
        calibration=calibration,
        test=test,
        pretrain=pretrain,
    )


def iter_batches(
    records: Sequence[EyeRecord],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterator[EyeBatch]:
    order = list(range(len(records)))
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        chunk = order[start : start + batch_size]
        yield collate_eyes([records[i] for i in chunk])
