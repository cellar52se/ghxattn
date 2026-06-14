from __future__ import annotations

from ghxattn.cloth.model import ModelHParams
from ghxattn.creel.batching import EyeBatch, collate_eyes
from ghxattn.creel.synthetic import SyntheticCohort, SyntheticSpec


def tiny_hparams(**overrides: object) -> ModelHParams:
    base = dict(
        d_model=16,
        n_oct=28,
        raw_dim=16,
        cross_heads=4,
        weft_heads=4,
        weft_layers=2,
        dropout=0.0,
        attn_dropout=0.0,
    )
    base.update(overrides)
    return ModelHParams(**base)


def oct_batch(n: int = 6, seed: int = 0) -> EyeBatch:
    spec = SyntheticSpec(n_oct=28, raw_dim=16, max_visits=4)
    records = SyntheticCohort(spec, seed).generate("GRAPE", max(2, n))
    return collate_eyes(records[:n])
