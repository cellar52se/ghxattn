from __future__ import annotations

from ghxattn.dressing.loader import load_config


def test_main_defaults_match_paper() -> None:
    cfg = load_config([])
    assert cfg.model.d_model == 768
    assert cfg.model.cross_heads == 12
    assert abs(cfg.train.lr - 3e-4) < 1e-12
    assert abs(cfg.train.weight_decay - 1e-2) < 1e-12
    assert cfg.train.max_epochs == 150
    assert abs(cfg.experiment.alpha - 0.10) < 1e-12
    assert abs(cfg.experiment.tau - 0.5) < 1e-12


def test_smoke_overrides_apply() -> None:
    cfg = load_config(["experiment=_smoke"])
    assert cfg.model.d_model == 32
    assert cfg.train.max_epochs == 3
    assert cfg.data.n_per_site == 16
