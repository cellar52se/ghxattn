from __future__ import annotations

from pathlib import Path

from ghxattn.dressing.loader import load_config
from ghxattn.mill import commands


def test_training_smoke_loss_decreases(tmp_path: Path) -> None:
    cfg = load_config([f"output_dir={tmp_path}", "experiment=_smoke"])
    result = commands.weave(cfg)
    assert len(result.history) >= 2
    assert result.history[-1].prediction < result.history[0].prediction
    assert (tmp_path / "GRAPE" / "seed42.ckpt").exists()
