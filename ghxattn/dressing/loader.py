from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from ghxattn.dressing.schema import RootConfig
from ghxattn.dressing.store import register


def default_config_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "configs"


def load_config(overrides: Sequence[str], config_dir: Path | None = None) -> RootConfig:
    register()
    directory = (config_dir or default_config_dir()).resolve()
    with initialize_config_dir(version_base=None, config_dir=str(directory)):
        composed = compose(config_name="config", overrides=list(overrides))
    schema = OmegaConf.structured(RootConfig)
    merged = OmegaConf.merge(schema, composed)
    return cast(RootConfig, OmegaConf.to_object(merged))
