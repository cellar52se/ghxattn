from __future__ import annotations

from hydra.core.config_store import ConfigStore

from ghxattn.dressing.schema import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    RootConfig,
    TrainConfig,
)


def register() -> ConfigStore:
    store = ConfigStore.instance()
    store.store(name="ghxattn_base", node=RootConfig)
    store.store(group="model", name="ghxattn_schema", node=ModelConfig)
    store.store(group="data", name="synthetic_schema", node=DataConfig)
    store.store(group="train", name="main_schema", node=TrainConfig)
    store.store(group="experiment", name="experiment_schema", node=ExperimentConfig)
    return store
