from __future__ import annotations

from dataclasses import dataclass, field

from ghxattn.cloth.model import ModelHParams
from ghxattn.creel.synthetic import SyntheticSpec
from ghxattn.loom.trainer import TrainHParams


@dataclass
class ModelConfig:
    d_model: int = 768
    n_oct: int = 196
    raw_dim: int = 32
    structural_kind: str = "frozen_tokeniser"
    cross_heads: int = 12
    weft_heads: int = 8
    weft_layers: int = 4
    lambda_init: float = 1.0
    rho: float = 0.3
    gh_epsilon: float = 0.1
    dropout: float = 0.1
    attn_dropout: float = 0.2
    use_gh: bool = True
    use_residual: bool = True
    use_site_bn: bool = True
    use_temporal_dt: bool = True
    fusion_mode: str = "xattn"
    branch: str = "full"

    def to_hparams(self) -> ModelHParams:
        return ModelHParams(
            d_model=self.d_model,
            n_oct=self.n_oct,
            raw_dim=self.raw_dim,
            structural_kind=self.structural_kind,
            cross_heads=self.cross_heads,
            weft_heads=self.weft_heads,
            weft_layers=self.weft_layers,
            lambda_init=self.lambda_init,
            rho=self.rho,
            gh_epsilon=self.gh_epsilon,
            dropout=self.dropout,
            attn_dropout=self.attn_dropout,
            use_gh=self.use_gh,
            use_residual=self.use_residual,
            use_site_bn=self.use_site_bn,
            use_temporal_dt=self.use_temporal_dt,
            fusion_mode=self.fusion_mode,
            branch=self.branch,
        )


@dataclass
class DataConfig:
    n_per_site: int = 400
    pretrain_eyes: int = 300
    cal_fraction: float = 0.2
    holdout: str = "GRAPE"
    min_visits: int = 3
    max_visits: int = 7
    healthy_db: float = 30.0
    damage_to_slope: float = 0.9
    base_decline: float = 0.15
    noise: float = 0.4

    def to_spec(self, model: ModelConfig) -> SyntheticSpec:
        return SyntheticSpec(
            n_oct=model.n_oct,
            raw_dim=model.raw_dim,
            min_visits=self.min_visits,
            max_visits=self.max_visits,
            healthy_db=self.healthy_db,
            damage_to_slope=self.damage_to_slope,
            base_decline=self.base_decline,
            noise=self.noise,
        )


@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 32
    max_epochs: int = 150
    warmup_steps: int = 1000
    lambda_site: float = 0.1
    lambda_kd: float = 0.1
    beta_ramp_epochs: int = 10
    pretrain_epochs: int = 5
    early_stop_patience: int = 10
    grad_clip: float = 1.0

    def to_hparams(self, seed: int) -> TrainHParams:
        return TrainHParams(
            lr=self.lr,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            warmup_steps=self.warmup_steps,
            lambda_site=self.lambda_site,
            lambda_kd=self.lambda_kd,
            beta_ramp_epochs=self.beta_ramp_epochs,
            pretrain_epochs=self.pretrain_epochs,
            early_stop_patience=self.early_stop_patience,
            grad_clip=self.grad_clip,
            seed=seed,
        )


@dataclass
class ExperimentConfig:
    name: str = "main"
    alpha: float = 0.10
    tau: float = 0.5
    eval_batch_size: int = 32
    use_conformal: bool = True
    seeds: list[int] = field(default_factory=lambda: [42, 123, 256, 512, 1024])


@dataclass
class RootConfig:
    seed: int = 42
    command: str = "weave"
    output_dir: str = "outputs"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
