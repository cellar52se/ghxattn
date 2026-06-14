# Repository Plan

## Layout

The package is organised around a weaving metaphor: two yarn systems (structural warp,
functional weft) are interlaced by a shuttle under the control of a reed, and the resulting
cloth is finished and inspected.

    ghxattn/
      creel/        yarn supply: synthetic cohort, assembly, reliability + vendor filters,
                    Garway-Heath sectoral bundle map, eye-level batching
      warp/         structural OCT branch: frozen tokeniser / guarded RETFound adapter,
                    site-aware vendor batch normalisation
      weft/         functional VF branch: dt-aware temporal encoding, 4-layer Transformer
      reed/         Garway-Heath topographic prior tensor and learned-weight gate
      shuttle/      cross-modal attention with constrained + residual fusion paths
      cloth/        VF decoder, MD-slope head, assembled GHXAttnMCID model
      tension/      composite loss, gradient-reversal site adversary, FairDist distillation
      fulling/      MCID-anchored per-site split-conformal, Levy-Prokhorov transfer
      inspection/   discrimination, pointwise, equity, coverage, agreement, significance,
                    discordance, concentration, protocol, baselines
      loom/         two-phase trainer, optimiser, schedules, checkpoint, seeding, early stop
      dressing/     Hydra structured config schema (frozen dataclasses) + ConfigStore
      mill/         Hydra CLI: weave / inspect / forecast / export
    configs/
      model/ data/ train/ experiment/
    tests/
    scripts/
    docs/
    assets/

## Module responsibilities

- `creel.synthetic` builds a deterministic cohort whose MD-slope label is a recoverable
  function of latent structural damage, so models can learn and overfit tests converge.
- `warp.encoder` returns structural tokens; the frozen tokeniser path needs no external
  weights, the guarded RETFound adapter activates only when the dependency and weights are
  present and otherwise raises rather than silently degrading.
- `reed.prior` precomputes the topographic prior once and exposes it as a fixed buffer.
- `shuttle.attention` owns the learned prior weight and the rho-scaled residual path.
- `cloth.model` wires warp + weft + shuttle + heads and exposes the Algorithm 1 forward.
- `tension` keeps each loss term independent so ablations toggle them via config.
- `fulling` is pure tensor / numpy and has no training dependency.
- `inspection` is read-only; statistics are validated against scikit-learn and analytic
  references in tests.
- `loom.trainer` runs the two phases and the seed loop; checkpoints store the seed.
- `dressing` schemas are frozen dataclasses so the merged Hydra config is statically typed.

## Dependencies (pinned)

    torch==2.3.*
    numpy>=1.26,<3
    scipy>=1.11
    scikit-learn==1.5.*
    hydra-core==1.3.*
    omegaconf>=2.3
    pyyaml>=6
    tqdm>=4.66

Dev: pytest, ruff, black, isort, mypy, pre-commit.

## Test coverage plan

| Kind | Target |
|---|---|
| shape | prior tensor, attention output, token counts |
| invariant | lambda -> 0 recovers data-driven attention; bundle-positive mass higher |
| overfit-single-batch | model memorises a tiny batch, loss -> 0 |
| gradient-flow | gradient reversal flips sign; every parameter receives a gradient |
| metric correctness | AUC vs scikit-learn; ES-AUC degeneracy; DeLong vs reference |
| conformal | empirical coverage approx 1 - alpha on synthetic residuals |
| determinism | identical seed -> identical output; per-vendor BN stats independent |
| regression | golden values on the synthetic cohort |
| style guard | no comments / no docstrings / no forbidden phrases / no emoji |
| end-to-end smoke | two training steps on `_smoke.yaml`, loss decreases |

## Compute budget (per Table S4, reported honestly)

| Quantity | Value |
|---|---|
| Hardware | 1x NVIDIA A100 80GB per seed |
| Parameters | 191.4 M |
| FLOPs per single-eye inference | 148.7 G |
| Inference latency | 147 +/- 11 ms per eye |
| Peak training GPU memory (batch 32) | 9.4 GB |
| Seeds | 5 |

The synthetic-cohort smoke configuration runs on CPU in seconds and is for testing only;
it must not be used for any reported number.
