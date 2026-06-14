# Implementation Map

Source code in this release carries no comments and no docstrings. All provenance linking
code to the manuscript lives here. Locations refer to the main text unless prefixed with S
(Supplementary).

## Architecture and equations

| Paper item | Location | File | Symbol |
|---|---|---|---|
| Garway-Heath topographic prior GH(i,j) in {0, eps, 1} | Def 1; Methods architecture | `ghxattn/reed/prior.py` | `garway_heath_prior`, `TopographicPrior` |
| Cross-modal attention alpha = softmax(QK^T/sqrt(d) + lambda*GH) | Methods Eq (attention); Alg 1 L4 | `ghxattn/shuttle/attention.py` | `CrossModalAttention` |
| Residual unconstrained path alpha-tilde, rho=0.3 fusion | Methods architecture; Alg 1 L5-L6 | `ghxattn/shuttle/attention.py` | `CrossModalAttention.forward` |
| Learned prior weight lambda init 1.0 | Methods architecture; S1 | `ghxattn/shuttle/attention.py` | `CrossModalAttention.log_lambda` |
| dt-aware sinusoidal temporal encoding (years) | Methods architecture; Fig 2 stage 2 | `ghxattn/weft/temporal.py` | `DeltaTimeEncoding` |
| Functional VF encoder, 4-layer Transformer over 52 x T | Methods architecture; S1 layers | `ghxattn/weft/encoder.py` | `FunctionalEncoder` |
| Structural OCT encoder, RETFound ViT-L/16 frozen, d=768 | Methods architecture | `ghxattn/warp/encoder.py` | `StructuralEncoder`, `FrozenTokeniser`, `RetfoundAdapter` |
| Site-aware vendor BatchNorm | Methods architecture; Methods preprocessing | `ghxattn/warp/site_norm.py` | `SiteAwareBatchNorm` |
| VF decoder head -> 52 dB; MD-slope head -> dB/yr | Methods architecture; Alg 1 L7-L8 | `ghxattn/cloth/heads.py` | `VisualFieldDecoder`, `SlopeHead` |
| Assembled forward pass with conformal interval | Alg 1 | `ghxattn/cloth/model.py` | `GHXAttnMCID.forward` |

## Losses and training

| Paper item | Location | File | Symbol |
|---|---|---|---|
| Composite loss L = L_pred + lambda_site L_site + lambda_KD L_KD | Methods training; Alg 2 L10 | `ghxattn/tension/objective.py` | `CompositeLoss` |
| L_pred = MSE(vf) + MSE(slope) | Methods training; Alg 2 L6 | `ghxattn/tension/objective.py` | `prediction_loss` |
| Gradient-reversal site discriminator L_site | Methods training; Alg 2 L8 | `ghxattn/tension/adversary.py` | `GradientReversal`, `SiteDiscriminator` |
| FairDist KL distillation L_KD on latent token distribution | Methods training; Alg 2 L9 | `ghxattn/tension/distill.py` | `fairdist_kl`, `RaceConditionedTeacher` |
| beta_GR ramp 0 -> 1 over first 10 epochs | Methods training; Alg 2 | `ghxattn/loom/schedule.py` | `grad_reversal_beta` |
| Two-phase loop (FairDist pretrain -> joint train) | Alg 2 | `ghxattn/loom/trainer.py` | `Trainer.fit`, `Trainer.pretrain` |
| AdamW wd 1e-2, cosine + 5% warmup | Methods training; S1 | `ghxattn/loom/optim.py` | `build_optimizer`, `cosine_warmup` |
| Early stop on validation MD-slope MSE | Methods training | `ghxattn/loom/trainer.py` | `EarlyStopping` |
| Atomic checkpoint, seed in checkpoint | R4 engineering bar | `ghxattn/loom/checkpoint.py` | `save_atomic`, `load_checkpoint` |
| set_seed single utility | R4 engineering bar | `ghxattn/loom/seeding.py` | `set_seed` |

## Conformal calibration

| Paper item | Location | File | Symbol |
|---|---|---|---|
| Per-site split-conformal interval C(x) | Def 2; Alg 3 | `ghxattn/fulling/conformal.py` | `SplitConformal`, `calibrate_site` |
| MCID-anchored residuals, tau=0.5, alpha=0.10 | Def 2; Methods calibration | `ghxattn/fulling/conformal.py` | `nonconformity`, `quantile_index` |
| Levy-Prokhorov cross-site coverage diagnostic | Thm 1; Remark 1; S8 deltas | `ghxattn/fulling/transfer.py` | `levy_prokhorov_distance`, `coverage_transfer_bound` |

## Metrics, statistics, evaluation

| Paper item | Location | File | Symbol |
|---|---|---|---|
| Per-site fast-progression AUC (MD-slope <= -0.5 dB/yr) | Table 1, 4; Methods | `ghxattn/inspection/discrimination.py` | `fast_progression_auc` |
| Pointwise MAE (PMAE) at 24-month horizon | Table 1; S5 | `ghxattn/inspection/pointwise.py` | `pointwise_mae` |
| Equity-Scaled AUC ES-AUC, beta=0.5 | Def 3; Table 3 | `ghxattn/inspection/equity.py` | `equity_scaled_auc` |
| Empirical conformal coverage and mean width | Table 5; Alg 3 | `ghxattn/inspection/coverage.py` | `empirical_coverage`, `mean_width` |
| Cohen's kappa vs clinical baselines | Table 6; Methods | `ghxattn/inspection/agreement.py` | `cohens_kappa` |
| McNemar exact test | Table 5; Methods | `ghxattn/inspection/significance.py` | `mcnemar_exact` |
| DeLong AUC confidence interval | Methods statistics | `ghxattn/inspection/significance.py` | `delong_ci` |
| Stratified bootstrap (1000 resamples) | Methods statistics | `ghxattn/inspection/significance.py` | `stratified_bootstrap` |
| Per-bundle Garway-Heath MAE | S5 | `ghxattn/inspection/pointwise.py` | `per_bundle_mae`, `BUNDLE_MAP` |
| Early-macular vs late-peripheral discordance ratio | Table 6; H4 | `ghxattn/inspection/discordance.py` | `discordance_ratio` |
| Attention concentration ratio C(lambda) | Lemma 1; Table 2 | `ghxattn/inspection/concentration.py` | `concentration_ratio` |
| Leave-site-out evaluation loop | Table 4; Methods | `ghxattn/inspection/protocol.py` | `leave_site_out`, `evaluate_site` |

## Baselines

| Paper item | Location | File | Symbol |
|---|---|---|---|
| MD-slope ordinary linear regression baseline | Table 1, 6 | `ghxattn/inspection/baselines.py` | `MdSlopeOLR` |
| GPA decision-rule proxy | Table 6 | `ghxattn/inspection/baselines.py` | `GuidedProgressionProxy` |
| Consensus chart-review decision-rule proxy | Table 6 | `ghxattn/inspection/baselines.py` | `ChartReviewProxy` |
| Twelve deep-learning comparators | Table 1 | `ghxattn/inspection/baselines.py` | `COMPARATOR_REGISTRY` (slots; see deviations) |

## Data

| Paper item | Location | File | Symbol |
|---|---|---|---|
| Synthetic cohort generator (paired OCT/VF/slope/site/race) | Methods datasets (synthetic stand-in) | `ghxattn/creel/synthetic.py` | `SyntheticCohort`, `synthesize_eye` |
| Cohort assembly, reliability filtering, vendor filtering | Methods datasets; preprocessing | `ghxattn/creel/assembly.py` | `assemble_splits`, `reliability_filter` |
| Garway-Heath sectoral bundle map | Methods architecture; S5 | `ghxattn/creel/bundles.py` | `BUNDLE_SECTORS`, `bundle_indicator` |
| Eye-level batch collation | Methods training | `ghxattn/creel/batching.py` | `EyeBatch`, `collate_eyes` |

## Theory (proofs not in code)

| Paper item | Location | Realised empirical quantity |
|---|---|---|
| Lemma 1 attention concentration monotonicity | Lemma 1 | `concentration_ratio` test asserts monotone in lambda |
| Theorem 1 Levy-Prokhorov coverage transfer | Thm 1; S0 | `coverage_transfer_bound` returns 1 - alpha - 2 delta |
| Theorem 2 Rademacher sample-complexity optimality | Thm 2; S0 | `rademacher_excess` proxy on bundle-indicator Frobenius norm |
| Corollary 1 equity-scaled coverage transfer | Cor 1 | `coverage_transfer_bound` strata variant |

## Reported tables coverage

| Table | Content | Produced by |
|---|---|---|
| T1 | Multi-baseline comparison | `inspection/protocol.py` + `baselines.py` |
| T2 | Component ablation | `configs/experiment/ablation_*.yaml` (use_gh / use_site_bn / use_temporal_dt / use_residual / fusion_mode / branch / use_conformal / lambda_kd toggles) + `inspection/concentration.py` |
| T3 | Subgroup-stratified equity | `inspection/equity.py` |
| T4 | Cross-site leave-site-out | `inspection/protocol.py` |
| T5 | Significance + MCID compliance | `inspection/significance.py` + `fulling/conformal.py` |
| T6 | Clinical-baseline head-to-head | `inspection/baselines.py` + `inspection/discordance.py` |
| S1 | Hyperparameter table | `configs/` defaults |
| S3 | Per-seed performance | seed loop in `loom/trainer.py` |
| S4 | Computational cost | `docs/repo-plan.md` budget section |
| S5 | Per-bundle MAE | `inspection/pointwise.py` |
| S8 | Calibration diagnostics | `inspection/coverage.py` |
| S9 | Cross-horizon AUC | `configs/experiment/supplementary_horizons.yaml` |
| S10 | External UWHVF validation | `configs/experiment/supplementary_external_uwhvf.yaml` |
