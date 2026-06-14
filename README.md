# ghxattn — a weaver's draft for GH-XAttn-MCID

This repository is laid out like a loom. Two yarn systems are interlaced into one cloth: the
**warp** carries the structural optical-coherence-tomography signal, the **weft** carries the
longitudinal visual-field sequence, and a **shuttle** passes between them under the control of
a **reed** — the Garway-Heath topographic prior that decides which structural sectors a visual-
field location is allowed to draw from. The finished cloth is a 24-month visual-field forecast
with an MCID-anchored prediction interval.

Read this file as a weaving draft: thread the loom, set the tie-up, follow the treadling, then
read the drawdown.

```
   weft (visual-field sequence)  ──┐
                                   ├─ shuttle (cross-modal attention + reed bias) ─ cloth ─ heads
   warp (OCT structural tokens) ───┘
```

## Threading — dressing the loom (install)

Pick one path.

pip

    python -m pip install -e .

conda

    conda env create -f environment.yml
    conda activate ghxattn
    python -m pip install -e .

docker

    docker build -t ghxattn:latest .
    docker run --rm ghxattn:latest weave experiment=_smoke

The structural branch reads RETFound when the optional `retfound` extra and its weights are
present; otherwise it falls back to a fixed orthonormal tokeniser so every command runs without
licence-gated weights. See `docs/deviations.md`.

## Tie-up — selecting the pattern (configuration)

Configuration is composed with Hydra. The tie-up is the set of config groups that combine into
one run:

    configs/
      config.yaml            the loom frame (defaults list)
      model/ghxattn.yaml     warp, weft, reed, shuttle, heads
      data/synthetic.yaml    the yarn supply
      train/main.yaml        the treadling schedule
      experiment/*.yaml      named patterns (main, ablations, supplementary)

Change any thread from the command line:

    ghxattn weave train.lr=1e-4 model.cross_heads=8 data.holdout=GRAPE

Each reported ablation is its own pattern, for example:

    ghxattn weave experiment=ablation_no_gh_prior
    ghxattn weave experiment=ablation_concat_fusion
    ghxattn weave experiment=ablation_functional_only

The `experiment=_smoke` pattern shrinks every dimension for a fast check and must never be used
for a reported number.

## Treadling — the working sequence (commands)

Four treadles drive the loom:

    ghxattn weave     experiment=main data.holdout=HarvardGDP    # two-phase training + checkpoint
    ghxattn inspect   experiment=main data.holdout=HarvardGDP    # leave-site-out evaluation
    ghxattn forecast  experiment=main data.holdout=HarvardGDP    # per-eye forecast + interval
    ghxattn export    experiment=main data.holdout=HarvardGDP    # write a portable weight bundle

`weave` runs the FairDist pretraining pass on the race-balanced cross-sectional set, then the
joint pass on the union of the three longitudinal cohorts with the gradient-reversal site
discriminator ramped over the first ten epochs.

## Drawdown — the cloth that comes off the loom (expected results)

The numbers below are what our full RETFound-plus-real-cohort training reports under the leave-
site-out protocol. The bundled synthetic cohort exercises the same code path but does not
reproduce these values; it exists so the loom runs end to end without restricted data.

| Endpoint | Treadle | Reported value |
|---|---|---|
| Harvard-GDP fast-progression AUC | `inspect data.holdout=HarvardGDP` | 0.872 +/- 0.012 |
| UWHVF fast-progression AUC | `inspect data.holdout=UWHVF` | 0.864 +/- 0.011 |
| GRAPE fast-progression AUC | `inspect data.holdout=GRAPE` | 0.901 +/- 0.014 |
| UWHVF pointwise MAE (24 mo) | `inspect data.holdout=UWHVF` | 2.07 +/- 0.12 dB |
| Harvard-GDP conformal coverage (alpha 0.10) | `inspect data.holdout=HarvardGDP` | 0.895 +/- 0.013 |
| Harvard-GDP equity-scaled AUC | `inspect data.holdout=HarvardGDP` | 0.731 +/- 0.021 |
| Garway-Heath attention concentration | `inspect experiment=main` | 0.71 +/- 0.04 |

Removing the reed (the topographic prior) drops Harvard-GDP AUC by 5.8 points, the largest single
ablation, matching `experiment=ablation_no_gh_prior`.

## Yarn & stock — where the thread comes from (data)

| Cohort | Role | Eyes / VFs | Vendor | Licence |
|---|---|---|---|---|
| UWHVF | visual-field-only longitudinal | 7,248 / 28,943 | Humphrey 24-2 | CC-BY 4.0 |
| Harvard-GDP | multimodal longitudinal | 1,962 / 4,812 | Cirrus + Spectralis | CC-BY-NC-ND 4.0 |
| GRAPE | cross-vendor longitudinal | 263 / 1,115 | Topcon | CC-BY 4.0 |
| Harvard-GF | fair-encoder pretraining only | 3,300 subjects | 3D OCT | CC-BY-NC-ND 4.0 |

Source URLs and versions are listed in `docs/project-context.md`. The cross-sectional Harvard-GF
set is firewalled from every training, calibration, and test split for the longitudinal task. The
public cohorts are not redistributed here; point the data adapters in `ghxattn/creel` at the
original sources to run on real data.

## Loom specs — power and runtime (compute)

| Quantity | Value |
|---|---|
| Hardware | 1x NVIDIA A100 80GB per seed |
| Parameters | 191.4 M |
| FLOPs (single-eye inference) | 148.7 G |
| Inference latency | 147 +/- 11 ms per eye |
| Peak training GPU memory (batch 32) | 9.4 GB |
| Seeds | {42, 123, 256, 512, 1024} |

## Finishing — setting the cloth (calibration and evaluation)

Per-site split-conformal calibration anchors the prediction interval at the EMGT MD-slope MCID
floor of -0.5 dB/year. A 20% per-site calibration partition is held out; nonconformity scores are
absolute MD-slope residuals restricted to eyes whose true slope crosses the 0.5 dB/year magnitude.
The `inspect` treadle reports per-site empirical coverage and mean interval width alongside the
discrimination metrics, with a Levy-Prokhorov diagnostic for cross-site transfer.

## Care label — handling and limits (ethics)

This is a computational retrospective analysis on publicly available, de-identified cohorts. No
institutional review board approval was required, as no new data collection or patient interaction
occurred; the original cohorts were collected and de-identified by their respective dataset
originators under their own ethical oversight. The system is a complement to, not a replacement
for, the standard-of-care progression triple of Guided Progression Analysis, MD-slope linear
regression, and ophthalmologist chart review. The findings are evidence at the retrospective tier
and should inform research prioritisation rather than clinical practice; African-ancestry
longitudinal representation is limited and the conformal coverage guarantee holds only within the
documented cross-site distribution-shift envelope.

## Quality gates

    make lint     # ruff + black + isort
    make type     # mypy
    make test     # pytest
    make smoke    # two-step training on the synthetic cohort
