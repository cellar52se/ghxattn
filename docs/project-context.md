# Project Context — GH-XAttn-MCID

Resolved configuration for the canonical code release accompanying our manuscript on
multimodal Transformer forecasting of glaucomatous visual-field loss.

    package            : ghxattn                              [HIGH]
    domain             : computational ophthalmology -        [HIGH]
                         longitudinal visual-field progression
                         forecasting from paired OCT + VF
    framework          : PyTorch 2.3 (bf16 mixed precision)   [HIGH]
                         scikit-learn 1.5; R 4.4 pROC (stats ref only)
    venue              : Scientific Reports                   [HIGH]
                         (AI in Medical Imaging Collection)
    primary_datasets   : 3 longitudinal + 1 cross-sectional   [HIGH]
                         (fair-pretraining firewall)
    compute_target     : 1x NVIDIA A100 80GB per seed         [HIGH]
    hparams_reference  : Table S1 + Methods training section  [HIGH]
    supp_path          : none (S1-S10 inline in main text)
    extra_signals      : RETFound frozen encoder; learned GH
                         prior weight; 5 seeds; FairDist teacher;
                         gradient-reversal site discriminator

    NEEDS_USER_DECISION: 0

## Model

GH-XAttn-MCID fuses a frozen RETFound OCT structural encoder with a longitudinal
visual-field functional encoder through a cross-modal attention block whose pre-softmax
score matrix is biased by an offline Garway-Heath topographic prior. The system pairs
MCID-anchored per-site split-conformal calibration, site-aware vendor-specific batch
normalisation, and FairDist cross-sectional to longitudinal equity distillation driven by
a gradient-reversal site discriminator. Evaluation is leave-site-out across three
institutionally-independent public cohorts.

## Architecture quantities

    structural encoder : RETFound ViT-L/16, frozen, d_model=768, N_OCT~196 tokens
    functional encoder : 4-layer Transformer over the 52 x T VF tensor
    cross-modal heads  : 12
    GH prior           : GH(i,j) in {0, eps=0.1, 1}, 52 x N_OCT, learned weight lambda init 1.0
    residual path      : rho = 0.3 unconstrained attention path
    temporal encoding  : sinusoidal over inter-visit interval dt in years, max context T=7
    heads              : 52-point VF decoder (dB) + MD-slope head (dB/yr)
    parameters         : 191.4 M
    inference          : 147 +/- 11 ms per eye, 9.4 GB peak GPU, 148.7 GFLOPs

## Training

Two phases. Phase one performs FairDist equity-distillation pretraining of the OCT encoder
on the race-balanced Harvard-GF cross-sectional set; a race-conditioned teacher supervises
the student latent token distribution via Kullback-Leibler divergence. Phase two unfreezes
the encoder and jointly optimises the full architecture on the union of UWHVF, Harvard-GDP,
and GRAPE longitudinal splits.

    composite loss     : L = L_pred + lambda_site * L_site + lambda_KD * L_KD
    L_pred             : MSE(vf_hat, vf) + MSE(slope_hat, slope)
    L_site             : CE(SiteDisc(GradReverse_beta(z)), site)
    L_KD               : KL(softmax(z) || softmax(teacher(x_oct)))
    grad-reversal beta : 0 -> 1 ramp over first 10 epochs
    optimiser          : AdamW, weight_decay 1e-2
    schedule           : cosine with linear warmup over 5% of steps (1000 steps)
    learning rate      : 3e-4
    dropout            : 0.1 transformer / 0.2 attention
    batch size         : 32 eyes
    epochs             : <= 150, early stop on validation MD-slope MSE
    precision          : bf16 mixed precision
    seeds              : {42, 123, 256, 512, 1024}
    hardware           : single NVIDIA A100 80GB per seed

## Conformal calibration

Per-site split-conformal anchored at the EMGT MD-slope MCID floor. For each site a 20%
calibration partition is held out; nonconformity scores are absolute MD-slope residuals
restricted to eyes whose true MD slope crosses the MCID magnitude tau = 0.5 dB/yr. The
(1 - alpha) quantile at alpha = 0.10 defines the symmetric prediction interval. Per-site
empirical coverage and mean width are reported with a Levy-Prokhorov cross-site transfer
diagnostic.

## Datasets

    UWHVF       VF-only longitudinal; 28,943 tests / 7,248 eyes / 3,871 patients;
                Humphrey 24-2; Seattle USA; CC-BY 4.0;
                https://github.com/uw-biomedical-ml/uwhvf
    Harvard-GDP multimodal OCT + VF longitudinal; 4,812 VFs / 1,962 eyes / 1,000 subjects;
                Cirrus + Spectralis OCT; Boston USA; CC-BY-NC-ND 4.0;
                https://ophai.hms.harvard.edu/datasets/harvard-gdp1000
    GRAPE       OCT + VF longitudinal; 1,115 follow-ups / 263 eyes / 144 patients;
                Topcon OCT, Humphrey HFA II; Hangzhou China; CC-BY 4.0;
                https://springernature.figshare.com/articles/dataset/_/22359164
    Harvard-GF  cross-sectional 3D OCT; 3,300 subjects, race-balanced Asian/Black/White;
                fair-encoder pretraining only, firewalled from train/cal/test;
                CC-BY-NC-ND 4.0; https://ophai.hms.harvard.edu/datasets/harvard-gf3300

The cross-sectional Harvard-GF set never enters a training, calibration, or test split for
the longitudinal task; this firewall preserves cross-sectional to longitudinal transfer as
a directed pretraining intervention rather than a hidden information leak.

## Compliance and reproducibility

This release ships a deterministic synthetic cohort generator in `ghxattn/creel`. The
public datasets above are not redistributed; users point the data adapters at the original
public sources under the listed licences. The synthetic generator emits paired structural
volumes, visual-field sequences, MD-slope labels, site identifiers, and race strata with
recoverable structure so that the full pipeline, unit tests, and overfit checks run without
access to restricted data.

Every reported number is produced by re-implementation and re-evaluation on the harmonised
leave-site-out splits, not extrapolated or copied from an external cohort.
