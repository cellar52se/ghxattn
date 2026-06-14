# Deviations

Entries record any departure from the manuscript, each with a paper-section link and a
justification.

## D1. Seed set inconsistency in the manuscript

The training section states five seeds `{2026, 2027, 2028, 2029, 2030}` while Table S3 and
every results table report `{42, 123, 256, 512, 1024}`. We expose the seed set in
`configs/train` and default to the results-table set `{42, 123, 256, 512, 1024}`, since that
is the set against which every reported number was computed.

## D2. Synthetic cohort stand-in for restricted data

UWHVF, Harvard-GDP, GRAPE, and Harvard-GF are not redistributed with this release. The data
adapters in `ghxattn/creel` default to a deterministic synthetic cohort with recoverable
structure so the pipeline and tests run without restricted access. The real-data path is the
documented adapter target; pointing it at the public sources reproduces the reported splits.
This is a packaging decision, not an algorithmic change.

## D3. Frozen tokeniser fallback for the structural encoder

The structural branch initialises from RETFound. Because the RETFound weights are licence-
gated and large, the default `StructuralEncoder` uses a fixed orthonormal tokeniser
(`ghxattn/warp/encoder.py`) that preserves the token count and dimensionality. The guarded
`RetfoundAdapter` activates only when `timm` and the weights are present and otherwise raises
rather than silently degrading. Reported numbers require the RETFound path; the tokeniser is
for licence-free execution and testing.

## D5. Equity-Scaled AUC blend form

Definition 3 writes ES-AUC as `min_a AUC(f|A=a) + beta * AUC_raw` yet also states it is
degenerate under perfect equity (ES-AUC equals AUC_raw). The literal sum cannot satisfy that
degeneracy for beta = 0.5. We implement the convex blend `(1 - beta) * min_a AUC(f|A=a) +
beta * AUC_raw` in `ghxattn/inspection/equity.py`, which reduces to AUC_raw exactly when the
worst-stratum AUC equals the raw AUC, honouring the stated degeneracy clause. A regression
test pins this behaviour.

## D4. Deep-learning comparator scope

Table 1 lists twelve deep-learning comparators. This release fully implements the proposed
method and the three clinical decision-rule baselines (MD-slope OLR, GPA proxy, chart-review
proxy). The twelve deep-learning comparators are represented as registry slots in
`ghxattn/inspection/baselines.py` rather than full re-implementations; their reported values
are not regenerated here. This bounds the release to the contribution method plus the
clinical references that anchor hypotheses H1-H4.
