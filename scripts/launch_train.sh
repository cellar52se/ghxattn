#!/usr/bin/env bash
set -euo pipefail

HOLDOUT="${1:-HarvardGDP}"
SEED="${2:-42}"

ghxattn weave experiment=main data.holdout="${HOLDOUT}" seed="${SEED}"

# multi-GPU example (single node, 4 ranks):
# torchrun --standalone --nproc_per_node=4 -m ghxattn.mill weave experiment=main data.holdout="${HOLDOUT}"

# slurm example:
# srun --gres=gpu:a100:1 ghxattn weave experiment=main data.holdout="${HOLDOUT}" seed="${SEED}"
