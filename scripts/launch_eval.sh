#!/usr/bin/env bash
set -euo pipefail

for HOLDOUT in UWHVF HarvardGDP GRAPE; do
  ghxattn inspect experiment=main data.holdout="${HOLDOUT}"
done
