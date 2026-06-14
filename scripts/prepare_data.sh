#!/usr/bin/env bash
set -euo pipefail

# The public cohorts are not redistributed with this repository. Download them from their
# original sources (see docs/project-context.md) into the directory below, then point the
# data adapters in ghxattn/creel at the extracted files.

DATA_ROOT="${1:-data}"
mkdir -p "${DATA_ROOT}"/{uwhvf,harvard_gdp,grape,harvard_gf}

echo "place UWHVF under      ${DATA_ROOT}/uwhvf"
echo "place Harvard-GDP under ${DATA_ROOT}/harvard_gdp"
echo "place GRAPE under      ${DATA_ROOT}/grape"
echo "place Harvard-GF under ${DATA_ROOT}/harvard_gf"
