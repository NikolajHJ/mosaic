#!/usr/bin/env bash
# One authenticated cumulative epoch segment;100 remains the final endpoint.
set -euo pipefail
: "${LSB_JOBID:?DTU allocation required}"
: "${LATENTMOL_SOURCE_SHA256:?Immutable source required}"
python -m latentmol.decoder_density_long_study --protocol "$1" --sha256 "$2" \
  --output-root "$3" --stage "$4" "${@:5}"
