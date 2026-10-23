#!/usr/bin/env bash
# One declared case per allocation; no historical checkpoint is overwritten.
set -euo pipefail
: "${LSB_JOBID:?DTU allocation required}"
: "${LATENTMOL_SOURCE_SHA256:?Immutable release required}"
python -m latentmol.decoder_density_study --protocol "$1" --sha256 "$2" \
  --output-root "$3" --stage "$4" "${@:5}"
