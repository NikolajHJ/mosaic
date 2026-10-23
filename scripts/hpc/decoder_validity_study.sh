#!/usr/bin/env bash
# One serial allocation; an actual native smoke gates six fixed-endpoint cases.
# No scientific advantage has been established merely by submitting this script.
set -euo pipefail
: "${LSB_JOBID:?DTU allocation required}"
: "${LATENTMOL_SOURCE_SHA256:?Immutable release required}"
python -m latentmol.decoder_finetune --config "$1" --sha256 "$2" --output-root "$3"
