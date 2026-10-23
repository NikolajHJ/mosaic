#!/usr/bin/env bash
# One serial pilot allocation after the existing source070 native study ends.
# Native search/guidance remains unvalidated until retained actual receipts.
set -euo pipefail
: "${LSB_JOBID:?DTU allocation required}"
: "${LATENTMOL_SOURCE_SHA256:?Immutable release required}"
python -m latentmol.search_decoder_study --config "$1" --sha256 "$2" --output-root "$3"
