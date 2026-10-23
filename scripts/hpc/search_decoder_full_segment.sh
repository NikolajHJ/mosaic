#!/usr/bin/env bash
# One training seed/two fresh arms per serial23h allocation. Both processes
# below run sequentially; this decoder-only job has no concurrent CUDA worker.
set -euo pipefail
: "${LSB_JOBID:?DTU allocation required}"
: "${LATENTMOL_SOURCE_SHA256:?Immutable source required}"
python scripts/search_full_cuda_gate.py
python -m latentmol.search_decoder_full --config "$1" --sha256 "$2" --output-root "$3" --segment "$4"
