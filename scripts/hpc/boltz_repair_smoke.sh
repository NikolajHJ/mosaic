#!/usr/bin/env bash
# NEW native plumbing; prepare and gate before a scientific comparison.
# Serial cases, separate DeFoG subprocess per declared refresh attempt. A
# completed all-fallback trajectory is not a passed repair-guidance test.
set -euo pipefail
: "${LSB_JOBID:?DTU allocation required}"
: "${LATENTMOL_SOURCE_SHA256:?Immutable source required}"
COMPARISON="$1"
COMPARISON_SHA="$2"
RUN_ROOT="$3"
python -m latentmol.repair_guidance.study run --comparison "$COMPARISON" \
  --sha256 "$COMPARISON_SHA" --output "$RUN_ROOT" --child-timeout-seconds 10800
