#!/usr/bin/env bash
# One serial allocation, four matched fixed-mask short trajectories. Untested
# native integration until the saved parity/component derivative gates pass.
set -euo pipefail
: "${LSB_JOBID:?DTU allocation required}"
: "${LATENTMOL_SOURCE_SHA256:?Immutable source required}"
COMPARISON="$1"
COMPARISON_SHA="$2"
RUN_ROOT="$3"
python scripts/run_dbmol_comparison.py --comparison "$COMPARISON" \
  --sha256 "$COMPARISON_SHA" --output "$RUN_ROOT" --child-timeout-seconds 5400 \
  --cases carbon-c13-binary-fixed_initial_mask carbon-c13-dbmol-fixed_initial_mask \
  prior-0-binary-fixed_initial_mask prior-0-dbmol-fixed_initial_mask
