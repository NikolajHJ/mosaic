#!/usr/bin/env bash
# CPU engineering regression then reference-pocket preparation, in DTU only.
set -euo pipefail
: "${LSB_JOBID:?DTU allocation required}"
: "${LATENTMOL_SOURCE_SHA256:?Immutable source required}"
PROTOCOL="$1"
PROTOCOL_SHA="$2"
PREPARED_ROOT="$3"
RUN_ROOT="$4"
python -m latentmol.experiments --run-root "$RUN_ROOT" --name dbmol-cpu-tests -- \
  bash scripts/hpc/boltz_dbmol_tests.sh
python scripts/prepare_dbmol_comparison.py --protocol "$PROTOCOL" \
  --sha256 "$PROTOCOL_SHA" --output "$PREPARED_ROOT" --steps 3
