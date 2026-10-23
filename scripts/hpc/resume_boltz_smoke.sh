#!/usr/bin/env bash
# Validate an adapter revision against the existing native pose/features.
set -euo pipefail
TASK_PREPARED="${1:?Prepared tensor snapshot required}"
TASK_AE="${2:?AE checkpoint required}"
python -m latentmol.experiments --run-root "$LATENTMOL_WORK_ROOT/runs" --name tests-before-boltz-recovery -- \
    python -m pytest -q tests --junitxml="$LATENTMOL_WORK_ROOT/logs/tests-${LSB_JOBID}.xml"
bash scripts/hpc/boltz_smoke.sh "$TASK_PREPARED" "$TASK_AE"
