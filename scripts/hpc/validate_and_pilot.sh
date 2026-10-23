#!/usr/bin/env bash
# Validate a new immutable source snapshot once before its first pilot.
set -euo pipefail
python -m latentmol.experiments --run-root "$LATENTMOL_WORK_ROOT/runs" --name tests-before-pilot -- \
    python -m pytest -q --junitxml="$LATENTMOL_WORK_ROOT/logs/tests-${LSB_JOBID}.xml"
bash scripts/hpc/pilot.sh "${1:?Choose ae or vae}"
