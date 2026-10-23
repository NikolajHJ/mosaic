#!/usr/bin/env bash
# Execute through run_job.sh. A failed test prevents creation of training data.
set -euo pipefail
python -m latentmol.experiments --run-root "$LATENTMOL_WORK_ROOT/runs" --name tests -- \
    python -m pytest -q --junitxml="$LATENTMOL_WORK_ROOT/logs/tests-${LSB_JOBID}.xml"
python -m latentmol.experiments --run-root "$LATENTMOL_WORK_ROOT/runs" --name prepare-zinc5k -- \
    python -m latentmol.data --input "$LATENTMOL_WORK_ROOT/raw/zinc250k.csv" \
    --output "$LATENTMOL_WORK_ROOT/data/zinc5k-seed17-v1" --limit 5000 --seed 17

