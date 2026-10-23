#!/usr/bin/env bash
# Execute through run_job.sh in a GPU allocation, after data preparation passed.
set -euo pipefail
python -m latentmol.experiments --run-root "$LATENTMOL_WORK_ROOT/runs" --name tests-before-ae -- \
    python -m pytest -q --junitxml="$LATENTMOL_WORK_ROOT/logs/tests-${LSB_JOBID}.xml"
python -m latentmol.train_and_evaluate --config configs/ae_smoke.yaml \
    --data "$LATENTMOL_WORK_ROOT/data/zinc5k-seed17-v1" \
    --run-root "$LATENTMOL_WORK_ROOT/runs" --smoke

