#!/usr/bin/env bash
# Submit through run_job.sh in an LSF CPU allocation. Existing caches are immutable.
set -euo pipefail
python -m latentmol.experiments --run-root "$LATENTMOL_WORK_ROOT/runs" --name tests-full-data -- \
    python -m pytest -q tests/test_chemistry.py tests/test_full_data.py \
    --junitxml="$LATENTMOL_WORK_ROOT/logs/tests-full-data-${LSB_JOBID}.xml"
python -m latentmol.experiments --run-root "$LATENTMOL_WORK_ROOT/runs" --name prepare-zinc215k -- \
    python -m latentmol.data --input "$LATENTMOL_WORK_ROOT/raw/zinc250k.csv" \
    --output "$LATENTMOL_WORK_ROOT/data/zinc215k-seed17-v1" \
    --train-count 215000 --val-count 7347 --seed 17 \
    --reserve-cache "$LATENTMOL_WORK_ROOT/data/zinc5k-seed17-v1"
