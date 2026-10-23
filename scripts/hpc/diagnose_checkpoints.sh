#!/usr/bin/env bash
# Reevaluate explicit training artifacts with richer failure diagnostics.
# Run through run_job.sh inside an allocated GPU job. Never discover a latest run.
set -euo pipefail
TASK_ROOT="/work3/$USER/latentmol"
if [ "$#" -eq 0 ]; then
    echo 'Supply one or more absolute training-run directories.' >&2
    exit 2
fi
python -m latentmol.experiments --run-root "$TASK_ROOT/runs" --name tests-before-diagnostics -- \
    python -m pytest -q tests
for TASK_TRAIN in "$@"; do
    for TASK_CHECKPOINT in best last; do
        python -m latentmol.evaluate --checkpoint "$TASK_TRAIN/$TASK_CHECKPOINT.pt" \
            --data "$TASK_ROOT/data/zinc5k-seed17-v1" --run-root "$TASK_ROOT/runs" \
            --split val --limit 256 --seed 17 --device cuda --novelty
    done
    python -m latentmol.evaluate --checkpoint "$TASK_TRAIN/best.pt" \
        --data "$TASK_ROOT/data/zinc5k-seed17-v1" --run-root "$TASK_ROOT/runs" \
        --split train --limit 256 --seed 17 --device cuda --novelty
done
