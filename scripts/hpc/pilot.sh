#!/usr/bin/env bash
# Call as: bash scripts/hpc/pilot.sh ae|vae, through run_job.sh.
set -euo pipefail
case "${1:-}" in
    ae|vae) TASK_MODE="$1" ;;
    *) printf 'Usage: pilot.sh ae|vae\n' >&2; exit 2 ;;
esac
python -m latentmol.train_and_evaluate --config "configs/${TASK_MODE}_pilot.yaml" \
    --data "$LATENTMOL_WORK_ROOT/data/zinc5k-seed17-v1" \
    --run-root "$LATENTMOL_WORK_ROOT/runs"
