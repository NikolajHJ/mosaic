#!/usr/bin/env bash
# Independent AE/VAE GPU allocations share one immutable expanded cache.
set -euo pipefail
case "${1:-}" in
    ae|vae) TASK_MODE="$1" ;;
    *) printf 'Usage: full_training.sh ae|vae\n' >&2; exit 2 ;;
esac
# Keep the preflight in this script so LSF never re-expands an inline bash -c
# command before run_job.sh has established the work root and Python environment.
python -m latentmol.experiments --run-root "$LATENTMOL_WORK_ROOT/runs" \
    --name "gpu-rng-before-full-${TASK_MODE}" -- \
    python -m pytest -q \
    tests/training/test_training.py::test_cuda_rng_state_is_restored_from_weights_only_checkpoint
python -m latentmol.full_training --config "configs/${TASK_MODE}_full.yaml" \
    --data "$LATENTMOL_WORK_ROOT/data/zinc215k-seed17-v1" \
    --run-root "$LATENTMOL_WORK_ROOT/runs" \
    --initial-epochs 60 --extension-epochs 20 --window 10 \
    --final-limit 1000000 --neighborhood-limit 256 --prior-limit 1024
