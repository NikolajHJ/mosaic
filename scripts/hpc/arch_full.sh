#!/usr/bin/env bash
# Staged full-data comparison; separate exclusive GPU job for each arm/mode.
set -euo pipefail
: "${LSB_JOBID:?HPC allocation required}"
case "${1:-}" in residual|pair_refine|deep_pair_refine|global_bond|wide_global_bond|stable_deep_pair_refine) TASK_VARIANT="$1" ;; *) exit 2 ;; esac
case "${2:-}" in ae|vae) TASK_MODE="$2" ;; *) exit 2 ;; esac
# Exercise CUDA checkpoint RNG restoration on the actual allocated GPU before
# committing a long comparison run. The CPU architecture suite cannot do this.
python -m latentmol.experiments --run-root "$LATENTMOL_WORK_ROOT/runs" \
    --name "gpu-rng-before-arch-${TASK_VARIANT}-${TASK_MODE}" -- \
    python -m pytest -q \
    tests/training/test_training.py::test_cuda_rng_state_is_restored_from_weights_only_checkpoint
python -m latentmol.full_training \
    --config "configs/arch_${TASK_VARIANT}_${TASK_MODE}_full.yaml" \
    --data "$LATENTMOL_WORK_ROOT/data/zinc215k-seed17-v1" \
    --run-root "$LATENTMOL_WORK_ROOT/runs" \
    --initial-epochs 60 --extension-epochs 20 --window 10 \
    --final-limit 1000000 --neighborhood-limit 256 --prior-limit 1024
