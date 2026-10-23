#!/usr/bin/env bash
# Two-epoch matched engineering smoke after the architecture test gate passes.
set -euo pipefail
: "${LSB_JOBID:?HPC allocation required}"
case "${1:-}" in residual|pair_refine|deep_pair_refine|global_bond|wide_global_bond|stable_deep_pair_refine) TASK_VARIANT="$1" ;; *) exit 2 ;; esac
case "${2:-}" in ae|vae) TASK_MODE="$2" ;; *) exit 2 ;; esac
python -m latentmol.train_and_evaluate \
    --config "configs/arch_${TASK_VARIANT}_${TASK_MODE}_smoke.yaml" \
    --data "$LATENTMOL_WORK_ROOT/data/zinc215k-seed17-v1" \
    --run-root "$LATENTMOL_WORK_ROOT/runs" --smoke
