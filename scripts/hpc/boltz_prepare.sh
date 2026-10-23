#!/usr/bin/env bash
# Invoke through boltz_run.sh in a GPU allocation; reduced sampling is a smoke.
# Usage: bash scripts/hpc/boltz_prepare.sh UNIQUE_ATTEMPT_DIR AE_CHECKPOINT
set -euo pipefail
: "${LSB_JOBID:?HPC allocation required}"
: "${BOLTZ_CACHE:?Invoke through boltz_run.sh}"
TASK_ATTEMPT="${1:?Unique output directory required}"
TASK_AE="${2:?AE checkpoint required}"
if [ -e "$TASK_ATTEMPT" ]; then
    printf 'Output already exists: %s\n' "$TASK_ATTEMPT" >&2
    exit 2
fi
mkdir -p "$TASK_ATTEMPT"
python -m latentmol.boltz.prepare example --output "$TASK_ATTEMPT/input"
# The native CLI runs a real confidence-model pose followed by affinity. Both
# diffusion schedules are deliberately shortened to 20 steps for engineering.
# No score from this run is treated as a calibrated binding result.
boltz predict "$TASK_ATTEMPT/input/tyrosine_smoke.yaml" \
    --cache "$BOLTZ_CACHE" --out_dir "$TASK_ATTEMPT/prediction" \
    --recycling_steps 0 --sampling_steps 20 --diffusion_samples 1 \
    --sampling_steps_affinity 20 --diffusion_samples_affinity 1 \
    --num_workers 0 --preprocessing-threads 1 --no_kernels --seed 17 --num_subsampled_msa 128
python -m latentmol.boltz.prepare snapshot \
    --prediction-dir "$TASK_ATTEMPT/prediction/boltz_results_tyrosine_smoke" \
    --cache "$BOLTZ_CACHE" --output "$TASK_ATTEMPT/prepared.pt" --ae-checkpoint "$TASK_AE"
