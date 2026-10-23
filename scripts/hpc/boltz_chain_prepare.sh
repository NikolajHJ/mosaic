#!/usr/bin/env bash
# Native C13 pose/features for both frozen AE and VAE pilots.
# Invoke through boltz_run.sh within one GPU allocation.
# Usage: bash scripts/hpc/boltz_chain_prepare.sh UNIQUE_ATTEMPT_DIR AE_CHECKPOINT
set -euo pipefail
: "${LSB_JOBID:?HPC allocation required}"
: "${BOLTZ_CACHE:?Invoke through boltz_run.sh}"
TASK_ATTEMPT="${1:?Unique output directory required}"
TASK_AE="${2:?AE checkpoint required for graph schema}"
if [ -e "$TASK_ATTEMPT" ]; then
    printf 'Output already exists: %s\n' "$TASK_ATTEMPT" >&2
    exit 2
fi
mkdir -p "$TASK_ATTEMPT"
python -m latentmol.boltz.prepare chain-example --output "$TASK_ATTEMPT/input"
# Reduced schedules are an engineering pose/feature preflight. A subsequent
# native rescoring study must state its full sampling policy independently.
boltz predict "$TASK_ATTEMPT/input/carbon_chain_smoke.yaml" \
    --cache "$BOLTZ_CACHE" --out_dir "$TASK_ATTEMPT/prediction" \
    --recycling_steps 0 --sampling_steps 20 --diffusion_samples 1 \
    --sampling_steps_affinity 20 --diffusion_samples_affinity 1 \
    --num_workers 0 --preprocessing-threads 1 --no_kernels --seed 17 \
    --subsample_msa --num_subsampled_msa 128
python -m latentmol.boltz.prepare snapshot \
    --prediction-dir "$TASK_ATTEMPT/prediction/boltz_results_carbon_chain_smoke" \
    --cache "$BOLTZ_CACHE" --output "$TASK_ATTEMPT/prepared.pt" --ae-checkpoint "$TASK_AE"
