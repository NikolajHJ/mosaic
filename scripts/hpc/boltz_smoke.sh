#!/usr/bin/env bash
# Usage through boltz_run.sh: bash scripts/hpc/boltz_smoke.sh PREPARED_PT AE_CHECKPOINT
set -euo pipefail
: "${LSB_JOBID:?HPC allocation required}"
: "${BOLTZ_CACHE:?Invoke through boltz_run.sh}"
python -m latentmol.boltz.smoke \
    --prepared "${1:?Prepared complex required}" \
    --ae-checkpoint "${2:?AE checkpoint required}" \
    --affinity-checkpoint "$BOLTZ_CACHE/boltz2_aff.ckpt" \
    --run-root "$LATENTMOL_WORK_ROOT/runs" --recycling-steps 0
