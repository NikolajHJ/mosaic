#!/usr/bin/env bash
# CPU post-hoc audit of the immutable failed Stage3 pilot.
set -euo pipefail
: "${LSB_JOBID:?HPC allocation required}"
python -m latentmol.denoiser_diagnostics \
    --pilot "$LATENTMOL_WORK_ROOT/runs/20260928T092202Z_stage3-schema-denoiser-pilot_ac19a401" \
    --data "$LATENTMOL_WORK_ROOT/data/zinc215k-seed17-v1" \
    --run-root "$LATENTMOL_WORK_ROOT/runs" --device cpu
