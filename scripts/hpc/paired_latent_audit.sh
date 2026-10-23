#!/usr/bin/env bash
# Execute the existing Stage 1 interpolation pairing on HPC without new model calls.
set -euo pipefail
: "${LSB_JOBID:?HPC allocation required}"
python -m latentmol.experiments --run-root "$LATENTMOL_WORK_ROOT/runs" \
    --name tests-paired-latent-audit -- \
    python -m pytest -q tests/test_latent_comparison.py \
    --junitxml="$LATENTMOL_WORK_ROOT/logs/tests-paired-audit-${LSB_JOBID}.xml"
python -m latentmol.latent_comparison \
    --ae-interpolations "$LATENTMOL_WORK_ROOT/runs/20260928T063833Z_ae-latent-stage1_580abd75/interpolations.json" \
    --vae-interpolations "$LATENTMOL_WORK_ROOT/runs/20260928T064117Z_vae-latent-stage1_bd6a6a7a/interpolations.json" \
    --run-root "$LATENTMOL_WORK_ROOT/runs"
