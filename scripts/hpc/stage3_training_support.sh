#!/usr/bin/env bash
# Train-only candidate-support diagnostic; CPU allocation required.
set -euo pipefail
: "${LSB_JOBID:?HPC allocation required}"
python -m latentmol.experiments --run-root "$LATENTMOL_WORK_ROOT/runs" \
    --name tests-stage3-training-support -- \
    python -m pytest -q tests/test_boltz_training_support.py \
    --junitxml="$LATENTMOL_WORK_ROOT/logs/tests-stage3-training-support-${LSB_JOBID}.xml"
python -m latentmol.boltz.training_support \
    --dataset-dir "$LATENTMOL_WORK_ROOT/data/zinc215k-seed17-v1" \
    --chemical-audit-dir "$LATENTMOL_WORK_ROOT/runs/20260928T085606Z_selected-molecule-chemical-audit_3d54051c" \
    --run-root "$LATENTMOL_WORK_ROOT/runs"
