#!/usr/bin/env bash
# CPU-only audit through boltz_run.sh in an HPC LSF job.
set -euo pipefail
: "${LSB_JOBID:?HPC allocation required}"
python -m latentmol.experiments --run-root "$LATENTMOL_WORK_ROOT/runs" \
    --name tests-chemical-audit -- \
    python -m pytest -q tests/test_boltz_chemical_audit.py \
    --junitxml="$LATENTMOL_WORK_ROOT/logs/tests-chemical-audit-${LSB_JOBID}.xml"
python -m latentmol.boltz.chemical_audit \
    --ae-selection "$LATENTMOL_WORK_ROOT/runs/20260928T082423Z_ae-native-fresh-pose_972c2d83/frozen_selection.json" \
    --vae-selection "$LATENTMOL_WORK_ROOT/runs/20260928T083844Z_vae-native-fresh-pose_253bb4de/frozen_selection.json" \
    --run-root "$LATENTMOL_WORK_ROOT/runs"
