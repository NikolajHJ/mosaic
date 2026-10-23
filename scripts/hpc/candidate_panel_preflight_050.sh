#!/usr/bin/env bash
# CPU-only corrected provenance/cohort check. No model or scorer forward.
set -euo pipefail
: "${LSB_JOBID:?HPC allocation required}"
: "${LATENTMOL_WORK_ROOT:?run_job.sh must establish the work root}"

python -m latentmol.experiments \
    --run-root "$LATENTMOL_WORK_ROOT/runs" \
    --name candidate-panel-tests-050 -- \
    python -m pytest -q tests/test_candidate_panel.py

python -m latentmol.candidate_panel \
    --plan configs/next_test_panel_stable_provisional.json \
    --data-manifest "$LATENTMOL_WORK_ROOT/data/zinc215k-seed17-v1/manifest.json" \
    --run-root "$LATENTMOL_WORK_ROOT/runs"
