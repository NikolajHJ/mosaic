#!/usr/bin/env bash
# CPU-only provenance/cohort check. No model forward or external scorer call.
set -euo pipefail
: "${LSB_JOBID:?HPC allocation required}"
: "${LATENTMOL_WORK_ROOT:?run_job.sh must establish the work root}"

python -m latentmol.experiments \
    --run-root "$LATENTMOL_WORK_ROOT/runs" \
    --name candidate-panel-tests-049 -- \
    python -m pytest -q tests/test_candidate_panel.py

python -m latentmol.candidate_panel \
    --plan configs/next_test_panel_stable_provisional.json \
    --data-manifest "$LATENTMOL_WORK_ROOT/data/zinc215k-seed17-v1/manifest.json" \
    --run-root "$LATENTMOL_WORK_ROOT/runs"
