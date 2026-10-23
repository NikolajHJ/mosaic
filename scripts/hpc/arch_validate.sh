#!/usr/bin/env bash
# New architecture regressions in a recorded DTU HPC CPU allocation.
set -euo pipefail
: "${LSB_JOBID:?HPC allocation required}"
python -m latentmol.experiments --run-root "$LATENTMOL_WORK_ROOT/runs" \
    --name "tests-atom-pair-architecture" -- \
    python -m pytest -q \
    tests/test_refined_decoder.py tests/test_architecture_study.py \
    tests/test_architecture_followup.py \
    tests/test_models.py tests/test_training.py \
    tests/test_full_workflow.py tests/test_graph_policy.py tests/test_scoring.py \
    --junitxml="$LATENTMOL_WORK_ROOT/logs/tests-atom-pair-${LSB_JOBID}.xml"
