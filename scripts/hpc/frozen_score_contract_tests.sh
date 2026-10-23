#!/usr/bin/env bash
# Verify the frozen-model branch boundary on an immutable source release.
set -euo pipefail
: "${LSB_JOBID:?HPC allocation required}"
: "${LATENTMOL_WORK_ROOT:?Invoke through boltz_run.sh}"

python -m latentmol.experiments \
    --run-root "$LATENTMOL_WORK_ROOT/runs" \
    --name tests-frozen-score-contract-035 -- \
    python -m pytest -q \
    tests/test_scoring.py \
    tests/test_graph_policy.py \
    tests/test_boltz_scoring.py \
    tests/test_boltz_adapter.py \
    tests/test_boltz_latent_pilot.py \
    --junitxml="$LATENTMOL_WORK_ROOT/logs/tests-frozen-score-contract-035-${LSB_JOBID}.xml"
