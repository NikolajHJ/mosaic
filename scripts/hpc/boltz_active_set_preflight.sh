#!/usr/bin/env bash
# Invoke through boltz_run.sh in a CPU LSF allocation. No native Boltz weights
# are loaded. Both fixture regressions and the selected frozen decoder's exact
# carbon-chain start must pass before GPU context preparation is authorized.
set -euo pipefail
: "${LSB_JOBID:?HPC allocation required}"
: "${LATENTMOL_SOURCE_SHA256:?Immutable source archive identity required}"
: "${LATENTMOL_WORK_ROOT:?Invoke through boltz_run.sh}"
TASK_CHECKPOINT="${1:?Selected graph checkpoint required}"
TASK_CHECKPOINT_SHA="${2:?Selected checkpoint SHA256 required}"
TASK_MODE="${3:?ae or vae required}"
TASK_COUNT="${4:?Initial carbon count required}"
TASK_RECEIPT="${5:?New success-receipt path required}"
if [ -e "$TASK_RECEIPT" ]; then
    printf 'Receipt already exists: %s\n' "$TASK_RECEIPT" >&2
    exit 2
fi
python -m latentmol.experiments --run-root "$LATENTMOL_WORK_ROOT/runs" \
    --name active-set-fixtures -- python -m pytest -q \
    tests/test_boltz_active_set.py tests/test_boltz_active_set_runtime.py \
    tests/test_boltz_active_set_rescore.py tests/test_boltz_adapter.py \
    tests/test_boltz_scoring.py tests/test_graph_policy.py \
    tests/test_boltz_chain_prepare.py tests/test_boltz_native_rescore.py \
    tests/test_multimodel.py tests/test_multimodel_external.py
python -m latentmol.boltz.active_set_preflight \
    --checkpoint "$TASK_CHECKPOINT" --sha256 "$TASK_CHECKPOINT_SHA" \
    --mode "$TASK_MODE" --count "$TASK_COUNT" \
    --run-root "$LATENTMOL_WORK_ROOT/runs" --receipt "$TASK_RECEIPT"
