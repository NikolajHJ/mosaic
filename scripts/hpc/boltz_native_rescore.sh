#!/usr/bin/env bash
# Execute from an immutable release via boltz_run.sh after a completed pilot.
set -euo pipefail
: "${LSB_JOBID:?HPC allocation required}"
: "${BOLTZ_CACHE:?Invoke through boltz_run.sh}"
TASK_MODE="${1:?Provide ae or vae}"
TASK_PILOT="${2:?Provide the completed pilot run path}"
case "$TASK_MODE" in ae|vae) ;; *) exit 2 ;; esac
python -m latentmol.experiments --run-root "$LATENTMOL_WORK_ROOT/runs" \
    --name "tests-native-rescore-${TASK_MODE}" -- \
    python -m pytest -q tests/test_boltz_native_rescore.py \
    --junitxml="$LATENTMOL_WORK_ROOT/logs/tests-native-rescore-${TASK_MODE}-${LSB_JOBID}.xml"
python -m latentmol.boltz.native_rescore \
    --pilot "$TASK_PILOT" \
    --template "$LATENTMOL_WORK_ROOT/boltz-preparation/source006-attempt001/input/tyrosine_smoke.yaml" \
    --cache "$BOLTZ_CACHE" --run-root "$LATENTMOL_WORK_ROOT/runs" \
    --mode "$TASK_MODE"
