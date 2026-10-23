#!/usr/bin/env bash
# Diagnostic only: no molecule update, tolerance relaxation or automatic rerun.
# Launch through boltz_run.sh from an immutable release inside a DTU GPU job.
set -euo pipefail
: "${LSB_JOBID:?HPC allocation required}"
: "${LATENTMOL_SOURCE_SHA256:?Immutable source archive identity required}"
if [ "$#" -ne 2 ]; then
    printf 'Usage: %s FAILED_PARENT_RUN RUN_ROOT\n' "$0" >&2
    exit 2
fi
python -m pytest -q tests/test_boltz_active_set_diagnostic.py
python -m latentmol.boltz.active_set_diagnostic --parent-run "$1" --run-root "$2"
