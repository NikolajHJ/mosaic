#!/usr/bin/env bash
# One bounded allocation; original saved states are compared serially.
# Launch through boltz_run.sh from an immutable release after native FD audit.
# Numerical/native verification remains pending until documented in the ledger.
set -euo pipefail
: "${LSB_JOBID:?DTU HPC allocation required}"
: "${LATENTMOL_SOURCE_SHA256:?Immutable comparison source required}"
if [ "$#" -ne 3 ]; then
    printf 'Usage: %s PROTOCOL_JSON VALIDATION_ROOT VALIDATION_ARCHIVE_SHA256\n' "$0" >&2
    exit 2
fi
python -m pytest -q tests/test_optimization.py tests/test_boltz_step_control.py tests/test_step_control_comparison.py tests/test_boltz_active_set_runtime.py
python scripts/compare_saved_boltz_steps.py --protocol "$1" --validation-root "$2" --validation-archive-sha256 "$3"
