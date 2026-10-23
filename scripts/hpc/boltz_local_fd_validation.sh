#!/usr/bin/env bash
# Run via boltz_run.sh inside one immutable DTU GPU allocation. The protocol
# lists bounded original failed states; children run serially, never optimize.
# Native-model validation is pending until recorded in the experiment ledger.
set -euo pipefail
if [[ -z "${LSB_JOBID:-}" || -z "${LATENTMOL_SOURCE_SHA256:-}" ]]; then
    echo "Require DTU job and immutable source archive identity" >&2
    exit 2
fi
if [[ "$#" -ne 1 ]]; then
    echo "Usage: boltz_local_fd_validation.sh ABSOLUTE_PROTOCOL_JSON" >&2
    exit 2
fi
python -m pytest -q tests/test_boltz_local_fd.py tests/test_boltz_active_set.py tests/test_boltz_active_set_runtime.py tests/test_gate_replay_study.py
python scripts/validate_saved_boltz_gates.py --protocol "$1"
