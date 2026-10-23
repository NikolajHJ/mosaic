#!/usr/bin/env bash
# One bounded GPU allocation; explicit failed gates replay sequentially.
# Launch through boltz_run.sh from the new immutable diagnostic release.
set -euo pipefail
: "${LSB_JOBID:?HPC allocation required}"
: "${LATENTMOL_SOURCE_SHA256:?Immutable source archive identity required}"
if [ "$#" -ne 1 ]; then
    printf 'Usage: %s REPLAY_PROTOCOL_JSON\n' "$0" >&2
    exit 2
fi
python -m pytest -q tests/test_boltz_active_set_diagnostic.py tests/test_gate_replay_study.py
python -m latentmol.boltz.gate_replay_study --protocol "$1"
