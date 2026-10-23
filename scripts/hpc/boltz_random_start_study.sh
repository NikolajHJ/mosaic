#!/usr/bin/env bash
# New source056 workflow: fixtures first, then serial contexts/seeds in this job.
# Use boltz_run.sh from the immutable release. Never submit child jobs/arrays.
set -euo pipefail
: "${LSB_JOBID:?HPC allocation required}"
: "${LATENTMOL_SOURCE_SHA256:?Immutable source archive required}"
if [ "$#" -ne 2 ]; then
    printf 'Usage: %s CONFIG CONFIG_SHA256\n' "$0" >&2
    exit 2
fi
python -m pytest -q tests/test_boltz_active_set.py tests/test_boltz_active_set_runtime.py \
    tests/test_boltz_active_set_rescore.py tests/test_boltz_active_set_diagnostic.py \
    tests/test_boltz_chain_prepare.py tests/test_random_start_study.py \
    tests/test_initialization_report.py
python -m latentmol.boltz.random_start_study --config "$1" --sha256 "$2"
