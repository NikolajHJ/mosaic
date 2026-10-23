#!/usr/bin/env bash
# Run only through boltz_run.sh within an immutable DTU GPU allocation.
# Source/protocol validation and early-stall support require HPC evidence.
set -euo pipefail
: "${LSB_JOBID:?DTU allocation required}"
: "${LATENTMOL_SOURCE_SHA256:?Immutable release required}"
if [ "$#" -lt 2 ]; then
    printf 'Usage: %s PROTOCOL_JSON SHA256 [--resume | --smoke-root PATH]\n' "$0" >&2
    exit 2
fi
python -m latentmol.boltz.stability_study --protocol "$1" --sha256 "$2" "${@:3}"
