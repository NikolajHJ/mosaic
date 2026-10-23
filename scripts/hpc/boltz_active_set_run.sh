#!/usr/bin/env bash
# UNTESTED. Execute from the immutable source release inside a DTU GPU job.
# Usage: bash scripts/hpc/boltz_active_set_run.sh VERIFIED_CONFIG.yaml
set -euo pipefail
: "${LSB_JOBID:?HPC allocation required}"
: "${LATENTMOL_SOURCE_SHA256:?Immutable source archive identity required}"
CONFIG="${1:?Verified active-set config required}"
if [ "$#" -ne 1 ] || [ ! -f "$CONFIG" ]; then
    printf 'Require exactly one existing config file\n' >&2
    exit 2
fi
python -m latentmol.boltz.active_set --config "$CONFIG"
