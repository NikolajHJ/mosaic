#!/usr/bin/env bash
# A diagnostic CPU backend override, never a pass of the original CUDA gate.
# Launch with boltz_run.sh from a NEW immutable release in an LSF CPU allocation.
set -euo pipefail
: "${LSB_JOBID:?HPC allocation required}"
: "${LATENTMOL_SOURCE_SHA256:?Immutable diagnostic archive identity required}"
if [ "$#" -ne 3 ]; then
    printf 'Usage: %s FAILED_PARENT_RUN ORIGINAL_PRODUCTION_RELEASE RUN_ROOT\n' "$0" >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES=''
python -m pytest -q tests/test_boltz_active_set_diagnostic.py tests/test_gate_replay_study.py
python scripts/diagnose_native_attention_mask.py --run-root "$3/attention-mask" --device cpu
timeout -k 10s 3300s python scripts/diagnose_saved_boltz_gate.py --parent-run "$1" --production-release "$2" \
    --run-root "$3" --arm active_set --step 0 --device-override cpu --probe-profile focused_cpu
