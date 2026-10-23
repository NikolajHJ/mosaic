#!/usr/bin/env bash
# Allocated CPU engineering + real pinned native-builder parity, zero inference.
set -euo pipefail
: "${LSB_JOBID:?CPU validation requires a DTU allocation}"
test "$#" -eq 2
TASK_NATIVE_SOURCE="$1"
TASK_PARITY_ROOT="$2"
python -m pytest -q tests/test_defog_charge_policy.py tests/test_defog_contract.py tests/test_defog_candidate_repair.py tests/test_repair_guidance_provider.py tests/test_repair_guidance.py tests/test_repair_weight_schedule.py tests/test_defog_charge_repair_diagnostic.py
"$LATENTMOL_WORK_ROOT/env/defog311/bin/python" -m latentmol.defog.charge_policy_diagnostic --source "$TASK_NATIVE_SOURCE" --run-root "$TASK_PARITY_ROOT"
