#!/usr/bin/env bash
# New workflow: numerical/model validation occurs in this allocation only.
set -euo pipefail
TASK_CONFIG="$1"
TASK_CONFIG_SHA="$2"
TASK_RESUME="${3:-}"
printf '%s  %s\n' "$TASK_CONFIG_SHA" "$TASK_CONFIG" | sha256sum --check
python -m pytest -q tests/test_protein_panel.py tests/test_boltz_active_set.py \
  tests/test_boltz_active_set_runtime.py tests/test_boltz_active_set_rescore.py \
  tests/test_boltz_chain_prepare.py tests/test_random_start_study.py
TASK_ARGS=()
if [ "$TASK_RESUME" = "resume" ]; then TASK_ARGS=(--resume); fi
python -m latentmol.boltz.protein_panel --config "$TASK_CONFIG" \
  --sha256 "$TASK_CONFIG_SHA" "${TASK_ARGS[@]}"
