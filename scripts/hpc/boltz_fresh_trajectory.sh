#!/usr/bin/env bash
# New native evaluation; fixture validation precedes every allocation.
set -euo pipefail
TASK_CONFIG="$1"; TASK_CONFIG_SHA="$2"; TASK_MODE="${3:-}"
printf '%s  %s\n' "$TASK_CONFIG_SHA" "$TASK_CONFIG" | sha256sum --check
python -m pytest -q tests/test_fresh_trajectory.py tests/test_processed_identity.py \
  tests/test_boltz_active_set_rescore.py tests/test_random_start_study.py tests/test_protein_panel.py \
  tests/test_config_identity.py
TASK_ARGS=()
if [ "$TASK_MODE" = resume ] && [ -d /work3/s234818/latentmol/runs/source058-fresh-trajectory ]; then
  TASK_ARGS=(--resume)
fi
python -m latentmol.boltz.fresh_trajectory --config "$TASK_CONFIG" --sha256 "$TASK_CONFIG_SHA" "${TASK_ARGS[@]}"
