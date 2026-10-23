#!/usr/bin/env bash
set -euo pipefail
TASK_CONFIG="$1"; TASK_CONFIG_SHA="$2"
python -m pytest -q tests/test_fresh_trajectory.py tests/test_processed_identity.py \
  tests/test_boltz_active_set_rescore.py tests/test_random_start_study.py tests/test_protein_panel.py
python -m latentmol.boltz.trajectory_preflight --config "$TASK_CONFIG" --sha256 "$TASK_CONFIG_SHA" \
  --output-root /work3/s234818/latentmol/runs/source058-preflight \
  --historical /work3/s234818/latentmol/runs/20261002T094641Z_boltz-active-set-fresh-pose_ed5ca94d
