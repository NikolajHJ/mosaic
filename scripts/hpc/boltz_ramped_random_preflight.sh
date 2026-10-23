#!/usr/bin/env bash
# HPC fixtures and config preparation; no native model inference in CPU gate.
set -euo pipefail
bash scripts/hpc/boltz_repair_tests.sh
python -m pytest -q tests/test_repair_weight_schedule.py tests/test_ramped_random_study.py
python -m pytest -q tests/test_gpu_process_probe.py
python -m pytest -q tests/test_dbmol_model.py
python -m latentmol.repair_guidance.random_study prepare --base "$1" --base-sha256 "$2" \
  --starts "$3" --starts-sha256 "$4" --guidance "$PWD/configs/defog_ramped_random.example.yaml" \
  --guidance-sha256 "$5" --output "$6" --campaign-root "$7"
# Reproduce the frame-shape rejection on the same authenticated saved context
# catalog. Native helper parity here is CPU engineering evidence; real head
# parity, derivatives and accepted repair-guided updates remain GPU gates.
python scripts/diagnose_native_graph_contracts.py --config "$1" --config-sha256 "$2" \
  --output "$6/frame-contracts" --counts 13 22
