#!/usr/bin/env bash
# Fixtures then metadata-only matched config preparation; no native sampling.
set -euo pipefail
bash scripts/hpc/boltz_repair_tests.sh
python -m latentmol.repair_guidance.study prepare --base-config "$1" --base-sha256 "$2" \
  --guidance "$PWD/configs/defog_repair_guidance.example.yaml" --guidance-sha256 "$3" \
  --output "$4" --steps 6 --weights 0 0.01
