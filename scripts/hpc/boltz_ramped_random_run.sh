#!/usr/bin/env bash
# One allocation: real native six-update gate, then ten serial 100-update seeds.
# Untested native combination until this job supplies retained gate evidence.
set -euo pipefail
: "${LSB_JOBID:?DTU allocation required}"
: "${LATENTMOL_SOURCE_SHA256:?Immutable release required}"
# The frozen Boltz process retains CUDA while its isolated DeFoG child runs.
# LSF-managed MPS must admit both contexts; test this before loading weights.
# This probe process exits before the separate scientific controller starts.
python -m latentmol.gpu_process_probe --protocol "$1" --protocol-sha256 "$2" \
  --output "${LATENTMOL_WORK_ROOT}/runs/${PWD##*/}-gpu-process-${LSB_JOBID}.json"
python -m latentmol.repair_guidance.random_study run --protocol "$1" --sha256 "$2"
