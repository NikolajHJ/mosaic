#!/usr/bin/env bash
# Small CPU engineering check before any new pretrained-model inference.
set -euo pipefail
TASK_RELEASE="$1"; TASK_SOURCE_SHA="$2"; TASK_CONFIG="$3"; TASK_CONFIG_SHA="$4"; TASK_RECEIPT="$5"
[[ "$TASK_RELEASE" = /* && "$TASK_CONFIG" = /* && "$TASK_RECEIPT" = /* ]]
[[ "$TASK_SOURCE_SHA" =~ ^[0-9a-f]{64}$ && "$TASK_CONFIG_SHA" =~ ^[0-9a-f]{64}$ ]]
cd "$TASK_RELEASE"
printf '%s  %s\n%s  %s\n' "$TASK_SOURCE_SHA" "${TASK_RELEASE}.tar.gz" "$TASK_CONFIG_SHA" "$TASK_CONFIG" | sha256sum --check
mkdir "$TASK_RECEIPT"
bsub -J fresh-trajectory-preflight-058 -q hpc -n 4 -R 'rusage[mem=4GB]' -R 'span[hosts=1]' -W 1:00 \
  -oo /work3/s234818/latentmol/logs/fresh-trajectory-preflight-058.%J.out \
  -eo /work3/s234818/latentmol/logs/fresh-trajectory-preflight-058.%J.err \
  bash scripts/hpc/boltz_run.sh python -m latentmol.experiments \
  --run-root /work3/s234818/latentmol/runs --name fresh-trajectory-preflight-workflow \
  -- bash scripts/hpc/fresh_trajectory_preflight.sh "$TASK_CONFIG" "$TASK_CONFIG_SHA" \
  | tee "$TASK_RECEIPT/bsub.txt"
