#!/usr/bin/env bash
# One long serial study, with an optional later boundary-resume allocation.
# No arrays, seed workers or submissions by the compute-node controller.
set -euo pipefail
if [ "$#" -lt 5 ] || [ "$#" -gt 7 ]; then
  printf 'Usage: %s RELEASE ARCHIVE_SHA CONFIG CONFIG_SHA RECEIPT [AFTER_JOB] [resume]\n' "$0" >&2
  exit 2
fi
TASK_RELEASE="$1"; TASK_SOURCE_SHA="$2"; TASK_CONFIG="$3"; TASK_CONFIG_SHA="$4"
TASK_RECEIPT="$5"; TASK_AFTER="${6:-}"; TASK_RESUME="${7:-}"
[[ "$TASK_RELEASE" = /* && "$TASK_CONFIG" = /* && "$TASK_RECEIPT" = /* ]]
[[ "$TASK_SOURCE_SHA" =~ ^[0-9a-f]{64}$ && "$TASK_CONFIG_SHA" =~ ^[0-9a-f]{64}$ ]]
TASK_DEP=()
if [ -n "$TASK_AFTER" ]; then
  [[ "$TASK_AFTER" =~ ^[0-9]+$ ]]
  TASK_DEP=(-w "ended($TASK_AFTER)")
fi
[[ -z "$TASK_RESUME" || "$TASK_RESUME" = resume ]]
cd "$TASK_RELEASE"
printf '%s  %s\n%s  %s\n' "$TASK_SOURCE_SHA" "${TASK_RELEASE}.tar.gz" \
  "$TASK_CONFIG_SHA" "$TASK_CONFIG" | sha256sum --check
unset PYTHONHOME PYTHONPATH
module load python3/3.11.9
PYTHONPATH="$TASK_RELEASE/src" /work3/s234818/latentmol/env/boltz311/bin/python - <<'PY'
import json
from pathlib import Path
from latentmol.experiments import source_manifest
assert source_manifest(Path.cwd())['sha256'] == json.loads(Path('release_manifest.json').read_text())['sha256']
PY
mkdir "$TASK_RECEIPT"
TASK_LABEL="vae-protein-panel-057${TASK_RESUME:+-resume}"
bsub -J "$TASK_LABEL" -q gpua40 -n 4 -R 'rusage[mem=8GB]' -R 'span[hosts=1]' \
  -gpu 'num=1:mode=exclusive_process' -W 23:00 "${TASK_DEP[@]}" \
  -oo "/work3/s234818/latentmol/logs/${TASK_LABEL}.%J.out" \
  -eo "/work3/s234818/latentmol/logs/${TASK_LABEL}.%J.err" \
  bash scripts/hpc/boltz_run.sh python -m latentmol.experiments \
  --run-root /work3/s234818/latentmol/runs --name protein-panel-workflow \
  -- bash scripts/hpc/boltz_protein_panel.sh "$TASK_CONFIG" "$TASK_CONFIG_SHA" "$TASK_RESUME" \
  | tee "$TASK_RECEIPT/bsub.txt"
