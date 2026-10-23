#!/usr/bin/env bash
# PREPARED, NOT SUBMITTED. Use only after the user restores DTU access.
# Exactly one LSF job; contexts, tests, ten seeds and native follow-up are serial.
# New workflow, pending HPC fixture/model validation; inspect retained failures.
set -euo pipefail
if [ "$#" -lt 5 ] || [ "$#" -gt 6 ]; then
    printf 'Usage: %s RELEASE_DIR ARCHIVE_SHA256 CONFIG CONFIG_SHA256 RECEIPT_DIR [AFTER_JOB]\n' "$0" >&2
    exit 2
fi
TASK_RELEASE="$1"
TASK_ARCHIVE_SHA="$2"
TASK_CONFIG="$3"
TASK_CONFIG_SHA="$4"
TASK_RECEIPT="$5"
TASK_AFTER_JOB="${6:-}"
[[ "$TASK_RELEASE" = /* && "$TASK_CONFIG" = /* && "$TASK_RECEIPT" = /* ]]
[[ "$TASK_ARCHIVE_SHA" =~ ^[0-9a-f]{64}$ && "$TASK_CONFIG_SHA" =~ ^[0-9a-f]{64}$ ]]
TASK_DEPENDENCY=()
if [ -n "$TASK_AFTER_JOB" ]; then
    [[ "$TASK_AFTER_JOB" =~ ^[0-9]+$ ]]
    TASK_DEPENDENCY=(-w "ended($TASK_AFTER_JOB)")
fi
cd "$TASK_RELEASE"
printf '%s  %s\n%s  %s\n' "$TASK_ARCHIVE_SHA" "${TASK_RELEASE}.tar.gz" \
    "$TASK_CONFIG_SHA" "$TASK_CONFIG" | sha256sum --check
# Pure file/config checks on the login host, no model loading or arithmetic.
unset PYTHONHOME PYTHONPATH
module load python3/3.11.9
TASK_PYTHON=/work3/s234818/latentmol/env/boltz311/bin/python
PYTHONPATH="$TASK_RELEASE/src" "$TASK_PYTHON" - "$TASK_CONFIG" <<'PY'
import json
from pathlib import Path
import sys
import yaml
from latentmol.experiments import source_manifest
expected = json.loads(Path('release_manifest.json').read_text())['sha256']
assert source_manifest(Path.cwd())['sha256'] == expected, 'Extracted release changed'
protocol = yaml.safe_load(Path(sys.argv[1]).read_text())
assert not Path(protocol['output_root']).exists(), 'Study output root already exists'
PY
# A failed submission still leaves this one-shot receipt for inspection. Do not
# retry blindly after an uncertain bsub result: first reconcile scheduler state.
mkdir "$TASK_RECEIPT"
mkdir -p /work3/s234818/latentmol/logs
bsub -J vae-random-starts-056 -q gpua40 -n 4 \
    -R 'rusage[mem=8GB]' -R 'span[hosts=1]' \
    -gpu 'num=1:mode=exclusive_process' -W 12:00 "${TASK_DEPENDENCY[@]}" \
    -oo /work3/s234818/latentmol/logs/vae-random-starts-056.%J.out \
    -eo /work3/s234818/latentmol/logs/vae-random-starts-056.%J.err \
    bash scripts/hpc/boltz_run.sh python -m latentmol.experiments \
    --run-root /work3/s234818/latentmol/runs --name random-start-study-workflow \
    -- bash scripts/hpc/boltz_random_start_study.sh "$TASK_CONFIG" "$TASK_CONFIG_SHA" \
    | tee "$TASK_RECEIPT/bsub.txt"
