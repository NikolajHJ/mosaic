#!/usr/bin/env bash
# Validation in the AE environment, including exact resume and small workflow.
# Pass cuda in a GPU allocation to require actual CUDA regressions, not skips.
set -euo pipefail
: "${LSB_JOBID:?HPC allocation required}"
case "${1:-cpu}" in
    cpu) ;;
    cuda) python -c 'import torch; assert torch.cuda.is_available(), "CUDA validation requires a GPU allocation"' ;;
    *) printf 'Usage: validate_full_training.sh [cpu|cuda]\n' >&2; exit 2 ;;
esac
mkdir -p "$LATENTMOL_WORK_ROOT/test-artifacts"
"$HOME/.local/bin/uv" pip install --python "$LATENTMOL_WORK_ROOT/env/ae/bin/python" \
    'matplotlib==3.10.1' 'numpy==2.1.3'
"$HOME/.local/bin/uv" pip freeze --python "$LATENTMOL_WORK_ROOT/env/ae/bin/python" \
    > "$LATENTMOL_WORK_ROOT/logs/environment-full-${LSB_JOBID}.txt"
python -m latentmol.experiments --run-root "$LATENTMOL_WORK_ROOT/runs" --name tests-full-training -- \
    python -m pytest -q tests --junitxml="$LATENTMOL_WORK_ROOT/logs/tests-full-${LSB_JOBID}.xml" \
    --basetemp="$LATENTMOL_WORK_ROOT/test-artifacts/tests-${LSB_JOBID}"
python -m latentmol.data_audit --cache "$LATENTMOL_WORK_ROOT/data/zinc215k-seed17-v1" \
    --parent "$LATENTMOL_WORK_ROOT/data/zinc5k-seed17-v1" --run-root "$LATENTMOL_WORK_ROOT/runs" \
    --train-count 215000 --val-count 7347 --test-count 7347
