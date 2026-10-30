#!/bin/bash
set -euo pipefail
source_root=$1
arm=$2
mode=${3:-run}
plan=${4:-$source_root/plan.json}
[[ "$mode" == run || "$mode" == smoke || "$mode" == startup ]]
module load python3/3.12.11
unset PYTHONHOME
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_COMPILATION_CACHE=false
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=0
if [[ "$mode" != startup ]]; then
  [[ "${LSB_QUEUE:-}" == gpul40s || "${LSB_QUEUE:-}" == gpuv100 ]]
  module load cuda/12.9.1 cudnn/v9.13.0.50-prod-cuda-12.X
  case "$arm" in
    protenix*)
      export JAX_PLATFORMS=cuda,cpu
      export XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_exclude_nondeterministic_ops=true' ;;
    dbmol) export JAX_ENABLE_X64=1 ;;
  esac
fi
base=/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01
if [[ "$mode" != startup ]]; then
  source "$base/protenix-extended-release-03/source/scripts/hpc/boltz_node_tmpdir.sh"
fi
export PYTHONPATH="$source_root"
export MOSAIC_GPU_PROFILE=$(/work3/s234818/mosaic/envs/protenix-tiny/bin/python -c 'import json,sys;from gpu_policy import selected;print(selected(json.load(open(sys.argv[1]))["arms"][sys.argv[2]])["id"])' "$plan" "$arm")
python=/work3/s234818/mosaic/envs/protenix-tiny/bin/python
cd "$source_root"
if [[ "$mode" == startup ]]; then
  exec "$python" -u exploratory_continuation.py --plan "$plan" --arm "$arm" --startup-only
fi
if [[ "$mode" == smoke ]]; then
  exec "$python" -u exploratory_continuation.py --plan "$plan" --arm "$arm" --smoke-updates 3
fi
if [[ -f "$source_root/test_gradient_scaled_noise.py" ]]; then
  tests=$($python -c 'import json,sys;from pathlib import Path;print(Path(json.load(open(sys.argv[1]))["root"])/sys.argv[2]/"noise-unit-tests")' "$plan" "$arm")
  mkdir "$tests"
  env JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' "$python" -m pytest \
    "$source_root/test_gradient_scaled_noise.py" -q -o addopts= -p no:cacheprovider \
    --junitxml="$tests/tests.xml" >"$tests/stdout.log" 2>&1
fi
exec "$python" -u exploratory_continuation.py --plan "$plan" --arm "$arm"
