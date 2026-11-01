#!/bin/bash
set -euo pipefail
source_root=$1
kind=$2
arm=$3
plan=$4
internal_seconds=$5
module load python3/3.12.11
unset PYTHONHOME
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_COMPILATION_CACHE=false
export JAX_ENABLE_X64=0 JAX_PLATFORMS=cpu
python=/work3/s234818/mosaic/envs/protenix-tiny/bin/python
export MOSAIC_GPU_PROFILE=$($python -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p["arms"][sys.argv[2] if sys.argv[2]!="-" else p["probe_arms"][0]]["gpu_profile"])' "$plan" "$arm")
original=$($python -c 'import json,sys; print(next(iter(json.load(open(sys.argv[1]))["arms"].values()))["science_source"]["path"])' "$plan")
export PYTHONPATH="$source_root:$source_root/src:$original/scripts"
cd "$source_root"
if [[ "$kind" == probe || "$kind" == run ]]; then
  expected_queue=$($python -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p["arms"][sys.argv[2] if sys.argv[2]!="-" else p["probe_arms"][0]]["queue"])' "$plan" "$arm")
  [[ "${LSB_QUEUE:-}" == "$expected_queue" ]]
  module load cuda/12.9.1 cudnn/v9.13.0.50-prod-cuda-12.X
  export JAX_PLATFORMS=cuda,cpu
  export XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_exclude_nondeterministic_ops=true'
  source "$original/scripts/hpc/boltz_node_tmpdir.sh"
fi
if [[ "$kind" == run ]]; then
  exec "$python" -u protenix_chemistry_screen.py --plan "$plan" --arm "$arm" --internal-seconds "$internal_seconds"
fi
exec "$python" -u chemistry_screen_control.py --plan "$plan" --kind "$kind" --internal-seconds "$internal_seconds"
