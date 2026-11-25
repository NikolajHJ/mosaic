#!/usr/bin/env bash
# Invoke after the allocation launcher has bound CUDA_VISIBLE_DEVICES.
set +e
set +o pipefail
source /etc/profile
set -eo pipefail
type module >/dev/null
: "${LSB_JOBID:?Run fresh scoring in an allocated GPU job}"
dir=$(cd "$(dirname "$0")" && pwd)
source_root=$(cd "$dir/../../.." && pwd)
args=("$@")
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) model=$2; shift 2 ;;
        --work) work=$(realpath -m -- "$2"); shift 2 ;;
        *) shift ;;
    esac
done
: "${work:?Missing --work}" "${model:?Missing --model}"
root=/work3/s234818/latentmol
if [[ $model == protenix ]]; then
    module load python3/3.12.11 cuda/12.9.1 cudnn/v9.13.0.50-prod-cuda-12.X
    export PYTHONPATH="$source_root/src:$root/protenix_occupancy/environment-attempt003/deps"
    export PROTENIX_ROOT_DIR="$root/protenix_occupancy/data" LAYERNORM_TYPE=torch
else
    module load python3/3.11.9
    unset PYTHONPATH
fi
unset PYTHONHOME
threads=${LSB_DJOB_NUMPROC:-4}
export OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads OPENBLAS_NUM_THREADS=$threads
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$work/runtime-cache" MPLCONFIGDIR="$work/runtime-cache/mpl"
export TORCH_HOME="$XDG_CACHE_HOME/torch" TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda" TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
exec "$root/env/boltz311/bin/python" -u "$dir/fresh_scores.py" "${args[@]}"
