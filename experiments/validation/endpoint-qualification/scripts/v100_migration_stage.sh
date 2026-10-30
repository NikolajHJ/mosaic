#!/bin/bash
set -euo pipefail
source_root=$1
mode=$2
module load python3/3.12.11
unset PYTHONHOME
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=0
export PYTHONPATH="$source_root"
if [[ "$mode" == compare ]]; then
  [[ "${LSB_QUEUE:-}" == gpuv100 ]]
  module load cuda/12.9.1 cudnn/v9.13.0.50-prod-cuda-12.X
else
  [[ "$mode" == migrate && "${LSB_QUEUE:-}" == hpc ]]
fi
cd "$source_root"
sha256sum -c checksums.sha256 >/dev/null
exec /work3/s234818/mosaic/envs/protenix-tiny/bin/python -u v100_migration.py "$mode" --source "$source_root"
