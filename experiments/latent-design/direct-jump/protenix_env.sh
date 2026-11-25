#!/usr/bin/env bash
# Source inside the allocation before invoking the direct-graph runner.
: "${LSB_JOBID:?Direct Protenix requires an allocated GPU}"
module load python3/3.12.11 cuda/12.9.1 cudnn/v9.13.0.50-prod-cuda-12.X
unset PYTHONHOME PYTHONOPTIMIZE
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export JAX_PLATFORMS=cuda JAX_ENABLE_X64=0 JAX_ENABLE_COMPILATION_CACHE=false
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
protenix_campaign=/work3/s234818/mosaic/runs/stage3-protenix-defog-lr60-3seeds-20261112a
export DIRECT_JUMP_MOSAIC_SOURCE="$protenix_campaign/projection-kkt-retry-20261117a/source"
export DIRECT_JUMP_PROTENIX_ASSETS="$protenix_campaign/assets.json"
export DIRECT_JUMP_PYTHON=/work3/s234818/mosaic/envs/protenix-tiny/bin/python
protenix_study="$DIRECT_JUMP_MOSAIC_SOURCE/experiments/learned-prior/protenix-defog-composition-30000"
export PYTHONPATH="$DIRECT_JUMP_MOSAIC_SOURCE/src:$protenix_study/scripts${PYTHONPATH:+:$PYTHONPATH}"
