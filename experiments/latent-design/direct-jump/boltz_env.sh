#!/usr/bin/env bash
# Source from the allocated worker before running the experiment CLI.
export DIRECT_JUMP_PYTHON=/work3/s234818/latentmol/env/boltz311/bin/python
export OMP_NUM_THREADS=${LSB_DJOB_NUMPROC:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=0

# Prefer the campaign snapshot on PYTHONPATH. This existing sealed source
# also contains the native probability-to-affinity implementation.
_direct_jump_boltz_src=/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01/boltz-dbmol-pilot-release-02/source/src
export PYTHONPATH=${PYTHONPATH:+$PYTHONPATH:}$_direct_jump_boltz_src
unset _direct_jump_boltz_src
