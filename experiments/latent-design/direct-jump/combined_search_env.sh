#!/usr/bin/env bash
set -eo pipefail
if [[ "$1" != "--routed" ]]; then
  # Capture the allocation environment before the router loads the JAX modules.
  export COMBINED_NESSO_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
  if [[ "${LSB_QUEUE:-}" == "gpua100i" ]]; then
    # a100sh manages two shared GPUs and picks one for the interactive shell.
    # Its per-task list spans the host and is not a batch MPS single-device map.
    : "${LSB_JOBID:?requires an interactive LSF job}" "${DCCGPU:?a100sh did not assign a GPU}"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$DCCGPU}"
    [[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+$ ]] || { echo "Expected one a100sh GPU" >&2; exit 1; }
    export XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_exclude_nondeterministic_ops=true'
    echo "a100sh allocation $LSB_JOBID, GPU $CUDA_VISIBLE_DEVICES" >&2
    exec bash "$0" --routed "$@"
  else
    exec bash scripts/hpc/protenix_latent_run.sh bash "$0" --routed "$@"
  fi
fi
shift
exp="$PWD/experiments/latent-design/direct-jump"
source "$exp/protenix_env.sh"
export PYTHONPATH="$PWD/src:$DIRECT_JUMP_MOSAIC_SOURCE/src/mosaic/small_molecule/Stage2:$PYTHONPATH"
export DIRECT_REPAIR_PYTHON=/work3/s234818/latentmol/env/boltz311/bin/python
export COMBINED_NESSO_ASSETS=/work3/s234818/latentmol/assets/nesso1-brd4-20261120a
export NESSO_ASSETS="$COMBINED_NESSO_ASSETS" NESSO_PYTHON="$COMBINED_NESSO_ASSETS/env/bin/python"
export TOKENIZERS_PARALLELISM=false
exec "$DIRECT_JUMP_PYTHON" -u "$exp/combined_search.py" "$@"
