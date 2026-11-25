#!/usr/bin/env bash
# New latent-design allocations only; does not change the running VAE shell.
set -euo pipefail
: "${LSB_JOBID:?Run the native Protenix bridge inside an allocated GPU job}"
experiment_import_root=""
if [[ "${1:-}" == "--nesso-blend" ]]; then
  shift
  export COMBINED_NESSO_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
  experiment_import_root="$PWD"
fi
module load python3/3.12.11
module load cuda/12.9.1 cudnn/v9.13.0.50-prod-cuda-12.X
unset PYTHONHOME
export PYTHONPATH="$PWD/src${experiment_import_root:+:$experiment_import_root}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_COMPILATION_CACHE=false JAX_ENABLE_X64=0
export JAX_PLATFORMS=cuda,cpu
export XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_exclude_nondeterministic_ops=true'
ulimit -c 0
# Torch runs latent optimization on CPU. Native workers share the allocated GPU.
# Translate LSF task routing BEFORE any Torch/JAX import. Reuse Mosaic's pinned,
# standard-library-only resolver rather than guessing a physical GPU ordinal.
exec /usr/bin/python3 - "$@" <<'PY'
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

argv = sys.argv[1:]
if argv.count("--plan") != 1 or argv.count("--plan-sha256") != 1:
    raise ValueError("GPU launch requires its exact campaign plan")
raw_plan = Path(argv[argv.index("--plan") + 1]).read_bytes()
if hashlib.sha256(raw_plan).hexdigest() != argv[argv.index("--plan-sha256") + 1]:
    raise ValueError("GPU launch campaign plan changed")
plan = json.loads(raw_plan)
helper = Path(plan["cuda_ownership_helper"])
if hashlib.sha256(helper.read_bytes()).hexdigest() != plan["cuda_ownership_sha256"]:
    raise ValueError("CUDA ownership helper changed")
spec = importlib.util.spec_from_file_location("_latent_protenix_cuda_routing", helper)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
if plan.get("nesso_blend") and os.environ.get("LSB_QUEUE") == "gpua100i":
    assigned = os.environ["DCCGPU"]
    selected = os.environ.get("CUDA_VISIBLE_DEVICES") or assigned
    if not selected.isdecimal() or not assigned.isdecimal() or selected != assigned:
        raise ValueError("Expected one scheduler-assigned interactive GPU")
    proof = {"selected_cuda_visible_devices": selected, "authority": "a100sh"}
else:
    proof = module.resolve_visibility(os.environ)
environment = dict(os.environ)
environment["CUDA_VISIBLE_DEVICES"] = proof["selected_cuda_visible_devices"]
print(json.dumps(dict(event="cuda_visibility_resolved", job_id=os.environ["LSB_JOBID"], **proof)),
      file=sys.stderr, flush=True)
os.execvpe(argv[0], argv, environment)
PY
