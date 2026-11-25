#!/usr/bin/env bash
# A frozen Torch scorer and the shared restart engine in one allocation.
set -euo pipefail
: "${LSB_JOBID:?Run inside an allocated GPU job}"
module load python3/3.11.9
unset PYTHONHOME
export PYTHONPATH="$PWD/src"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
ulimit -c 0
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
raw = Path(argv[argv.index("--plan") + 1]).read_bytes()
if hashlib.sha256(raw).hexdigest() != argv[argv.index("--plan-sha256") + 1]:
    raise ValueError("GPU campaign plan changed")
plan = json.loads(raw)
helper = Path(plan["cuda_ownership_helper"])
if hashlib.sha256(helper.read_bytes()).hexdigest() != plan["cuda_ownership_sha256"]:
    raise ValueError("CUDA ownership helper changed")
spec = importlib.util.spec_from_file_location("_restart_cuda_routing", helper)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
proof = module.resolve_visibility(os.environ)
environment = dict(os.environ, CUDA_VISIBLE_DEVICES=proof["selected_cuda_visible_devices"])
archive = Path(os.environ["DBMOL_LOCAL_SOURCE"]).parent / "source.tar.gz"
environment["LATENTMOL_SOURCE_SHA256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
print(json.dumps(dict(event="cuda_visibility_resolved", job_id=os.environ["LSB_JOBID"], **proof)),
      file=sys.stderr, flush=True)
os.execvpe(argv[0], argv, environment)
PY
