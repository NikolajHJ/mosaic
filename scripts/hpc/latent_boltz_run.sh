#!/usr/bin/env bash
# Frozen VAE CPU decoder and one native Torch GPU owner, in a new allocation.
set -euo pipefail
: "${LSB_JOBID:?Allocated GPU job required}"
module load python3/3.11.9
unset PYTHONHOME
export PYTHONPATH="$PWD/src:$PWD/tests"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
ulimit -c 0
exec /usr/bin/python3 - "$@" <<'PY'
import hashlib,importlib.util,json,os,sys
from pathlib import Path
argv=sys.argv[1:]
raw=Path(argv[argv.index('--plan')+1]).read_bytes()
if hashlib.sha256(raw).hexdigest()!=argv[argv.index('--plan-sha256')+1]:raise ValueError('Plan changed')
plan=json.loads(raw);path=Path(plan['cuda_ownership_helper'])
if hashlib.sha256(path.read_bytes()).hexdigest()!=plan['cuda_ownership_sha256']:raise ValueError('CUDA helper changed')
spec=importlib.util.spec_from_file_location('_owned_cuda',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
proof=m.resolve_visibility(os.environ)
env=dict(os.environ,CUDA_VISIBLE_DEVICES=proof['selected_cuda_visible_devices'])
print(json.dumps({'event':'cuda_visibility_resolved',**proof}),flush=True)
os.execvpe(argv[0],argv,env)
PY
