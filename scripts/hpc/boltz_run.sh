#!/usr/bin/env bash
# Run from an immutable source release, inside an LSF allocation. No local runs.
set -euo pipefail
: "${LSB_JOBID:?This entrypoint must run in an LSF job}"
module load python3/3.11.9
unset PYTHONHOME PYTHONPATH
export LATENTMOL_WORK_ROOT="${LATENTMOL_WORK_ROOT:-/work3/$USER/latentmol}"
TASK_ARCHIVE="${LATENTMOL_SOURCE_ARCHIVE:-${PWD}.tar.gz}"
test -f "$TASK_ARCHIVE"
export LATENTMOL_SOURCE_SHA256="$(sha256sum "$TASK_ARCHIVE" | cut -d ' ' -f 1)"
source "$LATENTMOL_WORK_ROOT/env/boltz311/bin/activate"
export PYTHONPATH="$PWD/src"
export PYTHONUNBUFFERED=1
export BOLTZ_CACHE="$LATENTMOL_WORK_ROOT/cache/boltz2"
export UV_CACHE_DIR="$LATENTMOL_WORK_ROOT/cache/uv"
export TMPDIR="$LATENTMOL_WORK_ROOT/cache/tmp"
export OMP_NUM_THREADS="${LSB_DJOB_NUMPROC:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$TMPDIR"
ulimit -c 0
printf 'JOB=%s HOST=%s SOURCE=%s\n' "$LSB_JOBID" "$(hostname)" "$PWD"
"$@"
