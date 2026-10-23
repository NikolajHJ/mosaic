#!/usr/bin/env bash
# Run from an immutable source release, inside an LSF allocation.
set -euo pipefail
: "${LSB_JOBID:?This entrypoint must run in an LSF job}"
module load python3/3.11.9
export LATENTMOL_WORK_ROOT="${LATENTMOL_WORK_ROOT:-/work3/$USER/latentmol}"
# LSF may filter submission-shell environment variables. Derive provenance from
# the immutable release layout instead of assuming exports survive submission.
TASK_ARCHIVE="${LATENTMOL_SOURCE_ARCHIVE:-${PWD}.tar.gz}"
if [ ! -f "$TASK_ARCHIVE" ]; then
    printf 'Missing immutable source archive: %s\n' "$TASK_ARCHIVE" >&2
    exit 2
fi
export LATENTMOL_SOURCE_SHA256="$(sha256sum "$TASK_ARCHIVE" | cut -d ' ' -f 1)"
source "$LATENTMOL_WORK_ROOT/env/ae/bin/activate"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${LSB_DJOB_NUMPROC:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export UV_CACHE_DIR="$LATENTMOL_WORK_ROOT/cache/uv"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR="$LATENTMOL_WORK_ROOT/cache/matplotlib"
export MPLBACKEND=Agg
mkdir -p "$MPLCONFIGDIR"
ulimit -c 0
printf 'JOB=%s HOST=%s SOURCE=%s\n' "$LSB_JOBID" "$(hostname)" "$PWD"
"$@"
