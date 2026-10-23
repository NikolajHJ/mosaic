#!/usr/bin/env bash
# Candidate-start native DeFoG worker, isolated from Boltz's environment.
# Implemented but numerically unvalidated. Run only inside an immutable HPC
# release; the parent optimizer supplies unique JSON request/receipt paths.
set -euo pipefail
: "${LSB_JOBID:?DTU HPC allocation required}"
task_root="${LATENTMOL_WORK_ROOT:-/work3/$USER/latentmol}"
task_python="$task_root/env/defog311/bin/python"
test -x "$task_python"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
exec "$task_python" -m latentmol.defog.candidate_repair "$@"
