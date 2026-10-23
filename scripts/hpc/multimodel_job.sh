#!/usr/bin/env bash
# UNTESTED ON HPC. Run only from an immutable extracted source release.
# This script does not install native packages or fetch weights; prepare a
# compatible environment and pinned native inputs first (docs28).
set -euo pipefail
: "${LSB_JOBID:?Run inside an LSF allocation}"
: "${LATENTMOL_ENV_DIR:?Path to the prepared compatible Python environment}"
: "${LATENTMOL_SOURCE_ARCHIVE:?Path to immutable source tarball}"
: "${LATENTMOL_EXPECTED_ARCHIVE_SHA256:?Expected tarball digest}"
: "${LATENTMOL_WORK_ROOT:=/work3/$USER/latentmol}"
TASK_STAGE="${1:?Usage: multimodel_job.sh STAGE CONFIG [ARM ...]}"
TASK_CONFIG="${2:?A configured study YAML is required}"
shift 2
test "$(sha256sum "$LATENTMOL_SOURCE_ARCHIVE" | cut -d ' ' -f 1)" = "$LATENTMOL_EXPECTED_ARCHIVE_SHA256"
# Compare the packaged manifest itself with the extracted manifest before code
# execution; the Python check below then verifies every recorded source byte.
test "$(tar -xOf "$LATENTMOL_SOURCE_ARCHIVE" release_manifest.json | sha256sum | cut -d ' ' -f 1)" = "$(sha256sum release_manifest.json | cut -d ' ' -f 1)"
export LATENTMOL_SOURCE_SHA256="$LATENTMOL_EXPECTED_ARCHIVE_SHA256"
unset PYTHONHOME PYTHONPATH
module load "${LATENTMOL_PYTHON_MODULE:-python3/3.11.9}"
source "$LATENTMOL_ENV_DIR/bin/activate"
export PYTHONPATH="$PWD/src"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${LSB_DJOB_NUMPROC:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
ulimit -c 0
mkdir -p "$LATENTMOL_WORK_ROOT/runs" "$LATENTMOL_WORK_ROOT/diagnostics"
TASK_RECEIPT_DIR=$(mktemp -d "$LATENTMOL_WORK_ROOT/diagnostics/multimodel-${LSB_JOBID}-XXXXXX")
# release_manifest.json is created by scripts/package_multimodel_release.py.
python -c 'import json,pathlib; from latentmol.experiments import source_manifest; p=pathlib.Path.cwd(); assert source_manifest(p)["sha256"] == json.loads((p/"release_manifest.json").read_text())["sha256"], "Extracted source differs from release manifest"'
TASK_ARMS=()
if [ "$#" -gt 0 ]; then TASK_ARMS=(--arms "$@"); fi
case "$TASK_STAGE" in
  tests)
    python -m latentmol.experiments --run-root "$LATENTMOL_WORK_ROOT/runs" \
      --name multimodel-contract-tests --source-root "$PWD" -- \
      python -m pytest -q tests/test_multimodel.py tests/test_multimodel_external.py \
      tests/test_scoring.py tests/test_boltz_scoring.py tests/test_graph_policy.py
    ;;
  validate|pipeline|plan)
    python -m latentmol.multimodel "$TASK_STAGE" --config "$TASK_CONFIG" \
      --run-root "$LATENTMOL_WORK_ROOT/runs" --source-root "$PWD" \
      --allow-experimental --output "$TASK_RECEIPT_DIR/receipt.json" "${TASK_ARMS[@]}"
    ;;
  *) echo 'Use tests, plan, validate or pipeline. For run/confirm/diagnose use the documented CLI with explicit evidence paths.' >&2; exit 2 ;;
esac
