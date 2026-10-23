#!/usr/bin/env bash
# UNTESTED submission template; NOT executed during the current HPC outage.
# Invocation is an explicit future submission, not a status check or auto-retry.
set -euo pipefail
TASK_RELEASE="${1:?Extracted immutable release directory}"
TASK_ARCHIVE="${2:?Immutable archive path}"
TASK_SHA="${3:?Expected archive SHA256}"
TASK_ENV="${4:?Prepared compatible venv directory}"
TASK_CONFIG="${5:?Study config path within release, e.g. configs/multimodel_boltz.yaml}"
TASK_STAGE="${6:-validate}"
TASK_ROOT="${LATENTMOL_WORK_ROOT:-/work3/$USER/latentmol}"
test "$(sha256sum "$TASK_ARCHIVE" | cut -d ' ' -f 1)" = "$TASK_SHA"
test -d "$TASK_RELEASE"
test -f "$TASK_RELEASE/$TASK_CONFIG"
test -x "$TASK_ENV/bin/python"
test "$(tar -xOf "$TASK_ARCHIVE" release_manifest.json | sha256sum | cut -d ' ' -f 1)" = "$(sha256sum "$TASK_RELEASE/release_manifest.json" | cut -d ' ' -f 1)"
TASK_ARMS=("${@:7}")
mkdir -p "$TASK_ROOT/logs" "$TASK_ROOT/diagnostics"
TASK_RECEIPTS=$(mktemp -d "$TASK_ROOT/diagnostics/multimodel-submit-XXXXXX")
export LATENTMOL_ENV_DIR="$TASK_ENV"
export LATENTMOL_SOURCE_ARCHIVE="$TASK_ARCHIVE"
export LATENTMOL_EXPECTED_ARCHIVE_SHA256="$TASK_SHA"
export LATENTMOL_WORK_ROOT="$TASK_ROOT"
cd "$TASK_RELEASE"
# Queue, GPU model, memory and wall time need refinement after the first native
# validation. Defaults follow earlier project A10 work, not a measured ensemble.
bsub -J "multimodel-$TASK_STAGE" -q "${LATENTMOL_QUEUE:-gpua10}" -n 4 \
  -R 'rusage[mem=8GB]' -R 'span[hosts=1]' -gpu 'num=1:mode=exclusive_process' \
  -W "${LATENTMOL_WALLTIME:-2:00}" \
  -o "$TASK_ROOT/logs/multimodel.%J.out" -e "$TASK_ROOT/logs/multimodel.%J.err" \
  bash scripts/hpc/multimodel_job.sh "$TASK_STAGE" "$TASK_CONFIG" "${TASK_ARMS[@]}" | tee "$TASK_RECEIPTS/submission.txt"
printf 'Record this receipt and exact source/config hashes in docs/05_experiment_ledger.md: %s\n' "$TASK_RECEIPTS"
