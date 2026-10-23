#!/usr/bin/env bash
# Source052 is a new diagnostic release; source051 and its results stay immutable.
set -euo pipefail
TASK_ROOT=/work3/s234818/latentmol
TASK_RELEASE="$TASK_ROOT/releases/source-052-equal-repair-quality"
TASK_RECEIPTS="$TASK_ROOT/diagnostics/graded-quality-052"
TASK_EXPECTED_SHA="${1:?Pass locally verified archive SHA256}"
test "$(sha256sum "$TASK_RELEASE.tar.gz" | cut -d ' ' -f 1)" = "$TASK_EXPECTED_SHA"
mkdir -p "$TASK_RECEIPTS"
test ! -e "$TASK_RECEIPTS/submission.txt"
cd "$TASK_RELEASE"
TASK_SUBMISSION=$(bsub -J graded-quality-052 -q hpc -n 4 \
    -R "rusage[mem=4GB]" -R "span[hosts=1]" -W 1:00 \
    -o "$TASK_ROOT/logs/graded-quality-052.%J.out" \
    -e "$TASK_ROOT/logs/graded-quality-052.%J.err" \
    bash scripts/hpc/run_job.sh bash scripts/hpc/graded_quality_052.sh)
printf '%s\n' "$TASK_SUBMISSION" | tee "$TASK_RECEIPTS/submission.txt"
