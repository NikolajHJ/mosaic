#!/usr/bin/env bash
# One immutable source051 CPU allocation; tests gate the actual decoder study.
set -euo pipefail
TASK_ROOT=/work3/s234818/latentmol
TASK_RELEASE="$TASK_ROOT/releases/source-051-graded-quality"
TASK_RECEIPTS="$TASK_ROOT/diagnostics/graded-quality-051"
TASK_EXPECTED_SHA="${1:?Pass the locally verified source archive SHA256}"
test "$(sha256sum "$TASK_RELEASE.tar.gz" | cut -d ' ' -f 1)" = "$TASK_EXPECTED_SHA"
mkdir -p "$TASK_RECEIPTS"
test ! -e "$TASK_RECEIPTS/submission.txt"
cd "$TASK_RELEASE"
TASK_SUBMISSION=$(bsub -J graded-quality-051 -q hpc -n 4 \
    -R "rusage[mem=4GB]" -R "span[hosts=1]" -W 0:30 \
    -o "$TASK_ROOT/logs/graded-quality-051.%J.out" \
    -e "$TASK_ROOT/logs/graded-quality-051.%J.err" \
    bash scripts/hpc/run_job.sh bash scripts/hpc/graded_quality_051.sh)
printf '%s\n' "$TASK_SUBMISSION" | tee "$TASK_RECEIPTS/submission.txt"
