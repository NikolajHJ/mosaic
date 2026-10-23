#!/usr/bin/env bash
# One-time DTU submission after source049 and the pinned plan are verified.
set -euo pipefail

TASK_ROOT=/work3/s234818/latentmol
TASK_RELEASE="$TASK_ROOT/releases/source-049-next-test-plumbing"
TASK_ARCHIVE="$TASK_RELEASE.tar.gz"
TASK_PLAN="$TASK_RELEASE/configs/next_test_panel_stable_provisional.json"
TASK_DIAGNOSTICS="$TASK_ROOT/diagnostics/candidate-panel-049"
TASK_RECEIPT="$TASK_DIAGNOSTICS/submission.txt"
test ! -e "$TASK_RECEIPT"
test "$(sha256sum "$TASK_ARCHIVE" | cut -d ' ' -f 1)" = \
    4f093c2f46cbebca60e1075cf107ff628031b1a4220b14d382c294fbaa7adf4d
test "$(sha256sum "$TASK_PLAN" | cut -d ' ' -f 1)" = \
    07947a9cd10cb3dd9e12272a4f5f86b5ae7a36e02ab6e8d2d0a0f3861bba02c4
test -f "$TASK_ROOT/data/zinc215k-seed17-v1/manifest.json"

cd "$TASK_RELEASE"
TASK_SUBMISSION=$(bsub \
    -J candidate-panel-049 -q hpc -n 2 \
    -R "rusage[mem=4GB]" -R "span[hosts=1]" -W 0:30 \
    -o "$TASK_ROOT/logs/candidate-panel-049.%J.out" \
    -e "$TASK_ROOT/logs/candidate-panel-049.%J.err" \
    bash scripts/hpc/run_job.sh bash scripts/hpc/candidate_panel_preflight_049.sh)
printf '%s\n' "$TASK_SUBMISSION" | tee "$TASK_RECEIPT"
