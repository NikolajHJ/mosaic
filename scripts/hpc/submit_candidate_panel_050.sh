#!/usr/bin/env bash
# One-time retry: source049 failed its error-contract test before panel sealing.
set -euo pipefail

TASK_ROOT=/work3/s234818/latentmol
TASK_RELEASE="$TASK_ROOT/releases/source-050-next-test-plumbing-fix"
TASK_ARCHIVE="$TASK_RELEASE.tar.gz"
TASK_PLAN="$TASK_RELEASE/configs/next_test_panel_stable_provisional.json"
TASK_DIAGNOSTICS="$TASK_ROOT/diagnostics/candidate-panel-050"
TASK_RECEIPT="$TASK_DIAGNOSTICS/submission.txt"
test ! -e "$TASK_RECEIPT"
test "$(sha256sum "$TASK_ARCHIVE" | cut -d ' ' -f 1)" = \
    18c98bde779ed479f76a569fbcca02c3bddc0cf6a9f3a709c70daf347a1e0d4b
test "$(sha256sum "$TASK_PLAN" | cut -d ' ' -f 1)" = \
    07947a9cd10cb3dd9e12272a4f5f86b5ae7a36e02ab6e8d2d0a0f3861bba02c4
test -f "$TASK_ROOT/data/zinc215k-seed17-v1/manifest.json"

cd "$TASK_RELEASE"
TASK_SUBMISSION=$(bsub \
    -J candidate-panel-050 -q hpc -n 2 \
    -R "rusage[mem=4GB]" -R "span[hosts=1]" -W 0:30 \
    -o "$TASK_ROOT/logs/candidate-panel-050.%J.out" \
    -e "$TASK_ROOT/logs/candidate-panel-050.%J.err" \
    bash scripts/hpc/run_job.sh bash scripts/hpc/candidate_panel_preflight_050.sh)
printf '%s\n' "$TASK_SUBMISSION" | tee "$TASK_RECEIPT"
