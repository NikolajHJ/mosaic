#!/usr/bin/env bash
# Submit once from the DTU login node after the immutable release is verified.
set -euo pipefail

TASK_ROOT=/work3/s234818/latentmol
TASK_RELEASE="$TASK_ROOT/releases/source-048-wide-vae-continuation"
TASK_ARCHIVE="$TASK_RELEASE.tar.gz"
TASK_PARENT="$TASK_ROOT/runs/20260930T141333Z_vae-train_7f1a1117/last.pt"
TASK_DIAGNOSTICS="$TASK_ROOT/diagnostics/wide-vae-continuation-048"
TASK_RECEIPT="$TASK_DIAGNOSTICS/submission.txt"
test ! -e "$TASK_RECEIPT"
test "$(sha256sum "$TASK_ARCHIVE" | cut -d ' ' -f 1)" = \
    c1c556ecc775920699d58664646507baf0a44f5f607e7d2690f7c047a05c04de
test "$(sha256sum "$TASK_PARENT" | cut -d ' ' -f 1)" = \
    ea5eeddeddf3fa10e74237af6051e6be9d11054c090f3703d74c572ab504073c
test "$(sha256sum "$TASK_RELEASE/configs/arch_wide_global_bond_vae_continue180.yaml" | cut -d ' ' -f 1)" = \
    2369304fdbf1d8d0f507060a139b6fa5f7584ca92ab7d324c50cfa5e819b05de
test -f "$TASK_ROOT/runs/20260930T041423Z_vae-full-workflow_1a3a89c8/history.json"

cd "$TASK_RELEASE"
TASK_SUBMISSION=$(bsub \
    -J wide-vae-continuation-048 -q gpua10 -n 4 \
    -gpu "num=1:mode=exclusive_process" \
    -R "rusage[mem=4GB]" -R "span[hosts=1]" -W 12:00 \
    -o "$TASK_ROOT/logs/wide-vae-continuation-048.%J.out" \
    -e "$TASK_ROOT/logs/wide-vae-continuation-048.%J.err" \
    bash scripts/hpc/run_job.sh bash scripts/hpc/continue_wide_vae_048.sh)
printf '%s\n' "$TASK_SUBMISSION" | tee "$TASK_RECEIPT"
