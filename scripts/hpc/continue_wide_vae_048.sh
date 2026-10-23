#!/usr/bin/env bash
# Exact source042 wide-VAE continuation: epoch 120 through epoch 180.
set -euo pipefail
: "${LSB_JOBID:?HPC allocation required}"
: "${LATENTMOL_WORK_ROOT:?run_job.sh must establish the work root}"

# The same GPU/runtime setting is required by the epoch-boundary resume gate.
python -m latentmol.experiments \
    --run-root "$LATENTMOL_WORK_ROOT/runs" \
    --name "gpu-rng-before-wide-vae-continuation" -- \
    python -m pytest -q \
    tests/test_training.py::test_cuda_rng_state_is_restored_from_weights_only_checkpoint

python scripts/hpc/continue_graph_training.py \
    --config configs/arch_wide_global_bond_vae_continue180.yaml \
    --data "$LATENTMOL_WORK_ROOT/data/zinc215k-seed17-v1" \
    --run-root "$LATENTMOL_WORK_ROOT/runs" \
    --resume-from "$LATENTMOL_WORK_ROOT/runs/20260930T141333Z_vae-train_7f1a1117/last.pt" \
    --expected-resume-sha256 ea5eeddeddf3fa10e74237af6051e6be9d11054c090f3703d74c572ab504073c \
    --expected-resume-epoch 120 \
    --parent-history "$LATENTMOL_WORK_ROOT/runs/20260930T041423Z_vae-full-workflow_1a3a89c8/history.json" \
    --neighborhood-limit 256 --prior-limit 1024
