#!/usr/bin/env bash
# Sequential matched AE/VAE ablation; baseline artifacts remain separate.
set -euo pipefail
for TASK_MODE in ae vae; do
    python -m latentmol.train_and_evaluate --config "configs/${TASK_MODE}_bond_balance.yaml" \
        --data "$LATENTMOL_WORK_ROOT/data/zinc5k-seed17-v1" \
        --run-root "$LATENTMOL_WORK_ROOT/runs"
done
