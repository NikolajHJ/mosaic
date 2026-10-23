#!/usr/bin/env bash
# Validate the release, evaluate VAE priors, then exercise the real Boltz path.
# This engineering stage preserves existing training artifacts and uses unique
# preparation outputs. Invoke via boltz_run.sh in an allocated GPU job.
set -euo pipefail
TASK_BASELINE_AE="$LATENTMOL_WORK_ROOT/runs/20260927T135410Z_ae-train_4b2da706/last.pt"
TASK_ATTEMPT="${1:?Provide a unique preparation output directory}"
python -m latentmol.experiments --run-root "$LATENTMOL_WORK_ROOT/runs" --name tests-before-boltz -- \
    python -m pytest -q tests --junitxml="$LATENTMOL_WORK_ROOT/logs/tests-${LSB_JOBID}.xml"
for TASK_TRAIN in 20260927T135938Z_vae-train_19abddb1 20260927T141215Z_vae-train_0875cdbf; do
    for TASK_CHECKPOINT in best last; do
        python -m latentmol.evaluate --checkpoint "$LATENTMOL_WORK_ROOT/runs/$TASK_TRAIN/$TASK_CHECKPOINT.pt" \
            --data "$LATENTMOL_WORK_ROOT/data/zinc5k-seed17-v1" --run-root "$LATENTMOL_WORK_ROOT/runs" \
            --split val --limit 256 --seed 17 --device cuda --novelty
    done
done
python -m latentmol.experiments --run-root "$LATENTMOL_WORK_ROOT/runs" --name boltz-preparation -- \
    bash scripts/hpc/boltz_prepare.sh "$TASK_ATTEMPT" "$TASK_BASELINE_AE"
bash scripts/hpc/boltz_smoke.sh "$TASK_ATTEMPT/prepared.pt" "$TASK_BASELINE_AE"
