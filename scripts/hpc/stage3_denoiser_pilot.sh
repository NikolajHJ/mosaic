#!/usr/bin/env bash
# Bounded graph-denoiser pilot initialized from the frozen AE; GPU allocation required.
set -euo pipefail
: "${LSB_JOBID:?HPC allocation required}"
TASK_CHECKPOINT="$LATENTMOL_WORK_ROOT/runs/20260928T002738Z_ae-train_2f61b913/best_validity.pt"
TASK_EXPECTED_SHA="1b1d2a32aedfadb376d5a5e2d8802d574ed5c1d72851ed79517acaef418e9b95"
TASK_COHORTS="$LATENTMOL_WORK_ROOT/runs/20260928T063833Z_ae-latent-stage1_580abd75/cohorts.json"
TASK_COHORTS_SHA="dd0645a76ce6074398e3d196e999116f11b134d8abbebf9726368ebc2c94d3e2"
test "$(sha256sum "$TASK_CHECKPOINT" | cut -d ' ' -f 1)" = "$TASK_EXPECTED_SHA"
test "$(sha256sum "$TASK_COHORTS" | cut -d ' ' -f 1)" = "$TASK_COHORTS_SHA"
python -m latentmol.experiments --run-root "$LATENTMOL_WORK_ROOT/runs" \
    --name tests-stage3-denoiser-pilot -- \
    python -m pytest -q tests/test_denoiser.py tests/test_losses.py \
    --junitxml="$LATENTMOL_WORK_ROOT/logs/tests-stage3-denoiser-${LSB_JOBID}.xml"
python -m latentmol.denoiser_study --checkpoint "$TASK_CHECKPOINT" \
    --data "$LATENTMOL_WORK_ROOT/data/zinc215k-seed17-v1" \
    --cohorts "$TASK_COHORTS" \
    --run-root "$LATENTMOL_WORK_ROOT/runs" --device cuda
