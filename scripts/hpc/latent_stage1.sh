#!/usr/bin/env bash
# Stage 1 uses existing selected checkpoints; it never reads the test split.
set -euo pipefail
TASK_MODE="${1:?Provide ae or vae}"
TASK_CHECKPOINT="${2:?Provide validation-selected checkpoint path}"
TASK_EXPECTED_SHA="${3:?Provide sealed checkpoint SHA-256}"
case "$TASK_MODE" in ae|vae) ;; *) exit 2 ;; esac
test "$(sha256sum "$TASK_CHECKPOINT" | cut -d ' ' -f 1)" = "$TASK_EXPECTED_SHA"
python -m latentmol.experiments --run-root "$LATENTMOL_WORK_ROOT/runs" --name "tests-latent-stage1-${TASK_MODE}" -- \
    python -m pytest -q tests/test_latent_priors.py tests/test_models.py tests/test_evaluation.py \
    --junitxml="$LATENTMOL_WORK_ROOT/logs/tests-latent-stage1-${LSB_JOBID}.xml"
python -m latentmol.latent_study --checkpoint "$TASK_CHECKPOINT" \
    --data "$LATENTMOL_WORK_ROOT/data/zinc215k-seed17-v1" \
    --run-root "$LATENTMOL_WORK_ROOT/runs" --mode "$TASK_MODE" --device cuda
