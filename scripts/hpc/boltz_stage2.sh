#!/usr/bin/env bash
# Execute after boltz_run.sh activates the pinned Boltz environment.
set -euo pipefail
: "${LSB_JOBID:?HPC allocation required}"
: "${BOLTZ_CACHE:?Invoke through boltz_run.sh}"
TASK_MODE="${1:?Provide ae or vae}"
case "$TASK_MODE" in
    ae)
        TASK_CHECKPOINT="$LATENTMOL_WORK_ROOT/runs/20260928T002738Z_ae-train_2f61b913/best_validity.pt"
        TASK_SHA="1b1d2a32aedfadb376d5a5e2d8802d574ed5c1d72851ed79517acaef418e9b95"
        TASK_REFERENCE="$LATENTMOL_WORK_ROOT/runs/20260928T063833Z_ae-latent-stage1_580abd75/latent_reference.pt"
        ;;
    vae)
        TASK_CHECKPOINT="$LATENTMOL_WORK_ROOT/runs/20260927T231648Z_vae-train_a0e7eba0/best_validity.pt"
        TASK_SHA="80b22b9c5a5e9010457b49246f830f0425c5578458eab28ddf2a10f5f632cee9"
        TASK_REFERENCE="$LATENTMOL_WORK_ROOT/runs/20260928T064117Z_vae-latent-stage1_bd6a6a7a/latent_reference.pt"
        ;;
    *) exit 2 ;;
esac
test "$(sha256sum "$TASK_CHECKPOINT" | cut -d ' ' -f 1)" = "$TASK_SHA"
python -m latentmol.experiments --run-root "$LATENTMOL_WORK_ROOT/runs" \
    --name "tests-boltz-stage2-${TASK_MODE}" -- \
    python -m pytest -q tests/test_boltz_adapter.py tests/test_boltz_latent_pilot.py \
    tests/test_latent_priors.py \
    --junitxml="$LATENTMOL_WORK_ROOT/logs/tests-boltz-stage2-${TASK_MODE}-${LSB_JOBID}.xml"
python -m latentmol.boltz.latent_pilot \
    --prepared "$LATENTMOL_WORK_ROOT/boltz-preparation/source006-attempt001/prepared.pt" \
    --affinity-checkpoint "$BOLTZ_CACHE/boltz2_aff.ckpt" \
    --checkpoint "$TASK_CHECKPOINT" \
    --reference "$TASK_REFERENCE" \
    --data-manifest "$LATENTMOL_WORK_ROOT/data/zinc215k-seed17-v1/manifest.json" \
    --run-root "$LATENTMOL_WORK_ROOT/runs" --mode "$TASK_MODE"
