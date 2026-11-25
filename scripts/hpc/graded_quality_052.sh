#!/usr/bin/env bash
# Equal-third diagnostic profile. Tests and a small real checkpoint preflight
# precede the full sealed panel; every invocation has a distinct recorded run.
set -euo pipefail
: "${LSB_JOBID:?HPC allocation required}"
: "${LATENTMOL_WORK_ROOT:?run_job.sh must establish work root}"

python -m latentmol.experiments --run-root "$LATENTMOL_WORK_ROOT/runs" \
    --name graded-quality-tests-052 -- python -m pytest -q \
    tests/evaluation/test_quality.py tests/evaluation/test_evaluation.py tests/test_architecture_study.py \
    tests/datasets/test_sampling.py tests/datasets/test_chemistry.py tests/test_candidate_panel.py

python -m latentmol.architecture_study \
    --checkpoint "$LATENTMOL_WORK_ROOT/runs/20261001T001617Z_vae-train_c008c090/best_validity.pt" \
    --data "$LATENTMOL_WORK_ROOT/data/zinc215k-seed17-v1" \
    --run-root "$LATENTMOL_WORK_ROOT/runs" --device cpu --split val \
    --molecules 16 --draws-per-molecule 4 --seed 20261001 --temperature 1.0 \
    --quality-config configs/quality_equal_v2.json

python -m latentmol.quality_panel \
    --panel "$LATENTMOL_WORK_ROOT/runs/20261001T094301Z_decoder-candidate-panel_adc34cfe/panel.json" \
    --panel-sha256 8daf20dae156a9407525d94bb1bf22782ede3ca8d1a886676c6136c18248764c \
    --data "$LATENTMOL_WORK_ROOT/data/zinc215k-seed17-v1" \
    --run-root "$LATENTMOL_WORK_ROOT/runs" --device cpu \
    --draws-per-molecule 16 --temperature 1.0 --quality-config configs/quality_equal_v2.json
