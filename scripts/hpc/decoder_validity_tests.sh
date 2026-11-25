#!/usr/bin/env bash
# DTU CPU engineering checks. These do not demonstrate native training efficacy.
set -euo pipefail
python -m pytest -q tests/test_chemistry_regularization.py tests/guidance/test_connectivity_regularization.py \
  tests/test_decoder_finetune.py tests/test_decoder_quality_evaluation.py \
  tests/datasets/test_sampling.py tests/test_probability_repair.py
