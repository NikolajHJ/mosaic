#!/usr/bin/env bash
# Numerical engineering fixtures only, on DTU CPU; no chemistry benefit claim.
set -euo pipefail
bash scripts/hpc/decoder_validity_tests.sh
python -m pytest -q tests/test_search_repair_guidance.py tests/test_search_decoder_study.py
