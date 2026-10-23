#!/usr/bin/env bash
# Full affected CPU fixtures, including real optimizer/RNG resume equivalence.
set -euo pipefail
: "${LSB_JOBID:?DTU allocation required}"
bash scripts/hpc/decoder_validity_tests.sh
python -m pytest -q tests/test_decoder_resume.py tests/test_search_repair_guidance.py tests/test_search_decoder_study.py
