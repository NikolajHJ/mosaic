#!/usr/bin/env bash
# Every numerical fixture runs on allocated DTU CPU, never on the local PC.
set -euo pipefail
: "${LSB_JOBID:?DTU allocation required}"
bash scripts/hpc/search_decoder_tests.sh
python -m pytest -q tests/test_search_decoder_full.py
