#!/usr/bin/env bash
# Allocated regression checks only; native full-width proof is a separate gate.
set -euo pipefail
: "${LSB_JOBID:?DTU allocation required}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
bash scripts/hpc/decoder_density_tests.sh
python -m pytest -q tests/test_decoder_long_resume.py tests/test_decoder_density_long_protocol.py tests/test_decoder_checkpoint_storage.py
