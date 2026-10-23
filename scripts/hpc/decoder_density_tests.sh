#!/usr/bin/env bash
# Allocated CPU fixtures; native50-slot profiling/training is a later GPU gate.
set -euo pipefail
: "${LSB_JOBID:?DTU allocation required}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
bash scripts/hpc/decoder_validity_tests.sh
python -m pytest -q tests/test_decoder_density.py tests/test_decoder_resume.py
