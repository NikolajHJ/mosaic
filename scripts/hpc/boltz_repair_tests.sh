#!/usr/bin/env bash
# Engineering fixtures, no pretrained model inference or scientific evidence.
set -euo pipefail
: "${LSB_JOBID:?DTU allocation required}"
: "${LATENTMOL_SOURCE_SHA256:?Immutable source required}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest -q tests/test_repair_guidance.py tests/test_repair_guidance_provider.py tests/test_repair_guidance_study.py \
  tests/test_defog_candidate_repair.py tests/datasets/test_sampling.py tests/test_graph_policy.py \
  tests/test_boltz_active_set_runtime.py tests/test_boltz_step_control.py \
  tests/test_dbmol_gradient_gate.py tests/test_config_identity.py
