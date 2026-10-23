#!/usr/bin/env bash
# HPC-only engineering gate. This code has not yet passed these tests.
set -euo pipefail
: "${LSB_JOBID:?DTU allocation required}"
: "${LATENTMOL_SOURCE_SHA256:?Immutable source required}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest -q \
  tests/test_optimization.py tests/test_boltz_step_control.py \
  tests/test_step_control_comparison.py tests/test_boltz_local_fd.py \
  tests/test_boltz_active_set.py tests/test_boltz_active_set_runtime.py \
  tests/test_boltz_active_set_diagnostic.py tests/test_gate_replay_study.py \
  tests/test_config_identity.py tests/test_random_start_study.py \
  tests/test_protein_panel.py tests/test_boltz_active_set_rescore.py \
  tests/test_trajectory_terminal.py tests/test_stability_study.py \
  tests/test_probability_repair.py tests/test_stability_submission.py
