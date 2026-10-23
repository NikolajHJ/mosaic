#!/usr/bin/env bash
# New profile: HPC engineering tests only; native checkpoint parity is separate.
set -euo pipefail
: "${LSB_JOBID:?DTU allocation required}"
: "${LATENTMOL_SOURCE_SHA256:?Immutable source required}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python -m pytest -q tests/test_dbmol_objective.py tests/test_dbmol_model.py \
  tests/test_dbmol_gradient_gate.py tests/test_dbmol_prepare.py \
  tests/test_boltz_scoring.py tests/test_boltz_adapter.py \
  tests/test_multimodel.py tests/test_config_identity.py \
  tests/test_boltz_active_set.py tests/test_boltz_active_set_runtime.py \
  tests/test_boltz_active_set_rescore.py tests/test_stability_study.py
