#!/usr/bin/env bash
# All fixtures and actual config parsing run on allocated DTU CPU resources.
set -euo pipefail
python -m pytest -q tests/test_defog_matched_pilot.py tests/test_defog_integration_smoke.py tests/test_dbmol_precision.py tests/test_repair_guidance.py tests/test_ramped_random_study.py tests/test_gpu_process_probe.py tests/test_boltz_active_set_rescore.py tests/test_probability_repair.py
python -m latentmol.repair_guidance.matched_pilot --plan "$1" --sha256 "$2" --validate-only
