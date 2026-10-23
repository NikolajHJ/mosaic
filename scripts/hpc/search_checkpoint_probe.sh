#!/usr/bin/env bash
# Saved-script recovery for the LSF multiline command quoting failure.
# Repeat only affected engineering fixtures, then actual FP32 CPU checkpoint.
set -euo pipefail
python -m pytest -q tests/test_search_repair_guidance.py tests/test_search_decoder_study.py
python scripts/search_actual_checkpoint_probe.py --config "$1" --sha256 "$2" --output-root "$3"
