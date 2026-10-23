#!/usr/bin/env bash
# One seed per serial allocation; GPU equivalence gate precedes first segment.
set -euo pipefail
: "${LSB_JOBID:?DTU allocation required}"
TASK_SEGMENT="$4"
if [ "$TASK_SEGMENT" = "0" ]; then
    python -m pytest -q tests/test_decoder_resume.py -k cuda
    python scripts/decoder_budget_native_gate.py --output-root "$3" --sha256 "$2" --create
else
    python scripts/decoder_budget_native_gate.py --output-root "$3" --sha256 "$2"
fi
python -m latentmol.decoder_budget_recovery --config "$1" --sha256 "$2" --output-root "$3" --segment "$TASK_SEGMENT"
