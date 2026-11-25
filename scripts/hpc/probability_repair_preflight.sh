#!/usr/bin/env bash
# Run regression fixtures, exact parent-metadata recovery and chemistry diagnostics
# on CPU HPC. This loads only the frozen historical graph decoder, not Boltz.
set -euo pipefail
TASK_PROTOCOL="$1"; TASK_PROTOCOL_SHA="$2"; TASK_GATE_EVIDENCE="$3"
printf '%s  %s\n' "$TASK_PROTOCOL_SHA" "$TASK_PROTOCOL" | sha256sum --check
python -m pytest -q tests/test_config_identity.py tests/test_probability_repair.py \
  tests/evaluation/test_quality.py tests/test_fresh_trajectory.py tests/test_processed_identity.py \
  tests/test_boltz_active_set_rescore.py tests/test_boltz_active_set_diagnostic.py \
  tests/test_gate_replay_study.py
python - "$TASK_GATE_EVIDENCE" <<'PY'
import json,sys
from pathlib import Path
from latentmol.config_identity import verify_config_identity
from latentmol.boltz.trajectory_selection import _pilot_candidates
rows=json.loads(Path(sys.argv[1]).read_text())
for row in rows:
    root=Path(row['pilot'])
    record=json.loads((root/'run.json').read_text())
    receipt=verify_config_identity(record,allow_legacy_active_set_contexts=True)
    verified,_,_,warnings=_pilot_candidates(root,record['source_archive_sha256'],record['config']['config_file_sha256'])
    assert verified['run_id']==record['run_id']
    print(json.dumps({'parent':row['case'],'identity':receipt,'warnings':warnings}),flush=True)
print('HISTORICAL_PARENT_READERS_PASSED',len(rows),flush=True)
PY
python -m latentmol.probability_repair_audit --protocol "$TASK_PROTOCOL" --sha256 "$TASK_PROTOCOL_SHA"
