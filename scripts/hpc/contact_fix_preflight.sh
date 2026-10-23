#!/usr/bin/env bash
# Numerical fixtures/config validation run only inside allocated DTU CPU jobs.
set -euo pipefail
python -m pytest -q tests/test_contact_projection.py tests/test_branch_local_fd.py tests/test_contact_precision_validation.py tests/test_contact_fix_campaign.py tests/test_dbmol_precision.py tests/test_dbmol_objective.py tests/test_dbmol_model.py tests/test_boltz_local_fd.py tests/test_defog_matched_pilot.py tests/test_defog_integration_smoke.py tests/test_repair_guidance.py tests/test_config_identity.py
python scripts/hpc/contact_fix_campaign.py --protocol "$1" --sha256 "$2" --mode prepare
python - "$1" <<'PY'
import json, subprocess, sys
from pathlib import Path
from latentmol.repair_guidance.matched_pilot import load_plan as load_pilot
protocol = json.loads(Path(sys.argv[1]).read_text())
if protocol["version"] == 3:
    # Engineering fixtures only. Actual native readiness must still pass both
    # exact old failures under the new explicitly named pointwise contract.
    subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_pointwise_ad.py",
                    "tests/test_pointwise_cuts.py", "tests/test_pointwise_contract.py",
                    "tests/test_pointwise_readiness.py", "tests/test_repair_guidance_study.py",
                    "tests/test_stability_submission.py"], check=True)
    from latentmol.boltz.pointwise_readiness import load_plan
elif protocol["version"] == 2:
    # These are allocated fixtures, not a native certificate. The actual
    # source106 completion and both candidate states are checked by readiness.
    subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_additive_fd.py",
                    "tests/test_additive_branch_fd.py", "tests/test_additive_contract.py", "tests/test_additive_readiness.py",
                    "tests/test_stability_submission.py"], check=True)
    from latentmol.boltz.additive_readiness import load_plan
else:
    from latentmol.boltz.contact_precision_validation import load_plan
for key, loader in (("validation_plan", load_plan), ("pilot_plan", load_pilot)):
    loader(Path(protocol[key]["path"]), protocol[key]["sha256"])
print("Authenticated readiness and new matched pilot plans; no native model execution")
PY
