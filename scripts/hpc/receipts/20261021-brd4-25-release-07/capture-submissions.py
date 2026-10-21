"""Authenticate existing native/submission records and copy their bytes only.

Run in a login shell with the existing Python 3.12 environment. This does not
execute a model, pilot, chemistry calculation, numerical test or new submission.
"""

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile


root = Path("/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-04")
source = Path("/work3/s234818/mosaic/releases/20261021-brd4-25-04/source")
manifest_sha = "e22aa03b95e630aae7695b0908a5d20eb26ff7f5ee9b79987c098ae86efb3e79"
output = root.with_name(root.name + "-submission-evidence-02")
output.mkdir(exist_ok=False)
sys.path.insert(0, str(source / "scripts"))
from campaign25_contract import sha, verify_readiness

paths = set()
outcomes = {}
for job, phase in (
    ("29769502", "validate-boltz2"),
    ("29769503", "release07-after-ready-boltz2"),
    ("29769504", "validate-boltz2_chemistry"),
    ("29769505", "release07-after-ready-boltz2_chemistry"),
    ("29769525", "run-boltz2"),
    ("29769540", "run-boltz2_chemistry"),
):
    query = subprocess.run(["bjobs", "-l", job], capture_output=True, text=True, check=True)
    raw = query.stdout + query.stderr
    (output / f"scheduler-{job}.txt").write_text(raw)
    # LSF can wrap inside field names as well as values. Match the exact
    # whitespace-normalized fields while retaining the complete original bytes.
    fields = re.sub(r"\s+", "", raw)
    actual = re.search(r"Job<([^>]+)>", fields)
    status = re.search(r"Status<([^>]+)>", fields)
    queue = re.search(r"Queue<([^>]+)>", fields)
    assert actual and re.sub(r"\s+", "", actual[1]) == job
    assert status and queue
    state = re.sub(r"\s+", "", status[1])
    expected_queue = "hpc" if "after-ready" in phase else "gpul40s"
    assert re.sub(r"\s+", "", queue[1]) == expected_queue
    if not phase.startswith("run-"):
        assert state == "DONE"
        stdout = root / phase / f"stdout.{job}.log"
        text = stdout.read_text()
        assert f"Subject: Job {job}:" in text and "Successfully completed." in text
    else:
        assert state in ("PEND", "RUN", "DONE", "EXIT")
        for required in ("num=1:mode=exclusive_process", "rusage[mem=8GB]", "23:00"):
            assert required in json.loads((root / phase / "submission.json").read_text())["args"]
        receipt = json.loads((root / phase / "submission.json").read_text())
        assert receipt["job_id"] == job and receipt["returncode"] == 0
        assert receipt["args"][receipt["args"].index("-n") + 1] == "4"
        assert "-w" not in receipt["args"]
    outcomes[job] = {"status": state, "queue": expected_queue, "phase": phase}
    paths.update(p for p in (root / phase).glob("*") if p.is_file())

assessments = {}
for arm in ("boltz2", "boltz2_chemistry"):
    decision_path = root / f"decision-{arm}.json"
    decision = json.loads(decision_path.read_text())
    assert decision["status"] == "passed" and decision["actual_scheduler_status"] == "DONE"
    assessment = verify_readiness(
        decision["assessment"], decision["assessment_sha256"],
        source / f"examples/configs/brd4_25_{arm}_10000.json", manifest_sha,
    )
    paths.add(decision_path)
    paths.add(Path(decision["assessment"]))
    paths.update(Path(p["path"]) for p in assessment["proofs"])
    assessments[arm] = {k: v for k, v in assessment.items() if k != "proofs"}

paths.update(root.glob("campaign-storage-gate-*.json"))
paths.update(root.glob("long-dispatch-started-*.json"))
paths.update(root / name for name in (
    "long-submissions.json", "chemistry-runtime-estimate.json",
    "prepared-certificate.json", "cpu-certificate.json", "launcher07-certificate.json",
))
for path in sorted(paths):
    assert path.is_file()
    # Original authenticated native proof files must remain within the campaign.
    relative = path.relative_to(root)
    target = output / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)

(output / "authentication.json").write_text(json.dumps({
    "outcomes": outcomes, "assessments": assessments,
    "source_manifest_sha256": manifest_sha,
    "scope": "Existing native proof byte/schema authentication and actual submissions; not completed trajectories",
}, indent=2) + "\n")
files = {
    p.relative_to(output).as_posix(): {"bytes": p.stat().st_size, "sha256": sha(p)}
    for p in sorted(output.rglob("*")) if p.is_file()
}
(output / "evidence_manifest.json").write_text(json.dumps({"files": files}, indent=2) + "\n")
archive = output.with_name(output.name + ".tar.gz")
with tarfile.open(archive, "w:gz") as handle:
    for path in sorted(output.rglob("*")):
        if path.is_file():
            handle.add(path, arcname=path.relative_to(output), recursive=False)
receipt = {"path": str(archive), "bytes": archive.stat().st_size,
           "sha256": sha(archive), "files": len(files),
           "manifest_sha256": sha(output / "evidence_manifest.json")}
archive.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({"receipt": receipt, "outcomes": outcomes, "assessments": assessments}))
