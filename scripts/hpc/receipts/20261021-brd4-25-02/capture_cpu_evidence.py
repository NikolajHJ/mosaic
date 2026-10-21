"""Capture scheduler, XML and byte evidence only; no numerical execution."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET

root, output, job, manifest, callback = sys.argv[1:]
root, output = Path(root), Path(output)
output.mkdir(exist_ok=False)
cpu = root / "cpu"
raw = subprocess.run(["bjobs", "-l", job], capture_output=True, text=True, check=True)
(cpu / "scheduler-final.txt").write_text(raw.stdout + raw.stderr)
import re
field = lambda name: re.sub(r"\s+", "", re.search(name + r"\s*<([^>]+)>", raw.stdout)[1])
assert field("Job") == job and field("Status") == "DONE" and field("Queue") == "hpc"
stdout = cpu / ("stdout." + job + ".log")
assert "Subject: Job " + job + ":" in stdout.read_text() and "Successfully completed." in stdout.read_text()
assessment = json.loads((cpu / "payload/assessment.json").read_text())
assert assessment["status"] == "passed" and assessment["source_manifest_sha256"] == manifest
count = 0
for proof in assessment["proofs"]:
    path = Path(proof["path"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == proof["sha256"]
    cases = ET.parse(path).findall(".//testcase")
    assert len(cases) == proof["passed"]
    assert not any(case.find(tag) is not None for case in cases for tag in ("failure", "error", "skipped"))
    count += len(cases)
assert count == assessment["passed"]
for path in sorted(cpu.rglob("*")):
    if path.is_file() and (path.parent == cpu or path.parent == cpu / "payload"):
        relative = path.relative_to(root)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
if callback != "none":
    raw = subprocess.run(["bjobs", "-l", callback], capture_output=True, text=True, check=True)
    text = raw.stdout + raw.stderr
    assert re.search(r"Status\s*<EXIT>", text)
    assert "Started " not in text
    (output / "cancelled-callback-scheduler-final.txt").write_text(text)
    for path in (root / "after-cpu").glob("user-hold-*.txt"):
        shutil.copyfile(path, output / path.name)
verification = {"status": "passed", "producer_job_id": job, "actual_scheduler": "DONE", "lsf_success": True, "source_manifest_sha256": manifest,
                "tests_passed": count, "scope": "XML/schema and byte authentication only; not native GPU validation"}
(output / "authentication.json").write_text(json.dumps(verification, indent=2) + "\n")
files = {p.relative_to(output).as_posix(): {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(output.rglob("*")) if p.is_file()}
(output / "evidence_manifest.json").write_text(json.dumps({"files": files}, indent=2) + "\n")
archive = output.with_name(output.name + ".tar.gz")
with tarfile.open(archive, "w:gz") as handle:
    for path in sorted(output.rglob("*")):
        if path.is_file():
            handle.add(path, arcname=path.relative_to(output), recursive=False)
receipt = {"path": str(archive), "bytes": archive.stat().st_size, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
           "files": len(files), "manifest_sha256": hashlib.sha256((output / "evidence_manifest.json").read_bytes()).hexdigest()}
archive.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({"receipt": receipt, "authentication": verification}))
