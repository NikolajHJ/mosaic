"""Capture saved campaign proof bytes and actual scheduler states; no model work.

Pass campaign root and a fresh sibling output name as arguments when piping to
the existing Python3.12 login-shell runtime. Preserve failed stages as evidence.
"""

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET

root = Path(sys.argv[1])
output = root.with_name(sys.argv[2])
assert root.parent == Path("/zhome/95/e/206121/mosaic-runs") and output.parent == root.parent
output.mkdir(exist_ok=False)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


paths, records, outcomes, producer_proofs = set(root.glob("*.json")), {}, {}, []
paths.update(root.glob("*.txt"))
for parent in sorted(p for p in root.iterdir() if p.is_dir()):
    paths.update(p for p in parent.glob("*") if p.is_file())
    receipt = parent / "submission.json"
    if receipt.exists():
        record = json.loads(receipt.read_text())
        if "job_id" in record:
            records[record["job_id"]] = {"phase": parent, "receipt": record}
    elif (parent / "submission.txt").exists():
        match = re.search(r"Job <(\d+)>", (parent / "submission.txt").read_text())
        if match:
            records[match[1]] = {"phase": parent}
    for sub in ("payload", "payload/prepared", "payload/study"):
        paths.update(p for p in (parent / sub).glob("*") if p.is_file() and p.suffix in (".json", ".xml", ".log", ".yaml", ".pt", ".npz"))
    if parent.name == "validate-protenix":
        paths.update(p for p in (parent / "payload/study").rglob("*") if p.is_file())
    assessment = parent / "payload/study/assessment.json"
    if assessment.exists():
        a = json.loads(assessment.read_text())
        assert a["status"] == "passed"
        for proof in a["proofs"]:
            assert sha(proof["path"]) == proof["sha256"]
            paths.add(Path(proof["path"]))
    certificate = parent / "payload/assessment.json"
    if certificate.exists():
        a = json.loads(certificate.read_text())
        assert a["status"] == "passed"
        for proof in a["proofs"]:
            path = Path(proof["path"])
            assert sha(path) == proof["sha256"]
            cases = ET.parse(path).findall(".//testcase")
            assert len(cases) == proof["passed"] and not any(c.find(t) is not None for c in cases for t in ("failure", "error", "skipped"))
            paths.add(path)

for job, record in records.items():
    command = subprocess.run(["bjobs", "-l", job], capture_output=True, text=True, check=True)
    raw = command.stdout + command.stderr
    (output / f"scheduler-{job}.txt").write_text(raw)
    normalized = re.sub(r"\s+", "", raw)
    assert f"Job<{job}>" in normalized
    status, queue = (re.search(p, normalized)[1] for p in (r"Status<([^>]+)>", r"Queue<([^>]+)>"))
    outcomes[job] = {"phase": record["phase"].name, "status": status, "queue": queue}
    if status == "DONE":
        text = (record["phase"] / f"stdout.{job}.log").read_text()
        assert f"Subject: Job {job}:" in text and "Successfully completed." in text
    receipt = record.get("receipt")
    if receipt and receipt["phase"] == "run":
        args = receipt["args"]
        assert receipt["returncode"] == 0 and "-w" not in args
        assert args[args.index("-q") + 1] == "gpul40s"
        assert args[args.index("-gpu") + 1] == "num=1:mode=exclusive_process"
        assert args[args.index("-n") + 1] == "4" and args[args.index("-W") + 1] == "23:00"

for path in sorted(paths):
    relative = path.relative_to(root)
    target = output / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)
(output / "authenticated-outcomes.json").write_text(json.dumps(outcomes, indent=2) + "\n")
manifest = {p.relative_to(output).as_posix(): {"bytes": p.stat().st_size, "sha256": sha(p)}
    for p in sorted(output.rglob("*")) if p.is_file()}
(output / "evidence_manifest.json").write_text(json.dumps({"files": manifest}, indent=2) + "\n")
archive = output.with_name(output.name + ".tar.gz")
with tarfile.open(archive, "w:gz") as handle:
    for path in sorted(output.rglob("*")):
        if path.is_file():
            handle.add(path, arcname=path.relative_to(output), recursive=False)
receipt = {"path": str(archive), "bytes": archive.stat().st_size, "sha256": sha(archive),
    "files": len(manifest), "manifest_sha256": sha(output / "evidence_manifest.json")}
archive.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps({"receipt": receipt, "outcomes": outcomes}))
