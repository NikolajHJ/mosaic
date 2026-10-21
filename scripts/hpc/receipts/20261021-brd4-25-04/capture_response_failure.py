"""Byte capture of source03 failure; no numerical model/chemistry work."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

root = Path("/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-03")
output = root.with_name(root.name + "-boltz-response-failure-evidence-01")
output.mkdir(exist_ok=False)
raw = subprocess.run(["bjobs", "-l", "29769078"], capture_output=True, text=True, check=True)
assert re.search(r"Status\s*<EXIT>", raw.stdout)
(output / "scheduler-final.txt").write_text(raw.stdout + raw.stderr)
summary = json.loads((root / "validate-boltz2/payload/study/summary.json").read_text())
assert summary["status"] == "failed" and summary["identity_error"] == "Boltz frozen identity changed"
reply = summary["frozen_identity_after"]
assert reply["status"] == "passed" and reply["model_digest"] == summary["model_digest"] and reply["context_digest"] == summary["context_digest"]
assert "id" in reply and "elapsed_seconds" in reply
paths = [p for p in (root / "validate-boltz2").rglob("*") if p.is_file()]
paths += [root / name for name in ("cpu-certificate.json", "prepared-certificate.json", "initial-submissions.json", "boltz-readiness-submissions.json", "decision-boltz2.json", "long-submissions.json")]
paths += list(root.glob("obsolete-*.txt"))
paths += [root / "prepare/payload/prepared" / name for name in ("context.pt", "preparation.json", "input.yaml")]
for path in paths:
    if not path.exists():
        continue
    target = output / path.relative_to(root)
    target.parent.mkdir(exist_ok=True, parents=True)
    shutil.copyfile(path, target)
files = {p.relative_to(output).as_posix(): {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(output.rglob("*")) if p.is_file()}
(output / "evidence_manifest.json").write_text(json.dumps({"files": files}, indent=2) + "\n")
archive = output.with_name(output.name + ".tar.gz")
with tarfile.open(archive, "w:gz") as handle:
    for path in sorted(output.rglob("*")):
        if path.is_file():
            handle.add(path, arcname=path.relative_to(output), recursive=False)
receipt = {"path": str(archive), "bytes": archive.stat().st_size, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "files": len(files), "manifest_sha256": hashlib.sha256((output / "evidence_manifest.json").read_bytes()).hexdigest()}
archive.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
