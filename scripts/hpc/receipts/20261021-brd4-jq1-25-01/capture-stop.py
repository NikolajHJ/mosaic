"""Copy intentional-stop and queue evidence without scientific execution."""

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

old = Path("/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-04")
root = old.with_name("20261021-brd4-jq1-25-stop-01")
output = root / "evidence"
output.mkdir(exist_ok=False)
files = list(root.glob("*.txt")) + list(root.glob("*.json"))
outcomes = {}
for arm, job in (("boltz2", "29769525"), ("boltz2_chemistry", "29769540")):
    command = subprocess.run(["bjobs", "-l", job], capture_output=True, text=True, check=True)
    raw = command.stdout + command.stderr
    fields = re.sub(r"\s+", "", raw)
    assert f"Job<{job}>" in fields and "Status<EXIT>" in fields
    (output / f"scheduler-final-{job}.txt").write_text(raw)
    phase = old / ("run-" + arm)
    selected = [p for p in phase.glob("*") if p.is_file()]
    selected += list((phase / "payload").glob("*.log"))
    selected += list((phase / "payload/study").glob("*.json"))
    selected += list((phase / "payload/study/case-0").glob("*.json"))
    selected += list((phase / "payload/study/case-0").glob("*.jsonl"))
    for path in selected:
        target = output / "parent" / path.relative_to(old)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    outcomes[job] = {"status": "EXIT", "intentional_user_stop": True,
        "artifact_directory_preserved": str(phase), "completed_10000_updates": False}
for path in files:
    shutil.copyfile(path, output / path.name)
(output / "outcomes.json").write_text(json.dumps(outcomes, indent=2) + "\n")
manifest = {p.relative_to(output).as_posix(): {"bytes": p.stat().st_size,
    "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    for p in sorted(output.rglob("*")) if p.is_file()}
(output / "evidence_manifest.json").write_text(json.dumps({"files": manifest}, indent=2) + "\n")
archive = root.with_suffix(".tar.gz")
with tarfile.open(archive, "w:gz") as handle:
    for path in sorted(output.rglob("*")):
        if path.is_file():
            handle.add(path, arcname=path.relative_to(output), recursive=False)
receipt = {"path": str(archive), "bytes": archive.stat().st_size,
    "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "files": len(manifest),
    "manifest_sha256": hashlib.sha256((output / "evidence_manifest.json").read_bytes()).hexdigest()}
archive.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
