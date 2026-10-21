"""Capture existing CPU/controller/failure bytes; no numerical execution."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile

root = Path("/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-04")
output = root.with_name(root.name + "-release07-evidence-01")
output.mkdir(exist_ok=False)
for job in ("29769499", "29769500"):
    command = subprocess.run(["bjobs", "-l", job], capture_output=True, text=True, check=True)
    raw = command.stdout + command.stderr
    status = re.search(r"Status\s*<([^>]+)>", raw)
    assert status and re.sub(r"\s+", "", status[1]) == "DONE"
    (output / f"scheduler-{job}.txt").write_text(raw)
    phase = "launcher07-cpu" if job == "29769499" else "launcher07-bootstrap"
    text = (root / phase / f"stdout.{job}.log").read_text()
    assert f"Subject: Job {job}:" in text and "Successfully completed." in text
sys.path.insert(0, "/work3/s234818/mosaic/releases/20261021-brd4-25-04/source/scripts")
from campaign25_contract import verify_cpu_certificate
seed = json.loads((root / "release07-seed-started.json").read_text())
verify_cpu_certificate(root / "launcher07-certificate.json", seed["launcher_cpu_certificate_sha256"], "80274d8d5f066af3e364f0356182ab0aa124302613800aa8b68fe3ed3cdec964")
original = json.loads((root / "dispatch-started.json").read_text())
verify_cpu_certificate(root / "cpu-certificate.json", original["cpu_certificate_sha256"], "e22aa03b95e630aae7695b0908a5d20eb26ff7f5ee9b79987c098ae86efb3e79")
paths = []
for phase in ("launcher07-cpu", "launcher07-bootstrap", "after-prepare"):
    paths += [p for p in (root / phase).glob("*") if p.is_file()]
paths += [p for p in (root / "launcher07-cpu/payload").glob("*") if p.is_file()]
paths += list(root.glob("old-protenix-callback-*.txt"))
paths += [root / n for n in ("cpu-certificate.json", "dispatch-started.json", "launcher07-certificate.json", "release07-seed-started.json", "release07-protenix-callback.json", "prepared-certificate.json", "boltz-readiness-submissions.json", "decision-protenix.json", "chemistry-launch-authorization.json")]
paths += list((root / "prepare").glob("scheduler*.txt"))
paths += [root / "prepare/payload/prepared" / n for n in ("preparation.json", "input.yaml", "context.pt")]
paths += [root / "validate-protenix/payload/study/summary.json", root / "validate-protenix/payload/native.log"]
for arm in ("protenix", "boltz2", "boltz2_chemistry"):
    phase = root / ("validate-" + arm)
    paths += [p for p in phase.glob("*") if p.is_file()]
    phase = root / ("release07-after-ready-" + arm)
    paths += [p for p in phase.glob("*") if p.is_file()]
for path in sorted(set(paths)):
    if path.is_file():
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
