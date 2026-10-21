"""Standard-library capture of an actual completed component timing job."""
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

root = Path("/zhome/95/e/206121/mosaic-runs/20261021-brd4-25-repair-timing-01")
job = "29769038"
output = root.with_name(root.name + "-evidence-02")
output.mkdir(exist_ok=False)
raw = subprocess.run(["bjobs", "-l", job], capture_output=True, text=True, check=True)
text = raw.stdout + raw.stderr
fields = {name: re.sub(r"\s+", "", re.search(name + r"\s*<([^>]+)>", text)[1]) for name in ("Job", "Status", "Queue")}
assert fields == {"Job": job, "Status": "DONE", "Queue": "hpc"}
stdout = (root / f"stdout.{job}.log").read_text()
assert f"Subject: Job {job}:" in stdout and "Successfully completed." in stdout
record = json.loads((root / "payload/timing/timing.json").read_text())
assert record["status"] == "recorded" and record["job_id"] == job and record["native_readiness"] == "not_evaluated"
assert record["science_manifest_sha256"] == "1b60f14134c607b2a55cd93dbb289ca41c45281380984694d53aaa690772056b"
assert [(c["n_atoms"], c["point"]) for c in record["cases"]] == [(n, p) for n in (6, 25) for p in ("initial_chain_uniform", "internal_fluorine_assay")]
for case in record["cases"]:
    assert len(case["timings"]) == 4 and [t["warmup"] for t in case["timings"]] == [True, False, False, False]
    assert all(math.isfinite(t[k]) and t[k] >= 0 for t in case["timings"] for k in ("total_seconds", "loss"))
    path = root / "payload/timing" / f"{case['n_atoms']}-{case['point']}.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == case["snapshot_sha256"]
for path in root.rglob("*"):
    if path.is_file():
        target = output / path.relative_to(root)
        target.parent.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(path, target)
(output / "scheduler-final.txt").write_text(text)
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
