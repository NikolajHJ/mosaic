"""Plan/submit exactly one CPU or native DBMol gate with durable LSF receipts.

No models or pocket calculations run on the login node. Run CPU/preparation
first; only after its receipt succeeds should native be submitted with the
actual comparison SHA256. A submitted gate is not numerical validation.
"""

import argparse
import json
from pathlib import Path

from latentmol.experiments import source_manifest
from submit_stability_study import authenticated, read, submit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("cpu", "native"), required=True)
    p.add_argument("--release", type=Path, default=Path.cwd())
    p.add_argument("--work-root", type=Path, default=Path("/work3/s234818/latentmol"))
    p.add_argument("--archive-sha256", required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--protocol-sha256", required=True)
    p.add_argument("--prepared-root", type=Path, required=True)
    p.add_argument("--comparison-sha256")
    p.add_argument("--receipts", type=Path, required=True)
    p.add_argument("--submit", action="store_true")
    a = p.parse_args()
    release, work = a.release.resolve(strict=True), a.work_root.resolve(strict=True)
    if not str(work).startswith("/work3/") or not release.is_relative_to(work / "releases"):
        raise ValueError("Require an immutable release under the declared DTU work root")
    for path in (a.protocol, a.prepared_root, a.receipts):
        if not path.is_absolute():
            raise ValueError("Protocol/prepared/receipt paths must be absolute")
    authenticated(Path(str(release) + ".tar.gz"), a.archive_sha256)
    authenticated(a.protocol, a.protocol_sha256)
    manifest_sha = source_manifest(release)["sha256"]
    if read(release / "release_manifest.json")["sha256"] != manifest_sha:
        raise ValueError("Executable source differs from its sealed manifest")
    comparison = a.prepared_root / "comparison.json"
    if a.stage == "cpu":
        if a.comparison_sha256 is not None:
            raise ValueError("CPU preparation has no comparison outcome yet")
        command = ["bash", "scripts/hpc/boltz_dbmol_preflight.sh", str(a.protocol),
                   a.protocol_sha256, str(a.prepared_root), str(work / "runs/source065-dbmol-cpu")]
        queue, cores, memory, limit = "hpc", 2, "4GB", "0:30"
    else:
        if not a.comparison_sha256:
            raise ValueError("Native gate requires the actual prepared comparison hash")
        authenticated(comparison, a.comparison_sha256)
        # Authenticate a completed CPU wrapper before releasing model compute.
        cpu_receipt = read(a.receipts / "cpu-receipt.json")
        cpu_intent = read(a.receipts / "cpu-intent.json")
        if cpu_receipt["intent"] != cpu_intent:
            raise ValueError("CPU receipt/intent differs")
        command = ["bash", "scripts/hpc/boltz_dbmol_smoke.sh", str(comparison),
                   a.comparison_sha256, str(work / "runs/source065-dbmol-native")]
        queue, cores, memory, limit = "gpua100", 4, "16GB", "6:15"
    label = f"source065-dbmol-{a.stage}"
    args = ["bsub", "-J", label, "-q", queue, "-n", str(cores),
            "-R", f"rusage[mem={memory}]", "-R", "span[hosts=1]", "-W", limit,
            "-oo", str(work / "logs" / f"{label}.%J.out"),
            "-eo", str(work / "logs" / f"{label}.%J.err")]
    if a.stage == "native":
        args += ["-R", "select[gpu80gb]", "-gpu", "num=1:mode=exclusive_process",
                 "-w", f"done({cpu_receipt['job_id']})"]
    job = ["env", f"LATENTMOL_SOURCE_ARCHIVE={release}.tar.gz", f"LATENTMOL_WORK_ROOT={work}",
           "bash", "scripts/hpc/boltz_run.sh", "python", "scripts/verify_stability_release.py",
           "--archive-sha256", a.archive_sha256, "--manifest-sha256", manifest_sha,
           "--", *command]
    identity = {"archive_sha256": a.archive_sha256, "manifest_sha256": manifest_sha,
                "protocol_sha256": a.protocol_sha256, "protocol": str(a.protocol),
                "prepared_root": str(a.prepared_root), "release": str(release)}
    plan = [{"stage": a.stage, "cwd": str(release), "arguments": args,
             "job_command": job, "dependency": None}]
    if a.submit:
        print(json.dumps(submit(plan, a.receipts, identity), indent=2))
    else:
        print(json.dumps({"identity": identity, "plan": plan, "status": "not_submitted"}, indent=2))


if __name__ == "__main__":
    main()
