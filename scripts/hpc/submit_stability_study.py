"""Verify and plan a serial source063 LSF workflow; --submit explicitly queues.

UNVALIDATED: run the CPU gate and native smoke before interpreting campaign
results. No SSH/VPN manipulation. Receipts prevent duplicate bsub retries after
unknown outcomes; inspect scheduler state before recovering a missing receipt.
"""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess

from latentmol.experiments import file_sha256, source_manifest, write_json


def read(path):
    return json.loads(path.read_text())


def authenticated(path, expected):
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or file_sha256(path) != expected:
        raise ValueError(f"Identity mismatch: {path}")


def workflow(release, work, protocol, protocol_sha, *, archive_sha, manifest_sha, continuation=True):
    """Return structured commands; never call a scheduler while constructing them."""
    stages = [("cpu", "hpc", 2, "4GB", "0:30", None, ["bash", "scripts/hpc/boltz_stability_tests.sh"]),
              ("smoke", "gpul40s", 4, "8GB", "0:45", "done:cpu",
               ["bash", "scripts/hpc/boltz_stability_study.sh", str(protocol), protocol_sha,
                "--smoke-root", str(work / "runs/source063-native-smoke")]),
              ("primary", "gpul40s", 4, "8GB", "23:00", "done:smoke",
               ["bash", "scripts/hpc/boltz_stability_study.sh", str(protocol), protocol_sha])]
    if continuation:
        stages.append(("continuation", "gpul40s", 4, "8GB", "23:00", "ended:primary",
                       ["bash", "scripts/hpc/boltz_stability_study.sh", str(protocol), protocol_sha, "--resume"]))
    result = []
    for stage, queue, cores, memory, limit, dependency, command in stages:
        label = f"source063-stability-{stage}"
        args = ["bsub", "-J", label, "-q", queue, "-n", str(cores),
                "-R", f"rusage[mem={memory}]", "-R", "span[hosts=1]", "-W", limit,
                "-oo", str(work / "logs" / f"{label}.%J.out"),
                "-eo", str(work / "logs" / f"{label}.%J.err")]
        if stage != "cpu":
            args += ["-gpu", "num=1:mode=exclusive_process"]
        job = ["env", f"LATENTMOL_SOURCE_ARCHIVE={release}.tar.gz", f"LATENTMOL_WORK_ROOT={work}",
               "bash", "scripts/hpc/boltz_run.sh", "python", "scripts/verify_stability_release.py",
               "--archive-sha256", archive_sha, "--manifest-sha256", manifest_sha,
               "--", "python", "-m", "latentmol.experiments",
               "--run-root", str(work / "runs"), "--name", label, "--", *command]
        result.append({"stage": stage, "cwd": str(release), "arguments": args,
                       "job_command": job, "dependency": dependency})
    return result


def submit(plan, receipt_root, identity):
    """Queue each stage once, retaining even failed or uncertain submissions."""
    plan = list(plan)
    # Reject malformed explicit external dependencies before issuing ANY bsub.
    # Only fixed scheduler predicates and positive ASCII job IDs are allowed;
    # no caller-supplied expression can enter LSF's dependency language.
    stages_seen = set()
    for row in plan:
        internal = row.get("dependency")
        if internal is not None:
            if type(internal) is not str or internal.count(":") != 1:
                raise ValueError("Invalid internal scheduler dependency")
            kind, preceding = internal.split(":")
            if kind not in ("done", "ended") or preceding not in stages_seen:
                raise ValueError("Invalid internal scheduler dependency")
        if row["stage"] == "continuation" and internal and "smoke" not in stages_seen:
            raise ValueError("Continuation requires its declared successful smoke dependency")
        stages_seen.add(row["stage"])
        external = row.get("external_dependencies", [])
        if not isinstance(external, list):
            raise ValueError("External scheduler dependencies must be an explicit list")
        for dependency in external:
            if (not isinstance(dependency, dict) or set(dependency) != {"kind", "job_id"}
                    or dependency["kind"] not in ("done", "ended")
                    or type(dependency["job_id"]) is not str
                    or not re.fullmatch(r"[1-9][0-9]*", dependency["job_id"])):
                raise ValueError("Invalid external scheduler dependency")
    receipt_root.mkdir(parents=True, exist_ok=True)
    identity_file = receipt_root / "identity.json"
    if identity_file.exists():
        if read(identity_file) != identity:
            raise ValueError("Submission directory belongs to different source or protocol")
    else:
        write_json(identity_file, identity)
    jobs = {}
    for row in plan:
        stage = row["stage"]
        args = list(row["arguments"])
        dependencies = []
        if row["dependency"]:
            kind, preceding = row["dependency"].split(":")
            if kind not in ("done", "ended") or not re.fullmatch(r"[1-9][0-9]*", jobs[preceding]):
                raise ValueError("Invalid internal scheduler dependency")
            dependency = f"{kind}({jobs[preceding]})"
            if stage == "continuation":
                # A failed primary may be continued, but a failed smoke must
                # never release the continuation's model workload.
                dependency += f" && done({jobs['smoke']})"
            dependencies.append(dependency)
        dependencies.extend(f"{item['kind']}({item['job_id']})" for item in row.get("external_dependencies", []))
        if dependencies:
            args += ["-w", " && ".join(dependencies)]
        args += [shlex.join(row["job_command"])]
        intent = {"command": args, "cwd": row["cwd"], "identity": identity}
        intent_path = receipt_root / f"{stage}-intent.json"
        receipt_path = receipt_root / f"{stage}-receipt.json"
        if receipt_path.exists():
            saved = read(receipt_path)
            if read(intent_path) != intent or saved.get("intent") != intent or not re.fullmatch(r"[0-9]+", saved.get("job_id", "")):
                raise ValueError("Existing submission receipt does not match this workflow")
            jobs[stage] = saved["job_id"]
            continue
        # Exclusive intent creation prevents repeated submission, including an
        # SSH disconnect after bsub accepts but before its reply is downloaded.
        with intent_path.open("x") as handle:
            json.dump(intent, handle, indent=2)
        try:
            completed = subprocess.run(args, cwd=row["cwd"], capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired) as exc:
            write_json(receipt_root / f"{stage}-unknown.json", {"intent": intent, "error": str(exc), "status": "unknown_not_retryable"})
            raise RuntimeError("Submission outcome is unknown; inspect bjobs before recovering this intent") from exc
        stdout, stderr = completed.stdout, completed.stderr
        write_json(receipt_root / f"{stage}-reply.json", {"returncode": completed.returncode, "stdout": stdout, "stderr": stderr})
        match = re.fullmatch(r"Job <([0-9]+)> is submitted to queue <[^>]+>\.\s*", stdout)
        if completed.returncode != 0 or not match:
            raise RuntimeError("No verified job receipt; retain reply/intent and inspect scheduler before retrying")
        jobs[stage] = match.group(1)
        write_json(receipt_path, {"intent": intent, "job_id": jobs[stage], "status": "submitted_not_passed"})
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release", type=Path, default=Path.cwd())
    p.add_argument("--work-root", type=Path, default=Path("/work3/s234818/latentmol"))
    p.add_argument("--archive-sha256", required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--protocol-sha256", required=True)
    p.add_argument("--receipts", type=Path, required=True)
    p.add_argument("--no-continuation", action="store_true")
    p.add_argument("--submit", action="store_true", help="Without this flag, print the verified plan only")
    a = p.parse_args()
    release = a.release.resolve(strict=True)
    work = a.work_root.resolve(strict=True)
    if not str(work).startswith("/work3/") or not release.is_relative_to(work / "releases"):
        raise ValueError("Require an immutable release inside the declared DTU work root")
    if not a.protocol.is_absolute() or not a.receipts.is_absolute():
        raise ValueError("Protocol and receipt paths must be absolute")
    authenticated(Path(str(release) + ".tar.gz"), a.archive_sha256)
    authenticated(a.protocol, a.protocol_sha256)
    manifest_sha = source_manifest(release)["sha256"]
    if read(release / "release_manifest.json")["sha256"] != manifest_sha:
        raise ValueError("Immutable release source differs")
    from latentmol.boltz.stability_study import load_protocol, verify_comparison
    protocol = load_protocol(a.protocol)
    verify_comparison(protocol["step_comparison"])
    if Path(protocol["output_root"]).resolve() != work / "runs/source063-stability-campaign":
        raise ValueError("Protocol is outside the declared campaign output root")
    identity = {"archive_sha256": a.archive_sha256, "manifest_sha256": manifest_sha,
                "protocol_sha256": a.protocol_sha256, "protocol": str(a.protocol), "release": str(release)}
    plan = workflow(release, work, a.protocol, a.protocol_sha256, archive_sha=a.archive_sha256,
                    manifest_sha=manifest_sha, continuation=not a.no_continuation)
    if a.submit:
        (work / "logs").mkdir(exist_ok=True)
        print(json.dumps(submit(plan, a.receipts, identity), indent=2))
    else:
        print(json.dumps({"identity": identity, "plan": plan, "status": "not_submitted"}, indent=2))


if __name__ == "__main__":
    main()
