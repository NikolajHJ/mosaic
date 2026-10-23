"""Metadata-only recovery submission: CPU gate, two serial GPUs, held pilot.

All scheduler actions retain exclusive intent/reply receipts. Uncertain actions
must be reconciled against LSF before retrying. Old releases stay immutable.
"""
import argparse
import json
from pathlib import Path
import subprocess

from latentmol.experiments import file_sha256, source_manifest, write_json
from submit_stability_study import authenticated, read, submit


def state(job):
    result = subprocess.run(["bjobs", "-a", "-noheader", "-o", "stat", job],
                            capture_output=True, text=True, timeout=30, check=True)
    return result.stdout.strip()


def action(root, name, command):
    intent = {"command": command}
    before, after = root / (name + "-intent.json"), root / (name + "-reply.json")
    if after.exists():
        saved = read(after)
        if read(before) != intent or saved["returncode"] != 0:
            raise ValueError("Prior scheduler action needs reconciliation")
        return
    with before.open("x") as handle:
        json.dump(intent, handle, indent=2)
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    write_json(after, {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
    if result.returncode:
        raise RuntimeError("Scheduler change rejected; retain intent/reply")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cpu", "native"), required=True)
    parser.add_argument("--release", type=Path, default=Path.cwd())
    parser.add_argument("--work-root", type=Path, default=Path("/work3/s234818/latentmol"))
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--retarget-search", action="store_true")
    args = parser.parse_args()
    if args.retarget_search and (args.stage != "native" or not args.submit):
        raise ValueError("Retarget requires native --submit before any scheduler action")
    release, work = args.release.resolve(strict=True), args.work_root.resolve(strict=True)
    if not str(work).startswith("/work3/") or not release.is_relative_to(work / "releases") or not args.receipts.resolve().is_relative_to(work):
        raise ValueError("Require immutable release/receipts inside declared DTU root")
    authenticated(Path(str(release) + ".tar.gz"), args.archive_sha256)
    manifest_sha = source_manifest(release)["sha256"]
    if read(release / "release_manifest.json")["sha256"] != manifest_sha:
        raise ValueError("Release source changed")
    path = release / "configs/decoder_budget_recovery.json"
    recovery, config_sha = read(path), file_sha256(path)
    base_path = Path(recovery["base_protocol"])
    authenticated(base_path, recovery["base_protocol_sha256"])
    base = read(base_path)
    for key in ("checkpoint", "parent_record", "parent_metrics"):
        authenticated(Path(recovery["resume"][key]), recovery["resume"][key + "_sha256"])
    authenticated(Path(base["initializer"]), base["initializer_sha256"])
    authenticated(Path(base["data"]) / "dataset.npz", base["dataset_sha256"])
    if read(Path(base["data"]) / "manifest.json")["split_sha256"] != base["split_sha256"]:
        raise ValueError("Protected split changed")
    authenticated(Path(recovery["base_campaign"]) / "original.json", recovery["base_original_sha256"])
    campaign = work / "runs" / (release.name + "-recovery")
    identity = {"archive_sha256": args.archive_sha256, "manifest_sha256": manifest_sha,
                "config_sha256": config_sha, "output_root": str(campaign), "resume": recovery["resume"]}
    if args.stage == "native":
        cpu = read(args.receipts / "cpu-receipt.json")
        if cpu["intent"]["identity"] != identity or state(cpu["job_id"]) != "DONE":
            raise ValueError("Native recovery requires matching completed CPU gate")
    plan = []
    for stage in (["cpu"] if args.stage == "cpu" else ["seed17", "seed23"]):
        label = release.name + "-" + stage
        arguments = ["bsub", "-J", label, "-q", "hpc" if stage == "cpu" else "gpuv100", "-n", "2" if stage == "cpu" else "4",
            "-R", "rusage[mem=4GB]" if stage == "cpu" else "rusage[mem=16GB]", "-R", "span[hosts=1]",
            "-W", "0:30" if stage == "cpu" else "23:00", "-oo", str(work / "logs" / (label + ".%J.out")),
            "-eo", str(work / "logs" / (label + ".%J.err"))]
        dependency = None
        if stage == "seed17":
            arguments += ["-gpu", "num=1:mode=exclusive_process", "-w", f"done({cpu['job_id']}) && ended({recovery['original_job']})"]
        elif stage == "seed23":
            arguments += ["-gpu", "num=1:mode=exclusive_process"]
            dependency = "ended:seed17"
        tail = ["bash", "scripts/hpc/decoder_budget_tests.sh"] if stage == "cpu" else [
            "bash", "scripts/hpc/decoder_budget_segment.sh", str(path), config_sha, str(campaign), "0" if stage == "seed17" else "1"]
        command = ["env", f"LATENTMOL_SOURCE_ARCHIVE={release}.tar.gz", f"LATENTMOL_WORK_ROOT={work}",
            "bash", "scripts/hpc/boltz_run.sh", "python", "scripts/verify_stability_release.py",
            "--archive-sha256", args.archive_sha256, "--manifest-sha256", manifest_sha, "--", "python", "-m", "latentmol.experiments",
            "--run-root", str(work / "runs" / (release.name + "-" + stage)), "--name", "decoder-budget-" + stage, "--", *tail]
        plan.append({"stage": stage, "cwd": str(release), "arguments": arguments, "job_command": command, "dependency": dependency})
    if not args.submit:
        print(json.dumps({"identity": identity, "plan": plan}, indent=2))
        return
    jobs = submit(plan, args.receipts, identity)
    print(json.dumps(jobs), flush=True)
    if args.retarget_search:
        pilot = recovery["search_pilot_job"]
        old_root = work / "submissions/source071-search-decoder-attempt001"
        old_receipt = read(old_root / "native-receipt.json")
        old_cpu = read(old_root / "cpu-receipt.json")
        if old_receipt["job_id"] != pilot:
            raise ValueError("Search pilot identity differs")
        if old_cpu["intent"]["identity"] != old_receipt["intent"]["identity"]:
            raise ValueError("Original search CPU/native identities differ")
        # LSF has already purged the original DONE CPU from its live cache.
        # Authenticate its completed wrapper instead of creating an unknown
        # done(old_job) dependency; retain the newly completed, superset gate
        # explicitly in LSF alongside the recovery's last segment.
        old_records = [path for path in (work / "runs/source-071-search-decoder-attempt001-cpu").glob("*/run.json")
                       if read(path)["environment"]["job_id"] == old_cpu["job_id"]]
        if len(old_records) != 1:
            raise ValueError("Original completed CPU wrapper is ambiguous/missing")
        old_record = read(old_records[0])
        if old_record["status"] != "completed" or old_record.get("returncode") != 0 or old_record["source_archive_sha256"] != old_cpu["intent"]["identity"]["archive_sha256"]:
            raise ValueError("Original search CPU gate was not completed on its source")
        if not (args.receipts / "release-search-reply.json").exists() and state(pilot) != "PSUSP":
            raise ValueError("Search pilot must be held before dependency replacement")
        gate_identity = {"original_cpu": old_cpu["job_id"], "original_completed_wrapper": str(old_records[0]),
                         "original_completed_wrapper_sha256": file_sha256(old_records[0]), "superset_cpu": cpu["job_id"]}
        gate_path = args.receipts / "retarget-search-gates.json"
        if gate_path.exists() and read(gate_path) != gate_identity:
            raise ValueError("Existing retarget gate identity differs")
        if not gate_path.exists(): write_json(gate_path, gate_identity)
        action(args.receipts, "retarget-search", ["bmod", "-w", f"done({cpu['job_id']}) && ended({jobs['seed23']})", pilot])
        action(args.receipts, "release-search", ["bresume", pilot])
        print("Search pilot retargeted/released behind recovery", flush=True)


if __name__ == "__main__":
    main()
