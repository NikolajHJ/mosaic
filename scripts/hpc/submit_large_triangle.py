"""Prepare one duplicate-safe LSF stage at a time from an immutable release.

Preparation/benchmark await HPC validation. Without --submit this only prints an authenticated plan.
No SSH, VPN control, downloads or model execution takes place on the login node.
Continuation stages are submitted after inspecting the preceding allocation.
"""

import argparse
import json
from pathlib import Path
import re
import subprocess

from latentmol.experiments import source_manifest, file_sha256
from large_triangle_stage import bindings
from large_triangle_evidence import require_stage_completion, require_submitted_stage, authenticate_prepared_reference
from submit_stability_study import authenticated, read, submit


STAGE_PREREQUISITES = {
    "cpu": (), "inventory": ("cpu",), "prepare": ("cpu",), "cache": ("cpu",),
    "benchmark": ("cpu", "prepare"), "segment": ("cpu", "prepare", "benchmark"),
    "recover": ("cpu", "prepare", "benchmark"), "final": ("cpu", "prepare", "benchmark"),
}


def required_stages(stage, *, reuse_cache=False):
    """Choose fresh cache validation instead of new preparation when declared."""
    return tuple("cache" if reuse_cache and name == "prepare" else name
                 for name in STAGE_PREREQUISITES[stage])


def require_completed_stage(receipts, stage, identity):
    """Verify durable allocated evidence, then reject any still-active job."""
    job = require_stage_completion(receipts, stage, identity)
    reply = subprocess.run(["bjobs", "-a", "-noheader", "-o", "stat", job], capture_output=True,
                           text=True, timeout=30)
    if reply.returncode and "not found" not in reply.stderr.lower():
        raise RuntimeError("Could not verify scheduler state: " + reply.stderr)
    if reply.stdout.strip() not in ("DONE", ""):
        raise ValueError("Prior allocation is not DONE; do not overlap stages")
    # It is already complete. Avoid a stale -w reference to an aged-out job.
    return None


def require_queued_prepare(receipts, identity):
    """Queue only after an authenticated same-release prepare submission.

    An active preparation gets an LSF done(job) dependency. A completed one
    still needs its durable artifact evidence, including after history expiry.
    No training stage uses this exception to the completed-stage policy.
    """
    job, receipt_sha = require_submitted_stage(receipts, "prepare", identity)
    reply = subprocess.run(["bjobs", "-a", "-noheader", "-o", "stat", job],
                           capture_output=True, text=True, timeout=30)
    state = reply.stdout.strip()
    if reply.returncode and "not found" not in reply.stderr.lower():
        raise RuntimeError("Could not verify preparation state: " + reply.stderr)
    if state in ("PEND", "RUN", "WAIT", "PSUSP", "USUSP", "SSUSP"):
        return job, receipt_sha, (job,)
    if state in ("DONE", ""):
        require_completed_stage(receipts, "prepare", identity)
        return job, receipt_sha, ()
    raise ValueError("Preparation is not active or successfully complete: " + state)


def require_ended_stage(receipts, stage, identity):
    """Recovery may follow EXIT as well as DONE, but never an active job."""
    receipt = read(receipts / (stage + "-receipt.json"))
    if receipt["intent"] != read(receipts / (stage + "-intent.json")) or receipt["intent"]["identity"] != identity:
        raise ValueError("Recovery predecessor receipt differs")
    job = receipt["job_id"]
    if not re.fullmatch("[0-9]+", job):
        raise ValueError("Need an authenticated numeric preceding job ID")
    reply = subprocess.run(["bjobs", "-a", "-noheader", "-o", "stat", job], capture_output=True,
                           text=True, timeout=30, check=True)
    if reply.stdout.strip() not in ("DONE", "EXIT"):
        raise ValueError("Recovery requires an ended predecessor allocation")
    return job


def workflow(stage, release, work, campaign, bindings_path, bindings_sha, archive_sha, manifest_sha,
             *, run_id, segment_id=None, dependency_ids=(), prepare_job_id=None, prepare_receipt_sha256=None):
    """Pure structured plan builder; scheduler mutations are outside this function."""
    cpu = stage in ("cpu", "inventory", "prepare", "cache")
    wall = "1:00" if stage in ("cpu", "cache") else "12:00" if stage in ("inventory", "prepare") else "3:00" if stage == "benchmark" else "23:00"
    memory = "8GB" if stage == "cpu" else "16GB"
    key = stage + "-" + run_id if stage in ("segment", "recover", "final") else stage
    label = release.name + "-" + key
    arguments = ["bsub", "-J", label, "-q", "hpc" if cpu else "gpua100", "-n", "4",
                 "-R", f"rusage[mem={memory}]", "-R", "span[hosts=1]", "-W", wall,
                 "-oo", str(work / "logs" / (label + ".%J.out")),
                 "-eo", str(work / "logs" / (label + ".%J.err"))]
    if not cpu:
        arguments += ["-R", 'select[gpu80gb]', "-gpu", "num=1:mode=exclusive_process:j_exclusive=yes"]
    if dependency_ids:
        arguments += ["-w", " && ".join(f"done({job})" for job in dependency_ids)]
    tail = ["python", "scripts/hpc/large_triangle_stage.py", stage, "--bindings", str(bindings_path),
            "--bindings-sha256", bindings_sha, "--run-id", run_id]
    if segment_id is not None:
        tail += ["--segment-id", segment_id]
    if prepare_job_id is not None:
        if stage != "benchmark" or not prepare_receipt_sha256:
            raise ValueError("Only benchmark may pin an authenticated preparation")
        tail += ["--prepare-job-id", prepare_job_id, "--prepare-receipt-sha256", prepare_receipt_sha256]
    command = ["env", f"LATENTMOL_SOURCE_ARCHIVE={release}.tar.gz", f"LATENTMOL_WORK_ROOT={work}",
               "bash", "scripts/hpc/boltz_run.sh", "python", "scripts/verify_stability_release.py",
               "--archive-sha256", archive_sha, "--manifest-sha256", manifest_sha, "--",
               "python", "-m", "latentmol.experiments", "--run-root", str(campaign / "job-records"),
               "--name", key, "--", *tail]
    return [{"stage": key, "cwd": str(release), "arguments": arguments,
             "job_command": command, "dependency": None}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cpu", "inventory", "prepare", "cache", "benchmark", "segment", "recover", "final"), required=True)
    parser.add_argument("--release", type=Path, default=Path.cwd())
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--bindings-sha256", required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--run-id", default="attempt001")
    parser.add_argument("--segment-id")
    parser.add_argument("--after-stage", help="Completed serial stage receipt, e.g. segment-001")
    parser.add_argument("--queue-after-prepare", action="store_true",
                        help="Benchmark only: queue behind an authenticated active preparation")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    if args.queue_after_prepare and args.stage != "benchmark":
        raise ValueError("Only benchmark may queue before preparation completes")
    if not re.fullmatch("[a-zA-Z0-9_-]+", args.run_id):
        raise ValueError("Run ID must be a simple unique label")
    if args.stage in ("segment", "recover") and (not args.segment_id or not re.fullmatch("[a-zA-Z0-9-]+", args.segment_id)):
        raise ValueError("Training/recovery needs the declared unique segment ID")
    value = bindings(args.bindings, args.bindings_sha256)
    reuse_cache = bool(value.get("prepared_reference"))
    if args.stage == "cache" and not reuse_cache:
        raise ValueError("Cache validation requires a declared prepared-cache reference")
    if reuse_cache and (args.stage in ("inventory", "prepare") or args.queue_after_prepare):
        raise ValueError("Prepared-cache reuse requires fresh cache validation, not preparation")
    work, release, campaign = Path(value["work_root"]).resolve(), args.release.resolve(strict=True), Path(value["campaign"])
    if args.receipts.resolve() != (campaign / "receipts").resolve():
        raise ValueError("Use the campaign's canonical receipts directory for runtime parity")
    if (not release.is_relative_to(work / "releases") or not args.receipts.is_absolute()
            or not args.receipts.resolve().is_relative_to(campaign)
            or not args.bindings.is_absolute() or not args.bindings.resolve().is_relative_to(work)):
        raise ValueError("Require immutable release, contained binding file and campaign receipts")
    authenticated(Path(str(release) + ".tar.gz"), args.archive_sha256)
    manifest_sha = source_manifest(release)["sha256"]
    if read(release / "release_manifest.json")["sha256"] != manifest_sha:
        raise ValueError("Release was edited after sealing")
    identity = {"release": str(release), "archive_sha256": args.archive_sha256,
                "manifest_sha256": manifest_sha, "bindings_sha256": args.bindings_sha256,
                "campaign": str(campaign)}
    gates = list(required_stages(args.stage, reuse_cache=reuse_cache))
    if reuse_cache:
        # Authenticate the prior preparation for every stage without importing
        # its old CPU gate or changing its source hashes. All current training
        # prerequisites remain current-source, durable completed allocations.
        authenticate_prepared_reference(value, read(release / "configs/large_triangle_protocol.json"), release)
    prepare_job = prepare_sha = None
    queued_dependencies = ()
    if args.queue_after_prepare:
        gates.remove("prepare")
        prepare_job, prepare_sha, queued_dependencies = require_queued_prepare(args.receipts, identity)
    if args.after_stage and args.stage != "recover":
        gates.append(args.after_stage)
    if args.stage == "segment" and (campaign / "progress.json").exists() and not args.after_stage:
        raise ValueError("Continuation needs --after-stage to serialize its allocation")
    if args.stage == "final" and not args.after_stage:
        raise ValueError("Final stage needs the completed final training/recovery receipt")
    ids = [job for stage in dict.fromkeys(gates)
           if (job := require_completed_stage(args.receipts, stage, identity)) is not None]
    ids.extend(queued_dependencies)
    if args.stage in ("segment", "final") and (campaign / "progress.json").exists():
        if not args.after_stage or not args.after_stage.startswith(("segment-", "recover-")):
            raise ValueError("Predecessor must be the completed training/recovery stage")
        progress = read(campaign / "progress.json")
        completion = read(campaign / "stage-completions" / (args.after_stage + ".json"))
        if completion["artifacts"].get(progress["checkpoint"]) != progress["checkpoint_sha256"]:
            raise ValueError("Predecessor did not produce the current progress checkpoint")
    if args.stage == "recover":
        if not args.after_stage:
            raise ValueError("Recovery needs --after-stage for the ended interrupted allocation")
        ended = require_ended_stage(args.receipts, args.after_stage, identity)
        intent = read(campaign / "segments" / args.segment_id / "intent.json")
        if intent["job_id"] != ended:
            raise ValueError("Recovery predecessor is not the declared interrupted segment job")
    if args.stage == "segment" and ((campaign / "active-segment.json").exists() or (campaign / "active-recovery.json").exists()):
        raise ValueError("Another segment is active or needs explicit failed-segment recovery")
    rows = workflow(args.stage, release, work, campaign, args.bindings, args.bindings_sha256,
                    args.archive_sha256, manifest_sha, run_id=args.run_id,
                    segment_id=args.segment_id, dependency_ids=ids,
                    prepare_job_id=prepare_job, prepare_receipt_sha256=prepare_sha)
    if args.submit:
        (work / "logs").mkdir(exist_ok=True)
        print(json.dumps(submit(rows, args.receipts, identity), indent=2))
    else:
        print(json.dumps({"status": "not_submitted", "identity": identity, "plan": rows}, indent=2))


if __name__ == "__main__":
    main()
