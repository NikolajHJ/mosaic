"""Allocated, fail-closed stage transitions for the three-arm 25-slot campaign.

Only this controller submits long runs, after actual scheduler outcomes, saved
native gates and a campaign-wide storage projection are authenticated. Waiting
validation jobs and conditional callbacks are never described as long runs.
"""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[4] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

from campaign25_contract import ARMS, REFERENCE_SHA, config25, sha, verify_readiness


QUEUES = {"protenix": "gpua10", "boltz2": "gpul40s", "boltz2_chemistry": "gpul40s"}
LONG_ARMS = ("protenix", "boltz2")
CHEMISTRY_HOLD = {
    "version": "chemistry-runtime-review-v1",
    "status": "held_for_runtime_review",
    "reason": "User requests a measured 25-slot runtime estimate before starting the full chemistry arm",
    "allowed_scope": "native validation and two disposable ten-update pilots only",
    "automatic_long_submission": False,
}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, record):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def parse_final(text, job):
    actual = re.search(r"Job\s*<([^>]+)>", text)
    status = re.search(r"Status\s*<([^>]+)>", text)
    queue = re.search(r"Queue\s*<([^>]+)>", text)
    if not actual or not status or re.sub(r"\s+", "", actual[1]) != str(job):
        raise ValueError("Scheduler job identity missing/different")
    result = re.sub(r"\s+", "", status[1])
    if result not in ("DONE", "EXIT"):
        raise ValueError("Actual producer has not ended")
    return result, None if queue is None else re.sub(r"\s+", "", queue[1])


def final_job(job, parent):
    command = subprocess.run(["bjobs", "-l", str(job)], capture_output=True, text=True, timeout=30)
    raw = command.stdout + command.stderr
    path = parent / "scheduler-final.txt"
    path.write_text(raw)
    if command.returncode:
        raise ValueError("Actual final scheduler outcome unavailable; no accounting bypass")
    status, queue = parse_final(raw, job)
    stdout = parent / f"stdout.{job}.log"
    text = stdout.read_text()
    if status == "DONE" and (f"Subject: Job {job}:" not in text or "Successfully completed." not in text):
        raise ValueError("DONE lacks actual LSF successful completion")
    return status, queue, [{"path": str(p), "sha256": sha(p)} for p in (path, stdout)]


def submit(source, root, manifest, phase, *, arm="none", dependency=None, cpu_job=None):
    if phase == "run" and arm not in LONG_ARMS:
        raise ValueError("Full chemistry optimization held for measured runtime review")
    parent = root / phase
    if arm != "none":
        parent = root / (phase + "-" + arm)
    parent.mkdir(exist_ok=False)
    gpu = phase in ("prepare", "validate", "run")
    queue = QUEUES.get(arm, "gpul40s") if gpu else "hpc"
    walltime = "23:00" if phase == "run" else ("02:00" if phase in ("prepare", "validate") else "00:20")
    command = ["bsub", "-q", queue, "-n", "4" if gpu else "1", "-R", "span[hosts=1]", "-R", "rusage[mem=8GB]" if gpu else "rusage[mem=2GB]", "-W", walltime,
               "-J", f"mosaic25-{phase}-{arm}", "-oo", str(parent / "stdout.%J.log"), "-eo", str(parent / "stderr.%J.log")]
    if gpu:
        command += ["-gpu", "num=1:mode=exclusive_process"]
    if dependency:
        command += ["-w", dependency]
    command += ["bash", "-l", str(_layout_path(source / "scripts/hpc/campaign25_stage.lsf")), str(source), str(root), manifest, phase, arm, str(parent), str(cpu_job or "0")]
    reply = subprocess.run(command, capture_output=True, text=True, timeout=60)
    record = {"submitted_at": datetime.now(timezone.utc).isoformat(), "args": command, "stdout": reply.stdout, "stderr": reply.stderr, "returncode": reply.returncode, "run": str(parent), "phase": phase, "arm": arm}
    match = re.search(r"Job <(\d+)>", reply.stdout)
    if match:
        record["job_id"] = match[1]
    write(parent / "submission.json", record)
    # Persist even an unknown submission outcome. Never blindly duplicate it.
    if reply.returncode or not match:
        raise RuntimeError("Submission failed/unknown; preserve its actual receipt")
    live = subprocess.run(["bjobs", "-l", match[1]], capture_output=True, text=True, timeout=30)
    (parent / "scheduler-at-submission.txt").write_text(live.stdout + live.stderr)
    return record


def dispatch_finished(source, root, manifest):
    """Called under one lock. Baselines do not wait for the chemistry pilot.

    Each actual baseline submission gets its own exclusive attempt marker. A
    chemistry readiness pass still cannot authorize its full optimization.
    """
    decisions = [read(root / ("decision-" + a + ".json")) for a in ARMS if (root / ("decision-" + a + ".json")).exists()]
    write(root / "chemistry-launch-hold.json", CHEMISTRY_HOLD)
    chemistry = next((d for d in decisions if d["arm"] == "boltz2_chemistry"), None)
    if chemistry:
        summary = root / "validate-boltz2_chemistry/payload/study/summary.json"
        estimate = {"launch_policy": CHEMISTRY_HOLD, "native_decision": chemistry, "runtime_available": False}
        if summary.exists():
            saved = read(summary)
            estimate.update(summary=str(summary), summary_sha256=sha(summary), native_status=saved["status"])
            if "runtime_gate" in saved:
                estimate.update(runtime_available=True, runtime_gate=saved["runtime_gate"],
                                scope="Saved disposable pilot measurement; not completed 10000-update runtime")
        write(root / "chemistry-runtime-estimate.json", estimate)
    jobs = []
    for arm in LONG_ARMS:
        marker = root / ("long-dispatch-started-" + arm + ".json")
        if marker.exists():
            receipt = root / ("run-" + arm) / "submission.json"
            if not receipt.exists() or "job_id" not in read(receipt) or read(receipt)["returncode"] != 0:
                raise RuntimeError("Previous long submission unknown/failed; no duplicate attempt")
            jobs.append(read(receipt))
    pending = [a for a in ARMS if not any(d["arm"] == a for d in decisions)]
    def record():
        write(root / "long-submissions.json", {"jobs": jobs, "failed_arms": [d for d in decisions if d["status"] != "passed"],
              "pending_readiness_arms": pending, "held_arms": [{"arm": "boltz2_chemistry", **CHEMISTRY_HOLD}],
              "scope": "Actual independent baseline submissions only; full chemistry held, not completed trajectories"})
    passed = [d for d in decisions if d["status"] == "passed" and d["arm"] in LONG_ARMS]
    eligible = [d for d in passed if not any(j["arm"] == d["arm"] for j in jobs)]
    if not eligible:
        record()
        return
    quota = subprocess.run(["getquota_zhome.sh"], capture_output=True, text=True, timeout=60, check=True).stdout
    from hpc.zhome_quota_preflight import parse_home
    q = parse_home(quota)
    current_bytes = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    available = min(q["hard_bytes_lower"] - q["used_bytes_upper"], shutil.disk_usage(root).free) - current_bytes
    required = sum(d["projected_bytes"] for d in passed) + 4 * 1024**3
    gate = {"status": "passed" if required <= available else "failed", "available_bytes_conservative": available, "required_bytes": required, "reserve_bytes": 4 * 1024**3, "campaign_bytes_subtracted": current_bytes, "quota_output": quota, "limitation": "cached quota is not a reservation; startup repeats fresh checks"}
    gate_path = root / ("campaign-storage-gate-" + "-".join(d["arm"] for d in eligible) + ".json")
    write(gate_path, gate)
    if gate["status"] != "passed":
        raise RuntimeError("Campaign-wide eligible-baseline artifact projection exceeds quota")
    for d in eligible:
        with (root / ("long-dispatch-started-" + d["arm"] + ".json")).open("x") as f:
            json.dump({"decision": d, "storage_gate": str(gate_path), "storage_gate_sha256": sha(gate_path), "chemistry_hold": CHEMISTRY_HOLD}, f)
        jobs.append(submit(source, root, manifest, "run", arm=d["arm"]))
        record()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("after-cpu", "after-prepare", "after-ready"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    source, root = args.source.resolve(), args.root.resolve()
    if not root.is_relative_to(Path.home() / "mosaic-runs") or not source.is_relative_to(Path("/work3/s234818/mosaic/releases")):
        raise ValueError("Unexpected campaign/source root")
    subprocess.run([sys.executable, str(_layout_path(source / "scripts/verify_stage0_release.py")), str(source), args.manifest], check=True)
    cpu_certificate = root / "cpu-certificate.json"
    if args.phase == "after-cpu":
        from check_boltz_cpu_proof import verify
        cpu = root / "cpu"
        status, queue, proofs = final_job(args.job, cpu)
        if status != "DONE" or queue != "hpc":
            raise ValueError("Actual CPU producer did not pass in hpc")
        assessment = verify(cpu / "payload", args.manifest)
        proofs += assessment["proofs"] + [{"path": str(cpu / "payload/assessment.json"), "sha256": sha(cpu / "payload/assessment.json")}, {"path": str(cpu / "payload/composition_reference.json"), "sha256": REFERENCE_SHA}]
        write(cpu_certificate, {"status": "passed", "source_manifest_sha256": args.manifest, "producer_job_id": args.job, "tests": assessment["passed"], "proofs": proofs})
        with (root / "dispatch-started.json").open("x") as f:
            json.dump({"source_manifest_sha256": args.manifest, "cpu_certificate_sha256": sha(cpu_certificate)}, f)
        jobs = []
        protenix = submit(source, root, args.manifest, "validate", arm="protenix")
        jobs.append(protenix)
        jobs.append(submit(source, root, args.manifest, "after-ready", arm="protenix", dependency=f"ended({protenix['job_id']})", cpu_job=protenix["job_id"]))
        prep = submit(source, root, args.manifest, "prepare")
        jobs.append(prep)
        jobs.append(submit(source, root, args.manifest, "after-prepare", dependency=f"ended({prep['job_id']})", cpu_job=prep["job_id"]))
        write(root / "initial-submissions.json", {"jobs": jobs, "long_runs_submitted": False})
    elif args.phase == "after-prepare":
        status, queue, proofs = final_job(args.job, root / "prepare")
        if status != "DONE":
            with (root / "dispatch.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                for arm in ("boltz2", "boltz2_chemistry"):
                    write(root / ("decision-" + arm + ".json"), {"arm": arm, "status": "blocked", "producer_job_id": args.job, "actual_scheduler_status": status, "reason": "shared fresh Boltz preparation failed; arm validation not submitted", "scheduler_proofs": proofs})
                dispatch_finished(source, root, args.manifest)
            raise RuntimeError("Fresh native preparation failed; preserve evidence")
        if queue != "gpul40s":
            raise ValueError("Fresh preparation used an unexpected queue")
        prepared = root / "prepare/payload/prepared"
        p = read(prepared / "preparation.json")
        if p["status"] != "prepared" or p["n_atoms"] != 25 or p["seed_smiles"] != "C" * 25 or p["context_sha256"] != sha(prepared / "context.pt"):
            raise ValueError("Fresh 25-slot context preparation differs")
        for path in (prepared / "context.pt", prepared / "preparation.json", prepared / "input.yaml"):
            proofs.append({"path": str(path), "sha256": sha(path)})
        write(root / "prepared-certificate.json", {"status": "passed", "producer_job_id": args.job, "source_manifest_sha256": args.manifest, "context_sha256": p["context_sha256"], "proofs": proofs})
        jobs = []
        for arm in ("boltz2", "boltz2_chemistry"):
            ready = submit(source, root, args.manifest, "validate", arm=arm)
            jobs.append(ready)
            jobs.append(submit(source, root, args.manifest, "after-ready", arm=arm, dependency=f"ended({ready['job_id']})", cpu_job=ready["job_id"]))
        write(root / "boltz-readiness-submissions.json", {"jobs": jobs, "long_runs_submitted": False})
    else:
        if args.arm is None:
            raise ValueError("Readiness callback requires an arm")
        arm = args.arm
        parent = root / ("validate-" + arm)
        status, queue, proofs = final_job(args.job, parent)
        decision = {"arm": arm, "producer_job_id": args.job, "actual_scheduler_status": status, "status": "failed", "scheduler_proofs": proofs}
        if status == "DONE":
            if queue != QUEUES[arm]:
                raise ValueError("Readiness used an unexpected GPU queue")
            path = parent / "payload/study/assessment.json"
            assessment = verify_readiness(path, sha(path), _layout_path(source / f"examples/configs/brd4_25_{arm}_10000.json"), args.manifest)
            decision.update(status="passed", assessment=str(path), assessment_sha256=sha(path), projected_bytes=assessment["storage_projection"]["projected_bytes"])
        with (root / "dispatch.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            write(root / ("decision-" + arm + ".json"), decision)
            dispatch_finished(source, root, args.manifest)


if __name__ == "__main__":
    main()
