"""Metadata-only LSF continuation for the segmented raw occupancy run."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


TERMINAL = {
    "completed", "stalled", "budget_exhausted", "budget_limited",
    "start_pool_exhausted", "native_budget_reached", "geometry_unavailable",
}


def write_once(path, value):
    """A lost scheduler response leaves an intent and requires reconciliation."""
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def command(argv, cwd=None):
    return subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=45,
                          env=dict(os.environ, LC_ALL="C"))


def require_done(job):
    for attempt in range(6):
        listed = command(["bjobs", "-a", "-noheader", "-o", "jobid stat", job])
        fields = listed.stdout.split()
        if listed.returncode == 0 and fields == [job, "DONE"]:
            return
        if listed.returncode == 0 and fields == [job, "EXIT"]:
            raise RuntimeError(f"Predecessor {job} exited")
        history = command(["bhist", "-l", job])
        events = list(re.finditer(r": (Done successfully\.|Exited[^\n]*|Completed <exit>[^\n]*)",
                                  history.stdout))
        if history.returncode == 0 and f"Job <{job}>" in history.stdout and events:
            last = events[-1]
            if not re.search(r": (Requeued|Submitted|Dispatched|Starting|Running)\b",
                             history.stdout[last.end():], re.I):
                if last.group(1) == "Done successfully.":
                    return
                raise RuntimeError(f"Predecessor {job} exited according to history")
        if attempt < 5:
            time.sleep(2)
    raise RuntimeError(f"Cannot verify that predecessor {job} finished DONE")


def submit(control, label, argv):
    intent = control / f"{label}-intent.json"
    receipt = control / f"{label}-submission.json"
    if receipt.exists():
        raise FileExistsError(f"Submission already recorded for {label}")
    write_once(intent, dict(argv=argv))
    reply = command(argv, cwd=control)
    match = re.fullmatch(r"Job <([0-9]+)> is submitted to queue <[^>]+>\.\s*", reply.stdout)
    write_once(receipt, dict(argv=argv, returncode=reply.returncode,
                             stdout=reply.stdout, stderr=reply.stderr,
                             job_id=match.group(1) if match else None))
    if reply.returncode or not match:
        raise RuntimeError(f"Submission outcome unresolved for {label}; check its receipts")
    return match.group(1)


def next_gpu_argv(plan, stage, state):
    wall, watchdog = plan["wall_seconds"], plan["run_seconds"]
    if not (type(wall) is int and type(watchdog) is int and 0 < wall < watchdog <= 86400):
        raise ValueError("Invalid worker or watchdog time")
    template = plan["gpu_argv"]
    if not isinstance(template, list) or not all(isinstance(item, str) for item in template):
        raise ValueError("GPU command must be an argv list")
    argv = [arg.replace("{stage}", stage).replace("{wall_seconds}", str(wall))
            .replace("{run_seconds}", str(watchdog)) for arg in template]
    if not argv or argv[0] != "bsub" or argv.count("-W") != 1 or "--resume" in argv:
        raise ValueError("Invalid staged GPU command")
    limit = re.fullmatch(r"([0-9]+):([0-5][0-9])", argv[argv.index("-W") + 1])
    if not limit or not watchdog < int(limit[1]) * 3600 + int(limit[2]) * 60 <= 86400:
        raise ValueError("GPU allocation must cover the watchdog and be at most 24 hours")
    return [*argv, "--resume", str(state)]


def advance(plan_path, plan_sha256, stage, job):
    plan_path = plan_path.resolve()
    data = plan_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != plan_sha256:
        raise ValueError("Continuation plan checksum differs")
    plan = json.loads(data)
    control = Path(plan["control"]).resolve()
    output = Path(plan["output_root"])
    maximum, target = plan["max_segments"], plan["target_updates"]
    if (plan_path.parent != control or not output.is_absolute() or not job.isdigit()
            or type(maximum) is not int or not 1 <= stage <= maximum
            or type(target) is not int or not 1 <= target <= 20000):
        raise ValueError("Invalid continuation scope")
    label = f"stage-{stage:03d}"
    require_done(job)
    result = json.loads((output / label / "result.json").read_text(encoding="utf-8"))
    state = (output / label / "state.json").resolve()
    completed, progress = result["completed_updates"], result["updates_this_segment"]
    if (str(result["job_id"]) != job or Path(result["state"]).resolve() != state
            or not state.is_relative_to((output / label).resolve())
            or type(completed) is not int or type(progress) is not int
            or not 0 <= progress <= completed <= target
            or result.get("start_updates", completed - progress) != completed - progress):
        raise ValueError("Worker result does not match the predecessor and budget")
    if json.loads(state.read_text(encoding="utf-8"))["completed_updates"] != completed:
        raise ValueError("Checkpoint and result report different progress")
    status = result["status"]
    summary = dict(job_id=job, completed_updates=completed, worker_status=status)
    if status in TERMINAL:
        if status == "completed" and completed != target:
            raise ValueError("Completed worker did not reach the target")
        write_once(control / f"{label}-advance.json", dict(**summary, status="terminal"))
        return
    if status != "segment_completed" or progress <= 0 or completed >= target:
        raise ValueError("No safe forward progress to continue")
    if stage == maximum:
        write_once(control / f"{label}-advance.json", dict(**summary, status="segment_cap"))
        return
    following = f"stage-{stage + 1:03d}"
    if (output / following).exists():
        raise FileExistsError("Successor output already exists")
    successor = submit(control, following, next_gpu_argv(plan, following, state))
    callback = submit(control, f"after-{following}", [
        "bsub", "-q", "hpc", "-n", "1", "-W", "00:15", "-J", f"raw-occ-after-{successor}",
        "-w", f"ended({successor})", "-oo", str(control / f"after-{following}.%J.out"),
        "-eo", str(control / f"after-{following}.%J.err"),
        sys.executable, str(control / "raw_occupancy_continue.py"),
        "--plan", str(plan_path), "--plan-sha256", plan_sha256,
        "--stage", str(stage + 1), "--job", successor,
    ])
    write_once(control / f"{label}-advance.json",
               dict(**summary, status="queued_next", successor=successor, callback=callback))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    advance(args.plan, args.plan_sha256, args.stage, args.job)


if __name__ == "__main__":
    main()
