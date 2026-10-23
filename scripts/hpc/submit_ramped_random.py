"""Submit CPU preparation, then one gated serial GPU study from a sealed release.

No model inference on the login node. --submit is explicit; durable intent and
receipt files prevent duplicate bsub after an unknown network outcome. Native
repair is unvalidated until the GPU job's own gate produces actual evidence.
"""

import argparse
import json
from pathlib import Path
import re
import subprocess
import yaml

from latentmol.experiments import file_sha256, source_manifest
from submit_stability_study import authenticated, read, submit


def gpu_request(*, managed_mps):
    """DTU-documented multi-process service, with whole-GPU job exclusivity.

    EXCLUSIVE_PROCESS without MPS admits only one CUDA process. The isolated
    DeFoG worker needs a second context while the frozen Boltz model is resident.
    LSF starts/stops MPS and preserves its device/pipe environment for clients.
    """
    return "num=1:mode=exclusive_process:mps=" + ("yes" if managed_mps else "no") + ":j_exclusive=yes"


def require_completed_stage(receipts, stage, identity):
    """Authenticate one exact workflow receipt before consulting LSF."""
    receipt, intent = read(receipts / (stage + "-receipt.json")), read(receipts / (stage + "-intent.json"))
    if receipt["intent"] != intent or intent["identity"] != identity:
        raise ValueError("Gate receipt differs from declared preparation")
    if not re.fullmatch(r"[0-9]+", receipt.get("job_id", "")):
        raise ValueError("Gate receipt lacks a verified numeric job ID")
    reply = subprocess.run(["bjobs", "-a", "-noheader", "-o", "stat", receipt["job_id"]],
                           capture_output=True, text=True, check=True, timeout=30)
    if reply.stdout.strip() != "DONE":
        raise ValueError("Require a DONE " + stage + " gate")
    return receipt["job_id"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cpu", "diagnostic", "native"), required=True)
    parser.add_argument("--release", type=Path, default=Path.cwd())
    parser.add_argument("--work-root", type=Path, default=Path("/work3/s234818/latentmol"))
    parser.add_argument("--archive-sha256", required=True)
    for key in ("base", "starts"):
        parser.add_argument("--" + key, type=Path, required=True)
        parser.add_argument("--" + key + "-sha256", required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    a = parser.parse_args()
    release, work = a.release.resolve(strict=True), a.work_root.resolve(strict=True)
    if not str(work).startswith("/work3/") or not release.is_relative_to(work / "releases"):
        raise ValueError("Require a sealed release under the declared DTU work root")
    for path in (a.base, a.starts, a.prepared_root, a.receipts):
        if not path.is_absolute() or not path.resolve().is_relative_to(work):
            raise ValueError("All input/output paths must stay inside the declared DTU root")
    authenticated(Path(str(release) + ".tar.gz"), a.archive_sha256)
    authenticated(a.base, a.base_sha256)
    authenticated(a.starts, a.starts_sha256)
    manifest_sha = source_manifest(release)["sha256"]
    if read(release / "release_manifest.json")["sha256"] != manifest_sha:
        raise ValueError("Source changed after sealing")
    guidance_sha = file_sha256(release / "configs/defog_ramped_random.example.yaml")
    provider = yaml.safe_load((release / "configs/defog_ramped_random.example.yaml").read_text())["provider"]
    if provider["device"] != "cuda":
        raise ValueError("This workflow requires the configured CUDA DeFoG worker")
    worker_python = provider["python"]  # Preserve venv spelling; never resolve its symlink.
    campaign_root = work / "runs" / (release.name + "-random-study")
    identity = {"release": str(release), "archive_sha256": a.archive_sha256, "manifest_sha256": manifest_sha,
        "base": str(a.base), "base_sha256": a.base_sha256, "starts": str(a.starts),
        "starts_sha256": a.starts_sha256, "guidance_sha256": guidance_sha,
        "prepared_root": str(a.prepared_root), "campaign_root": str(campaign_root),
        "native_gpu_request": gpu_request(managed_mps=True)}
    if a.submit:
        (work / "logs").mkdir(exist_ok=True)
    if a.stage == "cpu":
        if a.protocol_sha256 is not None:
            raise ValueError("CPU gate has not produced a protocol yet")
        command = ["bash", "scripts/hpc/boltz_ramped_random_preflight.sh", str(a.base), a.base_sha256,
            str(a.starts), a.starts_sha256, guidance_sha, str(a.prepared_root), str(campaign_root)]
        queue, cores, memory, wall = "hpc", 2, "4GB", "0:30"
        dependency = None
    else:
        if not a.protocol_sha256:
            raise ValueError("Native study requires authenticated CPU-prepared protocol")
        protocol_path = a.prepared_root / "protocol.json"
        authenticated(protocol_path, a.protocol_sha256)
        protocol = read(protocol_path)
        if (protocol["source_manifest_sha256"] != manifest_sha
                or protocol["source_archive_sha256"] != a.archive_sha256
                or protocol["base"]["sha256"] != a.base_sha256
                or protocol["starts_source"]["sha256"] != a.starts_sha256
                or Path(protocol["output_root"]) != campaign_root):
            raise ValueError("Prepared protocol belongs to another immutable workflow")
        cpu_job = require_completed_stage(a.receipts, "cpu", identity)
        if a.stage == "diagnostic":
            # Both allocated probes precede model loading. Preserve unexpected
            # old-policy success rather than forcing an expected failure claim.
            def row(stage, mps):
                label = release.name + "-" + stage
                arguments = ["bsub", "-J", label, "-q", "gpuv100", "-n", "4",
                    "-R", "rusage[mem=4GB]", "-R", "span[hosts=1]", "-W", "0:10",
                    "-gpu", gpu_request(managed_mps=mps),
                    "-oo", str(work / "logs" / (label + ".%J.out")),
                    "-eo", str(work / "logs" / (label + ".%J.err"))]
                command = ["python", "-m", "latentmol.gpu_process_probe", "--worker-python", worker_python,
                           "--output", str(work / "runs" / (release.name + "-" + stage + ".json"))]
                if not mps:
                    arguments += ["-w", f"done({cpu_job})"]
                    command += ["--negative-control"]
                job = ["env", f"LATENTMOL_SOURCE_ARCHIVE={release}.tar.gz", f"LATENTMOL_WORK_ROOT={work}",
                    "bash", "scripts/hpc/boltz_run.sh", "python", "scripts/verify_stability_release.py",
                    "--archive-sha256", a.archive_sha256, "--manifest-sha256", manifest_sha,
                    "--", "python", "-m", "latentmol.experiments", "--run-root", str(work / "runs" / label),
                    "--name", stage, "--", *command]
                return {"stage": stage, "cwd": str(release), "arguments": arguments,
                        "job_command": job, "dependency": None}
            first, second = row("exclusive-control", False), row("mps-probe", True)
            if not a.submit:
                second["dependency"] = "done(cpu) && ended(exclusive-control)"
                print(json.dumps({"identity": identity, "plan": [first, second]}, indent=2))
                return
            jobs = submit([first], a.receipts, identity)
            second["arguments"] += ["-w", f"done({cpu_job}) && ended({jobs['exclusive-control']})"]
            jobs.update(submit([second], a.receipts, identity))
            print(json.dumps(jobs), flush=True)
            return
        mps_job = require_completed_stage(a.receipts, "mps-probe", identity)
        probe = read(work / "runs" / (release.name + "-mps-probe.json"))
        if (probe["status"] != "passed" or probe["worker_python"] != worker_python
                or probe["parent"]["environment"]["LSB_JOBID"] != mps_job
                or probe["child"]["environment"]["LSB_JOBID"] != mps_job):
            raise ValueError("Native study requires this exact allocation's successful CUDA concurrency probe")
        command = ["bash", "scripts/hpc/boltz_ramped_random_run.sh", str(protocol_path), a.protocol_sha256]
        queue, cores, memory, wall = "gpua100", 4, "16GB", "23:00"
        dependency = f"done({cpu_job}) && done({mps_job})"
    label = release.name + "-" + a.stage
    args = ["bsub", "-J", label, "-q", queue, "-n", str(cores), "-R", f"rusage[mem={memory}]",
        "-R", "span[hosts=1]", "-W", wall,
        "-oo", str(work / "logs" / (label + ".%J.out")), "-eo", str(work / "logs" / (label + ".%J.err"))]
    if dependency is not None:
        args += ["-R", "select[gpu80gb]", "-gpu", gpu_request(managed_mps=True), "-w", dependency]
    job = ["env", f"LATENTMOL_SOURCE_ARCHIVE={release}.tar.gz", f"LATENTMOL_WORK_ROOT={work}",
        "bash", "scripts/hpc/boltz_run.sh", "python", "scripts/verify_stability_release.py",
        "--archive-sha256", a.archive_sha256, "--manifest-sha256", manifest_sha,
        "--", "python", "-m", "latentmol.experiments", "--run-root", str(work / "runs" / (release.name + "-" + a.stage)),
        "--name", "ramped-random-" + a.stage, "--", *command]
    plan = [{"stage": a.stage, "cwd": str(release), "arguments": args, "job_command": job, "dependency": None}]
    (work / "logs").mkdir(exist_ok=True)
    print(json.dumps(submit(plan, a.receipts, identity) if a.submit else
                     {"identity": identity, "plan": plan, "status": "not_submitted"}, indent=2))


if __name__ == "__main__":
    main()
