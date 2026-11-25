"""Small metadata-only LSF controller for the frozen VAE/Protenix experiment.

The GPU runner owns finite/shape/frozen-model checks. This controller checks
terminal allocations, checkpoint identity, bounded budgets and one writer per
stage. It never imports a numerical package on a login node. New native launch
and resume paths are experimental until their saved GPU results establish them.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def command(argv, *, cwd=None, env=None):
    result = subprocess.run(argv, capture_output=True, text=True, cwd=cwd, timeout=60, env=env)
    return dict(command=argv, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)


def require_done(job):
    # The pinned home helper also handles stale RUN / aged-out bjobs records.
    from vae_neighborhood_campaign import require_done as terminal_guard
    return terminal_guard(str(job))


def chunk_budget(elapsed, updates, remaining, *, seconds=82800):
    """Pilot whole-wall estimate, 30% margin and 20-minute reserve; <=23h."""
    if (isinstance(elapsed, bool) or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed) or elapsed <= 0
            or type(updates) is not int or updates <= 0
            or type(remaining) is not int or remaining <= 0
            or type(seconds) is not int or not 1200 < seconds <= 82800):
        raise ValueError("Invalid measured chunk budget")
    count = math.floor((seconds - 1200) / (1.3 * elapsed / updates))
    if count < 1:
        raise ValueError("Even one measured update does not fit the allocation")
    return min(count, remaining)


class Campaign:
    def __init__(self, path, sha):
        self.path, self.sha = Path(path), sha
        if digest(self.path) != sha:
            raise ValueError("Campaign plan changed")
        self.plan = read(path)
        if self.plan.get("version") != "vae-protenix-campaign-v1":
            raise ValueError("Unknown campaign version")
        self.control = Path(self.plan["control"])
        self.root, self.release = Path(self.plan["campaign"]), Path(self.plan["release"])
        if self.path.resolve() != self.control / "plan.json":
            raise ValueError("Use the immutable home-control plan")
        self.receipts = self.control / "receipts"
        self.control.mkdir(parents=True, exist_ok=True)
        expected = self.plan.get("controller_sha256")
        if expected and digest(self.control / "campaign.py") != expected:
            raise ValueError("Home controller changed")
        for path, expected in self.plan.get("input_sha256", {}).items():
            if digest(path) != expected:
                raise ValueError("Campaign configuration changed: " + path)
        if self.plan.get("target_updates") != 50000 or self.plan.get("pilot_updates") != 200:
            raise ValueError("Require the accepted 200-update pilot / 50k endpoint")

    def scratch(self, path, operation="json"):
        from hpc_launch import bounded_file
        return bounded_file(path, operation, timeout=30, spool=self.control)

    def source(self):
        require_done(self.plan["source_job"])
        value = self.scratch(self.plan["source_result"])
        if (value.get("status") != "completed_epoch" or value.get("epoch") != 45
                or value.get("study_sha256") != self.plan["source_study_sha256"]
                or value.get("frozen_anchors_unchanged") is not True
                or not value.get("terminal_checks")
                or not all(v is True for v in value["terminal_checks"].values())):
            raise ValueError("Epoch 45 has not completed its original checks")
        path, sha = value["checkpoint"], value["checkpoint_sha256"]
        if not re.fullmatch("[a-f0-9]{64}", sha) or self.scratch(path, "hash") != sha:
            raise ValueError("Completed epoch-45 checkpoint changed")
        result = dict(path=path, sha256=sha, source_job=str(self.plan["source_job"]),
                      source_study_sha256=self.plan["source_study_sha256"], epoch=45)
        receipt = self.control / "checkpoint-resolution.json"
        if receipt.exists():
            if read(receipt) != result:
                raise ValueError("Resolved decoder checkpoint identity changed")
        else:
            save(receipt, result)
        return result

    def cli(self, mode, *args):
        return ["/usr/bin/python3", str(self.control / "campaign.py"),
                "--plan", str(self.path), "--plan-sha256", self.sha, "--mode", mode, *args]

    def launch(self, stage, updates, resume=None):
        args = ["--stage", stage, "--updates", str(updates)]
        if resume:
            args += ["--resume", str(resume)]
        worker = [self.plan["torch_python"], "-u", str(self.control / "campaign.py"),
                  "--plan", str(self.path), "--plan-sha256", self.sha,
                  "--mode", "run-stage", *args]
        return ["/usr/bin/python3", str(self.control / "hpc_launch.py"),
                "--archive", str(self.release) + ".tar.gz", "--archive-sha256", self.plan["archive_sha256"],
                "--release", str(self.release), "--receipt", str(self.control / (stage + "-launch.json")),
                "--startup-seconds", "900", "--run-seconds", "6600" if stage == "pilot" else "85000",
                "--", "env", "PYTHONPATH=" + str(self.release / "src"),
                "bash", "scripts/hpc/protenix_latent_run.sh", *worker]

    def submit(self, stage, argv, *, gpu=False, dependency=None, walltime=None):
        receipt, intent = self.receipts / (stage + ".json"), self.receipts / (stage + "-intent.json")
        if receipt.exists():
            old = read(receipt)
            if old["plan_sha256"] != self.sha:
                raise ValueError("Submission belongs to another plan")
            return old["job_id"]
        if intent.exists():
            raise RuntimeError("Unknown submission outcome; reconcile before retry")
        name = self.plan.get("job_prefix", "vae-protenix-") + stage
        listing = command(["bjobs", "-u", "s234818", "-w"])
        if listing["returncode"] or any(name in line for line in listing["stdout"].splitlines()[1:]):
            raise RuntimeError("Cannot rule out an existing stage writer")
        args = ["bsub", "-J", name, "-q", self.plan.get("gpu_queue", "gpuv100") if gpu else "hpc",
                "-n", str(self.plan.get("gpu_slots", 8)) if gpu else "1", "-R", "span[hosts=1]", "-R", "rusage[mem=4096]",
            "-W", (walltime or ("2:00" if stage == "pilot" else "24:00")) if gpu else "0:15",
                "-env", "all,~CUDA_VISIBLE_DEVICES,~CUDA_MPS_PIPE_DIRECTORY,~CUDA_MPS_LOG_DIRECTORY,~CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
                "-oo", str(self.control / (stage + ".%J.out")), "-eo", str(self.control / (stage + ".%J.err"))]
        if gpu:
            args += ["-gpu", self.plan.get("gpu_request", "num=1:mode=exclusive_process:j_exclusive=yes:mps=no:gmem=30G")]
        if dependency:
            args += ["-w", dependency]
        args += argv
        save(intent, dict(plan_sha256=self.sha, command=args))
        # CPU callbacks can inherit a previous GPU's direct/task masks. Let LSF
        # establish the NEW allocation, then resolve it in the GPU wrapper.
        clean = {k: v for k, v in os.environ.items()
                 if not re.fullmatch(r"CUDA_VISIBLE_DEVICES[0-9]*", k)
                 and k not in ("CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY", "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")}
        reply = command(args, cwd=self.control, env=clean)
        save(self.receipts / (stage + "-reply.json"), reply)
        match = re.fullmatch(r"Job <([0-9]+)> is submitted to queue <[^>]+>\.\s*", reply["stdout"])
        if reply["returncode"] or not match:
            raise RuntimeError("Unknown submission outcome; reconcile saved intent/reply")
        job = match.group(1)
        save(receipt, dict(plan_sha256=self.sha, job_id=job, status="submitted_not_passed"))
        return job

    def queue(self):
        pilot = self.submit("pilot", self.launch("pilot", self.plan["pilot_updates"]), gpu=True,
                            dependency="done(" + str(self.plan["source_job"]) + ")")
        callback = self.submit("after-pilot", self.cli("after-pilot"), dependency="ended(" + pilot + ")")
        return dict(status="pilot_queued_after_epoch45", pilot=pilot, callback=callback)

    def validated_result(self, stage, job_stage=None):
        require_done(read(self.receipts / ((job_stage or stage) + ".json"))["job_id"])
        value = self.scratch(self.root / stage / "result.json")
        source = self.source()
        if (value.get("status") not in ("completed", "segment_completed")
                or value.get("checkpoint_sha256") != source["sha256"]
                or not value.get("technical_checks")
                or not all(v is True for v in value["technical_checks"].values())
                or type(value.get("completed_step")) is not int
                or type(value.get("updates_this_segment")) is not int
                or value["updates_this_segment"] <= 0
                or value["completed_step"] - value.get("start_step", -1) != value["updates_this_segment"]
                or value.get("state") != str(self.root / stage / "state.pt")):
            raise ValueError("Stage lacks complete finite/frozen/source success evidence")
        return value

    def queue_main(self, index, count, resume=None):
        stage = "main-" + str(index).zfill(4)
        job = self.submit(stage, self.launch(stage, count, resume), gpu=True)
        after = self.submit("after-" + stage, self.cli("advance", "--stage", stage),
                            dependency="ended(" + job + ")")
        return dict(stage=stage, job_id=job, callback=after, requested_updates=count, resume=resume)

    def after_pilot(self):
        value = self.validated_result("pilot")
        if value["completed_step"] != self.plan["pilot_updates"] or value.get("start_step") != 0:
            raise ValueError("Pilot did not finish the prescribed fresh updates")
        probes = value.get("weight_probes", {})
        if any(probes.get(k, {}).get("finite") is not True for k in ("1.0", "10.0")):
            raise ValueError("Missing finite chemistry-weight 1/10 pilot probes")
        high = self.validated_result("pilot-weight10", job_stage="pilot")
        if high.get("start_step") != 0 or high["completed_step"] != 20:
            raise ValueError("Independent weight-10 pilot did not finish its 20 updates")
        count = chunk_budget(value["elapsed_seconds"], value["updates_this_segment"], self.plan["target_updates"])
        result = self.queue_main(1, count)
        path = self.control / "after-pilot-result.json"
        if not path.exists():
            save(path, dict(status="main_queued_probe_discarded", **result))
        return result

    def advance(self, stage):
        if not re.fullmatch(r"main-[0-9]{4}", stage):
            raise ValueError("Invalid main stage")
        value = self.validated_result(stage)
        remaining = self.plan["target_updates"] - value["completed_step"]
        if remaining < 0:
            raise ValueError("Stage exceeded the authorized 50k endpoint")
        if not remaining:
            path = self.control / "completion.json"
            if not path.exists():
                save(path, dict(status="completed_50000", plan_sha256=self.sha,
                                state=value["state"], completed_step=value["completed_step"]))
            return read(path)
        pilot = self.scratch(self.root / "pilot" / "result.json")
        count = chunk_budget(pilot["elapsed_seconds"], pilot["updates_this_segment"], remaining)
        return self.queue_main(int(stage.split("-")[1]) + 1, count, value["state"])

    def run_stage(self, stage, updates, resume=None):
        if not os.environ.get("LSB_JOBID", "").isdigit():
            raise RuntimeError("Model worker requires an allocated job")
        if stage != "pilot" and not re.fullmatch(r"main-[0-9]{4}", stage):
            raise ValueError("Invalid GPU stage")
        if type(updates) is not int or not 1 <= updates <= self.plan["target_updates"]:
            raise ValueError("Invalid bounded update count")
        source = self.source()
        config = self.plan["pilot_config"] if stage == "pilot" else self.plan["full_config"]
        args = [self.plan["torch_python"], "-u", "-m", "latentmol.design.latent_protenix",
                "--config", config, "--checkpoint", source["path"], "--checkpoint-sha256", source["sha256"],
                "--output", str(self.root / stage), "--device", "cpu",
                "--mosaic-python", self.plan["mosaic_python"], "--mosaic-config", self.plan["mosaic_config"],
                "--max-updates", str(updates), "--wall-seconds", "6000" if stage == "pilot" else "82800"]
        if resume:
            args += ["--resume", resume]
        environment = dict(os.environ, LATENTMOL_SOURCE_SHA256=self.plan["archive_sha256"],
                           LATENTMOL_CAMPAIGN_SHA256=self.sha)
        result = subprocess.run(args, env=environment)
        if result.returncode:
            raise RuntimeError("GPU runner failed; preserve outputs and do not auto-restart")
        if stage == "pilot":
            # The high-weight stress test starts from the same random seed,
            # never from the normal pilot's updates. It shares the allocation,
            # but must not replace the launcher's first readiness marker.
            high = list(args)
            high[high.index("--config") + 1] = self.plan["high_weight_config"]
            high[high.index("--output") + 1] = str(self.root / "pilot-weight10")
            high[high.index("--max-updates") + 1] = "20"
            env = {k: v for k, v in environment.items()
                   if k not in ("DBMOL_LAUNCH_READY", "DBMOL_LAUNCH_NONCE")}
            result = subprocess.run(high, env=env)
            if result.returncode:
                raise RuntimeError("Weight-10 pilot failed; main run remains unsubmitted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--mode", choices=("queue", "after-pilot", "run-stage", "advance"), required=True)
    parser.add_argument("--stage")
    parser.add_argument("--updates", type=int)
    parser.add_argument("--resume")
    args = parser.parse_args()
    # Exact home-local helpers are supplied by the immutable launch plan.
    sys.path.insert(0, str(args.plan.parent))
    campaign = Campaign(args.plan, args.plan_sha256)
    try:
        if args.mode == "queue":
            result = campaign.queue()
        elif args.mode == "after-pilot":
            result = campaign.after_pilot()
        elif args.mode == "advance":
            result = campaign.advance(args.stage)
        else:
            result = campaign.run_stage(args.stage, args.updates, args.resume)
    except Exception as error:
        receipt = campaign.control / (args.mode + "-" + str(args.stage) + "-" + os.environ.get("LSB_JOBID", "local") + "-failure.json")
        if not receipt.exists():
            save(receipt, dict(status="failed_no_automatic_retry", error=type(error).__name__ + ": " + str(error)))
        raise
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
