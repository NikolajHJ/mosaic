"""Authenticated CPU fixtures -> native benchmark -> serial chemistry phase.

The scheduler controller uses only byte/JSON/quota metadata. Failed allocations,
failed scientific gates, unexpected identities and uncertain submissions stop
the chain. A submitted dependency is never reported as a passed experiment.
Scientific imports/array/model operations occur only inside allocated workers.
UNTESTED until the allocated CPU/controller fixtures pass.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

from latentmol.experiments import file_sha256, source_manifest, write_json
from latentmol.chemistry_runtime import (composition_prior_weight, validate_execution_policy,
                                        protocol_config_path, validate_failure_guidance, validate_runtime_policy,
                                        validate_handoff_policy, epoch_fit_policy)
from submit_stability_study import authenticated as _authenticated, submit
from triangle_continuation_health import strict_json
from latentmol.chemistry_short_continuation import (continuation_end_epoch,
    expected_stop_reason, require_training_branch, validate_short_continuation)


FIXTURES = ["tests/test_atom_preserving_chemistry.py", "tests/test_chemistry_continuation.py",
            "tests/test_chemistry_campaign.py", "tests/test_chemistry_resume_cuda.py",
            "tests/test_chemistry_runtime.py"]
OPTIMIZED_FIXTURES = ["tests/test_atom_preserving_diagnostics.py", "tests/test_atom_preserving_fast.py",
                      "tests/test_repair_worker_pool.py", "tests/test_chemistry_execution_validation.py"]
# A dependency may become eligible just before a status replica leaves RUN.
# Requery only that transient state, never approve it or retry other failures.
# Four five-second queries plus three two-second waits reserve at most 26s.
COMPLETION_QUERIES = 4
COMPLETION_QUERY_TIMEOUT = 5
COMPLETION_SETTLE_SECONDS = 2
OPTIMIZED_COMPLETION_SECONDS = 120
OPTIMIZED_COMPLETION_QUERIES = 17  # 17*5s queries + 16*2s waits <= 120s.
OPTIMIZED_EXECUTION = {"backend": "optimized", "workers": 12, "torch_threads": 4, "gpu_cpus": 16}


def execution_settings(plan, *, allocated=False):
    """Validate explicit resources using metadata only; old plans stay serial."""
    value = validate_execution_policy(plan.get("execution"))
    if value is None:
        return None
    if allocated:
        slots = os.environ.get("LSB_DJOB_NUMPROC", "")
        if not slots.isdecimal() or int(slots) != value["gpu_cpus"]:
            raise ValueError("Optimized chemistry requires its declared allocated CPU slots")
        if any(os.environ.get(key) != str(value["torch_threads"])
               for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")):
            raise ValueError("Main OMP/MKL threads differ from authenticated parent runtime")
    return dict(value)


def authenticated(path, expected):
    """Normalize JSON paths for the existing Path-based byte-authentication API."""
    return _authenticated(Path(path), expected)


def read(path):
    return strict_json(Path(path).read_text())


def validate_plan(path, sha):
    """Authenticate scheduler inputs without importing numerical modules."""
    authenticated(path, sha)
    p = read(path); work = Path(p["work_root"]).resolve(strict=True)
    execution_settings(p)
    from latentmol.repair_worker_pool import validate_classification_execution
    validate_classification_execution(p.get("classification_execution"))
    from latentmol.chemistry_worker_startup import validate_worker_startup
    validate_worker_startup(p.get("worker_startup"))
    from latentmol.chemistry_optimizer_step import validate_optimizer_step_control
    validate_optimizer_step_control(p.get("optimizer_step_control"))
    validate_runtime_policy(p["protocol"])
    epoch_fit_policy(p)
    composition_prior_weight(p["protocol"])
    validate_failure_guidance(p["protocol"].get("failure_guidance"))
    from latentmol.chemistry_restart_policy import validate_guarded_restart
    validate_guarded_restart(p)
    validate_short_continuation(p)
    starting = validate_handoff_policy(p)
    protocol_config_path(p)
    if not str(work).startswith("/work3/") or p["protocol"]["total_epochs"] != 100:
        raise ValueError("Unsupported DTU chemistry campaign")
    for key in ("release", "campaign", "data", "parent_checkpoint", "parent_release", "config", "parent_run_record"):
        if not Path(p[key]).is_absolute() or not Path(p[key]).resolve().is_relative_to(work):
            raise ValueError(f"Outside authorized work root: {key}")
    release = Path(p["release"])
    if not release.resolve().is_relative_to(work / "releases") or not Path(p["campaign"]).resolve().is_relative_to(work / "runs"):
        raise ValueError("Require unique sealed source and campaign paths")
    authenticated(str(release) + ".tar.gz", p["archive_sha256"])
    if source_manifest(release)["sha256"] != p["manifest_sha256"]:
        raise ValueError("Immutable scientific release changed")
    for key in ("parent_checkpoint", "config", "parent_run_record"):
        authenticated(p[key], p[key + "_sha256"])
    authenticated(str(p["parent_release"]) + ".tar.gz", p["parent_archive_sha256"])
    if starting is not None:
        for key, sha in (("path", "sha256"), ("source_plan", "source_plan_sha256"), ("completion", "completion_sha256")):
            if not Path(starting[key]).is_absolute() or not Path(starting[key]).resolve().is_relative_to(work):
                raise ValueError("Starting checkpoint escaped the declared work root")
            authenticated(starting[key], starting[sha])
    for artifact, digest in p["readiness_artifacts"].items():
        authenticated(artifact, digest)
    return p


def quota_receipt(minimum_gib):
    """Official per-user work3 quota, not global filesystem free space."""
    reply = subprocess.run(["getquota_work3.sh"], capture_output=True, text=True, timeout=30, check=True)
    units = {"Byte": Decimal(1), "KiB": Decimal(1024), "MiB": Decimal(1024)**2,
             "GiB": Decimal(1024)**3, "TiB": Decimal(1024)**4}
    remaining = []
    def amount(value):
        number, unit = value.strip().split(); result = Decimal(number) * units[unit]
        if not result.is_finite() or result < 0:
            raise ValueError("Invalid quota amount")
        return result
    for line in reply.stdout.splitlines():
        if line.strip().startswith("s234818|"):
            fields = line.split("||")[1].split("|")
            if len(fields) != 2:
                raise ValueError("Unrecognized official quota row")
            used, hard = (v.strip() for v in fields)
            if hard != "unlimited":
                available = (amount(hard) - amount(used)) / (Decimal(1024)**3)
                if available < Decimal(str(minimum_gib)):
                    raise ValueError("Insufficient work3 quota for serial chemistry phase")
                remaining.append(str(available))
    if not remaining:
        raise ValueError("Expected finite user quota missing")
    return {"utc": datetime.now(timezone.utc).isoformat(), "remaining_gib": remaining,
            "minimum_gib": minimum_gib, "stdout": reply.stdout, "stderr": reply.stderr}


class Campaign:
    """One GPU segment at a time, with durable submission intents and callbacks."""

    def __init__(self, path, sha):
        self.path, self.sha = Path(path).resolve(), sha
        self.plan = validate_plan(path, sha)
        self.work, self.release, self.root = (Path(self.plan[k]) for k in ("work_root", "release", "campaign"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.receipts = self.root / "receipts"
        self.identity = {"plan_sha256": sha, "archive_sha256": self.plan["archive_sha256"],
                         "manifest_sha256": self.plan["manifest_sha256"], "parent_sha256": self.plan["parent_checkpoint_sha256"]}

    def worker(self, command, label):
        execution = execution_settings(self.plan)
        # boltz_run intentionally defaults OMP/MKL to all allocated slots.
        # Override AFTER that wrapper so the parent keeps four Torch threads,
        # while the declared detached search workers each use one CPU thread.
        threads = (["env", "OMP_NUM_THREADS=4", "MKL_NUM_THREADS=4", "OPENBLAS_NUM_THREADS=4"]
                   if execution else [])
        return ["env", f"LATENTMOL_SOURCE_ARCHIVE={self.release}.tar.gz", f"LATENTMOL_WORK_ROOT={self.work}",
                "bash", "scripts/hpc/boltz_run.sh", *threads, "python", "scripts/verify_stability_release.py",
                "--archive-sha256", self.plan["archive_sha256"], "--manifest-sha256", self.plan["manifest_sha256"],
                "--", "python", "-m", "latentmol.experiments", "--run-root", str(self.root / "allocations"),
                "--name", label, "--", *command]

    def controller(self, mode, **values):
        command = [str(self.work / "env/boltz311/bin/python"), "scripts/hpc/chemistry_campaign.py",
                   "--plan", str(self.path), "--plan-sha256", self.sha, "--mode", mode]
        for name, value in values.items():
            command += ["--" + name.replace("_", "-"), str(value)]
        # No scientific imports: source package only for byte-only receipt helpers.
        return ["env", "PYTHONPATH=" + str(self.release / "src"), *command]

    def arguments(self, stage, gpu=False, limit="0:15", dependency=None):
        name = "source104-chemistry-" + stage
        execution = execution_settings(self.plan)
        # The optimized CPU preflight also spawns detached search workers.
        # LSF memory is per slot: old16*4GB=64GB; new32*4GB=128GB.
        heavy = execution is not None and (gpu or stage == "cpu")
        slots = str(execution["gpu_cpus"]) if heavy else ("4" if gpu else "2")
        memory = "rusage[mem=4GB]" if heavy or not gpu else "rusage[mem=16GB]"
        command = ["bsub", "-J", name, "-q", "gpua100" if gpu else "hpc", "-n", slots,
                   "-R", memory, "-R", "span[hosts=1]", "-W", limit,
                   "-oo", str(self.work / "logs" / (name + ".%J.out")), "-eo", str(self.work / "logs" / (name + ".%J.err"))]
        if gpu:
            command += ["-R", "select[gpu80gb]", "-gpu", "num=1:mode=exclusive_process:j_exclusive=yes"]
        if dependency:
            command += ["-w", dependency]
        return command

    def queue(self, stage, command, *, gpu=False, limit="0:15", dependency=None):
        row = {"stage": stage, "cwd": str(self.release), "arguments": self.arguments(stage, gpu, limit, dependency),
               "job_command": command, "dependency": None}
        return submit([row], self.receipts, self.identity)[stage]

    def receipt_job(self, stage):
        saved = read(self.receipts / (stage + "-receipt.json"))
        intent = read(self.receipts / (stage + "-intent.json"))
        if (saved.get("intent") != intent or intent.get("identity") != self.identity
                or not re.fullmatch("[0-9]+", saved.get("job_id", ""))):
            raise ValueError("Submission receipt identity differs")
        return saved["job_id"]

    def complete(self, stage):
        """Require DONE; allow bounded RUN-only scheduler propagation settling.

        A successful status query still reporting RUN is observed again, for a
        strictly bounded interval. PEND, EXIT, unknown statuses, query failures
        and exhausted RUN observations stop immediately without consulting a
        completed-artifact fallback. Only a genuinely missing scheduler record
        may use the existing unique persisted allocation evidence below.
        """
        job = self.receipt_job(stage)
        optimized = execution_settings(self.plan) is not None
        attempts = OPTIMIZED_COMPLETION_QUERIES if optimized else COMPLETION_QUERIES
        deadline = time.monotonic() + OPTIMIZED_COMPLETION_SECONDS if optimized else None
        reply, status = None, ""
        for attempt in range(attempts):
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                break
            reply = subprocess.run(["bjobs", "-a", "-noheader", "-o", "stat", job],
                                   capture_output=True, text=True,
                                   timeout=COMPLETION_QUERY_TIMEOUT if remaining is None else min(COMPLETION_QUERY_TIMEOUT, remaining))
            status = reply.stdout.strip()
            if reply.returncode == 0 and status == "DONE":
                return job
            if reply.returncode == 0 and status == "RUN" and attempt + 1 < attempts:
                pause = (COMPLETION_SETTLE_SECONDS if deadline is None
                         else min(COMPLETION_SETTLE_SECONDS, max(0., deadline - time.monotonic())))
                time.sleep(pause)
                continue
            break
        if reply is None:
            raise ValueError("Predecessor scheduler observation budget exhausted before a successful query")
        if status or (reply.returncode and "not found" not in reply.stderr.lower()):
            raise ValueError(f"Predecessor {stage} did not finish successfully: {status}")
        matches = [read(p) for p in (self.root / "allocations").glob("*/run.json")
                   if read(p).get("environment", {}).get("job_id") == job]
        if len(matches) != 1 or matches[0].get("status") != "completed" or matches[0].get("returncode") != 0:
            raise ValueError("Aged-out predecessor lacks unique completed allocation evidence")
        return job

    def queue_initial(self):
        """Only fixtures and disposable benchmark queued now; training is gated."""
        (self.work / "logs").mkdir(exist_ok=True)
        write_json(self.root / "initial_quota.json", quota_receipt(2))
        optimized = execution_settings(self.plan) is not None
        cpu = self.queue("cpu", self.worker(self.controller("cpu"), "source104-chemistry-cpu"),
                         limit="1:00" if optimized else "0:30")
        # Run the real CUDA resume regressions in the same authenticated native
        # allocation. The marker makes missing CUDA a failure, never a skip;
        # shell-quote each argument and run the benchmark only after pytest exits 0.
        cuda_fixtures = ["env", "LATENTMOL_REQUIRE_CUDA_RESUME_FIXTURES=1", "python", "-m", "pytest", "-q",
                         "tests/test_chemistry_resume_cuda.py"]
        benchmark_command = ["python", "-m", "latentmol.chemistry_continuation", "benchmark",
                             "--plan", str(self.path), "--plan-sha256", self.sha]
        native_command = ["bash", "-c", shlex.join(cuda_fixtures) + " && " + shlex.join(benchmark_command)]
        benchmark = self.queue("benchmark", self.worker(native_command, "source104-chemistry-benchmark"),
            gpu=True, limit="2:00", dependency="done(" + cpu + ")")
        callback = self.queue("after-benchmark", self.controller("advance", after_stage="benchmark"),
                              dependency="ended(" + benchmark + ")")
        return {"cpu": cpu, "benchmark": benchmark, "promotion_callback": callback,
                "status": "submitted_not_passed_no_training_yet"}

    def cpu(self):
        if not os.environ.get("LSB_JOBID"):
            raise ValueError("CPU fixtures require allocated LSF execution")
        optimized = execution_settings(self.plan, allocated=True) is not None
        command = [sys.executable, "-m", "pytest", "-q", *FIXTURES,
                   *(OPTIMIZED_FIXTURES if optimized else [])]
        reply = subprocess.run(command, cwd=self.release)
        receipt = {"status": "passed" if reply.returncode == 0 else "failed", "returncode": reply.returncode,
                   "job_id": os.environ["LSB_JOBID"], "identity": self.identity, "command": command}
        if optimized and reply.returncode == 0:
            profile_path = self.root / "repair_profile.json"
            profile_command = [sys.executable, "scripts/benchmark_atom_preserving_search.py",
                               "--plan", str(self.path), "--plan-sha256", self.sha,
                               "--output", str(profile_path)]
            receipt["profile_command"] = profile_command
            try:
                profile_reply = subprocess.run(profile_command, cwd=self.release)
                receipt["profile_returncode"] = profile_reply.returncode
                if profile_reply.returncode:
                    raise ValueError("Allocated reference/optimized repair profiler failed")
                profile = read(profile_path)
                if (profile.get("status") != "passed" or profile.get("identity") != self.identity
                        or profile.get("job_id") != receipt["job_id"]):
                    raise ValueError("Repair profiler identity/status differs from this CPU allocation")
                receipt["profile_sha256"] = file_sha256(profile_path)
            except Exception as exc:
                receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(self.root / "cpu_gate.json", receipt)
        if receipt["status"] != "passed":
            raise RuntimeError("Chemistry CPU/controller fixtures failed")

    def require_cpu(self):
        self.complete("cpu")
        value = read(self.root / "cpu_gate.json")
        if (value.get("status") != "passed" or value.get("returncode") != 0 or value.get("identity") != self.identity
                or value.get("job_id") != self.receipt_job("cpu")):
            raise ValueError("Allocated CPU gate identity differs")
        if execution_settings(self.plan) is not None:
            if value.get("profile_returncode") != 0:
                raise ValueError("Allocated repair profiler has not passed")
            profile_path = self.root / "repair_profile.json"
            authenticated(profile_path, value["profile_sha256"])
            profile = read(profile_path)
            if (profile.get("status") != "passed" or profile.get("identity") != self.identity
                    or profile.get("job_id") != value["job_id"]):
                raise ValueError("Allocated repair profiler binding differs")

    def advance(self, after_stage):
        """Fail closed; serial segment chain and reserved final evaluation."""
        audit = {"identity": self.identity, "after_stage": after_stage, "status": "stopped"}
        decision = self.root / ("advance-after-" + after_stage + ".json")
        # A callback may be re-entered after SSH loss. Do not overwrite its
        # scientific decision or schedule another trajectory under a new name.
        if decision.exists():
            saved = read(decision)
            if saved.get("identity") != self.identity:
                raise ValueError("Existing callback belongs to another phase")
            return saved
        with (self.root / ("advance-after-" + after_stage + "-intent.json")).open("x") as stream:
            json.dump(audit, stream)
        try:
            self.require_cpu()
            optimized = execution_settings(self.plan) is not None
            if not optimized:
                self.complete(after_stage)
            gate = read(self.root / "benchmark_gate.json")
            if (gate.get("status") != "passed" or gate.get("plan_sha256") != self.sha
                    or gate.get("parent_sha256") != self.plan["parent_checkpoint_sha256"]
                    or not gate.get("gates") or not all(type(v) is bool and v for v in gate["gates"].values())):
                raise ValueError("Benchmark did not pass all exact-source scientific promotion gates")
            step_control = self.plan.get("optimizer_step_control")
            if step_control is not None:
                from latentmol.chemistry_optimizer_step import (validate_optimizer_step_receipt,
                    authenticate_optimizer_step_evidence)
                if (gate.get("optimizer_step_control") != step_control
                        or gate.get("native_resume_probe", {}).get("optimizer_step_control") != step_control
                        or gate.get("native_execution_parity", {}).get("optimizer_step_control") != step_control
                        or len(gate.get("updates", [])) != self.plan["protocol"]["benchmark_updates"]):
                    raise ValueError("Native benchmark lost its amended optimizer-step identity")
                for update in gate["updates"]:
                    validate_optimizer_step_receipt(update.get("optimizer_step_control"), step_control)
                audit["optimizer_step_control"] = dict(step_control)
            adopted = (self.plan.get("runtime_adoption") is not None
                       or self.plan.get("valid_reference_adoption") is not None)
            if adopted:
                # This is an explicitly derived runtime-policy decision over
                # authenticated saved evidence, never a new GPU benchmark or a
                # rewrite of the predecessor's failed gate. The same verifier
                # is mandatory in the numerical trainer and at every resume.
                if self.plan.get("valid_reference_adoption") is not None:
                    from latentmol.chemistry_valid_reference_adoption import authenticate_adoption
                else:
                    from latentmol.chemistry_runtime_adoption import authenticate_adoption
                authenticate_adoption(dict(self.plan, _plan_sha256=self.sha),
                    file_sha256(self.root / "failure_readiness.json"))
            if optimized:
                proof = gate.get("native_execution_parity", {})
                if (gate["gates"].get("exact_reference_execution") is not True
                        or proof.get("passed") is not True or proof.get("status") != "passed"
                        or proof.get("restoration", {}).get("passed") is not True
                        or not proof.get("checks") or not all(type(v) is bool and v for v in proof["checks"].values())
                        or gate.get("execution") != self.plan["execution"]):
                    raise ValueError("Optimized execution lacks exact full-update reference parity")
            # Do not spend the new propagation allowance waiting on an already
            # failed scientific result. A passed artifact still never approves
            # RUN/EXIT: genuine scheduler DONE (or old-record proof) is required.
            if optimized and (not adopted or after_stage != "benchmark"):
                self.complete(after_stage)
            authenticated(self.root / "training_composition.json", gate["composition_sha256"])
            if not adopted:
                benchmark_run = Path(gate["run_dir"]).resolve(strict=True)
                benchmark_record = read(benchmark_run / "run.json")
                if (not benchmark_run.is_relative_to(self.root / "benchmark")
                        or benchmark_record.get("status") != "completed"
                        or benchmark_record.get("environment", {}).get("job_id") != self.receipt_job("benchmark")
                        or benchmark_record.get("source_archive_sha256") != self.plan["archive_sha256"]
                        or benchmark_record.get("source", {}).get("sha256") != self.plan["manifest_sha256"]
                        or file_sha256(benchmark_run / "gate.json") != file_sha256(self.root / "benchmark_gate.json")):
                    raise ValueError("Benchmark does not match unique completed native allocation")
            gate_sha = file_sha256(self.root / "benchmark_gate.json")
            audit["benchmark_gate_sha256"] = gate_sha
            readiness_args = []
            if (self.plan["protocol"].get("failure_guidance") is not None
                    or self.plan.get("short_continuation") is not None
                    or self.plan.get("valid_reference_continuation") is not None):
                from latentmol.failure_guidance_selection import authenticate_readiness
                failure_sha = file_sha256(self.root / "failure_readiness.json")
                authenticate_readiness(dict(self.plan, _plan_sha256=self.sha,
                    _benchmark_gate_sha256=gate_sha), failure_sha)
                readiness_args = ["--failure-readiness-sha256", failure_sha]
                audit["failure_readiness_sha256"] = failure_sha
            if self.plan.get("guarded_restart") is not None:
                from latentmol.chemistry_restart_policy import authenticate_stability
                stability = authenticate_stability(dict(self.plan, _plan_sha256=self.sha))
                audit["stability_gate_sha256"] = file_sha256(self.root / "stability_gate.json")
                audit["disposable_stability_updates"] = stability["completed_updates"]
            if (self.root / "active-phase.json").exists():
                raise ValueError("Another active/failed chemistry allocation blocks automatic continuation")
            resume, number = [], 1
            initial = self.plan.get("starting_checkpoint")
            if "epoch45_extension" in self.plan:
                from latentmol.chemistry_epoch45_extension import initial_checkpoint
                initial = initial_checkpoint(self.plan)
                if after_stage == "benchmark":
                    resume = ["--resume", initial["path"], "--resume-sha256", initial["sha256"]]
            epoch = initial["epoch"] if initial else self.plan["parent_epoch"]
            if after_stage != "benchmark":
                match = re.fullmatch("segment-([0-9]{3})", after_stage)
                if not match:
                    raise ValueError("Unknown predecessor stage")
                number = int(match[1])
                completion = read(self.root / f"segment-{number:03d}-completion.json")
                run = Path(completion["run_dir"]).resolve(strict=True)
                if not run.is_relative_to(self.root / "segments" / f"{number:03d}"):
                    raise ValueError("Completion escaped its immutable allocation")
                record, row = read(run / "run.json"), read(run / "latest_epoch.json")
                checkpoint = Path(completion["checkpoint"]).resolve(strict=True)
                if (completion.get("status") != "passed" or completion.get("plan_sha256") != self.sha
                        or not completion.get("epoch_complete") or record.get("status") != "completed"
                        or record.get("environment", {}).get("job_id") != self.receipt_job(after_stage)
                        or record.get("source_archive_sha256") != self.plan["archive_sha256"]
                        or record.get("source", {}).get("sha256") != self.plan["manifest_sha256"]
                        or record.get("config", {}).get("_plan_sha256") != self.sha
                        or row.get("epoch") != completion.get("completed_epoch")
                        or not row.get("health") or not all(type(v) is bool and v for v in row["health"].values())
                        or row.get("fresh_prior_points") != self.plan["protocol"]["prior_ratio"] * row.get("real_points", 0)
                        or not checkpoint.is_relative_to(run)
                        or row.get("checkpoint_sha256") != completion.get("checkpoint_sha256")):
                    raise ValueError("Persisted segment completion or health differs")
                previous_epoch, previous_sha = self.plan["parent_epoch"], self.plan["parent_checkpoint_sha256"]
                previous_updates = self.plan["parent_updates"]
                if initial:
                    previous_epoch, previous_sha, previous_updates = initial["epoch"], initial["sha256"], initial["updates"]
                if number > 1:
                    previous = read(self.root / f"segment-{number-1:03d}-completion.json")
                    previous_epoch, previous_sha, previous_updates = (previous[k] for k in ("completed_epoch", "checkpoint_sha256", "updates"))
                if (record.get("starting_epoch") != previous_epoch or record.get("resume_checkpoint_sha256") != previous_sha
                        or record.get("benchmark_gate_sha256") != gate_sha):
                    raise ValueError("Starting epoch/resume/checkpoint/benchmark ancestry differs")
                rows = [strict_json(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
                rows = [item for item in rows if "checkpoint_sha256" in item]
                expected_epochs = list(range(previous_epoch + 1, completion["completed_epoch"] + 1))
                if [item.get("epoch") for item in rows] != expected_epochs:
                    raise ValueError("Allocation epoch history is incomplete, duplicated or noncontiguous")
                train_count = self.plan["training_molecules"]
                updates_per_epoch = (train_count + 255) // 256
                for offset, item in enumerate(rows, 1):
                    if (item.get("real_points") != train_count or item.get("fresh_prior_points") != self.plan["protocol"]["prior_ratio"] * train_count
                            or item.get("updates") != previous_updates + offset * updates_per_epoch
                            or not item.get("health") or not all(type(v) is bool and v for v in item["health"].values())):
                        raise ValueError("Completed epoch sample/update/finite health history differs")
                    if step_control is not None:
                        if item.get("optimizer_step_control") != step_control:
                            raise ValueError("Completed epoch changed its optimizer-step policy")
                        authenticate_optimizer_step_evidence(item.get("optimizer_step_evidence"), step_control,
                            root=run, epoch=item["epoch"], start_update=previous_updates + (offset - 1) * updates_per_epoch,
                            end_update=item["updates"])
                final_row = dict(rows[-1]); final_row.pop("at", None)
                if final_row != row:
                    raise ValueError("Latest epoch artifact differs from final persisted epoch event")
                if (completion["updates"] != rows[-1]["updates"] or completion.get("segment") != number
                        or completion.get("stop_reason") != expected_stop_reason(self.plan, completion["completed_epoch"])):
                    raise ValueError("Completion update total differs from its epoch history")
                authenticated(checkpoint, completion["checkpoint_sha256"])
                epoch = completion["completed_epoch"]
                if not self.plan["parent_epoch"] < epoch <= continuation_end_epoch(self.plan):
                    raise ValueError("Completed epoch does not advance declared chemistry phase")
                resume = ["--resume", str(checkpoint), "--resume-sha256", completion["checkpoint_sha256"]]
                number += 1
            require_training_branch(self.plan)
            bounded = (self.plan.get("short_continuation") is not None
                       or self.plan.get("valid_reference_continuation") is not None)
            if epoch == continuation_end_epoch(self.plan) and bounded:
                # This is a deliberately bounded three-epoch branch, never an
                # epoch100 completion or permission to submit epoch40. The
                # authenticated segment includes its ordinary deep monitor.
                terminal = dict(status="completed_bounded_branch", identity=self.identity,
                    completed_epoch=epoch, updates=completion["updates"],
                    checkpoint=completion["checkpoint"], checkpoint_sha256=completion["checkpoint_sha256"],
                    completion_sha256=file_sha256(self.root / f"segment-{number-1:03d}-completion.json"),
                    global_epoch100_complete=False)
                if self.plan.get("short_continuation") is not None:
                    terminal["short_continuation"] = self.plan["short_continuation"]
                else:
                    terminal["valid_reference_continuation"] = self.plan["valid_reference_continuation"]
                write_json(self.root / "bounded-branch-completion.json", terminal)
                audit.update(status="short_continuation_completed", completed_epoch=epoch,
                             endpoint=continuation_end_epoch(self.plan), final_epoch100_evaluation="not_authorized_for_this_branch")
            elif epoch == 100:
                final = self.queue("final", self.worker(["python", "-m", "latentmol.chemistry_continuation", "final",
                    "--plan", str(self.path), "--plan-sha256", self.sha, "--benchmark-gate-sha256", gate_sha,
                    *readiness_args, *resume], "source104-chemistry-final"),
                    gpu=True, limit="6:00")
                audit.update(status="final_submitted_not_passed", job_id=final, completed_epoch=epoch)
            else:
                if (initial and self.plan["protocol"].get("failure_guidance") is None
                        and self.plan.get("short_continuation") is None
                        and self.plan.get("valid_reference_continuation") is None):
                    raise ValueError("Legacy handoff control is benchmark-only, not a training successor")
                if number > self.plan["protocol"]["maximum_segments"]:
                    raise ValueError("Declared serial allocation cap reached")
                # Reserve both remaining chemistry snapshots and unchanged
                # real-only campaign snapshots; do not assume quota is reserved.
                minimum = max(10., 2. + (100 - epoch) / 16 + (100 - self.plan["parent_epoch"]) / 16)
                write_json(self.root / f"quota-before-{number:03d}.json", quota_receipt(minimum))
                stage = f"segment-{number:03d}"
                job = self.queue(stage, self.worker(["python", "-m", "latentmol.chemistry_continuation", "train",
                    "--plan", str(self.path), "--plan-sha256", self.sha, "--benchmark-gate-sha256", gate_sha,
                    *readiness_args, "--segment", str(number), *resume],
                    "source104-chemistry-" + stage), gpu=True,
                    limit=validate_runtime_policy(self.plan["protocol"])["lsf_limit"])
                callback = self.queue("after-" + stage, self.controller("advance", after_stage=stage), dependency="ended(" + job + ")")
                audit.update(status="training_submitted_not_passed", job_id=job, callback_job_id=callback, completed_epoch=epoch)
        except Exception as exc:
            audit.update(error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            write_json(decision, audit)
        if (self.plan.get("short_continuation") is not None
                or self.plan.get("valid_reference_continuation") is not None):
            # The bounded suite consumes this result on its first callback,
            # not only on idempotent replay of the saved decision. Historical
            # callers retain their original None-returning interface.
            return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True); parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--mode", choices=("queue", "cpu", "advance"), required=True)
    parser.add_argument("--after-stage")
    args = parser.parse_args(); campaign = Campaign(args.plan, args.plan_sha256)
    if args.mode == "queue":
        print(json.dumps(campaign.queue_initial(), indent=2))
    elif args.mode == "cpu":
        campaign.cpu()
    elif args.after_stage:
        campaign.advance(args.after_stage)
    else:
        parser.error("advance requires --after-stage")


if __name__ == "__main__":
    main()
