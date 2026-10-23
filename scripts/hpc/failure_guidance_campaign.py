"""CPU gates -> healthy source121 boundary -> matched GPU suite -> continuation.

Controllers import metadata tools only. Frozen model/reference, optimizer and
all RNG are authenticated in allocated workers. Benchmark updates are never
promoted. Training continues serially only while BOTH gate families pass.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

from chemistry_campaign import Campaign, FIXTURES, OPTIMIZED_FIXTURES, authenticated, read, quota_receipt, validate_plan
from latentmol.experiments import file_sha256, write_json
from latentmol.failure_guidance_selection import authenticate_readiness, select_candidate, summarize_plan


VERSION = "failed-repair-campaign-v1"
IDS = ("legacy3", "fallback", "bond6")
EXTRA_FIXTURES = ["tests/test_async_repair_teacher.py", "tests/test_chemistry_speed_selection.py",
    "tests/test_chemistry_speed_campaign.py", "tests/test_repair_failure_guidance.py",
    "tests/test_failure_guidance_training.py", "tests/test_failure_guidance_selection.py",
    "tests/test_failure_guidance_campaign.py", "tests/test_deterministic_repair_events.py",
    "tests/test_failure_guidance_retry.py"]
SPEED_FIXTURES = ["tests/test_repair_progress_speed.py", "tests/test_failure_preparation_speed.py",
    "tests/test_repair_loss_validation_reuse.py", "tests/test_failure_loss_validation_reuse.py",
    "tests/test_failure_execution_predecessor.py"]


def require_terminal_metadata(reply, job, *, expected, confirmed_missing=False):
    """Recognize exact-ID LSF expiry, never a permission error or live state.

    DTU can return zero for an expired job. A missing observation is usable
    only alongside separately authenticated persisted terminal evidence.
    """
    if reply.returncode == 0 and reply.stdout.strip() == expected and not reply.stderr.strip():
        return "observed_" + expected
    if (confirmed_missing and not reply.stdout.strip()
            and reply.stderr.strip() == f"Job <{job}> is not found"):
        return "persisted_" + expected + "_now_aged_out"
    raise ValueError("Retry predecessor is live, unknown or lacks authenticated terminal evidence")


def authenticate_retry_origin(suite):
    """Same parent and scientific contract; no replay of failed native updates.

    This only permits the recorded deterministic-CUDA execution failure. All
    completed-source, scientific and quota gates remain in their original
    paths. No pending old callback is canceled or reinterpreted as passed.
    """
    policy = suite.plan.get("retry_origin")
    if (type(policy) is not dict or set(policy) != {"version", "suite", "suite_sha256", "artifacts"}
            or policy["version"] != "deterministic-repair-events-retry-v1"):
        raise ValueError("Require an explicit authenticated deterministic-event retry")
    original = RepairSuite(policy["suite"], policy["suite_sha256"])
    if (original.plan.get("retry_origin") is not None or original.policy != suite.policy
            or original.plan["protocol"] != suite.plan["protocol"]
            or original.plan["execution"] != suite.plan["execution"]
            or original.root == suite.root):
        raise ValueError("Retry changed the parent policy/objective/resources or reused its campaign")
    confirmation = suite.work / "submissions/source123-retirement-confirmation-001/confirmation.json"
    old_callback = original.receipt_job("after-native")
    callback_receipt = read(original.receipts / "after-native-receipt.json")
    callback_command = callback_receipt["intent"]["command"]
    callback_output = Path(callback_command[callback_command.index("-oo") + 1].replace("%J", old_callback))
    if not callback_output.resolve().is_relative_to(suite.work / "logs"):
        raise ValueError("Original callback completion log escaped its declared work root")
    required = {str(original.root / name) for name in
                ("starting-checkpoint.json", "bound-plans.json", "selection.json", "promotion.json",
                 "variants/fallback/failure_readiness.json", "variants/bond6/failure_readiness.json")}
    required.add(str(confirmation))
    required.add(str(callback_output))
    if type(policy["artifacts"]) is not dict or set(policy["artifacts"]) != required:
        raise ValueError("Retry lost its complete declared origin evidence")
    for path, sha in policy["artifacts"].items():
        authenticated(path, sha)
    selection, promotion = read(original.root / "selection.json"), read(original.root / "promotion.json")
    if (selection.get("suite_sha256") != original.sha
            or selection.get("job_id") != original.receipt_job("native")
            or selection.get("winner") is not None or selection.get("reason") != "no_passing_new_objective"
            or promotion.get("status") != "stopped_no_passing_new_objective"
            or promotion.get("selection") != selection):
        raise ValueError("Retry origin is not the preserved stopped comparison")
    original.complete("native")
    bound = read(original.root / "bound-plans.json")
    if (bound.get("suite_sha256") != original.sha
            or bound.get("binding_sha256") != file_sha256(original.root / "starting-checkpoint.json")
            or selection.get("bound_plans_sha256") != file_sha256(original.root / "bound-plans.json")
            or tuple(row.get("id") for row in bound.get("variants", [])) != IDS):
        raise ValueError("Original candidate plans or shared parent binding changed")
    candidate_shas = {row["id"]: row["sha256"] for row in bound["variants"]}
    for candidate in ("fallback", "bond6"):
        failure = read(original.root / "variants" / candidate / "failure_readiness.json")
        if (failure.get("status") != "failed" or failure.get("job_id") != selection["job_id"]
                or failure.get("plan_sha256") != candidate_shas[candidate]
                or failure.get("archive_sha256") != original.plan["archive_sha256"]
                or failure.get("manifest_sha256") != original.plan["manifest_sha256"]
                or "cumsum_cuda_kernel does not have a deterministic implementation" not in failure.get("error", "")):
            raise ValueError("Retry would reinterpret a different native failure")
    callback_log = callback_output.read_text()
    if (not re.search(r"^Subject: Job " + re.escape(old_callback) + r": .* Done$", callback_log, re.M)
            or "\nSuccessfully completed.\n" not in callback_log
            or "Exited with exit code" in callback_log):
        raise ValueError("Original callback lacks persisted LSF DONE evidence")
    source = suite.source()
    require_no_source_successor(source)
    proof = read(confirmation)
    callback = source.receipt_job("after-segment-001")
    if (proof.get("suite_sha256") != original.sha or proof.get("callback_job") != callback
            or callback != suite.policy["source_callback_job"]
            or proof.get("job_id") != original.receipt_job("retirement-readiness")
            or proof.get("status") != "retired_confirmed" or proof.get("observed") != "observed_EXIT"
            or proof.get("returncode") != 0 or proof.get("stdout", "").strip() != "EXIT"
            or proof.get("stderr") != ""):
        raise ValueError("Retry lost the original allocated cancellation confirmation")
    observations = {}
    for role, job, expected in (("source_callback", callback, "EXIT"),
                                ("old_native_callback", old_callback, "DONE")):
        args = ["bjobs", "-a", "-noheader", "-o", "stat", job]
        reply = subprocess.run(args, capture_output=True, text=True, timeout=15)
        observed = require_terminal_metadata(reply, job, expected=expected, confirmed_missing=True)
        observations[role] = dict(job_id=job, command=args, observed=observed,
            returncode=reply.returncode, stdout=reply.stdout, stderr=reply.stderr)
    return dict(status="authenticated_retry_origin", suite_sha256=suite.sha,
        original_suite_sha256=original.sha, original_artifacts=policy["artifacts"],
        starting_checkpoint=read(original.root / "starting-checkpoint.json"), observations=observations)


def algorithm_from_manifest(release):
    files = read(Path(release) / "release_manifest.json")["files"]
    package = {name[len("src/latentmol/"):]: digest for name, digest in files.items()
               if name.startswith("src/latentmol/") and name.endswith(".py")}
    return dict(files=package, sha256=hashlib.sha256(json.dumps(package, sort_keys=True).encode()).hexdigest())


def healthy_handoff(source, segment, expected_job):
    """Authenticate the COMPLETE source segment without invoking its callback."""
    stage = f"segment-{segment:03d}"
    require_no_source_successor(source)
    if segment != 1 or source.receipt_job(stage) != expected_job:
        raise ValueError("Only the declared source121 first completed segment may be handed off")
    source.complete(stage)
    root = source.root
    completion_path = root / (stage + "-completion.json")
    completion = read(completion_path)
    run = Path(completion["run_dir"]).resolve(strict=True)
    if not run.is_relative_to(root / "segments" / f"{segment:03d}") or (root / "active-phase.json").exists():
        raise ValueError("Source segment is active, failed or outside its declared campaign")
    record, latest = read(run / "run.json"), read(run / "latest_epoch.json")
    checkpoint = Path(completion["checkpoint"]).resolve(strict=True)
    parent_epoch, parent_updates = source.plan["parent_epoch"], source.plan["parent_updates"]
    if (completion.get("status") != "passed" or completion.get("epoch_complete") is not True
            or completion.get("plan_sha256") != source.sha or completion.get("segment") != segment
            or not parent_epoch < completion["completed_epoch"] < 100
            or completion.get("stop_reason") != "wall_budget_epoch_boundary"
            or record.get("status") != "completed"
            or record.get("environment", {}).get("job_id") != expected_job
            or record.get("source_archive_sha256") != source.plan["archive_sha256"]
            or record.get("source", {}).get("sha256") != source.plan["manifest_sha256"]
            or record.get("config", {}).get("_plan_sha256") != source.sha
            or record.get("starting_epoch") != parent_epoch
            or record.get("resume_checkpoint_sha256") != source.plan["parent_checkpoint_sha256"]
            or not checkpoint.is_relative_to(run)
            or file_sha256(run / "completion.json") != file_sha256(completion_path)
            or latest.get("epoch") != completion["completed_epoch"]
            or latest.get("checkpoint_sha256") != completion.get("checkpoint_sha256")
            or latest.get("checkpoint") != str(checkpoint)):
        raise ValueError("Completed source checkpoint/run ancestry differs")
    gate = root / "benchmark_gate.json"
    if record.get("benchmark_gate_sha256") != file_sha256(gate):
        raise ValueError("Source epoch changed its passed native benchmark binding")
    native = read(gate)
    if native.get("status") != "passed" or not native.get("gates") or not all(v is True for v in native["gates"].values()):
        raise ValueError("Source native readiness did not pass")
    rows = [read_json_line(line) for line in (run / "metrics.jsonl").read_text().splitlines() if line.strip()]
    rows = [row for row in rows if "checkpoint_sha256" in row]
    if [row.get("epoch") for row in rows] != list(range(parent_epoch + 1, completion["completed_epoch"] + 1)):
        raise ValueError("Source completed epoch history is incomplete or noncontiguous")
    count = source.plan["training_molecules"]
    for offset, row in enumerate(rows, 1):
        if (row.get("real_points") != count or row.get("fresh_prior_points") != count * 2
                or row.get("updates") != parent_updates + offset * ((count + 255) // 256)
                or not row.get("health") or not all(value is True for value in row["health"].values())):
            raise ValueError("Source completed epoch failed sample/update/health checks")
    final = dict(rows[-1]); final.pop("at", None)
    if final != latest or completion.get("updates") != final["updates"]:
        raise ValueError("Source latest checkpoint differs from its persisted healthy epoch")
    authenticated(checkpoint, completion["checkpoint_sha256"])
    return dict(path=str(checkpoint), sha256=completion["checkpoint_sha256"], source_plan=str(source.path),
        source_plan_sha256=source.sha, completion=str(completion_path), completion_sha256=file_sha256(completion_path),
        epoch=completion["completed_epoch"], updates=completion["updates"], algorithm=algorithm_from_manifest(source.release))


def read_json_line(line):
    return json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def require_no_source_successor(source):
    """Recheck on every bind/re-entry, including pending successor submissions."""
    paths = [source.root / "advance-after-segment-001-intent.json",
             source.root / "advance-after-segment-001.json",
             source.root / "receipts/segment-002-intent.json",
             source.root / "receipts/segment-002-receipt.json"]
    if any(path.exists() for path in paths):
        raise ValueError("A source continuation controller or successor has already acted")


class RepairSuite(Campaign):
    def __init__(self, path, sha):
        super().__init__(path, sha)
        policy = self.plan.get("failure_suite", {})
        if (policy.get("version") != VERSION or tuple(policy.get("variants", [])) != IDS
                or self.plan["protocol"]["prior_ratio"] != 2
                or self.plan.get("execution") != dict(backend="optimized", workers=12, torch_threads=4, gpu_cpus=16, overlap=True)):
            raise ValueError("Unsupported new-objective benchmark protocol")
        self.policy = policy

    def _stopped_execution(self, value, *, reference):
        """Shared metadata authentication; never import the old loss/controller."""
        old = RepairSuite(value["suite"], value["suite_sha256"])
        if ((reference and old.plan.get("execution_reference") is not None) or old.root == self.root
                or old.policy != self.policy or old.plan["protocol"] != self.plan["protocol"]
                or old.plan["execution"] != self.plan["execution"]
                or str(old.release) != value["release"]
                or old.plan["archive_sha256"] != value["archive_sha256"]
                or old.plan["manifest_sha256"] != value["manifest_sha256"]):
            raise ValueError("Execution optimization changed the comparison's scientific contract")
        callback = old.receipt_job("after-native")
        command = read(old.receipts / "after-native-receipt.json")["intent"]["command"]
        log = Path(command[command.index("-oo") + 1].replace("%J", callback))
        required_paths = {str(old.root / name) for name in
                          ("starting-checkpoint.json", "selection.json", "promotion.json")}
        if not reference:
            required_paths.add(str(old.root / "bound-plans.json"))
        required_paths.add(str(log))
        if not log.resolve().is_relative_to(self.work / "logs") or set(value["artifacts"]) != required_paths:
            raise ValueError("Execution reference lost its exact stopped-outcome evidence")
        for path, sha in value["artifacts"].items():
            authenticated(path, sha)
        selection, promotion = read(old.root / "selection.json"), read(old.root / "promotion.json")
        if (selection.get("suite_sha256") != old.sha
                or selection.get("job_id") != old.receipt_job("native") or selection.get("winner") is not None
                or selection.get("reason") != "no_passing_new_objective"
                or promotion.get("status") != "stopped_no_passing_new_objective"
                or promotion.get("selection") != selection
                or read(old.root / "starting-checkpoint.json") != value["starting_checkpoint"]):
            raise ValueError("Execution reference is not the preserved stopped comparison")
        old.complete("native")
        text = log.read_text()
        if (not re.search(r"^Subject: Job " + re.escape(callback) + r": .* Done$", text, re.M)
                or "\nSuccessfully completed.\n" not in text or "Exited with exit code" in text):
            raise ValueError("Execution reference lacks persisted callback DONE evidence")
        reply = subprocess.run(["bjobs", "-a", "-noheader", "-o", "stat", callback],
                               capture_output=True, text=True, timeout=15)
        require_terminal_metadata(reply, callback, expected="DONE", confirmed_missing=True)
        return old

    def execution_reference(self):
        """Authenticate the fixed historical parity oracle, not a chain of oracles."""
        value = self.plan.get("execution_reference")
        if value is None:
            return None
        required = {"version", "suite", "suite_sha256", "release", "archive_sha256",
                    "manifest_sha256", "module_sha256", "starting_checkpoint", "artifacts"}
        if (type(value) is not dict or set(value) != required
                or value["version"] != "failed-repair-execution-reference-v1"):
            raise ValueError("Require a complete source-bound execution reference")
        old = self._stopped_execution(value, reference=True)
        authenticated(old.release / "src/latentmol/repair_failure_guidance.py", value["module_sha256"])
        return value

    def execution_predecessor(self):
        """Recheck the immediate stopped campaign without replacing the A2 oracle."""
        value = self.plan.get("execution_predecessor")
        if value is None:
            return None
        required = {"version", "suite", "suite_sha256", "release", "archive_sha256",
                    "manifest_sha256", "starting_checkpoint", "artifacts"}
        if (type(value) is not dict or set(value) != required
                or value["version"] != "failed-repair-execution-predecessor-v1"):
            raise ValueError("Require a complete source-bound execution predecessor")
        old = self._stopped_execution(value, reference=False)
        changed = {"release", "campaign", "archive_sha256", "manifest_sha256", "execution_predecessor"}
        if ({key: part for key, part in old.plan.items() if key not in changed}
                != {key: part for key, part in self.plan.items() if key not in changed}
                or self.plan.get("execution_reference", {}).get("starting_checkpoint") != value["starting_checkpoint"]):
            raise ValueError("Execution predecessor changed source ancestry or the fixed parity reference")
        bound = read(old.root / "bound-plans.json")
        if (bound.get("suite_sha256") != old.sha
                or bound.get("binding_sha256") != file_sha256(old.root / "starting-checkpoint.json")
                or read(old.root / "selection.json").get("bound_plans_sha256") != file_sha256(old.root / "bound-plans.json")
                or tuple(row.get("id") for row in bound.get("variants", [])) != IDS):
            raise ValueError("Execution predecessor lost its shared bound plans")
        roots = [old.root]
        for row in bound["variants"]:
            plan = validate_plan(row["path"], row["sha256"])
            expected = old.root / "variants" / row["id"]
            if (Path(plan["campaign"]) != expected or plan.get("failure_variant_id") != row["id"]
                    or plan.get("failure_suite_sha256") != old.sha
                    or plan.get("starting_checkpoint") != value["starting_checkpoint"]
                    or plan["archive_sha256"] != old.plan["archive_sha256"]
                    or plan["manifest_sha256"] != old.plan["manifest_sha256"]):
                raise ValueError("Execution predecessor candidate changed source/start identity")
            roots.append(expected)
        patterns = ("advance-after-*.json", "receipts/segment-*.json", "receipts/after-segment-*.json",
                    "receipts/final-*.json", "segment-*-completion.json", "segments/*")
        if any((root / "active-phase.json").exists() or any(any(root.glob(pattern)) for pattern in patterns)
               for root in roots):
            raise ValueError("Execution predecessor already has active training or a successor intent/output")
        current = self.root / "starting-checkpoint.json"
        if current.exists() and read(current) != value["starting_checkpoint"]:
            raise ValueError("Execution predecessor differs from the current starting checkpoint")
        return value

    def controller(self, mode, **values):
        result = [str(self.work / "env/boltz311/bin/python"), "scripts/hpc/failure_guidance_campaign.py",
                  "--suite", str(self.path), "--suite-sha256", self.sha, "--mode", mode]
        for key, value in values.items():
            result += ["--" + key.replace("_", "-"), str(value)]
        return ["env", "PYTHONPATH=" + str(self.release / "src"), *result]

    def queue_initial(self):
        self.execution_predecessor()
        (self.work / "logs").mkdir(exist_ok=True)
        write_json(self.root / "initial_quota.json", quota_receipt(2))
        cpu = self.queue("cpu", self.worker(self.controller("cpu"), "source123-repair-cpu"), limit="1:00")
        after = self.queue("after-cpu", self.controller("after-cpu"), dependency="ended(" + cpu + ")")
        return dict(status="submitted_not_passed", cpu=cpu, after_cpu=after,
                    gpu="not_submitted_until_cpu_and_healthy_source_boundary", training="not_submitted")

    def cpu(self):
        if not os.environ.get("LSB_JOBID"):
            raise ValueError("All fixtures require an allocated CPU job")
        execution_reference = self.execution_reference()
        self.execution_predecessor()
        environment = dict(os.environ)
        if execution_reference is not None:
            environment["LATENTMOL_FAILURE_REFERENCE_MODULE"] = str(
                Path(execution_reference["release"]) / "src/latentmol/repair_failure_guidance.py")
        command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                   "--junitxml", str(self.root / "cpu-pytest.xml"), *FIXTURES, *OPTIMIZED_FIXTURES, *EXTRA_FIXTURES,
                   *(SPEED_FIXTURES if execution_reference is not None else [])]
        reply = subprocess.run(command, cwd=self.release, env=environment)
        gate = dict(status="passed" if reply.returncode == 0 else "failed", returncode=reply.returncode,
                    job_id=os.environ["LSB_JOBID"], identity=self.identity, command=command)
        if reply.returncode == 0:
            profile_path = self.root / "repair_profile.json"
            profile_command = [sys.executable, "scripts/benchmark_atom_preserving_search.py", "--plan", str(self.path),
                               "--plan-sha256", self.sha, "--output", str(profile_path)]
            result = subprocess.run(profile_command, cwd=self.release)
            gate.update(profile_command=profile_command, profile_returncode=result.returncode)
            if result.returncode:
                gate["status"] = "failed"
            else:
                profile = read(profile_path)
                if profile.get("status") != "passed" or profile.get("identity") != self.identity or profile.get("job_id") != gate["job_id"]:
                    raise ValueError("Allocated legacy repair profile lost its source identity")
                gate["profile_sha256"] = file_sha256(profile_path)
        if gate["status"] == "passed" and execution_reference is not None:
            path = self.root / "failure_preparation_profile.json"
            command = [sys.executable, "scripts/benchmark_failure_preparation.py", "--plan", str(self.path),
                       "--plan-sha256", self.sha, "--output", str(path)]
            result = subprocess.run(command, cwd=self.release, env=environment)
            gate.update(execution_profile_command=command, execution_profile_returncode=result.returncode)
            if result.returncode:
                gate["status"] = "failed"
            else:
                profile = read(path)
                if (profile.get("status") != "passed" or profile.get("identity") != self.identity
                        or profile.get("checkpoint_sha256") != execution_reference["starting_checkpoint"]["sha256"]
                        or profile.get("job_id") != gate["job_id"]):
                    raise ValueError("Execution profile lost its source/checkpoint/allocation identity")
                gate["execution_profile_sha256"] = file_sha256(path)
        write_json(self.root / "cpu_gate.json", gate)
        if gate["status"] != "passed":
            raise RuntimeError("New loss/controller/compatibility fixtures failed")

    def require_cpu(self):
        self.execution_predecessor()
        super().require_cpu()
        reference = self.plan.get("execution_reference")
        if reference is not None:
            gate = read(self.root / "cpu_gate.json")
            path = self.root / "failure_preparation_profile.json"
            authenticated(path, gate["execution_profile_sha256"])
            profile = read(path)
            if (profile.get("status") != "passed" or profile.get("identity") != self.identity
                    or profile.get("checkpoint_sha256") != reference["starting_checkpoint"]["sha256"]
                    or profile.get("job_id") != self.receipt_job("cpu")):
                raise ValueError("Mandatory execution profile is not an authenticated CPU pass")

    def source(self):
        return Campaign(self.policy["source_plan"], self.policy["source_plan_sha256"])

    def retire_original_callback(self):
        """Retire only the pending known callback, while its segment finishes."""
        if self.plan.get("retry_origin") is not None:
            proof = authenticate_retry_origin(self)
            path = self.root / "retry-origin-proof.json"
            if not path.exists():
                write_json(path, proof)
            return proof
        path = self.root / "source-callback-retirement.json"
        source = self.source()
        require_no_source_successor(source)
        job = source.receipt_job("after-segment-001")
        if job != self.policy["source_callback_job"]:
            raise ValueError("Original callback receipt changed")
        if path.exists():
            value = read(path)
            if (value.get("status") != "retired" or value.get("suite_sha256") != self.sha
                    or value.get("callback_job") != job or value.get("source_job") != self.policy["source_job"]
                    or value.get("source_plan_sha256") != source.sha or value.get("returncode") != 0
                    or value.get("command") != ["bkill", job] or value.get("before", "").strip() != "PEND"
                    or value.get("after", "").strip() != "EXIT"):
                raise ValueError("Original callback retirement is uncertain")
            return value
        query = ["bjobs", "-a", "-noheader", "-o", "stat", job]
        before = subprocess.run(query, capture_output=True, text=True, timeout=15)
        if before.returncode or before.stdout.strip() != "PEND":
            raise ValueError("Original callback must still be pending before retirement")
        intent = dict(suite_sha256=self.sha, source_plan_sha256=source.sha, callback_job=job,
                      source_job=self.policy["source_job"], command=["bkill", job], before=before.stdout,
                      utc=datetime.now(timezone.utc).isoformat())
        with (self.root / "source-callback-retirement-intent.json").open("x") as stream:
            json.dump(intent, stream)
        killed = subprocess.run(intent["command"], capture_output=True, text=True, timeout=30)
        after = subprocess.run(query, capture_output=True, text=True, timeout=15)
        value = dict(intent, status="retired" if killed.returncode == 0 and after.returncode == 0
                     and after.stdout.strip() == "EXIT" else "uncertain", stdout=killed.stdout,
                     stderr=killed.stderr, returncode=killed.returncode, after=after.stdout)
        write_json(path, value)
        if value["status"] != "retired":
            raise ValueError("Callback retirement is not proven; stop")
        require_no_source_successor(source)
        return value

    def after_cpu(self):
        self.require_cpu()
        if self.plan.get("retry_origin") is not None:
            # The old source is completed and may have aged out of LSF.
            # Authenticate it in healthy_handoff; do not depend on an expired
            # ended(old_job) expression in a fresh scheduler submission.
            return self.bind_parent()
        self.retire_original_callback()
        job = self.queue("bind-parent", self.controller("bind-parent"), dependency="ended(" + self.policy["source_job"] + ")")
        return dict(status="healthy_boundary_handoff_queued", job_id=job, source_job=self.policy["source_job"])

    def bind_parent(self):
        self.require_cpu()
        self.retire_original_callback()
        source = self.source()
        binding = healthy_handoff(source, 1, self.policy["source_job"])
        if (self.plan.get("retry_origin") is not None
                and binding != authenticate_retry_origin(self)["starting_checkpoint"]):
            raise ValueError("Retry changed the authenticated epoch-36 starting checkpoint")
        if (self.plan.get("execution_predecessor") is not None
                and binding != self.plan["execution_predecessor"]["starting_checkpoint"]):
            raise ValueError("Execution predecessor changed the authenticated starting checkpoint")
        write_json(self.root / "starting-checkpoint.json", binding)
        plans = self.root / "plans"; plans.mkdir(exist_ok=False)
        rows = []
        for candidate in IDS:
            relative = "configs/chemistry_failure_variants/" + candidate + ".json"
            plan = dict(source.plan, release=str(self.release), archive_sha256=self.plan["archive_sha256"],
                manifest_sha256=self.plan["manifest_sha256"], campaign=str(self.root / "variants" / candidate),
                protocol=read(self.release / relative), protocol_config=relative,
                failure_variant_id=candidate, failure_suite_path=str(self.path), failure_suite_sha256=self.sha,
                starting_checkpoint=binding, handoff_version="chemistry-objective-comparison-handoff-v1")
            path = plans / (candidate + ".json")
            write_json(path, plan)
            rows.append(dict(id=candidate, path=str(path), sha256=file_sha256(path)))
        write_json(self.root / "bound-plans.json", dict(suite_sha256=self.sha, variants=rows,
            starting_checkpoint_sha256=binding["sha256"], binding_sha256=file_sha256(self.root / "starting-checkpoint.json")))
        gpu = self.queue("native", self.worker(self.controller("native"), "source123-repair-native"), gpu=True, limit="6:00")
        after = self.queue("after-native", self.controller("after-native"), dependency="ended(" + gpu + ")")
        write_json(self.root / "native-queue.json", dict(gpu=gpu, callback=after, status="submitted_not_passed"))

    def bound_plans(self):
        value = read(self.root / "bound-plans.json")
        if value.get("suite_sha256") != self.sha or tuple(row.get("id") for row in value.get("variants", [])) != IDS:
            raise ValueError("Source-bound comparison plan family changed")
        authenticated(self.root / "starting-checkpoint.json", value["binding_sha256"])
        result = []
        for row in value["variants"]:
            p = validate_plan(row["path"], row["sha256"])
            if (p.get("failure_variant_id") != row["id"] or p.get("failure_suite_sha256") != self.sha
                    or p.get("starting_checkpoint") != read(self.root / "starting-checkpoint.json")
                    or p["archive_sha256"] != self.plan["archive_sha256"] or p["manifest_sha256"] != self.plan["manifest_sha256"]):
                raise ValueError("Bound candidate lost its shared source/start identity")
            result.append(dict(p, _plan_sha256=row["sha256"], _plan_path=row["path"]))
        return result

    def require_loss_validation(self):
        """New call-local execution reuse needs an actual same-native-job CUDA pass."""
        if self.plan.get("execution_predecessor") is None:
            return
        gate = read(self.root / "loss-validation-gate.json")
        reference = self.plan["execution_reference"]
        command = [str(self.work / "env/boltz311/bin/python"), "-m", "pytest", "-q", "-p", "no:cacheprovider",
                   "--junitxml", str(self.root / "loss-validation-pytest.xml"),
                   "tests/test_failure_loss_validation_reuse.py"]
        expected_environment = dict(LATENTMOL_REQUIRE_REPAIR_CUDA_FIXTURES="1",
            LATENTMOL_FAILURE_REFERENCE_MODULE=str(Path(reference["release"]) / "src/latentmol/repair_failure_guidance.py"))
        if (gate.get("status") != "passed" or gate.get("returncode") != 0
                or gate.get("suite_sha256") != self.sha or gate.get("identity") != self.identity
                or gate.get("job_id") != self.receipt_job("native") or gate.get("command") != command
                or gate.get("environment") != expected_environment
                or gate.get("reference_module_sha256") != reference["module_sha256"]):
            raise ValueError("Mandatory loss/CUDA parity is not bound to this native allocation")
        path = self.root / "loss-validation-pytest.xml"
        authenticated(path, gate["junit_sha256"])
        suites = list(ET.parse(path).getroot().iter("testsuite"))
        if (not suites or sum(int(row.get("tests", "0")) for row in suites) < 1
                or any(int(row.get(key, "0")) for row in suites for key in ("failures", "errors", "skipped"))):
            raise ValueError("Mandatory loss/CUDA parity must execute without failures or skips")

    def native(self):
        self.require_cpu()
        if not os.environ.get("LSB_JOBID"):
            raise ValueError("Native comparison requires allocated CUDA")
        event_command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                         "--junitxml", str(self.root / "deterministic-events-pytest.xml"),
                         "tests/test_deterministic_repair_events.py"]
        event_environment = dict(os.environ, LATENTMOL_REQUIRE_REPAIR_CUDA_FIXTURES="1")
        event_result = subprocess.run(event_command, cwd=self.release, env=event_environment)
        write_json(self.root / "deterministic-events-gate.json", dict(command=event_command,
            returncode=event_result.returncode, job_id=os.environ["LSB_JOBID"], suite_sha256=self.sha))
        if event_result.returncode:
            raise RuntimeError("Mandatory deterministic event loss/gradient fixtures failed")
        command = [sys.executable, "-m", "pytest", "-q", "tests/test_chemistry_resume_cuda.py",
                   "tests/test_failure_guidance_training.py", "-k", "cuda or frozen_cpu_references"]
        environment = dict(os.environ, LATENTMOL_REQUIRE_CUDA_RESUME_FIXTURES="1")
        check = subprocess.run(command, cwd=self.release, env=environment)
        write_json(self.root / "cuda-fixtures.json", dict(command=command, returncode=check.returncode,
            job_id=os.environ["LSB_JOBID"], suite_sha256=self.sha))
        if check.returncode:
            raise RuntimeError("Mandatory native CUDA fixtures failed")
        if self.plan.get("execution_predecessor") is not None:
            reference = self.execution_reference()
            path = self.root / "loss-validation-pytest.xml"
            command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                       "--junitxml", str(path), "tests/test_failure_loss_validation_reuse.py"]
            required_environment = dict(LATENTMOL_REQUIRE_REPAIR_CUDA_FIXTURES="1",
                LATENTMOL_FAILURE_REFERENCE_MODULE=str(Path(reference["release"]) / "src/latentmol/repair_failure_guidance.py"))
            result = subprocess.run(command, cwd=self.release, env=dict(os.environ, **required_environment))
            write_json(self.root / "loss-validation-gate.json", dict(command=command,
                returncode=result.returncode, status="passed" if result.returncode == 0 else "failed",
                job_id=os.environ["LSB_JOBID"], suite_sha256=self.sha, identity=self.identity,
                environment=required_environment, reference_module_sha256=reference["module_sha256"],
                junit_sha256=file_sha256(path) if path.exists() else None))
            self.require_loss_validation()
        rows = []
        for plan in self.bound_plans():
            result = subprocess.run([sys.executable, "-m", "latentmol.failure_guidance_benchmark",
                "--plan", plan["_plan_path"], "--plan-sha256", plan["_plan_sha256"]], cwd=self.release)
            rows.append(summarize_plan(plan, result.returncode))
            write_json(self.root / "observations.json", dict(suite_sha256=self.sha, rows=rows))
            if plan["failure_variant_id"] == "legacy3" and not rows[-1]["eligible"]:
                raise RuntimeError("Matched legacy native control failed")
        decision = select_candidate(rows)
        write_json(self.root / "selection.json", dict(decision, suite_sha256=self.sha,
            job_id=os.environ["LSB_JOBID"], rows=rows, bound_plans_sha256=file_sha256(self.root / "bound-plans.json")))

    def after_native(self):
        self.require_cpu(); self.complete("native")
        self.require_loss_validation()
        decision = read(self.root / "selection.json")
        if (decision.get("suite_sha256") != self.sha or decision.get("job_id") != self.receipt_job("native")
                or decision.get("bound_plans_sha256") != file_sha256(self.root / "bound-plans.json")
                or select_candidate(decision["rows"]) != {key: decision[key] for key in
                    ("status", "winner", "reason", "prior_ratio", "selection_epoch_seconds") if key in decision}):
            raise ValueError("Native selection identity or deterministic decision changed")
        if decision["winner"] is None:
            write_json(self.root / "promotion.json", dict(status="stopped_no_passing_new_objective", selection=decision))
            return
        selected = next(plan for plan in self.bound_plans() if plan["failure_variant_id"] == decision["winner"])
        campaign = FailureCampaign(selected["_plan_path"], selected["_plan_sha256"])
        try:
            campaign.advance("benchmark")
        except BaseException as error:
            # Keep a durable summary when unchanged quota/health gates stop
            # automatic promotion; eligibility is never a training receipt.
            decision_path = campaign.root / "advance-after-benchmark.json"
            write_json(self.root / "promotion.json", dict(status="stopped_promotion_error",
                winner=decision["winner"], error=f"{type(error).__name__}: {error}",
                decision=str(decision_path), decision_exists=decision_path.exists()))
            raise
        decision_path = campaign.root / "advance-after-benchmark.json"
        promotion = read(decision_path)
        write_json(self.root / "promotion.json", dict(status=promotion["status"], winner=decision["winner"],
            job_id=promotion.get("job_id"), callback_job_id=promotion.get("callback_job_id"), decision=str(decision_path)))


class FailureCampaign(Campaign):
    """Reuse the original serial health/identity/quota checks for the winner."""
    def __init__(self, path, sha):
        super().__init__(path, sha)
        self.suite = RepairSuite(self.plan["failure_suite_path"], self.plan["failure_suite_sha256"])
        selection = read(self.suite.root / "selection.json")
        if (self.plan["failure_variant_id"] == "legacy3" or self.plan["protocol"].get("failure_guidance") is None
                or selection.get("winner") != self.plan["failure_variant_id"]):
            raise ValueError("Only the selected new-objective candidate may train")

    def require_cpu(self):
        self.suite.require_cpu()
        self.suite.require_loss_validation()

    def receipt_job(self, stage):
        return self.suite.receipt_job("native") if stage == "benchmark" else super().receipt_job(stage)

    def complete(self, stage):
        return self.suite.complete("native") if stage == "benchmark" else super().complete(stage)

    def controller(self, mode, **values):
        result = super().controller(mode, **values)
        return [part.replace("scripts/hpc/chemistry_campaign.py", "scripts/hpc/failure_guidance_campaign.py") for part in result]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path); parser.add_argument("--suite-sha256")
    parser.add_argument("--plan", type=Path); parser.add_argument("--plan-sha256")
    parser.add_argument("--mode", choices=("queue", "cpu", "after-cpu", "bind-parent", "native", "after-native", "advance"), required=True)
    parser.add_argument("--after-stage")
    args = parser.parse_args()
    if args.mode == "advance":
        if args.plan is None or args.plan_sha256 is None or args.after_stage is None:
            parser.error("Serial continuation needs an exact plan/hash and predecessor")
        FailureCampaign(args.plan, args.plan_sha256).advance(args.after_stage)
    else:
        if args.suite is None or args.suite_sha256 is None:
            parser.error("Suite stages need an authenticated suite path/hash")
        suite = RepairSuite(args.suite, args.suite_sha256)
        result = suite.queue_initial() if args.mode == "queue" else getattr(suite, args.mode.replace("-", "_"))()
        if result is not None:
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
