"""Standard-library, fail-closed controller for the independent long bridge study.

No torch/model imports occur here. Every allocated worker and every scheduler
dispatch reauthenticates the immutable release and obtains fresh work3 quota.
Submission intents are exclusive: an uncertain bsub outcome is never retried.
All mutable control records live outside the sealed source release.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

try:
    from embedding_operator import authenticate, digest, quota, write_new
except ModuleNotFoundError:
    # importlib-based allocated fixtures do not add this script's directory.
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "_long_embedding_operator", Path(__file__).with_name("embedding_operator.py"))
    _operator = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_operator)
    authenticate, digest = _operator.authenticate, _operator.digest
    quota, write_new = _operator.quota, _operator.write_new


ROOT = Path("/work3/s234818/latentmol")
NAMESPACE = "source142-epoch36-guarded-embedding-long-attempt001"
CONFIG = "configs/epoch36_embedding_long.json"
SCRIPT = "scripts/hpc/embedding_long_campaign.py"
MODULE = "latentmol.boltz.embedding_long_experiment"
CASES = ("case-001", "case-002", "case-003")
ENDPOINT = 10000
OBSERVE_EVERY = 10
INTERNAL_SECONDS = 23 * 3600
SHA = re.compile(r"^[0-9a-f]{64}$")
SEGMENT = re.compile(r"^(case-00[123])-segment-([0-9]{4})$")
RESULT_LIMIT = 8 * 1024 * 1024


def read(path, limit=RESULT_LIMIT):
    path = Path(path)
    if path.stat().st_size > limit:
        raise ValueError("Metadata exceeds bounded read: " + str(path))
    return json.loads(path.read_text(), parse_constant=lambda value:
                      (_ for _ in ()).throw(ValueError("Nonfinite JSON: " + value)))


def integer(value, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError("Invalid bounded integer")
    return value


def finite(value, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError("Invalid finite profile value")
    return float(value)


def stage_for(node):
    if node in ("cpu", "fit", "pilot", "followup"):
        return node
    if SEGMENT.fullmatch(node):
        return "segment"
    raise ValueError("Unknown campaign node")


def junit_passed(path):
    tree = ET.parse(path)
    cases = list(tree.getroot().iter("testcase"))
    if not cases or any(list(case.iter(tag)) for case in cases for tag in ("failure", "error", "skipped")):
        raise ValueError("CPU JUnit requires cases and zero failures/errors/skips")
    for suite in tree.getroot().iter("testsuite"):
        for key in ("failures", "errors", "skipped"):
            if int(suite.get(key, "0")) != 0:
                raise ValueError("CPU JUnit aggregate is not passed")
    return len(cases)


def worker_environment(stage):
    environment = dict(os.environ)
    if stage == "cpu":
        environment.update(MKL_CBWR="COMPATIBLE", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    return environment


class Campaign:
    def __init__(self, release, campaign, archive_sha256, manifest_sha256, plan_sha256):
        self.release, self.root = Path(release).resolve(), Path(campaign).resolve()
        work = ROOT.resolve()
        attempt = re.fullmatch(r"source142-epoch36-guarded-embedding-long-attempt([0-9]{3})", self.root.name)
        if (self.release.parent != work / "releases" or self.root.parent != work / "runs"
                or not attempt or int(attempt.group(1)) < 1
                or self.release.name != self.root.name.replace("source142", "source-142", 1)):
            raise ValueError("Paths must use the exact independent source142 namespace")
        self.control = work / "submissions" / self.root.name
        if self.control.resolve().parent != work / "submissions":
            raise ValueError("Submission controls escape their allowlist")
        for value in (archive_sha256, manifest_sha256, plan_sha256):
            if not isinstance(value, str) or not SHA.fullmatch(value):
                raise ValueError("Expected a complete lowercase SHA256")
        self.identity = dict(source_sha256=archive_sha256, manifest_sha256=manifest_sha256,
                             plan_sha256=plan_sha256)
        self.archive_sha256, self.plan_sha256 = archive_sha256, plan_sha256
        self.check_source()

    def check_source(self):
        authenticate(self.release, self.archive_sha256, self.plan_sha256, CONFIG)
        if digest(self.release / "embedding-release-manifest.json") != self.identity["manifest_sha256"]:
            raise ValueError("Extracted manifest SHA256 differs from pinned identity")
        settings = read(self.release / CONFIG).get("long_campaign", {})
        if (settings.get("version") != "guarded-embedding-long-v1"
                or settings.get("case_ids") != list(CASES)
                or settings.get("updates_per_case") != ENDPOINT
                or settings.get("observe_every") != OBSERVE_EVERY
                or settings.get("segment_wall") != "24:00"
                or settings.get("segment_internal_seconds") != INTERNAL_SECONDS
                or finite(settings.get("pilot_safety_factor"), positive=True) < 1.5
                or integer(settings.get("segment_reserve_seconds"), 1800) >= INTERNAL_SECONDS
                or integer(settings.get("max_segments_per_case"), 1, 1000) < 1):
            raise ValueError("Long campaign must preserve the declared guarded endpoint/profile policy")
        self.settings = settings

    def prepare(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.control.mkdir(parents=True, exist_ok=True)
        if self.control.resolve().parent != ROOT.resolve() / "submissions":
            raise ValueError("Submission controls escape their allowlist")

    def inside(self, path, base):
        path, base = Path(path).resolve(), Path(base).resolve()
        if not path.is_relative_to(base) or path == base or not path.is_file():
            raise ValueError("Artifact outside its allowed stage directory: " + str(path))
        return path

    def artifacts(self, result):
        values = result.get("artifacts")
        if not isinstance(values, dict) or not values:
            raise ValueError("Missing stage artifacts")
        return values

    def asset(self, result, role):
        return self.artifacts(result).get(role)

    def artifact(self, result, role, directory):
        item = self.asset(result, role)
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not SHA.fullmatch(item.get("sha256", "")):
            raise ValueError("Missing authenticated artifact: " + role)
        raw = Path(item["path"])
        path = self.inside(raw if raw.is_absolute() else directory / raw, directory)
        if digest(path) != item["sha256"]:
            raise ValueError("Artifact hash mismatch: " + role)
        return path

    def node_intent(self, node):
        intent = read(self.control / (node + "-intent.json"))
        if (intent.get("identity") != self.identity or intent.get("node") != node
                or intent.get("campaign") != str(self.root) or intent.get("release") != str(self.release)):
            raise ValueError("Node intent binding mismatch")
        return intent

    def result(self, node):
        directory = self.root / node
        result = read(directory / "result.json")
        if any(result.get(key) != value for key, value in self.identity.items()):
            raise ValueError("Stage result source/manifest/plan mismatch")
        if result.get("stage") != stage_for(node):
            raise ValueError("Stage result identity mismatch")
        expected = ("completed_segment", "completed_case") if stage_for(node) == "segment" else ("passed",)
        if result.get("status") not in expected:
            raise ValueError("Predecessor did not pass: " + node)
        identities = result.get("frozen_identities")
        if not isinstance(identities, dict) or not identities or any(
                not isinstance(key, str) or not isinstance(value, str) or not SHA.fullmatch(value)
                for key, value in identities.items()):
            raise ValueError("Missing frozen identity hashes")
        if node != "cpu" and identities != self.result("cpu")["frozen_identities"]:
            raise ValueError("Frozen model/context identities changed")
        for role in self.artifacts(result):
            self.artifact(result, role, directory)
        if node == "cpu":
            count = junit_passed(self.artifact(result, "junit", directory))
            if result.get("tests_passed") != count:
                raise ValueError("CPU result/JUnit count mismatch")
        elif node == "fit":
            self.artifact(result, "checkpoint", directory)
            self.artifact(result, "calibration", directory)
            if result.get("fit_completed") is not True or result.get("calibration_passed") is not True:
                raise ValueError("Fit/calibration incomplete")
            self.model_identity(result)
        elif node == "pilot":
            fit = self.result("fit")
            if result.get("model_identity") != self.model_identity(fit):
                raise ValueError("Pilot fitted-model binding mismatch")
            cases = result.get("cases", [])
            if [case.get("case") for case in cases] != list(CASES):
                raise ValueError("Pilot must retain all three fixed cases in order")
            for case in cases:
                if (case.get("gates_passed") is not True or case.get("strict_valid_best") is not True
                        or integer(case.get("accepted_nonzero_steps"), 1) < 1):
                    raise ValueError("Pilot case gate/validity/progress failed")
                integer(case.get("completed_updates"), 1, ENDPOINT - 1)
                self.artifact(result, case["case"] + "_state", directory)
                self.profile(case.get("profile"))
        elif stage_for(node) == "segment":
            self.validate_segment(node, result)
        elif node == "followup" and result.get("all_requests_accounted") is not True:
            raise ValueError("Native followup lacks complete request accounting")
        return result

    def model_identity(self, fit):
        expected = {name + "_sha256": self.asset(fit, name)["sha256"] for name in ("checkpoint", "calibration")}
        if fit.get("model_identity") != expected:
            raise ValueError("Fitted checkpoint/calibration identity mismatch")
        return expected

    def profile(self, profile):
        if not isinstance(profile, dict):
            raise ValueError("Missing finite runtime/storage pilot profile")
        return {key: finite(profile.get(key), positive=key == "seconds_per_update") for key in
                ("seconds_per_update", "startup_seconds", "bytes_per_update", "fixed_bytes")}

    def validate_segment(self, node, result):
        intent = self.node_intent(node)
        request = intent["request"]
        start = integer(result.get("start_update"), 0, ENDPOINT - 1)
        end = integer(result.get("completed_updates"), start + 1, ENDPOINT)
        if (result.get("case") != request["case"] or start != request["start_update"]
                or end > start + request["max_updates"]
                or result.get("resume_state_sha256") != request["resume_state_sha256"]
                or result.get("model_identity") != self.model_identity(self.result("fit"))
                or result.get("guards_passed") is not True or result.get("guard_stall") is not False
                or result.get("observation_every") != OBSERVE_EVERY):
            raise ValueError("Segment resume/progress/model/guard binding mismatch")
        if result["status"] != ("completed_case" if end == ENDPOINT else "completed_segment"):
            raise ValueError("Segment completion status does not match fixed endpoint")
        self.artifact(result, "state", self.root / node)

    def require_done(self, node, *, check_scheduler=True):
        receipt = read(self.control / (node + "-receipt.json"))
        intent = self.node_intent(node)
        job = receipt.get("job_id", "")
        if (receipt.get("status") != "submitted_not_passed" or not re.fullmatch(r"[0-9]+", job)
                or receipt.get("intent_sha256") != digest(self.control / (node + "-intent.json"))):
            raise ValueError("Predecessor scheduler receipt is not authenticated")
        if check_scheduler:
            state = subprocess.run(["bjobs", "-a", "-noheader", "-o", "jobid stat job_name:200", job],
                                   text=True, capture_output=True, timeout=30)
            fields = state.stdout.split()
            if fields:
                if fields != [job, "DONE", intent["label"]] or state.returncode:
                    raise ValueError("Predecessor allocation is not exact-name DONE")
            elif not state.returncode or "not found" not in state.stderr.lower():
                raise ValueError("Predecessor scheduler status unavailable")
        execution = read(self.control / (node + "-execution.json"))
        if (execution.get("identity") != self.identity or execution.get("node") != node
                or execution.get("job_id") != job or execution.get("returncode") != 0
                or execution.get("status") != "completed"
                or execution.get("result_sha256") != digest(self.root / node / "result.json")):
            raise ValueError("Missing exact successful allocated execution evidence")

    def arguments(self, action, **values):
        command = ["python", SCRIPT, action, "--release", str(self.release),
                   "--campaign", str(self.root), "--archive-sha256", self.archive_sha256,
                   "--manifest-sha256", self.identity["manifest_sha256"], "--plan-sha256", self.plan_sha256]
        for key, value in values.items():
            command.extend(["--" + key.replace("_", "-"), str(value)])
        return command

    def dispatch(self, node, request=None):
        self.prepare()
        self.check_source()
        request = request or {}
        stage = stage_for(node)
        for suffix in ("intent", "receipt", "callback-intent", "callback-receipt"):
            if (self.control / (node + "-" + suffix + ".json")).exists():
                raise FileExistsError("Submission already attempted; no ambiguous retry")
        fresh = quota()
        if stage == "segment":
            projected = finite(request["projected_bytes"])
            if fresh["minimum_free_gib"] - projected / (1024 ** 3) < 5:
                raise RuntimeError("Projected segment storage lacks 5 GiB quota headroom")
        label = self.root.name + "-" + node
        bsub = ["bsub", "-J", label, "-q", "hpc" if stage == "cpu" else "gpua100",
                "-n", "4", "-R", "span[hosts=1]", "-R", "rusage[mem=8GB]" if stage == "cpu" else "rusage[mem=16GB]",
                "-W", "0:45" if stage == "cpu" else ("24:00" if stage in ("segment", "followup") else "4:00"),
                "-oo", str(ROOT / "logs" / (label + ".%J.out")),
                "-eo", str(ROOT / "logs" / (label + ".%J.err"))]
        if stage != "cpu":
            bsub += ["-R", "select[gpu80gb]", "-gpu", "num=1:mode=exclusive_process"]
        command = self.arguments("run-stage", node=node)
        intent = dict(identity=self.identity, release=str(self.release), campaign=str(self.root),
                      node=node, label=label, request=request, arguments=bsub,
                      command=command, fresh_quota=fresh,
                      dependencies=self.dependency_digests(stage))
        path = self.control / (node + "-intent.json")
        write_new(path, intent)
        receipt = self.submit(bsub, command, path, self.control / (node + "-receipt.json"))
        # A separate intent ensures a failed/uncertain callback submission cannot
        # silently cause a second worker submission or a duplicated continuation.
        self.check_source()
        callback_quota = quota()
        callback_label = label + "-after"
        callback = ["bsub", "-J", callback_label, "-q", "hpc", "-n", "1", "-W", "0:30",
                    "-R", "rusage[mem=2GB]", "-R", "span[hosts=1]", "-w", "ended(" + receipt["job_id"] + ")",
                    "-oo", str(ROOT / "logs" / (callback_label + ".%J.out")),
                    "-eo", str(ROOT / "logs" / (callback_label + ".%J.err"))]
        callback_command = self.arguments("advance", after=node)
        callback_path = self.control / (node + "-callback-intent.json")
        write_new(callback_path, dict(identity=self.identity, predecessor_job_id=receipt["job_id"],
            arguments=callback, command=callback_command, fresh_quota=callback_quota))
        after = self.submit(callback, callback_command, callback_path,
                            self.control / (node + "-callback-receipt.json"))
        return dict(worker=receipt, callback=after)

    def submit(self, arguments, command, intent_path, receipt_path):
        wrapped = ["env", "PYTHONDONTWRITEBYTECODE=1", "LATENTMOL_SOURCE_ARCHIVE=" + str(self.release) + ".tar.gz",
                   "bash", "scripts/hpc/boltz_run.sh", *command]
        script = "#!/bin/bash\nset -euo pipefail\ncd " + shlex.quote(str(self.release)) + "\nexec " + shlex.join(wrapped) + "\n"
        try:
            process = subprocess.run(arguments, input=script, text=True, capture_output=True, timeout=60)
        except (subprocess.TimeoutExpired, OSError) as error:
            write_new(receipt_path, dict(status="ambiguous_submission_no_retry", error=str(error),
                intent_sha256=digest(intent_path), job_id=None))
            raise RuntimeError("Submission outcome unknown; exclusive intent forbids retry") from error
        match = re.search(r"Job <([0-9]+)>", process.stdout)
        passed = process.returncode == 0 and match is not None
        receipt = dict(status="submitted_not_passed" if passed else "submission_failed_no_retry",
            returncode=process.returncode, stdout=process.stdout[-16000:], stderr=process.stderr[-16000:],
            job_id=match.group(1) if match else None, intent_sha256=digest(intent_path))
        write_new(receipt_path, receipt)
        if not passed:
            raise RuntimeError("LSF did not confirm submission; no automatic retry")
        return receipt

    def progress(self):
        pilot = self.result("pilot")
        states = {}
        for case in pilot["cases"]:
            name = case["case"]
            states[name] = dict(case=name, completed_updates=case["completed_updates"],
                state=str(self.artifact(pilot, name + "_state", self.root / "pilot")),
                state_sha256=self.asset(pilot, name + "_state")["sha256"],
                profile=self.profile(case["profile"]), segment_count=0)
        nodes = sorted({path.name.removesuffix("-receipt.json") for path in self.control.glob("case-*-segment-????-receipt.json")})
        for node in nodes:
            # The advancing predecessor gets its fresh exact-name DONE check
            # in advance(). Historical nodes use their exact allocated terminal
            # evidence, avoiding thousands of aged scheduler queries per turn.
            self.require_done(node, check_scheduler=False)
            result = self.result(node)
            state = states[result["case"]]
            request = self.node_intent(node)["request"]
            if (request["start_update"] != state["completed_updates"]
                    or request["resume_state_sha256"] != state["state_sha256"]):
                raise ValueError("Noncontiguous state history or budget reset")
            state.update(completed_updates=result["completed_updates"],
                state=str(self.artifact(result, "state", self.root / node)),
                state_sha256=self.asset(result, "state")["sha256"], segment_count=state["segment_count"] + 1)
        return states

    def next_segment(self, after):
        states = self.progress()
        if all(value["completed_updates"] == ENDPOINT for value in states.values()):
            return "followup", {}
        previous = SEGMENT.fullmatch(after)
        offset = CASES.index(previous.group(1)) + 1 if previous else 0
        case = next(CASES[(offset + index) % len(CASES)] for index in range(len(CASES))
                    if states[CASES[(offset + index) % len(CASES)]]["completed_updates"] < ENDPOINT)
        state = states[case]
        if state["segment_count"] >= self.settings["max_segments_per_case"]:
            raise RuntimeError("Guarded segment-count cap reached")
        profile = state["profile"]
        budget = INTERNAL_SECONDS - profile["startup_seconds"] - self.settings["segment_reserve_seconds"]
        maximum = min(ENDPOINT - state["completed_updates"], math.floor(budget / (self.settings["pilot_safety_factor"] * profile["seconds_per_update"])))
        if maximum < 1:
            raise RuntimeError("Conservative pilot profile cannot fit one update")
        request = dict(case=case, start_update=state["completed_updates"], max_updates=maximum,
            resume=state["state"], resume_state_sha256=state["state_sha256"],
            model_identity=self.model_identity(self.result("fit")), observation_every=OBSERVE_EVERY,
            endpoint=ENDPOINT, internal_seconds=INTERNAL_SECONDS,
            projected_bytes=profile["fixed_bytes"] + profile["bytes_per_update"] * maximum)
        return case + "-segment-" + f"{state['segment_count'] + 1:04d}", request

    def advance(self, after):
        stage = stage_for(after)
        self.prepare()
        self.check_source()
        transition = self.control / ("advance-after-" + after + ".json")
        if transition.exists():
            raise FileExistsError("Transition already attempted; inspect immutable evidence")
        self.require_done(after)
        result = self.result(after)
        if stage == "followup":
            value = dict(status="completed_bounded_campaign", identity=self.identity,
                         result_sha256=digest(self.root / after / "result.json"))
            write_new(transition, value)
            return value
        if stage == "cpu":
            node, request = "fit", {}
        elif stage == "fit":
            node, request = "pilot", {}
        else:
            node, request = self.next_segment(after)
        write_new(transition, dict(status="transition_attempted_not_passed", identity=self.identity,
            predecessor=after, predecessor_result_sha256=digest(self.root / after / "result.json"), next_node=node))
        return self.dispatch(node, request)

    def worker_command(self, node, intent):
        stage = stage_for(node)
        command = [sys.executable, "-m", MODULE, "--stage", stage,
                   "--config", str(self.release / CONFIG), "--plan-sha256", self.plan_sha256,
                   "--source-sha256", self.archive_sha256, "--manifest-sha256", self.identity["manifest_sha256"],
                   "--output", str(self.root / node), "--campaign", str(self.root),
                   "--seconds", str(40 * 60 if stage == "cpu" else (INTERNAL_SECONDS if stage in ("segment", "followup") else 3 * 3600 + 50 * 60))]
        if stage in ("pilot", "segment", "followup"):
            fit = self.result("fit")
            for role in ("checkpoint", "calibration"):
                command.extend(["--" + ("bridge" if role == "checkpoint" else role),
                    str(self.artifact(fit, role, self.root / "fit")),
                    "--" + ("bridge" if role == "checkpoint" else role) + "-sha256",
                    self.asset(fit, role)["sha256"]])
        if stage == "segment":
            request = intent["request"]
            if digest(self.inside(request["resume"], self.root)) != request["resume_state_sha256"]:
                raise ValueError("Saved resume state changed before allocated launch")
            for flag, key in (("case", "case"), ("resume-state", "resume"), ("segment-updates", "max_updates")):
                command.extend(["--" + flag, str(request[key])])
            command.extend(["--resume-sha256", request["resume_state_sha256"]])
        if stage == "followup":
            # Final worker reads only complete, authenticated per-case endpoints.
            states = self.progress()
            if any(state["completed_updates"] != ENDPOINT for state in states.values()):
                raise ValueError("Final native followup before all fixed endpoints")
            state_path = self.control / "final-states.json"
            if not state_path.exists():
                write_new(state_path, dict(identity=self.identity, cases=states))
            elif read(state_path) != dict(identity=self.identity, cases=states):
                raise ValueError("Final state manifest changed")
            command.extend(["--states", str(state_path), "--states-sha256", digest(state_path)])
        return command

    def dependency_digests(self, stage):
        predecessors = {"cpu": (), "fit": ("cpu",), "pilot": ("cpu", "fit"),
                        "segment": ("cpu", "fit", "pilot"), "followup": ("cpu", "fit", "pilot")}[stage]
        result = {}
        for predecessor in predecessors:
            self.result(predecessor)
            result[predecessor] = digest(self.root / predecessor / "result.json")
        return result

    def run_stage(self, node):
        self.check_source()
        job = os.environ.get("LSB_JOBID", "")
        if not re.fullmatch(r"[0-9]+", job):
            raise RuntimeError("Worker execution requires an allocated LSF job")
        intent = self.node_intent(node)
        receipt = read(self.control / (node + "-receipt.json"))
        if (receipt.get("job_id") != job or receipt.get("intent_sha256") != digest(self.control / (node + "-intent.json"))):
            raise ValueError("Allocation does not own this exact submission")
        execution_path = self.control / (node + "-execution.json")
        if execution_path.exists() or (self.root / node).exists():
            raise FileExistsError("Allocated stage already attempted; no state overwrite")
        if intent.get("dependencies") != self.dependency_digests(stage_for(node)):
            raise ValueError("Predecessor result changed after authenticated dispatch")
        launch_quota = quota()
        if stage_for(node) == "segment" and launch_quota["minimum_free_gib"] - intent["request"]["projected_bytes"] / 1024 ** 3 < 5:
            raise RuntimeError("Launch projected storage lacks 5 GiB headroom")
        write_new(self.control / (node + "-launch-quota.json"), launch_quota)
        command = self.worker_command(node, intent)
        output = self.root / node
        output.mkdir()
        started = time.time()
        timeout = 42 * 60 if stage_for(node) == "cpu" else (INTERNAL_SECONDS + 300 if stage_for(node) in ("segment", "followup") else 3 * 3600 + 55 * 60)
        returncode, error = -1, None
        # CPU parity's explicit kernel policy is installed before torch import;
        # GPU workers retain their existing native environment.
        environment = worker_environment(stage_for(node))
        try:
            with (output / "worker.stdout.log").open("wb") as stdout, (output / "worker.stderr.log").open("wb") as stderr:
                child = subprocess.Popen(command, cwd=self.release, env=environment,
                    stdout=stdout, stderr=stderr, start_new_session=True)
                try:
                    returncode = child.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=30)
                    raise
            if returncode == 0:
                self.result(node)
        except Exception as exception:
            error = str(exception)
        result_path = output / "result.json"
        execution = dict(identity=self.identity, node=node, job_id=job, returncode=returncode,
            status="completed" if returncode == 0 and error is None else "failed_not_passed",
            started_unix=started, finished_unix=time.time(), error=error,
            result_sha256=digest(result_path) if result_path.is_file() else None)
        write_new(execution_path, execution)
        if execution["status"] != "completed":
            raise RuntimeError("Allocated stage failed; preserved evidence: " + str(execution_path))
        return execution

    def collect(self):
        self.check_source()
        result = dict(identity=self.identity, campaign=str(self.root), files={}, scheduler={})
        if not self.control.exists():
            return result
        paths = sorted(self.control.glob("*.json"), key=lambda path: (path.stat().st_mtime_ns, path.name))
        if len(paths) > 25000:
            raise ValueError("Control inventory exceeds bounded campaign size")
        result["control_inventory"] = {path.name: dict(bytes=path.stat().st_size, sha256=digest(path)) for path in paths}
        # Fetch recent state only. Old per-node evidence remains directly
        # addressable by its immutable inventory hash, avoiding thousands of
        # serial scheduler probes or unbounded trajectory/log downloads.
        for path in paths[-32:]:
            value = read(path)
            result["files"][path.name] = value
        receipts = [path for path in paths if path.name.endswith("-receipt.json")][-8:]
        jobs = [read(path).get("job_id") for path in receipts]
        jobs = [job for job in jobs if isinstance(job, str) and re.fullmatch(r"[0-9]+", job)]
        if jobs:
            state = subprocess.run(["bjobs", "-a", "-l", *jobs], text=True, capture_output=True, timeout=45)
            result["scheduler"] = dict(job_ids=jobs, returncode=state.returncode,
                stdout=state.stdout[-64000:], stderr=state.stderr[-16000:])
        # Do not traverse arbitrary scientific tensors or unbounded trajectory logs.
        stage_paths = sorted(self.root.glob("*/result.json"), key=lambda path: (path.stat().st_mtime_ns, path.parent.name))
        result["stage_result_count"] = len(stage_paths)
        result["stage_results"] = {path.parent.name: read(path) for path in stage_paths[-8:]}
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("submit-initial", "advance", "collect", "run-stage"))
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    for name in ("archive-sha256", "manifest-sha256", "plan-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--after")
    parser.add_argument("--node")
    args = parser.parse_args(argv)
    campaign = Campaign(args.release, args.campaign, args.archive_sha256, args.manifest_sha256, args.plan_sha256)
    if args.action == "submit-initial":
        result = campaign.dispatch("cpu")
    elif args.action == "advance":
        if not args.after:
            parser.error("advance requires the exact --after node")
        result = campaign.advance(args.after)
    elif args.action == "run-stage":
        if not args.node:
            parser.error("run-stage requires the exact --node")
        result = campaign.run_stage(args.node)
    else:
        result = campaign.collect()
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
