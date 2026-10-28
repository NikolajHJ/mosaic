"""Control-only recovery around an authenticated, unchanged source142 worker.

The confirmed DTU bjobs JSON aged-out response is an exact per-job ERROR with
returncode zero. It is accepted only with independent successful LSF terminal
output and the original allocated execution/result proof. No fit is rerun.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import time
import xml.etree.ElementTree as ET


ROOT = Path("/work3/s234818/latentmol")
SCRIPT = "scripts/hpc/embedding_handoff_recovery.py"
CONFIG = "configs/embedding_handoff_recovery.json"
PARENT_CONFIG = "configs/epoch36_embedding_long.json"
PARENT_SCRIPT = "scripts/hpc/embedding_long_campaign.py"
VERSION = "embedding-handoff-recovery-v1"
POLICY = "lsf-json-exact-error-and-terminal-v1"
READINESS_VERSION = "embedding-handoff-readiness-v2"
SHA = re.compile(r"[0-9a-f]{64}")
MAX_JSON = 8 * 1024 * 1024


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    path = Path(path)
    if path.stat().st_size > MAX_JSON:
        raise ValueError("Recovery metadata exceeds bounded size")
    return json.loads(path.read_text(), parse_constant=lambda value:
                      (_ for _ in ()).throw(ValueError("Nonfinite JSON: " + value)))


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def sha_value(value):
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise ValueError("Expected a complete lowercase SHA256")
    return value


def verified_asset(item, *, below=None):
    if not isinstance(item, dict) or not isinstance(item.get("path"), str):
        raise ValueError("Malformed authenticated recovery asset")
    sha_value(item.get("sha256"))
    path = Path(item["path"]).resolve()
    if below is not None and (not path.is_relative_to(Path(below).resolve()) or path == Path(below).resolve()):
        raise ValueError("Recovery artifact escapes its allowlisted directory")
    if not path.is_file() or digest(path) != item["sha256"]:
        raise ValueError("Recovery asset hash mismatch: " + str(path))
    return path


def authenticate_release(release, source, manifest_sha, plan, config_relative):
    """Self-contained byte authentication before importing either controller."""
    release = Path(release).resolve()
    for value in (source, manifest_sha, plan):
        sha_value(value)
    if release.parent != ROOT.resolve() / "releases":
        raise ValueError("Recovery releases must stay in the exact HPC releases root")
    archive = Path(str(release) + ".tar.gz")
    if digest(archive) != source:
        raise ValueError("Release archive hash mismatch")
    manifest_path = release / "embedding-release-manifest.json"
    raw = manifest_path.read_bytes()
    if digest(manifest_path) != manifest_sha:
        raise ValueError("Release manifest hash mismatch")
    with tarfile.open(archive) as stream:
        sealed = stream.extractfile("embedding-release-manifest.json")
        if sealed is None or sealed.read() != raw:
            raise ValueError("Extracted manifest differs from authenticated archive")
    manifest = json.loads(raw)
    expected = set(manifest["members"]) | {"embedding-release-manifest.json"}
    actual = {path.relative_to(release).as_posix() for path in release.rglob("*")
              if path.is_file() and not {"__pycache__", ".pytest_cache"}.intersection(path.parts)}
    if actual != expected:
        raise ValueError("Release file set differs from authenticated manifest")
    for name, value in manifest["members"].items():
        verified_asset(dict(path=str(release / name), sha256=value), below=release)
    if digest(release / config_relative) != plan:
        raise ValueError("Recovery plan hash mismatch")
    return manifest


def original_identity(config):
    parent = config["parent"]
    return {key: parent[key] for key in ("source_sha256", "manifest_sha256", "plan_sha256")}


def authenticate_parent(config):
    if config.get("version") != VERSION:
        raise ValueError("Unknown handoff recovery protocol")
    parent = config["parent"]
    release, campaign = Path(parent["release"]).resolve(), Path(parent["campaign"]).resolve()
    attempt = re.fullmatch(r"source142-epoch36-guarded-embedding-long-attempt([0-9]{3})", campaign.name)
    if (not attempt or int(attempt.group(1)) < 1 or campaign.parent != ROOT.resolve() / "runs"
            or release.name != campaign.name.replace("source142", "source-142", 1)):
        raise ValueError("Recovery parent is outside the original source142 namespace")
    manifest = authenticate_release(release, parent["source_sha256"], parent["manifest_sha256"],
                                    parent["plan_sha256"], PARENT_CONFIG)
    controller_sha = sha_value(parent["controller_sha256"])
    if manifest["members"].get(PARENT_SCRIPT) != controller_sha or digest(release / PARENT_SCRIPT) != controller_sha:
        raise ValueError("Original controller hash mismatch")
    required = {"cpu_result", "cpu_execution", "fit_result", "fit_execution", "checkpoint", "calibration"}
    if not required <= set(parent["artifacts"]):
        raise ValueError("Incomplete saved parent CPU/fit/model proof")
    for role in required:
        base = ROOT / "submissions" / campaign.name if role.endswith("execution") else campaign
        verified_asset(parent["artifacts"][role], below=base)
    result = read(verified_asset(parent["artifacts"]["fit_result"], below=campaign))
    execution = read(verified_asset(parent["artifacts"]["fit_execution"], below=ROOT / "submissions" / campaign.name))
    if (result.get("status") != "passed" or result.get("stage") != "fit"
            or any(result.get(key) != value for key, value in original_identity(config).items())
            or execution.get("identity") != original_identity(config) or execution.get("node") != "fit"
            or execution.get("status") != "completed" or execution.get("returncode") != 0
            or execution.get("job_id") != parent.get("fit_job_id")
            or execution.get("result_sha256") != parent["artifacts"]["fit_result"]["sha256"]):
        raise ValueError("Saved fit is not an exact completed original-source pass")
    for role, fit_role in (("checkpoint", "checkpoint"), ("calibration", "calibration")):
        expected = parent["artifacts"][role]
        actual = result["artifacts"][fit_role]
        if Path(actual["path"]).resolve() != Path(expected["path"]).resolve() or actual["sha256"] != expected["sha256"]:
            raise ValueError("Recovery changed the saved fit checkpoint/calibration")
    return release, campaign


def parse_scheduler(response, job, label):
    """Parse only the observed LSF JSON contract, never infer missing from empty."""
    if response.get("returncode") != 0 or response.get("stderr", "").strip():
        raise ValueError("Scheduler command failed or returned unexpected stderr")
    try:
        value = json.loads(response["stdout"])
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError("Scheduler did not return structured LSF JSON") from error
    if (not isinstance(value, dict) or value.get("COMMAND") != "bjobs" or type(value.get("JOBS")) is not int or value.get("JOBS") != 1
            or not isinstance(value.get("RECORDS"), list) or len(value["RECORDS"]) != 1):
        raise ValueError("Scheduler response does not contain exactly the requested job")
    record = value["RECORDS"][0]
    if not isinstance(record, dict) or record.get("JOBID") != job:
        raise ValueError("Scheduler returned a different job ID")
    if "ERROR" in record:
        if set(record) != {"JOBID", "ERROR"} or record["ERROR"] != "Job <" + job + "> is not found":
            raise ValueError("Unknown scheduler error is not an aged-out job")
        return "aged_out"
    if record.get("STAT") != "DONE" or record.get("JOB_NAME") != label:
        raise ValueError("Predecessor allocation is not exact-job/exact-name DONE")
    return "DONE"


def terminal_passed(text, job, label, release):
    """Independent successful outer-LSF summary; execution alone is insufficient."""
    subjects = re.findall(r"^Subject: Job ([0-9]+): <([^>]+)> in cluster <[^>]+> ([^\r\n]+)$", text, re.M)
    if subjects != [(job, label, "Done")]:
        raise ValueError("Terminal LSF job/name/DONE subject mismatch")
    if (not re.search(r"^JOB=" + re.escape(job) + r" HOST=\S+ SOURCE=" + re.escape(str(release)) + r"$", text, re.M)
            or not re.search(r"^Successfully completed\.$", text, re.M)
            or not re.search(r"^Terminated at .+$", text, re.M)
            or not re.search(r"^Results reported at .+$", text, re.M)
            or re.search(r"^Exited with exit code|^TERM_", text, re.M)):
        raise ValueError("Terminal LSF output does not establish successful outer completion")


def _load_parent_module(release, controller_sha):
    """Bind even the helper dependency to original142, independent of sys.path."""
    dependency_path = release / "scripts/hpc/embedding_operator.py"
    spec = importlib.util.spec_from_file_location("_handoff_original_embedding_operator", dependency_path)
    dependency = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dependency)
    old = sys.modules.get("embedding_operator")
    sys.modules["embedding_operator"] = dependency
    try:
        spec = importlib.util.spec_from_file_location("_handoff_original_campaign_" + controller_sha[:12], release / PARENT_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if old is None:
            sys.modules.pop("embedding_operator", None)
        else:
            sys.modules["embedding_operator"] = old
    module.ROOT = ROOT
    return module


def make_parent_campaign(config):
    """Authenticate and expose the exact pilot argv for allocated readiness only."""
    release, campaign = authenticate_parent(config)
    module = _load_parent_module(release, config["parent"]["controller_sha256"])

    class RecoveryCampaign(module.Campaign):
        repair = None

        def check_source(self):
            super().check_source()
            if self.repair is not None:
                self.repair.verify()

        def ensure_ready(self):
            if self.repair is None:
                raise RuntimeError("Parent-only builder cannot mutate the scientific campaign")
            self.repair.ensure_ready()

        def dispatch(self, node, request=None):
            self.ensure_ready()
            if module.stage_for(node) in ("cpu", "fit"):
                raise ValueError("Recovery forbids CPU/fit dispatch and refitting")
            return super().dispatch(node, request)

        def advance(self, after):
            self.ensure_ready()
            if after == "cpu":
                raise ValueError("Recovery may not dispatch a replacement fit")
            return super().advance(after)

        def run_stage(self, node):
            self.ensure_ready()
            if module.stage_for(node) in ("cpu", "fit"):
                raise ValueError("Recovery forbids CPU/fit worker execution")
            return super().run_stage(node)

        def worker_command(self, node, intent):
            if self.repair is None and node != "pilot":
                raise RuntimeError("Parent-only metadata builder exposes only the exact pilot argv")
            return super().worker_command(node, intent)

        def arguments(self, action, **values):
            self.ensure_ready()
            command = ["python", str(self.repair.release / SCRIPT), action,
                "--release", str(self.release), "--campaign", str(self.root),
                "--archive-sha256", self.archive_sha256,
                "--manifest-sha256", self.identity["manifest_sha256"], "--plan-sha256", self.plan_sha256,
                *self.repair.arguments()]
            for key, value in values.items():
                command.extend(["--" + key.replace("_", "-"), str(value)])
            return command

        def require_done(self, node, *, check_scheduler=True):
            # Reuse every original saved allocated job/result check. Inherited
            # historical progress uses this path without new scheduler queries.
            super().require_done(node, check_scheduler=False)
            if not check_scheduler:
                return
            self.ensure_ready()
            intent = self.node_intent(node)
            receipt = read(self.control / (node + "-receipt.json"))
            job = receipt["job_id"]
            command = ["bjobs", "-a", "-json", "-o", "jobid stat job_name", job]
            evidence = dict(version=VERSION, identity=self.identity, node=node,
                repair_identity=self.repair.identity, command=command, job_id=job, label=intent["label"])
            evidence_path = self.repair.control / "scheduler" / (node + "-" + str(time.time_ns()) + ".json")
            try:
                run = subprocess.run(command, text=True, capture_output=True, timeout=30)
                evidence.update(returncode=run.returncode, stdout=run.stdout, stderr=run.stderr)
            except (subprocess.TimeoutExpired, OSError) as error:
                evidence.update(error_type=type(error).__name__, error=str(error))
                write_new(evidence_path, evidence)
                raise RuntimeError("Scheduler unavailable; no handoff") from error
            # Save exact raw responses before interpreting any status/error.
            write_new(evidence_path, evidence)
            state = parse_scheduler(evidence, job, intent["label"])
            if state == "aged_out":
                terminal = ROOT / "logs" / (intent["label"] + "." + job + ".out")
                if node == "fit":
                    pinned = self.repair.config["parent"]["artifacts"].get("fit_terminal")
                    if pinned is None or verified_asset(pinned, below=ROOT / "logs") != terminal.resolve():
                        raise ValueError("Original fit lacks its pinned successful outer LSF log")
                if not terminal.is_file() or terminal.stat().st_size > MAX_JSON:
                    raise ValueError("Missing bounded successful terminal LSF log")
                raw = terminal.read_text(errors="replace")
                proof = dict(scheduler_evidence=str(evidence_path), path=str(terminal),
                    sha256=digest(terminal), bytes=terminal.stat().st_size, text=raw)
                write_new(evidence_path.with_name(evidence_path.stem + "-terminal.json"), proof)
                terminal_passed(raw, job, intent["label"], self.release)

    parent = config["parent"]
    value = RecoveryCampaign(release, campaign, parent["source_sha256"], parent["manifest_sha256"], parent["plan_sha256"])
    # Validate every original completed stage/artifact/frozen identity contract
    # before exposing even a metadata command. No model/torch import occurs.
    value.result("cpu")
    value.result("fit")
    return value


class RepairContext:
    def __init__(self, args):
        self.args = args
        self.release = Path(args.repair_release).resolve()
        match = re.fullmatch(r"source-143-embedding-handoff-recovery-attempt([0-9]{3})", self.release.name)
        if not match or int(match.group(1)) < 1:
            raise ValueError("Repair must use the independent immutable source143 namespace")
        self.identity = dict(source_sha256=args.repair_archive_sha256,
            manifest_sha256=args.repair_manifest_sha256, plan_sha256=args.repair_plan_sha256)
        self.control = ROOT / "submissions" / self.release.name.replace("source-143", "source143", 1)
        if self.control.resolve().parent != ROOT.resolve() / "submissions":
            raise ValueError("Repair controls escape the HPC submissions root")
        self.verify()

    def verify(self):
        manifest = authenticate_release(self.release, self.identity["source_sha256"],
            self.identity["manifest_sha256"], self.identity["plan_sha256"], CONFIG)
        if (Path(__file__).resolve() != (self.release / SCRIPT).resolve()
                or manifest["members"].get(SCRIPT) != digest(Path(__file__))):
            raise ValueError("Executing recovery helper is outside its authenticated release")
        self.config = read(self.release / CONFIG)
        authenticate_parent(self.config)
        parent = self.config["parent"]
        supplied = dict(source_sha256=self.args.archive_sha256,
            manifest_sha256=self.args.manifest_sha256, plan_sha256=self.args.plan_sha256)
        if (supplied != original_identity(self.config)
                or Path(self.args.release).resolve() != Path(parent["release"]).resolve()
                or Path(self.args.campaign).resolve() != Path(parent["campaign"]).resolve()):
            raise ValueError("Recovery command differs from its pinned scientific parent")

    def ensure_ready(self):
        self.verify()
        if self.config.get("scheduler_policy") != POLICY:
            raise ValueError("Scheduler correction has not been selected from captured evidence")
        proof_path = verified_asset(dict(path=str(self.args.readiness), sha256=self.args.readiness_sha256),
                                    below=ROOT / "runs" / self.control.name)
        proof = read(proof_path)
        if (proof.get("version") != READINESS_VERSION or proof.get("status") != "passed"
                or any(proof.get(key) != value for key, value in self.identity.items())
                or proof.get("original_identity") != original_identity(self.config)
                or proof.get("original_controller_sha256") != self.config["parent"]["controller_sha256"]
                or not re.fullmatch(r"[0-9]+", str(proof.get("job_id", "")))):
            raise ValueError("Recovery lacks its own exact allocated CPU readiness")
        junit = proof.get("junit", {})
        path = verified_asset(junit, below=proof_path.parent)
        cases = list(ET.parse(path).getroot().iter("testcase"))
        if (not cases or junit.get("tests") != len(cases)
                or any(junit.get(key) != 0 for key in ("failures", "errors", "skipped"))
                or any(list(case.iter(tag)) for case in cases for tag in ("failure", "error", "skipped"))):
            raise ValueError("Recovery CPU JUnit is not a complete zero-skip pass")
        for suite in ET.parse(path).getroot().iter("testsuite"):
            if any(int(suite.get(key, "0")) != 0 for key in ("failures", "errors", "skipped")):
                raise ValueError("Recovery CPU JUnit aggregate failed")
        cli_path = verified_asset(proof.get("actual_pilot_cli_asset"), below=proof_path.parent)
        cli = read(cli_path)
        expected_command = make_parent_campaign(self.config).worker_command("pilot", {})
        manifest = read(Path(self.config["parent"]["release"]) / "embedding-release-manifest.json")
        worker_sha = manifest["members"].get("src/latentmol/boltz/embedding_long_experiment.py")
        imports = cli.get("cpu_module_imports")
        if (cli != proof.get("actual_pilot_cli") or cli.get("status") != "passed"
                or cli.get("argv") != expected_command or cli.get("actual_worker_sha256") != worker_sha
                or not isinstance(worker_sha, str) or not SHA.fullmatch(worker_sha)
                or cli.get("reached") != "native_smoke" or cli.get("stop") != "before_cuda_or_model_setup"
                or cli.get("cuda_visible_devices") != ""
                or cli.get("cuda_initialized_before") is not False or cli.get("cuda_initialized_after") is not False
                or not isinstance(imports, list) or any(not isinstance(name, str) for name in imports)
                or imports != sorted(set(imports)) or "torch" not in imports
                or not set(imports) <= {"torch", "numpy"}
                or any(type(cli.get(key)) is not int or cli[key] != 0 for key in
                       ("parent_writes", "job_submissions", "cuda_initializations", "native_model_loads", "updates"))):
            raise ValueError("Recovery lacks its exact saved pre-model pilot dispatch proof")
        # The independently pinned successful outer fit summary is mandatory
        # for this recovery, even if its live record happens to remain visible.
        parent = self.config["parent"]
        terminal = verified_asset(parent["artifacts"].get("fit_terminal"), below=ROOT / "logs")
        execution = read(verified_asset(parent["artifacts"]["fit_execution"]))
        label = Path(parent["campaign"]).name + "-fit"
        if terminal != (ROOT / "logs" / (label + "." + execution["job_id"] + ".out")).resolve():
            raise ValueError("Fit terminal path belongs to a different job")
        terminal_passed(terminal.read_text(errors="replace"), execution["job_id"], label, Path(parent["release"]))

    def arguments(self):
        values = dict(repair_release=self.release,
            repair_archive_sha256=self.identity["source_sha256"],
            repair_manifest_sha256=self.identity["manifest_sha256"],
            repair_plan_sha256=self.identity["plan_sha256"],
            readiness=self.args.readiness, readiness_sha256=self.args.readiness_sha256)
        return [part for key, value in values.items() for part in ("--" + key.replace("_", "-"), str(value))]


def build_campaign(args, *, require_readiness=True):
    repair = RepairContext(args)
    value = make_parent_campaign(repair.config)
    value.repair = repair
    if require_readiness:
        value.ensure_ready()
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    # No submit-initial action exists: the original CPU/fit are retained.
    parser.add_argument("action", choices=("advance", "collect", "run-stage"))
    for name in ("release", "campaign", "repair-release", "readiness"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("archive-sha256", "manifest-sha256", "plan-sha256", "repair-archive-sha256",
                 "repair-manifest-sha256", "repair-plan-sha256", "readiness-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--after")
    parser.add_argument("--node")
    args = parser.parse_args(argv)
    campaign = build_campaign(args)
    if args.action == "advance":
        if not args.after or args.after == "cpu":
            parser.error("Recovery advance requires fit or a later exact node")
        result = campaign.advance(args.after)
    elif args.action == "run-stage":
        if not args.node or args.node in ("cpu", "fit"):
            parser.error("Recovery may only run pilot, long segments or followup")
        result = campaign.run_stage(args.node)
    else:
        result = campaign.collect()
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
