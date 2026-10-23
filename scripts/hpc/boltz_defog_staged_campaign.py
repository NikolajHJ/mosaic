"""Submit independent long-comparison arms in serial allocations of at most24h.

This controller reads byte/JSON/scheduler metadata only. Scientific receipt
validation is a separate subprocess inside its allocated CPU callback. A DONE
job alone never approves an arm; the unchanged native stage reader must pass
before the next allocation is submitted. Unknown submissions retain exclusive
intents and are never retried automatically. Allocated HPC validation pending.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from latentmol.config_identity import normalize_json, verify_config_identity
from latentmol.experiments import file_sha256, source_manifest, write_json
from boltz_defog_long_job import TESTS, quota_receipt
from submit_stability_study import authenticated, submit


VERSION = "boltz-defog-staged-campaign-v1"
STAGE_VERSION = "long-comparison-allocation-stage-v1"
STAGES = ("control", "guided", "evaluation")
STAGE_BUDGETS = {
    "control": {"allocation_seconds": 64800, "internal_wall_seconds": 61200},
    "guided": {"allocation_seconds": 86400, "internal_wall_seconds": 84600},
    "evaluation": {"allocation_seconds": 21600, "internal_wall_seconds": 19800},
}
EXTRA_TESTS = ("tests/test_boltz_defog_long_stage.py", "tests/test_boltz_defog_staged_campaign.py")
_SHA = r"[0-9a-f]{64}"
_JOB = r"[1-9][0-9]*"


def strict_json(text):
    """Reject duplicate keys/nonfinite values using sealed core JSON policy.

    The Boltz source lineage deliberately does not inherit chemistry campaign
    helpers. Keep its parser stdlib-only and reuse the core finite/type walk;
    this also rejects overflowing JSON numbers such as ``1e400``.
    """
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("Duplicate JSON key: " + key)
            value[key] = item
        return value
    return normalize_json(json.loads(text, object_pairs_hook=pairs))


def read(path):
    return strict_json(Path(path).read_text(encoding="utf-8"))


def asset(ref):
    """Authenticate a small declared file without accepting symlink aliases."""
    if (type(ref) is not dict or set(ref) != {"path", "sha256"}
            or type(ref.get("path")) is not str or not Path(ref["path"]).is_absolute()
            or type(ref.get("sha256")) is not str or re.fullmatch(_SHA, ref["sha256"]) is None):
        raise ValueError("Malformed immutable campaign asset")
    path = Path(ref["path"])
    if path.is_symlink() or not path.is_file():
        raise ValueError("Missing or symlinked campaign asset")
    authenticated(path, ref["sha256"])
    return path


def ref(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": file_sha256(path)}


def successful_process(row):
    return (type(row) is dict and type(row.get("returncode")) is int and row["returncode"] == 0
            and row.get("timed_out") is False and row.get("failure") is None
            and row.get("interrupted") is False)


def _flag(command, flag, value):
    return command.count(flag) == 1 and command.index(flag) + 1 < len(command) and command[command.index(flag) + 1] == value


class StagedCampaign:
    """One durable stage submission and one allocated successor callback."""

    def __init__(self, path, sha):
        self.path = Path(path).resolve(strict=True)
        self.sha = sha
        self.manifest = read(asset({"path": str(self.path), "sha256": sha}))
        m = self.manifest
        fields = {"version", "name", "work_root", "release", "control", "plan", "bootstrap",
                  "cpu_readiness", "readiness_operator", "source_archive_sha256", "source_manifest_sha256"}
        if (type(m) is not dict or set(m) != fields or m.get("version") != VERSION
                or type(m.get("name")) is not str or re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", m["name"]) is None
                or any(type(m.get(key)) is not str or re.fullmatch(_SHA, m[key]) is None
                       for key in ("source_archive_sha256", "source_manifest_sha256"))):
            raise ValueError("Unsupported staged campaign manifest")
        self.work, self.release, self.root = (Path(m[key]).resolve(strict=True) for key in ("work_root", "release", "control"))
        if (not str(self.work).startswith("/work3/")
                or not self.release.is_relative_to(self.work / "releases")
                or not self.root.is_relative_to(self.work / "submissions")
                or self.path != self.root / "campaign.json"):
            raise ValueError("Campaign escaped its declared DTU source/control roots")
        authenticated(Path(str(self.release) + ".tar.gz"), m["source_archive_sha256"])
        if source_manifest(self.release)["sha256"] != m["source_manifest_sha256"]:
            raise ValueError("Staged campaign source bytes changed")
        self.plan = read(asset(m["plan"]))
        self.bootstrap = read(asset(m["bootstrap"]))
        asset(m["readiness_operator"])
        if (self.plan.get("source_archive_sha256") != m["source_archive_sha256"]
                or self.plan.get("source_manifest_sha256") != m["source_manifest_sha256"]
                or not Path(self.plan["output_root"]).is_absolute()
                or not Path(self.plan["output_root"]).resolve().is_relative_to(self.work / "runs")
                or self.bootstrap.get("archive_sha256") != m["source_archive_sha256"]
                or self.bootstrap.get("manifest_sha256") != m["source_manifest_sha256"]
                or Path(self.bootstrap["release"]).resolve() != self.release
                or Path(self.bootstrap["control"]).resolve() != self.root
                or Path(self.bootstrap["campaign"]).resolve() != Path(self.plan["output_root"]).resolve()
                or Path(m["plan"]["path"]).resolve() != self.root / "readiness/prepared/plan.json"
                or Path(m["cpu_readiness"]["path"]).resolve() != self.root / "readiness/completion.json"):
            raise ValueError("Prepared plan/bootstrap/source/output bindings differ")
        self.identity = {"campaign_sha256": sha, "plan_sha256": m["plan"]["sha256"],
            "archive_sha256": m["source_archive_sha256"], "manifest_sha256": m["source_manifest_sha256"]}
        self.receipts = self.root / "stage-submissions"

    def _controller(self, after):
        return [str(self.work / "env/boltz311/bin/python"), "scripts/hpc/boltz_defog_staged_campaign.py",
                "--campaign", str(self.path), "--sha256", self.sha, "advance", "--after-stage", after]

    def _stage_command(self, stage, previous=None):
        command = ["python", "scripts/run_boltz_defog_long_stage.py", "run", "--stage", stage,
            "--plan", self.manifest["plan"]["path"], "--sha256", self.manifest["plan"]["sha256"]]
        if previous is not None:
            command += ["--previous-receipt", previous["path"], "--previous-sha256", previous["sha256"]]
        return command

    def _worker(self, command, label):
        m = self.manifest
        return ["env", f"LATENTMOL_SOURCE_ARCHIVE={self.release}.tar.gz", f"LATENTMOL_WORK_ROOT={self.work}",
            "bash", "scripts/hpc/boltz_run.sh", "python", "scripts/verify_stability_release.py",
            "--archive-sha256", m["source_archive_sha256"], "--manifest-sha256", m["source_manifest_sha256"],
            "--", "python", "-m", "latentmol.experiments", "--run-root", str(self.root / "stage-allocations"),
            "--name", label, "--", *command]

    def arguments(self, stage, *, callback=False):
        """Keep each individual scheduler request at or below24h."""
        if stage not in STAGES:
            raise ValueError("Unknown staged allocation")
        minutes = 30 if callback else STAGE_BUDGETS[stage]["allocation_seconds"] // 60
        if not 0 < minutes <= 1440:
            raise ValueError("Stage allocation exceeds the user24h limit")
        label = self.manifest["name"] + ("-after-" if callback else "-") + stage
        arguments = ["bsub", "-J", label, "-q", "hpc" if callback else "gpua100", "-n", "4",
            "-R", "rusage[mem=4GB]" if callback else "rusage[mem=16GB]", "-R", "span[hosts=1]",
            "-W", f"{minutes // 60}:{minutes % 60:02d}", "-oo", str(self.work / "logs" / (label + ".%J.out")),
            "-eo", str(self.work / "logs" / (label + ".%J.err"))]
        if not callback:
            arguments += ["-R", "select[gpu80gb]", "-gpu", "num=1:mode=exclusive_process:mps=yes:j_exclusive=yes"]
        return arguments

    def receipt_job(self, stage):
        receipt = read(self.receipts / (stage + "-receipt.json"))
        intent = read(self.receipts / (stage + "-intent.json"))
        if (receipt.get("intent") != intent or intent.get("identity") != self.identity
                or type(receipt.get("job_id")) is not str or re.fullmatch(_JOB, receipt["job_id"]) is None):
            raise ValueError("Stage submission receipt/source identity differs")
        return receipt["job_id"]

    def _wrapper(self, job, directory, command=None):
        matches = []
        for path in Path(directory).glob("*/run.json"):
            value = read(path)
            if value.get("environment", {}).get("job_id") == job:
                matches.append((path, value))
        if len(matches) != 1:
            raise ValueError("Allocation lacks a unique persisted process wrapper")
        path, value = matches[0]
        verify_config_identity(value)
        if (value.get("status") != "completed" or type(value.get("returncode")) is not int or value["returncode"] != 0
                or value.get("source_archive_sha256") != self.manifest["source_archive_sha256"]
                or value.get("source", {}).get("sha256") != self.manifest["source_manifest_sha256"]
                or (command is not None and value.get("config", {}).get("command") != command)):
            raise ValueError("Allocation wrapper did not complete with the exact source/command")
        return path, value

    def _completed(self, job, directory, command=None):
        """Check DONE; only a genuinely aged-out job may use persisted completion.

        A bounded RUN requery handles scheduler propagation after an ended()
        dependency. EXIT/PEND/query errors fail immediately, regardless of files.
        """
        reply = None
        for attempt in range(4):
            reply = subprocess.run(["bjobs", "-a", "-noheader", "-o", "stat", job],
                capture_output=True, text=True, timeout=5)
            status = reply.stdout.strip()
            if reply.returncode == 0 and status == "DONE":
                kind = "scheduler_DONE"
                break
            if reply.returncode == 0 and status == "RUN" and attempt < 3:
                time.sleep(2)
                continue
            if not status and "not found" in reply.stderr.lower():
                kind = "aged_out_persisted_completion"
                break
            raise ValueError("Predecessor allocation did not finish successfully: " + status)
        else:
            raise ValueError("Predecessor still RUN after bounded scheduler observation")
        path, record = self._wrapper(job, directory, command)
        return {"job_id": job, "kind": kind, "stdout": reply.stdout, "stderr": reply.stderr,
                "returncode": reply.returncode, "wrapper": ref(path)}, record

    def require_cpu(self):
        """Recheck the real readiness result before any native stage can queue."""
        receipt = read(asset(self.manifest["cpu_readiness"]))
        if (receipt.get("status") != "passed" or receipt.get("source") != self.bootstrap
                or receipt.get("bootstrap_sha256") != self.manifest["bootstrap"]["sha256"]
                or receipt.get("operator_sha256") != self.manifest["readiness_operator"]["sha256"]
                or receipt.get("plan") != self.manifest["plan"]
                or type(receipt.get("job_id")) is not str or re.fullmatch(_JOB, receipt["job_id"]) is None
                or set(receipt.get("stages", {})) != {"preflight", "prepare", "validate"}
                or not set((*TESTS, *EXTRA_TESTS)).issubset(receipt.get("tests", []))):
            raise ValueError("Original allocated readiness/source/test coverage has not passed")
        for name, process in receipt["stages"].items():
            if not successful_process(process):
                raise ValueError("A readiness subprocess did not complete successfully")
            expected_log = self.root / "readiness" / (name + ".log")
            if Path(process.get("log", "")).resolve() != expected_log:
                raise ValueError("Readiness log escaped its declared output namespace")
            asset({"path": str(expected_log), "sha256": process.get("log_sha256")})
        cases = self.plan.get("cases", [])
        if [row.get("id") for row in cases] != ["boltz_only", "boltz_defog"]:
            raise ValueError("Prepared readiness changes matched arm membership")
        hashes = {}
        for row in cases:
            path = asset(row["config"])
            if path.resolve() != self.root / "readiness/prepared" / (row["id"] + ".yaml"):
                raise ValueError("Prepared arm configuration escaped its readiness namespace")
            hashes[row["id"]] = row["config"]["sha256"]
        if receipt.get("prepared_configurations") != hashes:
            raise ValueError("Prepared native configuration digests changed")
        observation, wrapper = self._completed(receipt["job_id"], self.root / "readiness-allocations")
        command = wrapper.get("config", {}).get("command")
        if (type(command) is not list or len(command) < 2
                or Path(command[1]).resolve() != Path(self.manifest["readiness_operator"]["path"]).resolve()
                or not _flag(command, "--bootstrap", self.manifest["bootstrap"]["path"])
                or not _flag(command, "--bootstrap-sha256", self.manifest["bootstrap"]["sha256"])
                or not _flag(command, "--operator-sha256", self.manifest["readiness_operator"]["sha256"])):
            raise ValueError("CPU readiness wrapper did not execute the pinned operator/bootstrap")
        return observation

    def _queue_stage(self, stage, previous=None):
        # User quota is checked at every promotion, not reserved by a past check.
        quota = quota_receipt()
        write_json(self.root / (stage + "-launch-quota.json"), quota)
        label = self.manifest["name"] + "-" + stage
        rows = [{"stage": stage, "cwd": str(self.release), "arguments": self.arguments(stage),
                 "job_command": self._worker(self._stage_command(stage, previous), label), "dependency": None},
                {"stage": "after-" + stage, "cwd": str(self.release), "arguments": self.arguments(stage, callback=True),
                 "job_command": self._worker(self._controller(stage), label + "-callback"), "dependency": "ended:" + stage}]
        return submit(rows, self.receipts, self.identity)

    def queue_control(self):
        readiness = self.require_cpu()
        jobs = self._queue_stage("control")
        result = {"identity": self.identity, "status": "submitted_not_passed", "readiness": readiness, "jobs": jobs}
        write_json(self.root / "control-queued.json", result)
        return result

    def _validate_stage(self, stage, stage_ref, stage_job):
        """Execute the authoritative saved-state reader only in this allocation."""
        directory = self.root / "stage-validation" / stage
        directory.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "scripts/run_boltz_defog_long_stage.py", "validate-receipt", "--stage", stage,
            "--plan", self.manifest["plan"]["path"], "--sha256", self.manifest["plan"]["sha256"],
            "--receipt", stage_ref["path"], "--receipt-sha256", stage_ref["sha256"]]
        expected = {"identity": self.identity, "stage": stage, "stage_job_id": stage_job, "receipt": stage_ref, "command": command}
        expected_cases = ["boltz_only"] if stage == "control" else ["boltz_only", "boltz_defog"]
        saved = directory / "result.json"
        if saved.exists():
            result = read(saved)
            if (any(result.get(key) != value for key, value in expected.items()) or result.get("status") != "passed"
                    or type(result.get("returncode")) is not int or result["returncode"] != 0
                    or result.get("callback_job_id") != os.environ["LSB_JOBID"]
                    or result.get("validation", {}).get("status") != "passed"
                    or result["validation"].get("stage") != stage
                    or result["validation"].get("case_ids") != expected_cases):
                raise ValueError("Existing stage validation is failed or belongs to another input")
            asset(result["log"])
            return result
        with (directory / "intent.json").open("x", encoding="utf-8") as stream:
            json.dump({**expected, "callback_job_id": os.environ["LSB_JOBID"]}, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        result = {**expected, "status": "failed", "callback_job_id": os.environ["LSB_JOBID"]}
        try:
            execution = subprocess.run(command, cwd=self.release, capture_output=True, text=True, timeout=900)
            log = directory / "validation.log"
            log.write_text(execution.stdout + "\nSTDERR:\n" + execution.stderr, encoding="utf-8")
            result.update(returncode=execution.returncode, log=ref(log))
            parsed = strict_json(execution.stdout)
            if (execution.returncode != 0 or parsed.get("status") != "passed" or parsed.get("stage") != stage
                    or parsed.get("case_ids") != expected_cases):
                raise ValueError("Original native stage receipt validation did not pass")
            result.update(status="passed", validation=parsed)
        except BaseException as error:
            result.update(error=str(error), error_type=type(error).__name__)
            raise
        finally:
            write_json(saved, result)
        return result

    def advance(self, stage):
        if stage not in STAGES:
            raise ValueError("Unknown predecessor stage")
        callback = self.receipt_job("after-" + stage)
        if os.environ.get("LSB_JOBID") != callback:
            raise ValueError("Promotion must run in its actual declared allocated callback")
        self.require_cpu()
        job = self.receipt_job(stage)
        previous = None if stage == "control" else ref(Path(self.plan["output_root"]) / "staged" / STAGES[STAGES.index(stage) - 1] / "receipt.json")
        observation, _ = self._completed(job, self.root / "stage-allocations", self._stage_command(stage, previous))
        stage_ref = ref(Path(self.plan["output_root"]) / "staged" / stage / "receipt.json")
        receipt = read(asset(stage_ref))
        source = {"source_manifest_sha256": self.manifest["source_manifest_sha256"], "source_archive_sha256": self.manifest["source_archive_sha256"]}
        if (receipt.get("version") != STAGE_VERSION or receipt.get("stage") != stage or receipt.get("status") != "passed"
                or receipt.get("job_id") != job or receipt.get("source") != source or receipt.get("plan") != self.manifest["plan"]
                or receipt.get("budget") != STAGE_BUDGETS[stage] or receipt.get("previous") != previous):
            raise ValueError("Saved native stage has inconsistent source/allocation/predecessor bindings")
        validation = self._validate_stage(stage, stage_ref, job)
        result = {"identity": self.identity, "after_stage": stage, "status": "validated_not_training_or_success_claim",
                  "scheduler": observation, "validation": ref(self.root / "stage-validation" / stage / "result.json"),
                  "receipt": stage_ref, "utc": datetime.now(timezone.utc).isoformat()}
        if stage == STAGES[-1]:
            result.update(status="comparison_terminal_validated", final_report=receipt["report"])
        else:
            following = STAGES[STAGES.index(stage) + 1]
            result.update(status="successor_submitted_not_passed", jobs=self._queue_stage(following, stage_ref))
        write_json(self.root / ("after-" + stage + ".json"), result)
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("queue-control")
    commands.add_parser("advance").add_argument("--after-stage", choices=STAGES, required=True)
    args = parser.parse_args(argv)
    campaign = StagedCampaign(args.campaign, args.sha256)
    result = campaign.queue_control() if args.command == "queue-control" else campaign.advance(args.after_stage)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
