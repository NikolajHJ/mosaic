"""Allocated contracts -> new native/stability proof -> serial VAE continuation.

Historical source127 is never restarted. All validation updates are disposable;
the existing serial controller restores the authenticated completed epoch36
checkpoint only after both native and longer stability gates pass.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET

from chemistry_campaign import Campaign, FIXTURES, OPTIMIZED_FIXTURES, authenticated, read, quota_receipt
from failure_guidance_campaign import EXTRA_FIXTURES, SPEED_FIXTURES
from chemistry_runtime_continuation import AMENDMENT_FIXTURES
from latentmol.chemistry_restart_policy import validate_guarded_restart
from latentmol.experiments import file_sha256, write_json


GUARDED_FIXTURES = ["tests/test_repair_guidance_v2.py", "tests/test_chemistry_classification.py",
    "tests/test_chemistry_early_health.py", "tests/test_chemistry_guarded_restart.py"]
STARTUP_FIXTURES = ["tests/test_chemistry_classification_pool.py",
    "tests/test_chemistry_classification_validation.py", "tests/test_chemistry_recovery_diagnostic.py",
    "tests/test_chemistry_recovery_controller.py", "tests/test_chemistry_worker_startup.py",
    "tests/test_chemistry_shared_budget.py"]


def cpu_command(executable, root):
    modules = list(dict.fromkeys([*FIXTURES, *OPTIMIZED_FIXTURES, *EXTRA_FIXTURES,
                                 *SPEED_FIXTURES, *AMENDMENT_FIXTURES, *GUARDED_FIXTURES]))
    return [executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--junitxml",
            str(root / "guarded-cpu-pytest.xml"), *modules]


class GuardedRestart(Campaign):
    """Reuse quota, native identity and whole-epoch serial progression unchanged."""

    def __init__(self, path, sha):
        super().__init__(path, sha)
        self.policy = validate_guarded_restart(self.plan)
        if self.policy is None:
            raise ValueError("This controller requires explicit corrected-objective validation")

    def cpu_contract_command(self, executable):
        """Let a new sealed diagnostic extend, rather than replace, contracts."""
        command = cpu_command(executable, self.root)
        if getattr(self, "plan", {}).get("worker_startup") is not None:
            command.extend(path for path in STARTUP_FIXTURES if path not in command)
        return command

    def controller(self, mode, **values):
        command = [str(self.work / "env/boltz311/bin/python"), "scripts/hpc/chemistry_guarded_restart.py",
                   "--plan", str(self.path), "--plan-sha256", self.sha, "--mode", mode]
        for key, value in values.items():
            command += ["--" + key.replace("_", "-"), str(value)]
        return ["env", "PYTHONPATH=" + str(self.release / "src"), *command]

    def require_predecessor_stopped(self):
        """Aged scheduler records require the authenticated collected terminal proof."""
        previous = self.plan["stopped_predecessor"]
        proof_path = Path(previous["terminal_evidence"])
        authenticated(proof_path, previous["terminal_evidence_sha256"])
        proof = read(proof_path)
        expected_jobs = {"training": "29758907", "callback": "29758908"}
        if previous.get("jobs") != expected_jobs:
            raise ValueError("Stopped predecessor identity changed")
        observations = {}
        for role, job in expected_jobs.items():
            saved = proof.get("queries", {}).get(job, {})
            text = saved.get("stdout", "")
            if (saved.get("command") != ["bjobs", "-l", job] or saved.get("returncode") != 0
                    or "TERM_OWNER: job killed by owner" not in text
                    or (role == "training" and "Status <EXIT>" not in text)):
                raise ValueError("Predecessor lacks exact saved user-termination evidence")
            command = ["bjobs", "-a", "-noheader", "-o", "stat", job]
            reply = subprocess.run(command, capture_output=True, text=True, timeout=15)
            direct_exit = reply.returncode == 0 and reply.stdout.strip() == "EXIT" and not reply.stderr.strip()
            expired = not reply.stdout.strip() and reply.stderr.strip() == f"Job <{job}> is not found"
            if not (direct_exit or expired):
                raise ValueError("Stopped predecessor is live or its scheduler state is uncertain")
            observations[role] = dict(job_id=job, command=command, returncode=reply.returncode,
                stdout=reply.stdout, stderr=reply.stderr,
                state="observed_EXIT" if direct_exit else "authenticated_EXIT_now_aged_out")
        receipt = dict(status="stopped_predecessor_authenticated", identity=self.identity,
                       terminal_evidence_sha256=previous["terminal_evidence_sha256"], observations=observations)
        # An observation may change from EXIT to aged out; preserve each phase.
        path = self.root / ("predecessor-stop-" + os.environ.get("LSB_JOBID", "initial") + ".json")
        if not path.exists():
            write_json(path, receipt)
        return receipt

    def queue_initial(self):
        self.require_predecessor_stopped()
        write_json(self.root / "initial_quota.json", quota_receipt(2))
        job = self.queue("cpu", self.worker(self.controller("cpu"), "guarded-chemistry-cpu"), limit="1:00")
        callback = self.queue("after-cpu", self.controller("after-cpu"), dependency="ended(" + job + ")")
        return dict(status="submitted_not_passed", cpu=job, after_cpu=callback,
                    native_stability="not_yet_submitted", training="not_yet_submitted")

    def cpu(self):
        if not os.environ.get("LSB_JOBID"):
            raise ValueError("Contract checks require allocated CPU execution")
        self.require_predecessor_stopped()
        command = self.cpu_contract_command(sys.executable)
        reference = self.plan["loss_v1_reference"]
        authenticated(reference["module"], reference["sha256"])
        environment = dict(os.environ, LATENTMOL_FAILURE_REFERENCE_MODULE=reference["module"])
        reply = subprocess.run(command, cwd=self.release, env=environment)
        junit = self.root / "guarded-cpu-pytest.xml"
        gate = dict(status="passed" if reply.returncode == 0 else "failed", returncode=reply.returncode,
            identity=self.identity, job_id=os.environ["LSB_JOBID"], command=command,
            junit_sha256=file_sha256(junit) if junit.exists() else None)
        if reply.returncode == 0:
            path = self.root / "repair_profile.json"
            profile_command = [sys.executable, "scripts/benchmark_atom_preserving_search.py", "--plan", str(self.path),
                               "--plan-sha256", self.sha, "--output", str(path)]
            profile_reply = subprocess.run(profile_command, cwd=self.release)
            gate.update(profile_command=profile_command, profile_returncode=profile_reply.returncode,
                        profile_sha256=file_sha256(path) if path.exists() else None)
            if profile_reply.returncode:
                gate["status"] = "failed"
        write_json(self.root / "cpu_gate.json", gate)
        if gate["status"] != "passed":
            raise RuntimeError("Corrected chemistry CPU contracts or repair execution profile failed")

    def require_cpu(self):
        super().require_cpu()
        gate = read(self.root / "cpu_gate.json")
        path = self.root / "guarded-cpu-pytest.xml"
        if (gate.get("command") != self.cpu_contract_command(str(self.work / "env/boltz311/bin/python"))
                or file_sha256(path) != gate.get("junit_sha256")):
            raise ValueError("Corrected chemistry CPU command or JUnit evidence changed")
        suites = list(ET.parse(path).getroot().iter("testsuite"))
        if (not suites or any(int(s.get(k, 0)) for s in suites for k in ("failures", "errors"))
                or sum(int(s.get("tests", 0)) - int(s.get("skipped", 0)) for s in suites) < 1):
            raise ValueError("Corrected chemistry contracts failed or were entirely skipped")

    def after_cpu(self):
        path = self.root / "after-cpu-decision.json"
        if path.exists():
            return read(path)
        intent = self.root / "after-cpu-intent.json"
        with intent.open("x") as stream:
            stream.write(self.sha + "\n")
        result = dict(status="stopped", identity=self.identity)
        try:
            self.require_cpu()
            self.require_predecessor_stopped()
            # Existing CUDA exact-resume fixtures are mandatory, not a CPU skip.
            reference = self.plan["loss_v1_reference"]
            authenticated(reference["module"], reference["sha256"])
            cuda = ["env", "LATENTMOL_REQUIRE_CUDA_RESUME_FIXTURES=1", "LATENTMOL_REQUIRE_REPAIR_CUDA_FIXTURES=1",
                    "LATENTMOL_FAILURE_REFERENCE_MODULE=" + reference["module"], "python", "-m", "pytest", "-q",
                    "-p", "no:cacheprovider", "--junitxml", str(self.root / "guarded-cuda-pytest.xml"),
                    "tests/test_chemistry_resume_cuda.py", "tests/test_failure_loss_validation_reuse.py",
                    "tests/test_repair_guidance_v2.py"]
            if self.plan.get("worker_startup") is not None:
                cuda += ["tests/test_chemistry_worker_startup.py", "tests/test_chemistry_recovery_diagnostic.py"]
            native = ["python", "-m", "latentmol.chemistry_guarded_validation", "--plan", str(self.path),
                      "--plan-sha256", self.sha]
            command = ["bash", "-c", shlex.join(cuda) + " && " + shlex.join(native)]
            job = self.queue("benchmark", self.worker(command, "guarded-chemistry-native-stability"),
                             gpu=True, limit="24:00")
            callback = self.queue("after-benchmark", self.controller("advance", after_stage="benchmark"),
                                  dependency="ended(" + job + ")")
            result.update(status="native_stability_submitted_not_passed", job_id=job, callback_job_id=callback,
                          validation_updates=self.policy["validation_updates"], training="not_yet_submitted")
            return result
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            write_json(path, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--mode", choices=("queue", "cpu", "after-cpu", "advance"), required=True)
    parser.add_argument("--after-stage")
    args = parser.parse_args()
    campaign = GuardedRestart(args.plan, args.plan_sha256)
    if args.mode == "queue":
        print(campaign.queue_initial())
    elif args.mode == "cpu":
        campaign.cpu()
    elif args.mode == "after-cpu":
        print(campaign.after_cpu())
    elif args.after_stage:
        print(campaign.advance(args.after_stage))
    else:
        parser.error("Serial advancement requires its exact predecessor")


if __name__ == "__main__":
    main()
