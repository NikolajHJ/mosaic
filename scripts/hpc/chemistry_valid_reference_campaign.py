"""Allocated adoption contracts -> guarded serial epochs37--40.

The completed source135 native/probe/2,500-update proof is authenticated with
its original plan. A new campaign restores original epoch36 and preserves all
checks. Mandatory CUDA fixtures are rerun before its first training allocation;
none of the old disposable weights or diagnostics becomes a checkpoint.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET

from chemistry_campaign import Campaign, read, quota_receipt
from chemistry_guarded_restart import GuardedRestart
from chemistry_independent_self_test import (FIXTURES, ITERATION_FIXTURES, VALID_REFERENCE_FIXTURES)
from latentmol.chemistry_valid_reference_training_policy import validate_valid_reference_continuation
from latentmol.chemistry_valid_reference_adoption import materialize_adoption, authenticate_adoption
from latentmol.experiments import file_sha256, write_json


ADOPTION_FIXTURES = ["tests/test_chemistry_optimizer_step.py",
    "tests/test_chemistry_valid_reference_training_policy.py", "tests/test_chemistry_valid_reference_adoption.py",
    "tests/test_chemistry_launch.py"]
CUDA_FIXTURES = list(dict.fromkeys([
    "tests/test_chemistry_resume_cuda.py", "tests/test_failure_loss_validation_reuse.py",
    "tests/test_repair_guidance_v2.py", "tests/test_chemistry_valid_self_guidance.py",
    "tests/test_chemistry_valid_self_training.py", "tests/test_chemistry_generation_comparison.py",
    "tests/test_chemistry_worker_startup.py", "tests/test_chemistry_recovery_diagnostic.py",
    *ITERATION_FIXTURES, *VALID_REFERENCE_FIXTURES, *ADOPTION_FIXTURES]))


class ValidReferenceCampaign(GuardedRestart):
    def __init__(self, path, sha):
        super().__init__(path, sha)
        if validate_valid_reference_continuation(self.plan) is None:
            raise ValueError("This controller requires the explicit validated epoch36->40 policy")

    def controller(self, mode, **values):
        command = [str(self.work / "env/boltz311/bin/python"), "scripts/hpc/chemistry_valid_reference_campaign.py",
            "--plan", str(self.path), "--plan-sha256", self.sha, "--mode", mode]
        for key, value in values.items():
            command += ["--" + key.replace("_", "-"), str(value)]
        return ["env", "PYTHONPATH=" + str(self.release / "src"), *command]

    def cpu_contract_command(self, executable):
        command = super().cpu_contract_command(executable)
        modules = [*FIXTURES, *ITERATION_FIXTURES, *VALID_REFERENCE_FIXTURES, *ADOPTION_FIXTURES]
        return [*command, *(path for path in dict.fromkeys(modules) if path not in command)]

    def cpu(self):
        super().cpu()
        # Explicit additive adoption is a new allocated CPU result. The old
        # native observations retain their old job/source/plan identities.
        materialize_adoption(dict(self.plan, _plan_sha256=self.sha))

    def after_cpu(self):
        self.require_cpu()
        self.require_predecessor_stopped()
        authenticate_adoption(dict(self.plan, _plan_sha256=self.sha))
        # Campaign.advance reauthenticates all proof/health/identity/quota
        # requirements. Only its historical benchmark-allocation lookup is
        # explicitly replaced by the additive source135 adoption certificate.
        return self.advance("benchmark")

    def advance(self, after_stage):
        self.require_predecessor_stopped()
        return Campaign.advance(self, after_stage)

    def worker(self, command, label):
        if "latentmol.chemistry_continuation" in command and "train" in command:
            self.check_launch(command)
        if ("latentmol.chemistry_continuation" in command and "train" in command
                and command[command.index("--segment") + 1] == "1"):
            command = ["bash", "-c", shlex.join(self.controller("cuda-check")) + " && " + shlex.join(command)]
        return super().worker(command, label)

    def check_launch(self, command):
        """Run generated argv through the real CLI before any GPU submission."""
        if not os.environ.get("LSB_JOBID", "").isdigit():
            raise ValueError("Launch-command preflight requires the allocated callback")
        number = int(command[command.index("--segment") + 1])
        output = self.root / f"launch-command-{number:03d}.json"
        check = [str(self.work / "env/boltz311/bin/python"), "scripts/hpc/check_chemistry_launch.py",
                 "--output", str(output), "--", *command]
        subprocess.run(check, cwd=self.release, env=dict(os.environ, PYTHONPATH=str(self.release / "src")),
                       check=True, timeout=180)
        receipt = read(output)
        from latentmol.chemistry_short_continuation import continuation_end_epoch
        if (receipt.get("status") != "passed_dispatch_not_training" or receipt.get("command") != command
                or receipt.get("plan_sha256") != self.sha or receipt.get("job_id") != os.environ["LSB_JOBID"]
                or receipt.get("segment") != number or receipt.get("start_epoch") != 36
                or receipt.get("end_epoch") != continuation_end_epoch(self.plan) or receipt.get("model_setup_started") is not False
                or receipt.get("updates") != 0):
            raise ValueError("Actual launch-command preflight receipt differs")

    def cuda_check(self):
        if not os.environ.get("LSB_JOBID", "").isdigit():
            raise ValueError("Mandatory native GPU fixtures require an allocated job")
        path = self.root / "training-launch-cuda-pytest.xml"
        receipt = self.root / "training-launch-cuda-gate.json"
        if path.exists() or receipt.exists():
            raise FileExistsError("Preserve the original first-allocation CUDA evidence")
        reference = self.plan["loss_v1_reference"]
        command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                   "--junitxml", str(path), *CUDA_FIXTURES]
        environment = dict(os.environ, LATENTMOL_REQUIRE_CUDA_RESUME_FIXTURES="1",
            LATENTMOL_REQUIRE_REPAIR_CUDA_FIXTURES="1", LATENTMOL_FAILURE_REFERENCE_MODULE=reference["module"])
        reply = subprocess.run(command, cwd=self.release, env=environment)
        suites = list(ET.parse(path).getroot().iter("testsuite")) if path.exists() else []
        checks = dict(exit_zero=reply.returncode == 0, junit_present=bool(suites),
            zero_errors_failures_skips=bool(suites) and not any(int(s.get(k, 0)) for s in suites
                for k in ("errors", "failures", "skipped")),
            full_mandatory_coverage=sum(int(s.get("tests", 0)) for s in suites) >= 518)
        value = dict(status="passed" if all(checks.values()) else "failed", identity=self.identity,
            job_id=os.environ["LSB_JOBID"], command=command, checks=checks,
            junit_sha256=file_sha256(path) if path.exists() else None,
            scope="Mandatory current-release CUDA contracts before original36 training; old full scientific proof independently authenticated")
        write_json(receipt, value)
        if not all(checks.values()):
            raise RuntimeError("Mandatory first-allocation CUDA launch fixtures failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--mode", choices=("queue", "cpu", "after-cpu", "advance", "cuda-check"), required=True)
    parser.add_argument("--after-stage")
    args = parser.parse_args()
    campaign = ValidReferenceCampaign(args.plan, args.plan_sha256)
    if args.mode == "queue":
        print(campaign.queue_initial())
    elif args.mode == "cpu":
        campaign.cpu()
    elif args.mode == "after-cpu":
        print(campaign.after_cpu())
    elif args.mode == "cuda-check":
        campaign.cuda_check()
    elif args.after_stage:
        print(campaign.advance(args.after_stage))
    else:
        parser.error("Advancement requires the exact predecessor stage")


if __name__ == "__main__":
    main()

