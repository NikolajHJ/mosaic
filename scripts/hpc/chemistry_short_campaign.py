"""Original-loss proof -> disposable self-guidance test -> epochs37--39.

Both arms restore epoch36 independently. The self-guidance experiment can never
promote its disposable weights, and its scientific failure does not invalidate a
separately passing baseline. Every transition authenticates immutable evidence;
an uncertain scheduler state stops the chain rather than overlapping allocations.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import time

from chemistry_campaign import authenticated, read, quota_receipt
from chemistry_guarded_restart import GuardedRestart
from latentmol.chemistry_restart_policy import authenticate_stability
from latentmol.chemistry_short_continuation import validate_short_continuation
from latentmol.experiments import file_sha256, write_json
from latentmol.failure_guidance_selection import authenticate_readiness


FIXTURES = ["tests/test_chemistry_short_continuation.py", "tests/test_chemistry_valid_self_guidance.py",
    "tests/test_chemistry_valid_self_training.py", "tests/test_chemistry_short_campaign.py",
    "tests/test_chemistry_generation_comparison.py"]


class ShortCampaign(GuardedRestart):
    def __init__(self, path, sha):
        super().__init__(path, sha)
        short = validate_short_continuation(self.plan)
        if short is None or short["purpose"] != "baseline_training":
            raise ValueError("Only the original-loss baseline owns the bounded training controller")
        binding = self.plan.get("self_test_plan")
        if type(binding) is not dict or set(binding) != {"path", "sha256"}:
            raise ValueError("Require the separately sealed self-guidance test plan")
        self.test_path = Path(binding["path"]).resolve(strict=True)
        if not self.test_path.is_relative_to(self.work / "submissions"):
            raise ValueError("Self-guidance test plan escaped authorized submissions")
        authenticated(self.test_path, binding["sha256"])
        self.test_sha = binding["sha256"]
        self.test = read(self.test_path)
        policy = validate_short_continuation(self.test)
        if policy is None or policy["purpose"] != "self_guidance_test":
            raise ValueError("The second arm must remain a disposable self-guidance test")
        if (any(self.test.get(key) != self.plan.get(key) for key in (
                "release", "archive_sha256", "manifest_sha256", "starting_checkpoint",
                "parent_checkpoint_sha256", "parent_epoch", "execution", "classification_execution",
                 "worker_startup", "optimizer_step_control", "dataset_sha256", "split_sha256", "config_sha256"))
                or self.test["campaign"] == self.plan["campaign"]):
            raise ValueError("Matched self-guidance test changed baseline inputs/source or reused its run")
        left, right = dict(self.plan["protocol"]), dict(self.test["protocol"])
        right.pop("valid_prior_self_guidance")
        if left != right:
            raise ValueError("Only the explicit valid-prior self loss may differ between arms")

    def controller(self, mode, **values):
        command = [str(self.work / "env/boltz311/bin/python"), "scripts/hpc/chemistry_short_campaign.py",
            "--plan", str(self.path), "--plan-sha256", self.sha, "--mode", mode]
        for key, value in values.items():
            command += ["--" + key.replace("_", "-"), str(value)]
        return ["env", "PYTHONPATH=" + str(self.release / "src"), *command]

    def cpu_contract_command(self, executable):
        return [*super().cpu_contract_command(executable), *FIXTURES]

    def advance(self, after_stage):
        if after_stage != "benchmark":
            return super().advance(after_stage)
        path = self.root / "self-test-decision.json"
        if path.exists():
            result = read(path)
            if result.get("identity") != self.identity:
                raise ValueError("Self-test decision belongs to a different sealed suite")
            return result
        with (self.root / "self-test-intent.json").open("x") as stream:
            stream.write(self.sha + "\n")
        result = dict(status="stopped", identity=self.identity, training="not_yet_submitted")
        try:
            self.require_cpu()
            self.require_predecessor_stopped()
            self.complete("benchmark")
            native = self.root / "failure_readiness.json"
            authenticate_readiness(dict(self.plan, _plan_sha256=self.sha), file_sha256(native))
            authenticate_stability(dict(self.plan, _plan_sha256=self.sha))
            # The baseline has passed; the matched weak arm is evaluated next
            # in a separate allocation, with no training callback of its own.
            write_json(self.root / "quota-before-self-test.json", quota_receipt(2))
            if (self.root / "active-phase.json").exists():
                raise ValueError("A live/failed training phase blocks the sequential self test")
            test_root = Path(self.test["campaign"])
            test_root.mkdir(parents=True, exist_ok=False)
            reference = self.plan["loss_v1_reference"]
            cuda = ["env", "LATENTMOL_REQUIRE_CUDA_RESUME_FIXTURES=1",
                "LATENTMOL_REQUIRE_REPAIR_CUDA_FIXTURES=1",
                "LATENTMOL_FAILURE_REFERENCE_MODULE=" + reference["module"],
                "python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--junitxml",
                str(test_root / "self-cuda-pytest.xml"), "tests/test_chemistry_valid_self_guidance.py",
                "tests/test_chemistry_valid_self_training.py", "tests/test_chemistry_resume_cuda.py"]
            native_command = ["python", "-m", "latentmol.chemistry_guarded_validation",
                "--plan", str(self.test_path), "--plan-sha256", self.test_sha]
            command = ["bash", "-c", shlex.join(cuda) + " && " + shlex.join(native_command)]
            job = self.queue("self-test", self.worker(command, "bounded-valid-self-test"),
                             gpu=True, limit="24:00")
            callback = self.queue("after-self-test", self.controller("after-self-test"),
                                  dependency="ended(" + job + ")")
            result.update(status="self_test_submitted_not_passed", job_id=job,
                callback_job_id=callback, self_test_plan_sha256=self.test_sha,
                test_weights_disposable=True, baseline_training_after_terminal_test=True)
            return result
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            write_json(path, result)

    def after_self_test(self):
        """An ended failed experiment is evidence, never permission to overlap it."""
        path = self.root / "after-self-test-decision.json"
        if path.exists():
            saved = read(path)
            if saved.get("identity") != self.identity:
                raise ValueError("Terminal self-test decision changed its suite")
            return saved
        with (self.root / "after-self-test-intent.json").open("x") as stream:
            stream.write(self.sha + "\n")
        result = dict(status="stopped", identity=self.identity)
        try:
            job = self.receipt_job("self-test")
            for attempt in range(4):
                command = ["bjobs", "-a", "-noheader", "-o", "stat", job]
                reply = subprocess.run(command, capture_output=True, text=True, timeout=5)
                state = reply.stdout.strip()
                if reply.returncode == 0 and not reply.stderr.strip() and state in ("DONE", "EXIT"):
                    break
                if attempt == 3 or state not in ("RUN", "PEND"):
                    raise ValueError("Self-test is not confirmed terminal; baseline promotion is blocked")
                time.sleep(2)
            result["self_test_terminal"] = dict(job_id=job, state=state, command=command)
            test_root = Path(self.test["campaign"])
            result["self_test_results"] = {
                name: dict(path=str(test_root/name), sha256=file_sha256(test_root/name),
                           status=read(test_root/name).get("status"))
                for name in ("failure_readiness.json", "stability_gate.json") if (test_root/name).exists()}
            # Reauthenticate the baseline independently. Never pass self-test
            # weights, optimizer moments or readiness into the baseline branch.
            promotion = super().advance("benchmark")
            result.update(status="baseline_promotion_evaluated", baseline=promotion,
                          weak_loss_automatically_adopted=False)
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
    parser.add_argument("--mode", choices=("queue", "cpu", "after-cpu", "advance", "after-self-test"), required=True)
    parser.add_argument("--after-stage")
    args = parser.parse_args()
    campaign = ShortCampaign(args.plan, args.plan_sha256)
    if args.mode == "queue":
        print(campaign.queue_initial())
    elif args.mode == "cpu":
        campaign.cpu()
    elif args.mode == "after-cpu":
        print(campaign.after_cpu())
    elif args.mode == "after-self-test":
        print(campaign.after_self_test())
    elif args.after_stage:
        print(campaign.advance(args.after_stage))
    else:
        parser.error("Advancement requires its exact predecessor")


if __name__ == "__main__":
    main()
