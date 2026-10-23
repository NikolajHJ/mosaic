"""Independent, disposable weak-self proof from authenticated epoch36.

Only this plan's CPU/native/2,500-update checks matter. A failed original-loss
baseline does not block submission, and this controller has no training or
promotion action, even when every weak-self check passes. New sealed plan/run
directories preserve all historical failed campaigns.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex

from chemistry_campaign import authenticated, read, quota_receipt
from chemistry_guarded_restart import GuardedRestart
from chemistry_short_campaign import FIXTURES as SHORT_FIXTURES
from latentmol.chemistry_short_continuation import validate_short_continuation
from latentmol.chemistry_target_weighting_policy import validate_target_weighting_test
from latentmol.chemistry_valid_reference_policy import validate_valid_reference_test
from latentmol.experiments import write_json


FIXTURES = [*SHORT_FIXTURES, "tests/test_chemistry_independent_self_test.py"]
ITERATION_FIXTURES = ["tests/test_chemistry_self_guidance_iteration.py",
    "tests/test_chemistry_objective_gradient_diagnostics.py", "tests/test_chemistry_self_strength_validation.py"]
TARGET_WEIGHTING_FIXTURES = ["tests/test_chemistry_target_weighting.py",
    "tests/test_chemistry_target_weighting_policy.py", "tests/test_chemistry_target_weighting_validation.py",
    "tests/test_chemistry_target_weighting_training.py"]
VALID_REFERENCE_FIXTURES = ["tests/test_chemistry_valid_reference_guidance.py",
    "tests/test_chemistry_valid_reference_policy.py", "tests/test_chemistry_valid_reference_validation.py",
    "tests/test_chemistry_valid_reference_training.py"]


class IndependentSelfTest(GuardedRestart):
    """Reuse allocated guards without inheriting a training callback chain."""

    def __init__(self, path, sha):
        super().__init__(path, sha)
        short = validate_short_continuation(self.plan)
        if short is None or short["purpose"] != "self_guidance_test":
            raise ValueError("Independent controller requires the disposable weak-self test purpose")

    def controller(self, mode, **values):
        if values or mode not in ("cpu", "after-cpu"):
            raise ValueError("Independent weak-self proof cannot promote or train")
        return ["env", "PYTHONPATH=" + str(self.release / "src"),
            str(self.work / "env/boltz311/bin/python"), "scripts/hpc/chemistry_independent_self_test.py",
            "--plan", str(self.path), "--plan-sha256", self.sha, "--mode", mode]

    def cpu_contract_command(self, executable):
        command = super().cpu_contract_command(executable)
        modules = list(FIXTURES)
        if self.plan.get("self_guidance_iteration") is not None:
            modules.extend(ITERATION_FIXTURES)
        if validate_target_weighting_test(self.plan) is not None:
            modules.extend(TARGET_WEIGHTING_FIXTURES)
        if validate_valid_reference_test(self.plan) is not None:
            modules.extend([*ITERATION_FIXTURES, *VALID_REFERENCE_FIXTURES])
        if self.plan.get("optimizer_step_control") is not None:
            modules.append("tests/test_chemistry_optimizer_step.py")
        return [*command, *(path for path in modules if path not in command)]

    def queue_initial(self):
        result = super().queue_initial()
        result.update(independent_weak_test=True, baseline_success_required=False,
                      training="forbidden", promotable=False)
        return result

    def advance(self, after_stage):
        raise ValueError("Independent weak-self proof cannot promote or train")

    def after_cpu(self):
        path = self.root / "after-cpu-decision.json"
        if path.exists():
            saved = read(path)
            if saved.get("identity") != self.identity:
                raise ValueError("Saved weak-self decision belongs to a different sealed plan")
            return saved
        # A lost scheduler response must not silently create a second GPU job.
        with (self.root / "after-cpu-intent.json").open("x") as stream:
            json.dump(dict(identity=self.identity, action="independent_disposable_self_test"), stream)
        result = dict(status="stopped", identity=self.identity, training="forbidden", promotable=False,
                      baseline_success_required=False)
        try:
            valid_reference = validate_valid_reference_test(self.plan)
            target_weighting = validate_target_weighting_test(self.plan)
            self.require_cpu()
            self.require_predecessor_stopped()
            if (self.root / "active-phase.json").exists():
                raise ValueError("A phase lock blocks the independent disposable self test")
            write_json(self.root / "quota-before-self-test.json", quota_receipt(2))
            reference = self.plan["loss_v1_reference"]
            authenticated(reference["module"], reference["sha256"])
            cuda = ["env", "LATENTMOL_REQUIRE_CUDA_RESUME_FIXTURES=1",
                "LATENTMOL_REQUIRE_REPAIR_CUDA_FIXTURES=1",
                "LATENTMOL_FAILURE_REFERENCE_MODULE=" + reference["module"],
                "python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--junitxml",
                str(self.root / "independent-self-cuda-pytest.xml"),
                "tests/test_chemistry_resume_cuda.py", "tests/test_failure_loss_validation_reuse.py",
                "tests/test_repair_guidance_v2.py", "tests/test_chemistry_valid_self_guidance.py",
                "tests/test_chemistry_valid_self_training.py", "tests/test_chemistry_generation_comparison.py"]
            if self.plan.get("worker_startup") is not None:
                cuda += ["tests/test_chemistry_worker_startup.py", "tests/test_chemistry_recovery_diagnostic.py"]
            if self.plan.get("optimizer_step_control") is not None:
                cuda.append("tests/test_chemistry_optimizer_step.py")
            if self.plan.get("self_guidance_iteration") is not None:
                cuda.extend(ITERATION_FIXTURES)
            if valid_reference is not None:
                cuda.extend(path for path in (*ITERATION_FIXTURES, *VALID_REFERENCE_FIXTURES) if path not in cuda)
                module = "latentmol.chemistry_valid_reference_validation"
            elif target_weighting is not None:
                cuda.extend(TARGET_WEIGHTING_FIXTURES)
                module = "latentmol.chemistry_target_weighting_validation"
            else:
                module = ("latentmol.chemistry_self_strength_validation" if self.plan.get("self_guidance_iteration") is not None
                          else "latentmol.chemistry_guarded_validation")
            native = ["python", "-m", module, "--plan", str(self.path),
                      "--plan-sha256", self.sha]
            command = ["bash", "-c", shlex.join(cuda) + " && " + shlex.join(native)]
            job = self.queue("independent-self-test", self.worker(command, "independent-valid-self-test"),
                             gpu=True, limit="24:00")
            result.update(status="independent_self_test_submitted_not_passed", job_id=job,
                validation_updates=self.policy["validation_updates"], disposable=True,
                native_checks_required=True, training_callback="none")
            if valid_reference is not None:
                result["valid_reference_test"] = dict(valid_reference)
            return result
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            write_json(path, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--mode", choices=("queue", "cpu", "after-cpu"), required=True)
    args = parser.parse_args()
    campaign = IndependentSelfTest(args.plan, args.plan_sha256)
    if args.mode == "queue":
        print(campaign.queue_initial())
    elif args.mode == "cpu":
        campaign.cpu()
    else:
        print(campaign.after_cpu())


if __name__ == "__main__":
    main()
