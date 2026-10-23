"""CPU contracts followed by a small disposable epoch-36 mechanism probe.

This controller intentionally has no training/promotion action. A successful
diagnostic is evidence for choosing a fix, never a replacement for the native
readiness or the required 2,500-update stability proof. Historical stopped and
failed campaigns are read only; all outputs belong to this new sealed plan.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex

from chemistry_guarded_restart import GuardedRestart
from chemistry_campaign import authenticated, read
from latentmol.experiments import write_json


RECOVERY_FIXTURES = ["tests/test_chemistry_classification_pool.py",
                     "tests/test_chemistry_classification_validation.py",
                     "tests/test_chemistry_recovery_diagnostic.py",
                     "tests/test_chemistry_recovery_controller.py"]


class RecoveryDiagnostic(GuardedRestart):
    """Reuse exact source, quota, CPU, scheduler and predecessor contracts."""

    def controller(self, mode, **values):
        if values or mode not in ("cpu", "after-cpu"):
            raise ValueError("A diagnostic controller cannot promote training")
        return ["env", "PYTHONPATH=" + str(self.release / "src"),
                str(self.work / "env/boltz311/bin/python"),
                "scripts/hpc/chemistry_recovery_diagnostic.py", "--plan", str(self.path),
                "--plan-sha256", self.sha, "--mode", mode]

    def cpu_contract_command(self, executable):
        return [*super().cpu_contract_command(executable), *RECOVERY_FIXTURES]

    def queue_initial(self):
        result = super().queue_initial()
        result.pop("native_stability", None)
        result.update(diagnostic="not_yet_submitted", training="not_submitted")
        return result

    def advance(self, after_stage):
        raise ValueError("Mechanism diagnostics cannot submit a continuation")

    def after_cpu(self):
        path = self.root / "after-cpu-decision.json"
        if path.exists():
            value = read(path)
            if value.get("identity") != self.identity:
                raise ValueError("Saved diagnostic decision has a different identity")
            return value
        # Exclusive intent prevents duplicate submission after a lost SSH reply.
        with (self.root / "after-cpu-intent.json").open("x") as stream:
            import json
            json.dump(dict(identity=self.identity, action="disposable_diagnostic_only"), stream)
        result = dict(status="stopped", identity=self.identity, training="not_submitted")
        try:
            self.require_cpu()
            self.require_predecessor_stopped()
            reference = self.plan["loss_v1_reference"]
            authenticated(reference["module"], reference["sha256"])
            cuda = ["env", "LATENTMOL_REQUIRE_CUDA_RESUME_FIXTURES=1",
                    "LATENTMOL_REQUIRE_REPAIR_CUDA_FIXTURES=1",
                    "LATENTMOL_FAILURE_REFERENCE_MODULE=" + reference["module"],
                    "python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--junitxml",
                    str(self.root / "recovery-cuda-pytest.xml"),
                    "tests/test_chemistry_resume_cuda.py", "tests/test_failure_loss_validation_reuse.py",
                    "tests/test_repair_guidance_v2.py", "tests/test_chemistry_recovery_diagnostic.py"]
            probe = ["python", "-m", "latentmol.chemistry_recovery_diagnostic",
                     "--plan", str(self.path), "--plan-sha256", self.sha]
            command = ["bash", "-c", shlex.join(cuda) + " && " + shlex.join(probe)]
            job = self.queue("recovery-diagnostic", self.worker(command, "chemistry-recovery-diagnostic"),
                             gpu=True, limit="1:00")
            result.update(status="diagnostic_submitted_not_passed", job_id=job,
                          disposable=True, training="not_submitted")
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
    campaign = RecoveryDiagnostic(args.plan, args.plan_sha256)
    if args.mode == "queue":
        print(campaign.queue_initial())
    elif args.mode == "cpu":
        campaign.cpu()
    else:
        print(campaign.after_cpu())


if __name__ == "__main__":
    main()
