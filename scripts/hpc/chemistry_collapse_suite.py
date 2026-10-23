"""Serial mechanism replay, guarded baseline proof and independent weak proof.

The mechanism allocation is separate from the 23-hour proof budget. Every GPU
predecessor must be terminal before another GPU is queued. A failed baseline is
retained and does not block the independent disposable weak experiment; it does
block training promotion after that experiment. No diagnostic weights survive.
UNTESTED until these allocated controller contracts pass.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

from chemistry_campaign import authenticated, read, quota_receipt
from chemistry_guarded_restart import GuardedRestart
from chemistry_short_campaign import ShortCampaign
from latentmol.chemistry_optimizer_step import validate_optimizer_step_control
from latentmol.experiments import file_sha256, write_json


FIXTURES = ["tests/test_chemistry_optimizer_step.py", "tests/test_chemistry_collapse_diagnostic.py",
            "tests/test_chemistry_independent_self_test.py", "tests/test_chemistry_collapse_suite.py"]


class CollapseSuite(ShortCampaign):
    """Extend the bounded controller; retain its independent promotion checks."""

    def __init__(self, path, sha):
        super().__init__(path, sha)
        control = validate_optimizer_step_control(self.plan.get("optimizer_step_control"))
        if control is None or self.test.get("optimizer_step_control") != control:
            raise ValueError("Collapse suite requires the same explicit quarter step in both proof arms")
        diagnostic = self.plan.get("collapse_diagnostic")
        if (type(diagnostic) is not dict or diagnostic.get("version") != "epoch36-collapse-finite-step-v1"
                or diagnostic.get("updates") != 100 or diagnostic.get("instrument_updates") != [39, 96]
                or "collapse_diagnostic" in self.test):
            raise ValueError("Only the baseline plan declares the exact original100-update mechanism replay")

    def controller(self, mode, **values):
        command = [str(self.work / "env/boltz311/bin/python"), "scripts/hpc/chemistry_collapse_suite.py",
            "--plan", str(self.path), "--plan-sha256", self.sha, "--mode", mode]
        for key, value in values.items():
            command += ["--" + key.replace("_", "-"), str(value)]
        return ["env", "PYTHONPATH=" + str(self.release / "src"), *command]

    def cpu_contract_command(self, executable):
        command = super().cpu_contract_command(executable)
        return [*command, *(path for path in FIXTURES if path not in command)]

    def _terminal(self, stage):
        """DONE and EXIT establish sequencing, never scientific success."""
        job = self.receipt_job(stage)
        for attempt in range(4):
            command = ["bjobs", "-a", "-noheader", "-o", "stat", job]
            reply = subprocess.run(command, capture_output=True, text=True, timeout=5)
            state = reply.stdout.strip()
            if reply.returncode == 0 and not reply.stderr.strip() and state in ("DONE", "EXIT"):
                return dict(job_id=job, state=state, command=command)
            if attempt == 3 or reply.returncode != 0 or reply.stderr.strip() or state not in ("RUN", "PEND"):
                raise ValueError("GPU predecessor is not confirmed terminal; serial suite is blocked")
            time.sleep(2)

    def _saved(self, path):
        if not path.exists():
            return None
        result = read(path)
        if result.get("identity") != self.identity:
            raise ValueError("Saved collapse-suite decision belongs to another sealed plan")
        return result

    def _intent(self, name):
        with (self.root / (name + "-intent.json")).open("x") as stream:
            json.dump(dict(identity=self.identity, action=name), stream)

    def _proof_command(self, path, sha, *, weak):
        reference = self.plan["loss_v1_reference"]
        authenticated(reference["module"], reference["sha256"])
        root = Path(self.test["campaign"]) if weak else self.root
        cuda = ["env", "LATENTMOL_REQUIRE_CUDA_RESUME_FIXTURES=1",
            "LATENTMOL_REQUIRE_REPAIR_CUDA_FIXTURES=1",
            "LATENTMOL_FAILURE_REFERENCE_MODULE=" + reference["module"],
            "python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--junitxml",
            str(root / ("self-cuda-pytest.xml" if weak else "guarded-cuda-pytest.xml")),
            "tests/test_chemistry_resume_cuda.py", "tests/test_failure_loss_validation_reuse.py",
            "tests/test_repair_guidance_v2.py", "tests/test_chemistry_optimizer_step.py",
            "tests/test_chemistry_valid_self_guidance.py", "tests/test_chemistry_valid_self_training.py",
            "tests/test_chemistry_generation_comparison.py"]
        if self.plan.get("worker_startup") is not None:
            cuda += ["tests/test_chemistry_worker_startup.py", "tests/test_chemistry_recovery_diagnostic.py"]
        native = ["python", "-m", "latentmol.chemistry_guarded_validation", "--plan", str(path),
                  "--plan-sha256", sha]
        return ["bash", "-c", shlex.join(cuda) + " && " + shlex.join(native)]

    def after_cpu(self):
        path = self.root / "after-cpu-decision.json"
        saved = self._saved(path)
        if saved is not None:
            return saved
        self._intent("after-cpu")
        result = dict(status="stopped", identity=self.identity, training="not_yet_submitted")
        try:
            self.require_cpu(); self.require_predecessor_stopped()
            if (self.root / "active-phase.json").exists():
                raise ValueError("A phase lock blocks the disposable mechanism allocation")
            write_json(self.root / "quota-before-mechanism.json", quota_receipt(2))
            # This replay intentionally uses original full AdamW updates. The
            # amended quarter step is evaluated separately by the later proof.
            cuda = ["env", "LATENTMOL_REQUIRE_REPAIR_CUDA_FIXTURES=1", "python", "-m", "pytest", "-q",
                    "-p", "no:cacheprovider", "--junitxml", str(self.root / "mechanism-cuda-pytest.xml"),
                    "tests/test_chemistry_collapse_diagnostic.py", "tests/test_chemistry_optimizer_step.py"]
            diagnostic = ["python", "-m", "latentmol.chemistry_collapse_diagnostic",
                          "--plan", str(self.path), "--plan-sha256", self.sha]
            command = ["bash", "-c", shlex.join(cuda) + " && " + shlex.join(diagnostic)]
            job = self.queue("collapse-mechanism", self.worker(command, "chemistry-collapse-mechanism"),
                             gpu=True, limit="2:00")
            callback = self.queue("after-mechanism", self.controller("after-mechanism"),
                                  dependency="ended(" + job + ")")
            result.update(status="mechanism_submitted_not_completed", job_id=job,
                callback_job_id=callback, mechanism_seconds=6600, disposable=True)
            return result
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            write_json(path, result)

    def _mechanism_result(self, terminal):
        path = self.root / "collapse_diagnostic.json"
        report = read(path)
        run_path = Path(report.get("run_dir", "")).resolve(strict=True)
        if not run_path.is_relative_to((self.root / "collapse_diagnostic").resolve(strict=True)):
            raise ValueError("Mechanism run escaped its sealed campaign")
        run = read(run_path / "run.json")
        if (report.get("version") != "epoch36-collapse-finite-step-v1"
                or any(report.get(key) != value for key, value in self.identity.items() if key != "parent_sha256")
                or report.get("job_id") != terminal["job_id"]
                or run.get("environment", {}).get("job_id") != terminal["job_id"]
                or run.get("source_archive_sha256") != self.plan["archive_sha256"]
                or run.get("source", {}).get("sha256") != self.plan["manifest_sha256"]
                or report.get("policy") != self.plan["collapse_diagnostic"]
                or type(report.get("training_updates_promoted")) is not int
                or report["training_updates_promoted"] != 0
                or report.get("promotable") is not False or report.get("epoch_complete") is not False
                or report.get("status") not in ("failed", "completed_disposable_not_training")
                or file_sha256(run_path / "diagnostic.json") != file_sha256(path)):
            raise ValueError("Mechanism result lost its original source, job or disposable identity")
        passed = (terminal["state"] == "DONE" and run.get("status") == "completed"
            and report.get("status") == "completed_disposable_not_training"
            and report.get("diagnostic_completed") is True and report.get("restoration", {}).get("passed") is True
            and len(report.get("updates", [])) == 100
            and [row.get("relative_update") for row in report["updates"]] == list(range(1, 101))
            and all(row.get("historical_receipt_matches") is True for row in report["updates"])
            and [row.get("relative_update") for row in report.get("spikes", [])] == [39, 96])
        return dict(path=str(path), sha256=file_sha256(path), status=report["status"], passed=passed)

    def after_mechanism(self):
        path = self.root / "after-mechanism-decision.json"
        saved = self._saved(path)
        if saved is not None:
            return saved
        self._intent("after-mechanism")
        result = dict(status="stopped", identity=self.identity, training="not_yet_submitted")
        try:
            self.require_cpu(); self.require_predecessor_stopped()
            terminal = self._terminal("collapse-mechanism")
            result.update(mechanism_terminal=terminal, mechanism=self._mechanism_result(terminal))
            if not result["mechanism"]["passed"]:
                # A failed mechanism proof cannot start baseline training. Its
                # separately restored weak experiment still has its own gates.
                result.update(status="mechanism_failed_weak_test_only", weak=self._queue_weak("mechanism_failed"))
                return result
            if (self.root / "active-phase.json").exists():
                raise ValueError("A phase lock blocks the corrected baseline proof")
            write_json(self.root / "quota-before-proof.json", quota_receipt(2))
            command = self._proof_command(self.path, self.sha, weak=False)
            job = self.queue("benchmark", self.worker(command, "guarded-quarter-step-proof"),
                             gpu=True, limit="24:00")
            callback = self.queue("after-benchmark", self.controller("advance", after_stage="benchmark"),
                                  dependency="ended(" + job + ")")
            result.update(status="native_stability_submitted_not_passed", job_id=job,
                callback_job_id=callback, validation_updates=self.policy["validation_updates"])
            return result
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            write_json(path, result)

    def _queue_weak(self, reason):
        if (self.root / "active-phase.json").exists():
            raise ValueError("A phase lock blocks the independent weak proof")
        authenticated(self.test_path, self.test_sha)
        write_json(self.root / "quota-before-self-test.json", quota_receipt(2))
        test_root = Path(self.test["campaign"])
        test_root.mkdir(parents=True, exist_ok=False)
        command = self._proof_command(self.test_path, self.test_sha, weak=True)
        job = self.queue("self-test", self.worker(command, "independent-quarter-step-valid-self-proof"),
                         gpu=True, limit="24:00")
        callback = self.queue("after-self-test", self.controller("after-self-test"), dependency="ended(" + job + ")")
        return dict(status="independent_self_test_submitted_not_passed", job_id=job,
            callback_job_id=callback, reason=reason, self_test_plan_sha256=self.test_sha,
            baseline_success_required=False, test_weights_disposable=True, weak_training="forbidden")

    def advance(self, after_stage):
        if after_stage != "benchmark":
            return super().advance(after_stage)
        path = self.root / "self-test-decision.json"
        saved = self._saved(path)
        if saved is not None:
            return saved
        self._intent("self-test")
        result = dict(status="stopped", identity=self.identity, training="not_yet_submitted")
        try:
            self.require_cpu(); self.require_predecessor_stopped()
            result["baseline_terminal"] = self._terminal("benchmark")
            result.update(self._queue_weak("baseline_terminal_independent_of_scientific_success"))
            return result
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            write_json(path, result)

    # ShortCampaign.after_self_test authenticates terminal weak execution and
    # calls GuardedRestart.advance, which independently reauthenticates every
    # baseline native/stability gate. Failed baselines therefore never train.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--mode", choices=("queue", "cpu", "after-cpu", "after-mechanism", "advance", "after-self-test"), required=True)
    parser.add_argument("--after-stage")
    args = parser.parse_args()
    campaign = CollapseSuite(args.plan, args.plan_sha256)
    if args.mode == "queue":
        print(campaign.queue_initial())
    elif args.mode == "cpu":
        campaign.cpu()
    elif args.mode == "after-cpu":
        print(campaign.after_cpu())
    elif args.mode == "after-mechanism":
        print(campaign.after_mechanism())
    elif args.mode == "after-self-test":
        print(campaign.after_self_test())
    elif args.after_stage:
        print(campaign.advance(args.after_stage))
    else:
        parser.error("Advancement requires its exact predecessor")


if __name__ == "__main__":
    main()
