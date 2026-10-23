"""Validate an authorized runtime amendment, then resume saved VAE training.

Only allocated CPU contract fixtures and authenticated saved evidence precede
submission. No new GPU benchmark is claimed or run. The ordinary serial trainer
still owns whole-epoch, health, provenance, quota and absolute-100-epoch checks.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from chemistry_campaign import Campaign, FIXTURES, OPTIMIZED_FIXTURES, read, quota_receipt
from failure_guidance_campaign import EXTRA_FIXTURES, SPEED_FIXTURES
from latentmol.chemistry_runtime import epoch_fit_policy
from latentmol.experiments import file_sha256, write_json


AMENDMENT_FIXTURES = ["tests/test_chemistry_epoch_fit_amendment.py",
    "tests/test_chemistry_runtime_adoption.py", "tests/test_chemistry_runtime_continuation.py"]


def cpu_command(executable, root):
    """Keep exact-source legacy fixtures; add explicit runtime/adoption guards."""
    return [executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--junitxml",
            str(root / "runtime-amendment-pytest.xml"), *FIXTURES, *OPTIMIZED_FIXTURES,
            *EXTRA_FIXTURES, *SPEED_FIXTURES, *AMENDMENT_FIXTURES]


class RuntimeContinuation(Campaign):
    """One new campaign, reusing immutable evidence and original serial machinery."""

    def __init__(self, path, sha):
        super().__init__(path, sha)
        policy = epoch_fit_policy(self.plan)
        adoption = self.plan.get("runtime_adoption")
        if (policy["factor"] != 1.25 or type(adoption) is not dict
                or adoption.get("version") != "chemistry-runtime-adoption-v1"
                or self.plan.get("failure_variant_id") != "bond6"
                or self.plan["protocol"].get("prior_ratio") != 2
                or self.plan["protocol"].get("failure_guidance", {}).get("progress_search", {}).get("max_edits") != 6):
            raise ValueError("Require the explicitly authorized bond6 runtime-only continuation")

    def controller(self, mode, **values):
        command = [str(self.work / "env/boltz311/bin/python"),
            "scripts/hpc/chemistry_runtime_continuation.py", "--plan", str(self.path),
            "--plan-sha256", self.sha, "--mode", mode]
        for key, value in values.items():
            command += ["--" + key.replace("_", "-"), str(value)]
        return ["env", "PYTHONPATH=" + str(self.release / "src"), *command]

    def queue_initial(self):
        write_json(self.root / "initial_quota.json", quota_receipt(2))
        job = self.queue("cpu", self.worker(self.controller("cpu"), "runtime-amendment-cpu"), limit="1:00")
        after = self.queue("after-cpu", self.controller("after-cpu"), dependency="ended(" + job + ")")
        return dict(status="submitted_not_passed", cpu=job, after_cpu=after,
                    gpu_benchmark="not_rerun_saved_evidence_required", training="not_yet_submitted")

    def cpu(self):
        if not os.environ.get("LSB_JOBID"):
            raise ValueError("Runtime and adoption fixtures require an allocated CPU job")
        # Byte checks only: this path never loads a model or alters old results.
        from latentmol.chemistry_runtime_adoption import materialize_adoption
        materialize_adoption(dict(self.plan, _plan_sha256=self.sha))
        source = self.plan["runtime_adoption"]
        old = read(source["source_plan"])
        suite = read(old["failure_suite_path"])
        reference = suite["execution_reference"]
        environment = dict(os.environ, LATENTMOL_FAILURE_REFERENCE_MODULE=str(
            Path(reference["release"]) / "src/latentmol/repair_failure_guidance.py"))
        command = cpu_command(sys.executable, self.root)
        reply = subprocess.run(command, cwd=self.release, env=environment)
        junit = self.root / "runtime-amendment-pytest.xml"
        gate = dict(status="passed" if reply.returncode == 0 else "failed",
            returncode=reply.returncode, job_id=os.environ["LSB_JOBID"], identity=self.identity,
            command=command, junit_sha256=file_sha256(junit) if junit.exists() else None,
            scope="CPU engineering checks; saved GPU scientific evidence is separately authenticated")
        write_json(self.root / "cpu_gate.json", gate)
        if reply.returncode:
            raise RuntimeError("Runtime amendment/adoption contract fixtures failed")

    def require_cpu(self):
        import xml.etree.ElementTree as ET
        self.complete("cpu")
        gate = read(self.root / "cpu_gate.json")
        expected = cpu_command(str(self.work / "env/boltz311/bin/python"), self.root)
        path = self.root / "runtime-amendment-pytest.xml"
        if (gate.get("status") != "passed" or gate.get("returncode") != 0
                or gate.get("identity") != self.identity or gate.get("job_id") != self.receipt_job("cpu")
                or gate.get("command") != expected or file_sha256(path) != gate.get("junit_sha256")):
            raise ValueError("Runtime amendment CPU checks lack their exact source/job/command")
        suites = list(ET.parse(path).getroot().iter("testsuite"))
        if (not suites or sum(int(row.get("tests", "0")) - int(row.get("skipped", "0")) for row in suites) < 1
                or any(int(row.get(key, "0")) for row in suites for key in ("failures", "errors"))):
            raise ValueError("Runtime amendment CPU checks did not pass")
        from latentmol.chemistry_runtime_adoption import authenticate_adoption
        authenticate_adoption(dict(self.plan, _plan_sha256=self.sha),
                              file_sha256(self.root / "failure_readiness.json"))

    def after_cpu(self):
        self.require_cpu()
        return self.advance("benchmark")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--mode", choices=("queue", "cpu", "after-cpu", "advance"), required=True)
    parser.add_argument("--after-stage")
    args = parser.parse_args()
    campaign = RuntimeContinuation(args.plan, args.plan_sha256)
    if args.mode == "queue":
        print(campaign.queue_initial())
    elif args.mode == "cpu":
        campaign.cpu()
    elif args.mode == "after-cpu":
        print(campaign.after_cpu())
    elif args.after_stage:
        print(campaign.advance(args.after_stage))
    else:
        parser.error("Serial advancement needs its exact preceding segment")


if __name__ == "__main__":
    main()
