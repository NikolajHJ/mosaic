"""Queue fixture-gated exact-step15 boundary observation on one A10080GB.

Run from an authenticated sealed release on the DTU login node. The default is
a dry command plan. Submission reuses durable intents/receipts and the existing
allocated source verifier; a failed fixture gate cannot release native work.
No VPN/SSH settings, model parameters or failed production gates are changed.
"""
import argparse
import json
from pathlib import Path

from latentmol.experiments import source_manifest
from submit_stability_study import authenticated, read, submit
from submit_failed_gate_attribution import workflow as original_workflow


FIXTURES = ["tests/test_operation_boundary_diagnostic.py", "tests/test_failed_gate_attribution.py",
            "tests/test_boltz_local_fd.py", "tests/test_saved_state_submission.py"]


def workflow(release, work, plan, plan_sha, archive_sha, manifest_sha):
    """Keep the existing native environment/GPU policy and source verifier."""
    cpu, native = original_workflow(release, work, plan, plan_sha, archive_sha, manifest_sha)
    index = cpu["job_command"].index("pytest")
    cpu["job_command"][index + 2:] = FIXTURES
    index = native["arguments"].index("-W")
    native["arguments"][index + 1] = "2:00"
    index = native["job_command"].index("latentmol.boltz.failed_gate_attribution")
    native["job_command"][index] = "latentmol.boltz.operation_boundary_diagnostic"
    return [cpu, native]


def validate_plan(plan, work, release):
    cases = plan.get("cases", [])
    if (plan.get("version") != 1 or [case.get("id") for case in cases] != ["boltz_only", "boltz_defog"]
            or any(case.get("step") != 15 or case.get("arm") != "fixed_initial_mask" for case in cases)
            or Path(plan["output_root"]) != work / "runs" / (release.name + "-replay")):
        raise ValueError("Require exact source091 step15 pair and unique release replay root")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=Path.cwd())
    parser.add_argument("--work-root", type=Path, default=Path("/work3/s234818/latentmol"))
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    release, work = args.release.resolve(strict=True), args.work_root.resolve(strict=True)
    if not str(work).startswith("/work3/") or not release.is_relative_to(work / "releases"):
        raise ValueError("Require a sealed release under the declared DTU work root")
    if any(not path.is_absolute() or not path.resolve().is_relative_to(work) for path in (args.plan, args.receipts)):
        raise ValueError("Plan and receipts must stay under declared work root")
    authenticated(Path(str(release) + ".tar.gz"), args.archive_sha256)
    authenticated(args.plan, args.plan_sha256)
    if (source_manifest(release)["sha256"] != args.manifest_sha256
            or read(release / "release_manifest.json")["sha256"] != args.manifest_sha256):
        raise ValueError("Source changed after sealing")
    plan = read(args.plan)
    validate_plan(plan, work, release)
    identity = {"release": str(release), "archive_sha256": args.archive_sha256,
        "manifest_sha256": args.manifest_sha256, "plan": str(args.plan), "plan_sha256": args.plan_sha256,
        "diagnostic": "operation_boundary_diagnostic",
        "scope": "Original affinity-path complete-cut observer; no optimizer, DeFoG call, production patch or historical gate approval"}
    jobs = workflow(release, work, args.plan, args.plan_sha256, args.archive_sha256, args.manifest_sha256)
    if args.submit:
        (work / "logs").mkdir(exist_ok=True)
        print(json.dumps(submit(jobs, args.receipts, identity), indent=2))
    else:
        print(json.dumps({"identity": identity, "plan": jobs, "status": "not_submitted"}, indent=2))


if __name__ == "__main__":
    main()
