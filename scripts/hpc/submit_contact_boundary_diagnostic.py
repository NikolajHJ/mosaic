"""Submit a gated, immutable contact-boundary observer, never optimization."""
import argparse
import json
from pathlib import Path

from latentmol.experiments import source_manifest
from submit_stability_study import authenticated, read, submit
from submit_structural_loss_diagnostic import validate_plan as validate_previous, workflow as previous_workflow

FIXTURES = ["tests/test_contact_boundary_observer.py", "tests/test_contact_boundary_diagnostic.py",
            "tests/test_contact_boundary_submission.py", "tests/test_structural_loss_diagnostic.py",
            "tests/test_structural_loss_submission.py", "tests/test_dbmol_objective.py",
            "tests/test_dbmol_precision.py", "tests/test_failed_gate_attribution.py",
            "tests/test_boltz_local_fd.py"]
SCOPE = ("Exact original contact path and complete frontiers; held input-ReLU and widened final-linear "
         "diagnostic controls only; no production patch, optimizer updates, new DeFoG sampling or historical gate approval")


def workflow(release, work, plan, plan_sha, archive_sha, manifest_sha):
    cpu, native = previous_workflow(release, work, plan, plan_sha, archive_sha, manifest_sha)
    index = cpu["job_command"].index("pytest")
    cpu["job_command"][index + 2:] = FIXTURES
    index = native["job_command"].index("latentmol.boltz.structural_loss_diagnostic")
    native["job_command"][index] = "latentmol.boltz.contact_boundary_diagnostic"
    return [cpu, native]


def validate_plan(plan, work, release):
    validate_previous(plan, work, release)
    for case in plan["cases"]:
        paths = case["parent_artifacts_sha256"]
        matches = [Path(path) for path in paths
                   if "source-102-structural-loss-diagnostic-attempt001-replay" in path
                   and Path(path).name == "attribution.json"]
        if len(matches) != 1 or str(matches[0].with_name("run.json")) not in paths:
            raise ValueError("Require one completed source102 report and allocated run per exact state")


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
        raise ValueError("Require a sealed release under the DTU work root")
    if any(not path.is_absolute() or not path.resolve().is_relative_to(work) for path in (args.plan, args.receipts)):
        raise ValueError("Plan and receipts must stay under the declared work root")
    authenticated(Path(str(release) + ".tar.gz"), args.archive_sha256)
    authenticated(args.plan, args.plan_sha256)
    if (source_manifest(release)["sha256"] != args.manifest_sha256
            or read(release / "release_manifest.json")["sha256"] != args.manifest_sha256):
        raise ValueError("Source changed after sealing")
    validate_plan(read(args.plan), work, release)
    identity = {"release": str(release), "archive_sha256": args.archive_sha256,
        "manifest_sha256": args.manifest_sha256, "plan": str(args.plan), "plan_sha256": args.plan_sha256,
        "diagnostic": "contact_boundary_diagnostic", "scope": SCOPE}
    jobs = workflow(release, work, args.plan, args.plan_sha256, args.archive_sha256, args.manifest_sha256)
    if args.submit:
        (work / "logs").mkdir(exist_ok=True)
        print(json.dumps(submit(jobs, args.receipts, identity), indent=2))
    else:
        print(json.dumps({"identity": identity, "plan": jobs, "status": "not_submitted"}, indent=2))


if __name__ == "__main__":
    main()
