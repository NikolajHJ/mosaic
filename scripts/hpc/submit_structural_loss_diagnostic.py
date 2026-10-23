"""Queue exact-state contact/confidence observers after allocated CPU fixtures.

This uses the existing immutable-source verifier and durable LSF intents. The
native allocation observes the two source091 step-15 states serially, without
optimization, new DeFoG calls, a production patch or historical gate approval.
Cached precision and held-selection variants are diagnostic counterfactuals.
"""
import argparse
import json
from pathlib import Path

from latentmol.experiments import source_manifest
from submit_stability_study import authenticated, read, submit
from submit_failed_gate_attribution import workflow as original_workflow
from submit_affinity_head_diagnostic import validate_plan as validate_saved_pair


FIXTURES = [
    "tests/test_structural_loss_diagnostic.py",
    "tests/test_structural_loss_submission.py",
    "tests/test_dbmol_objective.py",
    "tests/test_dbmol_precision.py",
    "tests/test_dbmol_model.py",
    "tests/test_failed_gate_attribution.py",
    "tests/test_boltz_local_fd.py",
    "tests/test_saved_state_submission.py",
]
SCOPE = (
    "Exact source091 pair: native structural frontier/VJP and cached contact/"
    "confidence reduction precision/selection controls; no production patch, "
    "optimizer updates, new DeFoG sampling or historical gate approval"
)


def workflow(release, work, plan, plan_sha, archive_sha, manifest_sha):
    """Reuse authenticated CPU→GPU execution, changing only observer selection."""
    cpu, native = original_workflow(release, work, plan, plan_sha, archive_sha, manifest_sha)
    index = cpu["job_command"].index("pytest")
    cpu["job_command"][index + 2:] = FIXTURES
    native["arguments"][native["arguments"].index("-W") + 1] = "2:00"
    index = native["job_command"].index("latentmol.boltz.failed_gate_attribution")
    native["job_command"][index] = "latentmol.boltz.structural_loss_diagnostic"
    return [cpu, native]


def validate_plan(plan, work, release):
    """Require the original pair plus completed, authenticated source093 replay."""
    validate_saved_pair(plan, work, release)
    for case in plan["cases"]:
        paths = case["parent_artifacts_sha256"]
        references = [path for path in paths
            if "source-093-first-suite-attempt001-replay" in path
            and Path(path).name == "attribution.json"]
        if len(references) != 1 or str(Path(references[0]).with_name("run.json")) not in paths:
            raise ValueError("Require unique authenticated source093 scalar/gradient replay and run identity")


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
    if any(not path.is_absolute() or not path.resolve().is_relative_to(work)
           for path in (args.plan, args.receipts)):
        raise ValueError("Plan and receipts must stay under the declared work root")
    authenticated(Path(str(release) + ".tar.gz"), args.archive_sha256)
    authenticated(args.plan, args.plan_sha256)
    if (source_manifest(release)["sha256"] != args.manifest_sha256
            or read(release / "release_manifest.json")["sha256"] != args.manifest_sha256):
        raise ValueError("Source changed after sealing")
    validate_plan(read(args.plan), work, release)
    identity = {"release": str(release), "archive_sha256": args.archive_sha256,
        "manifest_sha256": args.manifest_sha256, "plan": str(args.plan), "plan_sha256": args.plan_sha256,
        "diagnostic": "structural_loss_diagnostic", "scope": SCOPE}
    jobs = workflow(release, work, args.plan, args.plan_sha256, args.archive_sha256, args.manifest_sha256)
    if args.submit:
        (work / "logs").mkdir(exist_ok=True)
        print(json.dumps(submit(jobs, args.receipts, identity), indent=2))
    else:
        print(json.dumps({"identity": identity, "plan": jobs, "status": "not_submitted"}, indent=2))


if __name__ == "__main__":
    main()
