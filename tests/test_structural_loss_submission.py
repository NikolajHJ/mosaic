"""Allocated-HPC metadata fixtures; constructing jobs never submits them."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/hpc"))
from submit_structural_loss_diagnostic import SCOPE, validate_plan, workflow


WORK = Path("/work3/user")
RELEASE = WORK / "releases/observer"


def _plan():
    cases = []
    for name in ("boltz_only", "boltz_defog"):
        paths = {}
        helper = "/work3/user/releases/source-097-operation-boundary-recovery-attempt001/src/latentmol/boltz/operation_boundary_diagnostic.py"
        paths[helper] = "a" * 64
        for source in ("source-093-first-suite-attempt001", "source-097-operation-boundary-recovery-attempt001"):
            parent = "/work3/user/runs/" + source + "-replay/case/" + name
            paths[parent + "/attribution.json"] = "b" * 64
            paths[parent + "/run.json"] = "c" * 64
        cases.append({"id": name, "step": 15, "arm": "fixed_initial_mask",
                      "parent_artifacts_sha256": paths})
    return {"version": 1, "cases": cases,
            "output_root": str(WORK / "runs/observer-replay")}


def test_observer_preserves_cpu_dependency_source_verifier_and_gpu_environment():
    cpu, native = workflow(RELEASE, WORK, WORK / "plan.json", "a" * 64, "b" * 64, "c" * 64)
    assert "tests/test_structural_loss_diagnostic.py" in cpu["job_command"]
    assert "tests/test_structural_loss_submission.py" in cpu["job_command"]
    assert "latentmol.boltz.structural_loss_diagnostic" in native["job_command"]
    assert native["dependency"] == "done:cpu"
    assert native["arguments"][native["arguments"].index("-W") + 1] == "2:00"
    assert "select[gpu80gb]" in native["arguments"] and "-gpu" in native["arguments"]
    assert "--archive-sha256" in native["job_command"]
    assert "--manifest-sha256" in native["job_command"]
    assert native["cwd"] == str(RELEASE)
    assert "no production patch" in SCOPE and "historical gate approval" in SCOPE


def test_exact_original_pair_and_source093_replay_are_required():
    validate_plan(_plan(), WORK, RELEASE)
    for field, value in (("step", 20), ("arm", "adaptive_mask"), ("id", "different")):
        plan = _plan()
        plan["cases"][0][field] = value
        with pytest.raises(ValueError, match="exact source091"):
            validate_plan(plan, WORK, RELEASE)
    plan = _plan()
    plan["output_root"] = str(WORK / "runs/old")
    with pytest.raises(ValueError, match="unique replay"):
        validate_plan(plan, WORK, RELEASE)


@pytest.mark.parametrize("missing", ["run.json", "attribution.json"])
def test_missing_completed_source093_evidence_rejected(missing):
    plan = _plan()
    paths = plan["cases"][0]["parent_artifacts_sha256"]
    path = next(path for path in paths if "source-093-" in path and path.endswith(missing))
    del paths[path]
    with pytest.raises(ValueError, match="source093"):
        validate_plan(plan, WORK, RELEASE)


def test_duplicate_source093_case_reference_rejected():
    plan = deepcopy(_plan())
    paths = plan["cases"][0]["parent_artifacts_sha256"]
    duplicate = "/work3/user/runs/source-093-first-suite-attempt001-replay/extra/attribution.json"
    paths[duplicate] = "d" * 64
    paths[str(Path(duplicate).with_name("run.json"))] = "e" * 64
    with pytest.raises(ValueError, match="source093"):
        validate_plan(plan, WORK, RELEASE)
