"""Allocated-HPC metadata fixtures; constructing jobs never submits them."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/hpc"))
from submit_contact_boundary_diagnostic import SCOPE, validate_plan, workflow
from test_structural_loss_submission import _plan, WORK, RELEASE


def plan():
    value = _plan()
    for case in value["cases"]:
        parent = str(WORK / "runs/source-102-structural-loss-diagnostic-attempt001-replay" / case["id"])
        case["parent_artifacts_sha256"].update({parent + "/attribution.json": "d" * 64,
                                               parent + "/run.json": "e" * 64})
    return value


def test_preserve_dependency_source_verifier_gpu_and_budget():
    cpu, native = workflow(RELEASE, WORK, WORK / "plan.json", "a" * 64, "b" * 64, "c" * 64)
    assert "tests/test_contact_boundary_observer.py" in cpu["job_command"]
    assert "tests/test_contact_boundary_diagnostic.py" in cpu["job_command"]
    assert "tests/test_contact_boundary_submission.py" in cpu["job_command"]
    assert "latentmol.boltz.contact_boundary_diagnostic" in native["job_command"]
    assert native["dependency"] == "done:cpu"
    assert native["arguments"][native["arguments"].index("-W") + 1] == "2:00"
    assert "select[gpu80gb]" in native["arguments"]
    assert "--archive-sha256" in native["job_command"]
    assert "--manifest-sha256" in native["job_command"]
    assert "no production patch" in SCOPE and "historical gate approval" in SCOPE


@pytest.mark.parametrize("missing", ["run.json", "attribution.json"])
def test_missing_source102_proof_rejected(missing):
    value = plan()
    validate_plan(value, WORK, RELEASE)
    paths = value["cases"][0]["parent_artifacts_sha256"]
    del paths[next(path for path in paths if "source-102-" in path and path.endswith(missing))]
    with pytest.raises(ValueError, match="source102"):
        validate_plan(value, WORK, RELEASE)


def test_duplicate_source102_report_rejected():
    value = deepcopy(plan())
    duplicate = str(WORK / "runs/source-102-structural-loss-diagnostic-attempt001-replay/duplicate/attribution.json")
    value["cases"][0]["parent_artifacts_sha256"][duplicate] = "f" * 64
    with pytest.raises(ValueError, match="source102"):
        validate_plan(value, WORK, RELEASE)
