"""Independent native studies can be compared only with matched shared inputs."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


def _renderer():
    path = Path(__file__).resolve().parents[1] / "scripts/render_protenix_trajectory.py"
    spec = importlib.util.spec_from_file_location("trajectory_renderer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs(tmp_path, change=None):
    primary = {
        "status": "passed",
        "job_id": "first",
        "config_sha256": "first-config",
        "config": {"mixing_values": [0.15, 0.3], "checks": {"steps": 500}},
        "cases": [{"mixing": 0.15}, {"mixing": 0.3}],
        "fixture_sha256": "fixture",
        "hard_graph_sha256": "graph",
        "model": {"identity": "weights"},
        "numerical_profile": {"precision": "highest"},
        "model_digest_before": "weights",
        "model_digest_after": "weights",
        "context_digest_before": "context",
        "context_digest_after": "context",
    }
    other = deepcopy(primary)
    other.update(job_id="second", config_sha256="second-config")
    other["config"]["mixing_values"] = [0.5]
    other["cases"] = [{"mixing": 0.5}]
    if change == "context":
        other["context_digest_before"] = "different"
    elif change == "steps":
        other["config"]["checks"]["steps"] = 100
    elif change == "duplicate":
        other["cases"][0]["mixing"] = 0.3
        other["config"]["mixing_values"] = [0.3]
    elif change == "failed":
        other["status"] = "failed"
    paths = [tmp_path / "primary", tmp_path / "extra"]
    for path, study in zip(paths, (primary, other), strict=True):
        path.mkdir()
        (path / "summary.json").write_text(json.dumps(study))
    return paths


def test_combined_report_keeps_case_directories_and_source_receipts(tmp_path):
    paths = _inputs(tmp_path)
    combined, directories, receipts = _renderer()._load_studies(paths)
    assert combined["config"]["mixing_values"] == [0.15, 0.3, 0.5]
    assert combined["config_sha256"] is None
    assert directories == [
        paths[0] / "case-0",
        paths[0] / "case-1",
        paths[1] / "case-0",
    ]
    assert [receipt["job_id"] for receipt in receipts] == ["first", "second"]
    assert json.loads((paths[0] / "summary.json").read_text())["config"][
        "mixing_values"
    ] == [0.15, 0.3]


@pytest.mark.parametrize("change", ["context", "steps", "duplicate", "failed"])
def test_combined_report_rejects_unmatched_or_incomplete_studies(tmp_path, change):
    with pytest.raises(ValueError):
        _renderer()._load_studies(_inputs(tmp_path, change))
