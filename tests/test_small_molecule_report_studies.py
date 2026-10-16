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


def _mixed_inputs(tmp_path):
    paths = _inputs(tmp_path)
    primary = json.loads((paths[0] / "summary.json").read_text())
    primary["config"].update(seed=17, objective="native_whole_target")
    primary["config"]["checks"]["initial_step_size"] = 0.03
    primary["config"]["mixing_values"].append(0.5)
    primary["cases"].append({"mixing": 0.5})
    uniform = deepcopy(primary)
    uniform.update(job_id="uniform", config_sha256="uniform-config")
    uniform["config"].pop("mixing_values")
    uniform["config"]["initialization"] = {
        "kind": "uniform",
        "bond_existence_numerator": 3.0,
    }
    uniform["cases"] = [
        {"mixing": None, "initialization": uniform["config"]["initialization"]}
    ]
    noisy = deepcopy(uniform)
    noisy.update(job_id="noisy", config_sha256="noisy-config")
    noisy["config"]["initialization"].update(kind="noisy_uniform", noise_weight=0.5)
    paths.append(tmp_path / "noisy")
    paths[-1].mkdir()
    for path, study in zip(paths, (primary, uniform, noisy), strict=True):
        (path / "summary.json").write_text(json.dumps(study))
    return paths


def test_three_study_comparison_preserves_all_five_cases_and_native_receipts(tmp_path):
    paths = _mixed_inputs(tmp_path)
    renderer = _renderer()
    with pytest.raises(ValueError):
        renderer._load_studies(paths)
    combined, directories, receipts = renderer._load_studies(
        paths, allow_initialization_comparison=True
    )
    assert [case["mixing"] for case in combined["cases"]] == [
        0.15,
        0.3,
        0.5,
        None,
        None,
    ]
    assert [case["source_job_id"] for case in combined["cases"]] == ["first"] * 3 + [
        "uniform",
        "noisy",
    ]
    assert combined["config"]["initialization_comparison"]
    assert "mixing_values" not in combined["config"]
    assert combined["config_sha256"] is None
    assert directories == [paths[0] / f"case-{i}" for i in range(3)] + [
        paths[1] / "case-0",
        paths[2] / "case-0",
    ]
    assert [receipt["config_sha256"] for receipt in receipts] == [
        "first-config",
        "uniform-config",
        "noisy-config",
    ]
    assert len(set(renderer.CASE_COLORS)) >= 5
    assert json.loads((paths[0] / "summary.json").read_text())["config"][
        "mixing_values"
    ] == [0.15, 0.3, 0.5]


@pytest.mark.parametrize(
    "change", ["seed", "rate", "context", "failed", "duplicate", "saved_recipe"]
)
def test_initialization_comparison_keeps_shared_input_and_receipt_guards(
    tmp_path, change
):
    paths = _mixed_inputs(tmp_path)
    path = paths[-1] / "summary.json"
    study = json.loads(path.read_text())
    if change == "seed":
        study["config"]["seed"] = 99
    elif change == "rate":
        study["config"]["checks"]["initial_step_size"] = 0.09
    elif change == "context":
        study["context_digest_before"] = study["context_digest_after"] = "different"
    elif change == "failed":
        study["status"] = "failed"
    elif change == "duplicate":
        study["config"]["initialization"] = {
            "kind": "uniform",
            "bond_existence_numerator": 3.0,
        }
        study["cases"][0]["initialization"] = study["config"]["initialization"]
    else:
        study["cases"][0]["initialization"]["noise_weight"] = 0.1
    path.write_text(json.dumps(study))
    with pytest.raises(ValueError):
        _renderer()._load_studies(paths, allow_initialization_comparison=True)


def test_renderer_builds_five_case_html_from_synthetic_saved_graphs(
    tmp_path, monkeypatch
):
    """Engineering rendering only: fabricated losses establish no model result."""
    import hashlib
    import numpy as np

    paths = _mixed_inputs(tmp_path)
    for path in paths:
        summary_path = path / "summary.json"
        study = json.loads(summary_path.read_text())
        fixture = {
            "smiles": "CC",
            "protein_sequence": "ACDE",
            "protein_name": "SYNTHETIC RENDERING TEST; NO NATIVE MODEL",
            "contact_partner_scope": "whole_target",
        }
        fixture_bytes = json.dumps(fixture).encode()
        (path / "fixture.json").write_bytes(fixture_bytes)
        study.update(fixture_file="fixture.json", pocket_token_indices=[])
        study["fixture_sha256"] = hashlib.sha256(fixture_bytes).hexdigest()
        study["config"].update(fixture="unused", contact_cutoff_angstrom=8.0)
        study["config"]["checks"].update(steps=1, max_backtracks=10)
        for index, case in enumerate(study["cases"]):
            directory = path / f"case-{index}"
            directory.mkdir()
            case["label"] = case.get("initialization", {}).get(
                "kind", f"Chain {case['mixing']}"
            )
            case["states"] = []
            for step in (0, 1):
                X = np.full((2, 6), 1 / 6)
                E = np.array([[0.5, 0.125, 0.125, 0.125, 0.125]])
                np.savez(directory / f"step-{step}.npz", X=X, E_upper=E)
                case["states"].append(
                    {
                        "step": step,
                        "snapshot": f"step-{step}.npz",
                        "primal_loss": -0.1 - 0.01 * step,
                        "autodiff_loss": -0.1 - 0.01 * step,
                        "readout": {
                            "sanitized": False,
                            "canonical_smiles": None,
                            "conversion_error": "synthetic fixture",
                            "fragment_count": 2,
                        },
                    }
                )
        summary_path.write_text(json.dumps(study))
    monkeypatch.setenv("LSB_JOBID", "synthetic-render-test")
    import sys

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    assert sys.path[0].endswith("scripts")
    output = tmp_path / "report"
    _renderer().main(
        [
            "--output",
            str(output),
            "--allow-initialization-comparison",
            *[arg for path in paths for arg in ("--study", str(path))],
        ]
    )
    analysis = json.loads((output / "analysis_summary.json").read_text())
    assert len(analysis["cases"]) == 5
    assert len(analysis["input_studies"]) == 3
    assert analysis["initialization_comparison"]
    assert analysis["initial_smiles"] is None
    html = (output / "report.html").read_text(encoding="utf-8")
    assert "__DATA__" not in html and "__PROBABILITY_IMAGES__" not in html
    payload = json.loads(html.split("const DATA=", 1)[1].split(";\n", 1)[0])
    assert len({case["color"] for case in payload["cases"]}) == 5
    assert all(len(case["states"]) == 2 for case in payload["cases"])


def test_four_study_comparison_includes_canonical_recipe_without_changing_size(
    tmp_path,
):
    paths = _mixed_inputs(tmp_path)
    study = json.loads((paths[-1] / "summary.json").read_text())
    config_path = (
        Path(__file__).resolve().parents[1]
        / "examples/configs/brd4_canonical_diffuse_5000.json"
    )
    recipe = json.loads(config_path.read_text())["initialization"]
    study.update(job_id="canonical", config_sha256="canonical-config")
    study["config"]["initialization"] = recipe
    study["cases"] = [{"mixing": None, "initialization": recipe}]
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    (canonical / "summary.json").write_text(json.dumps(study))
    combined, directories, receipts = _renderer()._load_studies(
        paths + [canonical], allow_initialization_comparison=True
    )
    assert len(combined["cases"]) == 6
    assert len(receipts) == 4
    assert combined["cases"][-1]["initialization"] == recipe
    assert combined["cases"][-1]["source_job_id"] == "canonical"
    assert directories[-1] == canonical / "case-0"
    assert len(set(_renderer().CASE_COLORS)) == 6
