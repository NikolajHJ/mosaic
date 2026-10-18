"""Allocated presentation fixtures: saved-only rendering and identity failures."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


def renderer():
    path = (
        Path(__file__).resolve().parents[1] / "scripts/render_chemistry_trajectory.py"
    )
    spec = importlib.util.spec_from_file_location("chemistry_saved_renderer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_study(tmp_path):
    root = tmp_path / "study"
    (root / "case-0/snapshots").mkdir(parents=True)
    old = tmp_path / "historical"
    old.mkdir()
    states = []
    components = {
        k: 0.1
        for k in (
            "valid_full_nll",
            "extra_valid_nll",
            "partial_edited_nll",
            "local_bond_risk",
            "component_cut_risk",
            "nonempty_risk",
            "element_js",
        )
    }
    for step in range(2):
        path = root / f"case-0/snapshots/step-{step:04d}.npz"
        # Explicit invalid F graph; rendering must retain the raw labels.
        x = np.full((6, 6), 0.02)
        x[:, 4] = 0.9
        e = np.full((15, 5), 0.01)
        e[:, 3] = 0.96
        np.savez_compressed(path, X=x, E_upper=e)
        states.append(
            {
                "step": step,
                "snapshot": f"snapshots/step-{step:04d}.npz",
                "snapshot_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "readout": {"sanitized": False, "canonical_smiles": None},
                "contact_loss": -0.2 - step * 0.01,
                "components": components,
                "refreshed_surrogate": 0.1,
                "preparation_identity": "synthetic-reference",
                "chemistry_preparation": [
                    {
                        "raw_native_valid": False,
                        "original_status": "censored",
                        "extra_status": "partial",
                    }
                ],
            }
        )
    historical = {
        "status": "passed",
        "job_id": "29745503",
        "source_case_indices": [2],
        "model_digest_before": "model",
        "model_digest_after": "model",
        "context_digest_before": "context",
        "context_digest_after": "context",
        "cases": [
            {
                "states": [
                    {"step": i, "primal_loss": -0.2 - i * 0.00001} for i in range(5001)
                ]
            }
        ],
    }
    old_path = old / "summary.json"
    old_path.write_text(json.dumps(historical))
    summary = {
        "status": "passed",
        "job_id": "synthetic-cpu-fixture",
        "model_digest_before": "model",
        "model_digest_after": "model",
        "context_digest_before": "context",
        "context_digest_after": "context",
        "config": {
            "steps": 1,
            "model_digest": "model",
            "context_digest": "context",
            "historical_study": str(old),
            "historical_summary_sha256": hashlib.sha256(
                old_path.read_bytes()
            ).hexdigest(),
        },
    }
    case = {
        "status": "passed",
        "states": states,
        "counts": {"applied": 1},
        "updates": [
            {"step": 0, "fixed_before": 0.1, "fixed_after": 0.11, "fallback_used": True}
        ],
    }
    configuration = root / "configuration.json"
    configuration.write_text(json.dumps(summary["config"]))
    summary["config_sha256"] = hashlib.sha256(configuration.read_bytes()).hexdigest()
    summary["cases"] = [case]
    (root / "summary.json").write_text(json.dumps(summary))
    (root / "case-0/report.json").write_text(json.dumps(case))
    return root


def test_saved_raw_probability_viewer_and_separate_historical_scalar_overlay(
    tmp_path, monkeypatch
):
    root = fixture_study(tmp_path)
    output = tmp_path / "report"
    monkeypatch.setenv("LSB_JOBID", "allocated-presentation-fixture")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render",
            "--study",
            str(root),
            "--output",
            str(output),
            "--config",
            str(root / "configuration.json"),
        ],
    )
    renderer().main()
    html = (output / "report.html").read_text()
    raw = html.split('<script id="saved-data" type="application/json">')[1].split(
        "</script>"
    )[0]
    data = json.loads(raw)
    assert len(data["studies"]) == 1
    assert len(data["studies"][0]["frames"]) == 2
    frame = data["studies"][0]["frames"][0]
    assert np.argmax(frame["X"], -1).tolist() == [4] * 6
    assert not frame["readout"]["sanitized"]
    assert len(data["historical_contact"]["frames"]) == 5001
    assert "X" not in data["historical_contact"]["frames"][0]
    assert "data:image/svg+xml;base64," in html and "__LOSS_PLOT_BASE64__" not in html
    assert "strict/fallback" in data["limitations"] or "fallback" in data["limitations"]
    receipt = json.loads((output / "report_receipt.json").read_text())
    assert receipt["states"] == [2] and receipt["complete"]
    assert (
        hashlib.sha256((output / "report.html").read_bytes()).hexdigest()
        == receipt["files"]["report.html"]["sha256"]
    )


def test_renderer_does_not_waive_native_identity_even_for_partial_output(
    tmp_path, monkeypatch
):
    root = fixture_study(tmp_path)
    summary = json.loads((root / "summary.json").read_text())
    summary["model_digest_after"] = "changed"
    (root / "summary.json").write_text(json.dumps(summary))
    monkeypatch.setenv("LSB_JOBID", "allocated-presentation-fixture")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render",
            "--study",
            str(root),
            "--output",
            str(tmp_path / "report"),
            "--config",
            str(root / "configuration.json"),
            "--allow-partial",
        ],
    )
    with pytest.raises(ValueError, match="identity"):
        renderer().main()


def test_missing_saved_probability_state_is_not_invented(tmp_path, monkeypatch):
    root = fixture_study(tmp_path)
    (root / "case-0/snapshots/step-0001.npz").unlink()
    monkeypatch.setenv("LSB_JOBID", "allocated-presentation-fixture")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render",
            "--study",
            str(root),
            "--output",
            str(tmp_path / "report"),
            "--config",
            str(root / "configuration.json"),
        ],
    )
    with pytest.raises(FileNotFoundError):
        renderer().main()
