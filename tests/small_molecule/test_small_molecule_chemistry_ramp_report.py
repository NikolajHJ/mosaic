"""Allocated saved-only report fixtures, with failures kept distinct."""

# Shared test import paths; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[2] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__, include_tests=True)

import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

from test_small_molecule_chemistry_report import fixture_study


def setup_pair(tmp_path, monkeypatch):
    scripts = _layout_path(_repository_root(__file__) / "scripts")
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "ramp_renderer", _layout_path(scripts / "render_chemistry_ramp.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = fixture_study(tmp_path)
    config = json.loads((root / "configuration.json").read_text())
    config.update(
        label="synthetic saved fixture",
        repair_schedule={
            "initial": 0.001,
            "final": 50.0,
            "hold_updates": 0,
            "total_updates": 1,
            "version": "delayed-geometric-v1",
        },
    )
    (root / "configuration.json").write_text(json.dumps(config))
    case = json.loads((root / "case-0/report.json").read_text())
    case["repair_weight_schedule"] = config["repair_schedule"]
    for state in case["states"]:
        state["repair_weight"] = 50.0
        state["components"]["chemistry_loss"] = 0.1
        path = root / "case-0" / state["snapshot"]
        with np.load(path) as saved:
            arrays = dict(saved)
        np.savez_compressed(path, **arrays, repair_weight=np.asarray(50.0))
        state["snapshot_sha256"] = module.sha(path)
    for update in case["updates"]:
        update.update(repair_weight=50.0, attempts=[{"repair_weight": 50.0}])
    (root / "case-0/report.json").write_text(json.dumps(case))
    summary = json.loads((root / "summary.json").read_text())
    summary.update(
        config=config,
        config_sha256=module.sha(root / "configuration.json"),
        cases=[case],
        initialization={"recipe": {"kind": "fixture"}},
    )
    for field in (
        "native_contact_checks",
        "coupled_chemistry_probe",
        "high_weight_chemistry_probe",
        "pilot",
        "high_weight_pilot",
    ):
        summary[field] = {"status": "passed"}
    (root / "summary.json").write_text(json.dumps(summary))
    other = tmp_path / "study2"
    shutil.copytree(root, other)
    monkeypatch.setenv("LSB_JOBID", "allocated-report-fixture")
    output = tmp_path / "report"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render",
            "--study",
            str(root),
            "--study",
            str(other),
            "--config",
            str(root / "configuration.json"),
            "--config",
            str(other / "configuration.json"),
            "--output",
            str(output),
            "--allow-partial",
        ],
    )
    return module, root, other, output


def test_pair_report_retains_raw_invalid_graphs_and_schedule(tmp_path, monkeypatch):
    module, root, other, output = setup_pair(tmp_path, monkeypatch)
    module.main()
    html = (output / "report.html").read_text()
    data = json.loads(
        html.split('<script id="saved-data" type="application/json">')[1].split(
            "</script>"
        )[0]
    )
    assert data["complete"] and len(data["studies"]) == 2
    assert data["historical_contact"] is None
    assert data["studies"][0]["frames"][0]["repair_weight"] == 50.0
    assert not data["studies"][1]["frames"][0]["readout"]["sanitized"]
    assert np.argmax(data["studies"][0]["frames"][0]["X"], -1).tolist() == [4] * 6
    assert "__LOSS_PLOT_BASE64__" not in html


@pytest.mark.parametrize("tamper", ["trial", "identity", "budget"])
def test_pair_report_fails_closed_on_tampered_gates(tmp_path, monkeypatch, tamper):
    module, root, other, output = setup_pair(tmp_path, monkeypatch)
    summary = json.loads((other / "summary.json").read_text())
    case = json.loads((other / "case-0/report.json").read_text())
    if tamper == "trial":
        case["updates"][0]["attempts"][0]["repair_weight"] = 0.001
        summary["cases"] = [case]
    elif tamper == "identity":
        summary["context_digest_after"] = "changed"
    else:
        summary["high_weight_chemistry_probe"]["status"] = "failed"
    (other / "summary.json").write_text(json.dumps(summary))
    (other / "case-0/report.json").write_text(json.dumps(case))
    with pytest.raises(ValueError):
        module.main()


def test_completed_preview_retains_pending_identity_without_inventing_frames(
    tmp_path, monkeypatch
):
    module, root, other, output = setup_pair(tmp_path, monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render",
            "--study",
            str(root),
            "--config",
            str(root / "configuration.json"),
            "--config",
            str(other / "configuration.json"),
            "--pending-job",
            "29753668",
            "--output",
            str(output),
        ],
    )
    module.main()
    html = (output / "report.html").read_text()
    data = json.loads(
        html.split('<script id="saved-data" type="application/json">')[1].split(
            "</script>"
        )[0]
    )
    assert not data["complete"] and data["included_studies_complete"]
    assert len(data["studies"]) == 1 and len(data["studies"][0]["frames"]) == 2
    assert data["pending_studies"][0]["job_id"] == "29753668"
    assert "frames" not in data["pending_studies"][0]
    assert data["pending_studies"][0]["config_sha256"] == module.sha(
        other / "configuration.json"
    )
    assert (
        "not yet included" in data["status_text"]
        and "FAILED" not in data["status_text"]
    )
    receipt = json.loads((output / "report_receipt.json").read_text())
    assert receipt["input_jobs"] == ["synthetic-cpu-fixture"]
    assert receipt["states"] == [2] and len(receipt["pending_inputs"]) == 1


def test_pending_preview_cannot_hide_a_failed_included_native_gate(
    tmp_path, monkeypatch
):
    module, root, other, output = setup_pair(tmp_path, monkeypatch)
    summary = json.loads((root / "summary.json").read_text())
    summary["high_weight_chemistry_probe"]["status"] = "failed"
    (root / "summary.json").write_text(json.dumps(summary))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render",
            "--study",
            str(root),
            "--config",
            str(root / "configuration.json"),
            "--config",
            str(other / "configuration.json"),
            "--pending-job",
            "29753668",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(ValueError, match="gates"):
        module.main()
