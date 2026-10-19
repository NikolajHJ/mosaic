"""Allocated saved-only report fixtures, with failures kept distinct."""

import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

from test_small_molecule_chemistry_report import fixture_study


def setup_pair(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "ramp_renderer", scripts / "render_chemistry_ramp.py"
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
