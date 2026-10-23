"""Engineering fixtures for bounded replay orchestration; native probes use HPC."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from latentmol.boltz.gate_replay_study import load_protocol


def protocol(tmp_path):
    return {"version": 1, "name": "saved-failure-replays", "output_root": str(tmp_path / "outputs"),
            "timeout_seconds": 900, "cases": [
                {"id": "random-transition", "parent_run": str(tmp_path / "parent"),
                 "production_release": str(tmp_path / "production"), "arm": "active_set", "step": 4},
                {"id": "panel-control", "parent_run": str(tmp_path / "other-parent"),
                 "production_release": str(tmp_path / "other-production"), "arm": "fixed_initial_mask", "step": 0}]}


def write_protocol(tmp_path, record):
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(record))
    return path


def test_explicit_parent_arm_step_and_bounded_budget_are_retained(tmp_path):
    record = protocol(tmp_path)
    assert load_protocol(write_protocol(tmp_path, record)) == record


@pytest.mark.parametrize("field,value", [("timeout_seconds", True), ("timeout_seconds", 0),
    ("timeout_seconds", 3601), ("cases", []), ("version", True), ("output_root", "relative")])
def test_invalid_budget_or_protocol_is_rejected(tmp_path, field, value):
    record = protocol(tmp_path)
    record[field] = value
    with pytest.raises(ValueError):
        load_protocol(write_protocol(tmp_path, record))


@pytest.mark.parametrize("field,value", [("id", "../escape"), ("step", True), ("step", -1),
    ("arm", "guessed"), ("parent_run", "relative"), ("production_release", "")])
def test_case_cannot_guess_or_escape_saved_state(tmp_path, field, value):
    record = protocol(tmp_path)
    record["cases"][0][field] = value
    with pytest.raises(ValueError):
        load_protocol(write_protocol(tmp_path, record))


def test_duplicate_case_and_overlong_list_are_rejected(tmp_path):
    record = protocol(tmp_path)
    record["cases"].append(deepcopy(record["cases"][0]))
    with pytest.raises(ValueError):
        load_protocol(write_protocol(tmp_path, record))
    record = protocol(tmp_path)
    record["cases"] = [{**record["cases"][0], "id": f"case-{index}"} for index in range(17)]
    with pytest.raises(ValueError):
        load_protocol(write_protocol(tmp_path, record))


def test_bootstrap_rejects_modified_production_before_import(tmp_path):
    launcher = Path(__file__).resolve().parents[1] / "scripts/diagnose_saved_boltz_gate.py"
    spec = importlib.util.spec_from_file_location("_launcher_fixture", launcher)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parent, original = tmp_path / "parent", tmp_path / "original"
    parent.mkdir()
    (original / "src").mkdir(parents=True)
    (original / "src/changed.py").write_text("# changed file\n")
    (parent / "run.json").write_text(json.dumps({"source": {"files": {"src/changed.py": "0" * 64}}}))
    with pytest.raises(ValueError, match="Original production file changed"):
        module.bootstrap(original, parent)
