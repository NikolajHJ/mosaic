"""HPC-only engineering fixtures for campaign identity, factors and bookkeeping."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch
import yaml

from latentmol.boltz import active_set, stability_study as study
from latentmol.config_identity import config_sha256
from latentmol.experiments import file_sha256
from latentmol.optimization import StepControl
from test_boltz_active_set_runtime import _config, _Decoder, _Scorer, _invoke, _yaml_configuration


def ref(path):
    return {"path": str(path), "sha256": file_sha256(path)}


def protocol(tmp_path):
    raw = _yaml_configuration()
    raw["contexts"] = {"1": {"path": str(tmp_path / "context.pt"), "sha256": "b" * 64}}
    # Historical inputs are resolved ExperimentRun configs, not minimal YAML.
    # Exercise the real parser so diagnostics/objective/start defaults and
    # absolute paths match production records rather than guessed fixture data.
    configuration = tmp_path / "historical-configuration.yaml"
    configuration.write_text(yaml.safe_dump(raw))
    raw = active_set.ActiveSetConfig.load(configuration).record()
    parent = tmp_path / "parent.json"
    parent.write_text(json.dumps({"status": "failed", "config": raw, "config_sha256": config_sha256(raw)}))
    initial = tmp_path / "initial.pt"
    torch.save({"z": torch.zeros(1, 1)}, initial)
    template = tmp_path / "native.yaml"
    template.write_text("version: 1\n")
    dummy = tmp_path / "dummy.json"
    dummy.write_text("{}")
    return {"version": 1, "name": "fixture-stability", "output_root": str(tmp_path / "campaign"),
            "cache": str(tmp_path / "cache"), "confidence_checkpoint": ref(dummy),
            "step_comparison": {"run": ref(dummy), "cases": ref(dummy), "reports": [ref(dummy)] * 12},
            "starts": [{"id": "prior-4", "parent_record": ref(parent), "initial_latent": ref(initial),
                        "native_template": ref(template), "policies": list(study.POLICIES), "arms": list(study.ARMS)}],
            "steps": 100, "gradient_audit_interval": 50, "optimization_timeout_seconds": 1800,
            "repair_timeout_seconds": 300, "rescore_timeout_seconds": 3600,
            "allocation_seconds": 79200, "repair_search": {"max_edits": 3, "checks_per_depth": 256, "beam_width": 4}}


def test_factors_share_exact_saved_inputs_and_only_declared_update_rule_changes(tmp_path):
    p = protocol(tmp_path)
    configs = [study.build_config(p, p["starts"][0], policy, "fixed_initial_mask", tmp_path / policy)
               for policy in study.POLICIES]
    assert [c["optimizer"]["learning_rate"] for c in configs] == [400., 50., 400.]
    for c in configs:
        assert c["initial_latent"] == p["starts"][0]["initial_latent"]
        assert c["optimizer"]["steps"] == 100 and c["gradient_audit_interval"] == 50
        assert c["fd_policy"] == "local_v1" and c["diagnostics"]["hard_samples"] == 8
        assert c["contexts"] == configs[0]["contexts"] and c["objective"] == configs[0]["objective"]
    assert "step_control" not in configs[0] and "step_control" not in configs[1]
    assert configs[2]["step_control"]["version"] == "armijo_v1"
    rows = study.expand_cases(p)
    assert len(rows) == 6 and len({r["id"] for r in rows}) == 6
    parent = json.loads(Path(p["starts"][0]["parent_record"]["path"]).read_text())
    assert parent["status"] == "failed"  # Historical result was not resumed/reclassified.


@pytest.mark.parametrize("change", ["version", "duplicate_start", "unknown_factor", "timeout", "extra"])
def test_protocol_rejects_ambiguous_or_unbounded_cases(tmp_path, change):
    p = protocol(tmp_path)
    if change == "version": p["version"] = True
    if change == "duplicate_start": p["starts"].append(deepcopy(p["starts"][0]))
    if change == "unknown_factor": p["starts"][0]["policies"] = ["unrecognized"]
    if change == "timeout": p["optimization_timeout_seconds"] = 0
    if change == "extra": p["unknown"] = True
    path = tmp_path / "protocol.json"; path.write_text(json.dumps(p))
    with pytest.raises(ValueError): study.load_protocol(path)


def test_saved_initial_z_is_exact_and_authenticates_shape_dtype_finiteness_and_file(tmp_path):
    path = tmp_path / "start.pt"
    value = torch.randn(1, 5, dtype=torch.float64)
    torch.save({"z": value}, path)
    assert torch.equal(active_set.load_initial_latent(ref(path), 5, "cpu"), value)
    for malformed in (torch.zeros(2, 5), torch.zeros(1, 5, dtype=torch.long), torch.full((1, 5), float('nan'))):
        torch.save({"z": malformed}, path)
        with pytest.raises(ValueError): active_set.load_initial_latent(ref(path), 5, "cpu")
    torch.save({"z": value}, path)
    stale = ref(path); path.write_bytes(b"changed")
    with pytest.raises(ValueError): active_set.load_initial_latent(stale, 5, "cpu")


def test_new_config_keys_are_opt_in_and_do_not_change_old_config_identity(tmp_path):
    config = _config(tmp_path)
    assert "initial_latent" not in config.record() and "gradient_audit_interval" not in config.record()
    assert config_sha256(config.record()) == config_sha256(replace(config, initial_latent=None, gradient_audit_interval=None).record())
    raw = _yaml_configuration(); raw["gradient_audit_interval"] = True
    path = tmp_path / "configuration.yaml"; path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="gradient_audit_interval"): active_set.ActiveSetConfig.load(path)


def test_periodic_derivative_checks_cover_intermediate_and_endpoint_without_duplicate_start(tmp_path):
    cfg = replace(_config(tmp_path), gradient_audit_interval=1)
    summary = _invoke("fixed_initial_mask", _Decoder(), _Scorer(), cfg, tmp_path)
    gates = json.loads((tmp_path / "fixed_initial_mask" / "gates.json").read_text())
    assert set(gates) == {"step-0000", "step-0001", "step-0002"}
    assert summary["gated_contexts"] == 3
    assert all(g["passed"] and g["native_seed_parity"] for g in gates.values())


def test_periodic_checks_include_endpoint_when_interval_does_not_divide_budget(tmp_path):
    cfg = replace(_config(tmp_path), gradient_audit_interval=3)
    _invoke("fixed_initial_mask", _Decoder(), _Scorer(), cfg, tmp_path)
    gates = json.loads((tmp_path / "fixed_initial_mask" / "gates.json").read_text())
    assert set(gates) == {"step-0000", "step-0002"}


def test_middle_stall_has_a_terminal_gradient_check_and_keeps_budget_explicit(tmp_path, monkeypatch):
    cfg = replace(_config(tmp_path), step_control=StepControl(version="armijo_v1", max_trials=2), gradient_audit_interval=2)
    original = active_set.score_at
    accepted_first = {"done": False}
    def score(decoder, point, branch, config, counters, purpose, **kwargs):
        if purpose == "optimization" and accepted_first["done"] and float(point[0, 0]) > 0:
            accepted_first["block"] = True
        if purpose == "step_control" and accepted_first.get("block"):
            return {"loss": 100., "loss_dtype": "torch.float64"}, None
        observed = original(decoder, point, branch, config, counters, purpose, **kwargs)
        if purpose == "step_control": accepted_first["done"] = True
        return observed
    monkeypatch.setattr(active_set, "score_at", score)
    summary = _invoke("fixed_initial_mask", _Decoder(), _Scorer(), cfg, tmp_path)
    assert summary["stalled"] and summary["accepted_updates"] == 1
    assert summary["endpoint"]["step"] == 1 and summary["requested_steps"] == 2
    gates = json.loads((tmp_path / "fixed_initial_mask" / "gates.json").read_text())
    assert set(gates) == {"step-0000", "step-0001"}


def test_case_summary_separates_stalls_budget_completion_and_process_failures():
    rows = [{"status": "evaluated", "trajectory": {"termination": {"stalled": True, "completed_budget": False}}},
            {"status": "rescore_failed", "trajectory": {"termination": {"stalled": False, "completed_budget": True}}},
            {"status": "optimization_failed"}, {"status": "not_attempted"}]
    result = study.study_summary(rows)
    assert result["terminal_verified"] == 2 and result["completed_budget"] == 1 and result["stalled"] == 1
    assert result["optimization_failed"] == result["rescore_failed"] == result["not_attempted"] == 1


def test_resume_seal_detects_artifact_tampering_and_path_escape(tmp_path):
    attempt = tmp_path / "cases" / "fixture" / "attempt-001"; attempt.mkdir(parents=True)
    artifact = attempt / "result.json"; artifact.write_text("{}")
    row = {"id": "fixture", "attempt_dir": str(attempt)}
    study.publish_terminal(row, attempt, tmp_path); study.verify_seal(row, tmp_path)
    tampered_row = {**row, "status": "altered"}
    with pytest.raises(ValueError): study.verify_seal(tampered_row, tmp_path)
    artifact.write_text("changed")
    with pytest.raises(ValueError): study.verify_seal(row, tmp_path)
    row["attempt_dir"] = str(tmp_path)
    with pytest.raises(ValueError): study.verify_seal(row, tmp_path)


def test_controller_preserves_failed_case_and_continues_without_reclassifying_bad_start(tmp_path, monkeypatch):
    p = protocol(tmp_path)
    p["starts"][0]["arms"] = ["fixed_initial_mask"]
    path = tmp_path / "protocol.json"; path.write_text(json.dumps(p))
    (tmp_path / "release_manifest.json").write_text(json.dumps({"sha256": "a" * 64}))
    monkeypatch.chdir(tmp_path); monkeypatch.setenv("LSB_JOBID", "fixture")
    monkeypatch.setenv("LATENTMOL_SOURCE_SHA256", "b" * 64)
    monkeypatch.setattr(study, "source_manifest", lambda root: {"sha256": "a" * 64})
    monkeypatch.setattr(study, "verify_comparison", lambda gate: None)
    monkeypatch.setattr(study, "trajectory_report", lambda pilot, arm:
        {"termination": {"endpoint_step": 0, "completed_budget": False, "stalled": True}})
    def run_logged(command, logfile, **kwargs):
        command = list(map(str, command)); logfile.write_text("fixture")
        record = {"status": "completed", "run_id": "fixture-child", "source": {"sha256": "a" * 64},
                  "source_archive_sha256": "b" * 64, "config": {}}
        if "latentmol.boltz.active_set" in command:
            config_path = Path(command[command.index("--config") + 1])
            cfg = yaml.safe_load(config_path.read_text())
            record["config"]["config_file_sha256"] = file_sha256(config_path)
            pilot = Path(cfg["output_root"]) / "child"; pilot.mkdir(parents=True)
            # First policy lies about its initial z; later policies must proceed.
            torch.save({"z": torch.ones(1, 1) if "-legacy-" in cfg["name"] else torch.zeros(1, 1)}, pilot / "start.pt")
        else:
            key = "--run-root" if "--run-root" in command else "--protocol"
            if key == "--protocol":
                pilot = Path(json.loads(Path(command[command.index(key) + 1]).read_text())["output_root"]) / "child"
            else: pilot = Path(command[command.index(key) + 1]) / "child"
            pilot.mkdir(parents=True)
            record["config"]["parent_run_id"] = "fixture-child"
            record["config"]["confidence_checkpoint_sha256"] = p["confidence_checkpoint"]["sha256"]
            if key == "--protocol":
                record["config"]["protocol_sha256"] = file_sha256(Path(command[command.index(key) + 1]))
            (pilot / "summary.json").write_text("{}")
        record["config_sha256"] = config_sha256(record["config"])
        (pilot / "run.json").write_text(json.dumps(record))
        return {"returncode": 0, "timed_out": False, "run_paths": [str(pilot)], "log": str(logfile)}
    monkeypatch.setattr(study, "run_logged", run_logged)
    root = study.run(path, file_sha256(path))
    rows = json.loads((root / "cases.json").read_text())
    assert rows[0]["status"] == "optimization_failed" and "trajectory" not in rows[0]
    assert all(r["status"] == "evaluated" for r in rows[1:])
    assert study.study_summary(rows)["terminal_verified"] == 2
    before = (root / "cases.json").read_bytes()
    study.run(path, file_sha256(path), resume=True)
    assert (root / "cases.json").read_bytes() == before
    # Simulate a crash after receipt publication but before table publication.
    stale = json.loads(before)
    for row in stale:
        for key in list(row):
            if key not in {"id", "start", "policy", "arm", "attempts"}:
                row.pop(key)
        row["status"] = "optimizing"
    (root / "cases.json").write_text(json.dumps(stale))
    study.run(path, file_sha256(path), resume=True)
    assert (root / "cases.json").read_bytes() == before


@pytest.mark.parametrize("change", ["outside_root", "source", "config", "parent", "confidence"])
def test_completed_child_is_not_credited_when_provenance_differs(tmp_path, change):
    child = tmp_path / "children/one"; child.mkdir(parents=True)
    config = {"config_file_sha256": "c" * 64, "parent_run_id": "parent", "confidence_checkpoint_sha256": "d" * 64}
    record = {"status": "completed", "source": {"sha256": "a" * 64}, "source_archive_sha256": "b" * 64,
              "config": config, "config_sha256": config_sha256(config)}
    expected_root = tmp_path / "children"
    if change == "outside_root": expected_root = tmp_path / "other"
    if change == "source": record["source_archive_sha256"] = "e" * 64
    if change == "config": config["config_file_sha256"] = "e" * 64
    if change == "parent": config["parent_run_id"] = "other"
    if change == "confidence": config["confidence_checkpoint_sha256"] = "e" * 64
    record["config_sha256"] = config_sha256(config)
    (child / "run.json").write_text(json.dumps(record))
    execution = {"run_paths": [str(child)], "returncode": 0, "timed_out": False}
    with pytest.raises(ValueError):
        study.verified_child(execution, expected_root,
            {"source_manifest_sha256": "a" * 64, "source_archive_sha256": "b" * 64},
            config_file_sha256="c" * 64, parent_run_id="parent", confidence_sha256="d" * 64)


def test_same_count_slot_swap_is_not_reported_as_within_fixed_input_loss_increase(tmp_path, monkeypatch):
    from latentmol.boltz import active_set_rescore
    folder = tmp_path / "active_set"; folder.mkdir()
    monkeypatch.setattr(active_set_rescore, "select_endpoints", lambda path:
        {"arms": {"active_set": {"termination": {"endpoint_step": 1}}}})
    for step, slot in enumerate([0, 1]):
        row = {"step": step, "score": {"loss": step, "gradient_norm": None},
               "context_sha256": "a" * 64, "selection": {"count": 1, "active_slots": [slot]},
               "natural_selection": {"count": 1}, "latent_displacement": 0., "argmax_smiles": "C"}
        (folder / f"metrics-{step:04d}.json").write_text(json.dumps(row))
    (tmp_path / "oracle_cost.json").write_text("{}")
    report = study.trajectory_report(tmp_path, "active_set")
    assert not report["within_context_score_increases"] and not report["late_same_context"]
    assert report["fixed_input_score_jumps"] == [{"from_step": 0, "to_step": 1, "signed_loss_jump": 1}]
