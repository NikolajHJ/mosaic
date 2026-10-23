"""Allocated-HPC fixtures only: cadence, fixed observers and failed-state evidence."""
from copy import deepcopy
import hashlib
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from latentmol import chemistry_continuation as continuation
from latentmol import chemistry_early_health as health
from latentmol.chemistry_classification import VERSION as CLASSIFICATION_VERSION, classification_summary
from latentmol.train import capture_rng_state, state_digest
from test_chemistry_continuation import _tiny_models
from test_repair_failure_guidance import output_for


def protocol():
    return dict(monitor_prior_count=1024, validation_count=256, evaluation_seed=904017,
        failure_guidance={"loss": {"version": "repair-failure-guidance-v2"}},
        early_health=dict(version=health.VERSION, first_check_updates=32, interval_updates=100,
                          max_examples_per_failure=4, max_strict_validity_drop=.02))


def _classification(strict=641, raw=641):
    records = []
    for index in range(1024):
        valid, chemical = index < strict, index >= raw
        classification = dict(version=CLASSIFICATION_VERSION, graph_sanitization_valid=not chemical,
            raw_chemical_valid=not chemical, smiles_parse_valid=None if chemical else True,
            schema_roundtrip_valid=None if chemical else valid, strict_teacher_valid=valid,
            failure_kind="none" if valid else "chemical" if chemical else "representation",
            reason=None if valid else "sanitization" if chemical else "representation_roundtrip")
        records.append(dict(candidate_id=index, classification=classification))
    return classification_summary(records)


def observation(strict=641, raw=641):
    return dict(population=dict(attempts=1024, raw_argmax_valid_count=raw,
        raw_argmax_validity=raw / 1024, expected_size=22., nonempty_fraction=1.,
        sulfur_frequency=.015, element_js=.001, charge_js=.0001),
        reconstruction=dict(molecules=256, exact_reconstruction=1., validity=1.),
        classification=_classification(strict, raw))


def plan(tmp_path):
    return dict(protocol=protocol(), campaign=str(tmp_path / "campaign"),
        _plan_sha256="a" * 64, archive_sha256="b" * 64, manifest_sha256="c" * 64,
        dataset_sha256="d" * 64, split_sha256="e" * 64,
        starting_checkpoint=dict(epoch=36, updates=237456, sha256="f" * 64))


def baseline(tmp_path, monkeypatch):
    monkeypatch.setenv("LSB_JOBID", "123456")
    p = plan(tmp_path)
    record, sha = health.materialize_early_health_baseline(p, observation(),
        completed_epoch=36, updates=237456, model_sha256="1" * 64)
    p["_early_health_baseline_sha256"] = sha
    return p, record, sha


def test_legacy_has_no_observation_or_cadence():
    assert health.validate_early_health_policy({}) is None
    assert not health.early_health_due(None, 100, 0)


@pytest.mark.parametrize("change", [
    lambda p: p.update(early_health=None),
    lambda p: p["early_health"].update(version="unknown"),
    lambda p: p["early_health"].update(extra=True),
    lambda p: p["early_health"].update(first_check_updates=True),
    lambda p: p["early_health"].update(first_check_updates=0),
    lambda p: p["early_health"].update(first_check_updates=101),
    lambda p: p["early_health"].update(interval_updates=101),
    lambda p: p["early_health"].update(interval_updates=100.),
    lambda p: p["early_health"].update(max_examples_per_failure=9),
    lambda p: p["early_health"].update(max_strict_validity_drop=float("nan")),
    lambda p: p["early_health"].update(max_strict_validity_drop=.021),
    lambda p: p.update(monitor_prior_count=256),
    lambda p: p.update(validation_count=255),
    lambda p: p.update(failure_guidance=None),
    lambda p: p["failure_guidance"]["loss"].update(version="repair-failure-guidance-v1"),
])
def test_policy_cannot_silently_change_panels_cadence_or_old_objective(change):
    value = protocol(); change(value)
    with pytest.raises(ValueError):
        health.validate_early_health_policy(value)


def test_cadence_is_relative_to_original_handoff_not_absolute_or_segment_restart():
    value = health.validate_early_health_policy(protocol())
    expected = [32, 100, 200, 300]
    assert [i for i in range(301) if health.early_health_due(value, 237456 + i, 237456)] == expected
    assert health.early_health_due(value, 237456 + 6600, 237456)
    value["interval_updates"] = 17
    assert protocol()["early_health"]["interval_updates"] == 100
    with pytest.raises(ValueError):
        health.early_health_due(value, 1, 2)


def test_baseline_is_exclusive_exact_source_bound_and_not_a_rolling_epoch(tmp_path, monkeypatch):
    p, record, sha = baseline(tmp_path, monkeypatch)
    assert health.require_early_health_baseline(p, sha) == record
    assert record["job_id"] == "123456" and record["completed_epoch"] == 36
    with pytest.raises(FileExistsError):
        health.materialize_early_health_baseline(p, observation(), completed_epoch=36,
                                               updates=237456, model_sha256="1" * 64)
    assert health.require_early_health_baseline(p, sha) == record


@pytest.mark.parametrize("field", ["_plan_sha256", "archive_sha256", "manifest_sha256", "dataset_sha256", "split_sha256"])
def test_baseline_cannot_cross_source_or_plan_bindings(tmp_path, monkeypatch, field):
    p, _, sha = baseline(tmp_path, monkeypatch); p[field] = "9" * 64
    with pytest.raises(ValueError, match="identity differs"):
        health.require_early_health_baseline(p, sha)


@pytest.mark.parametrize("change", [
    lambda p: p["starting_checkpoint"].update(epoch=37),
    lambda p: p["starting_checkpoint"].update(sha256="0" * 64),
    lambda p: p["protocol"].update(evaluation_seed=1),
    lambda p: p["protocol"]["early_health"].update(first_check_updates=16),
])
def test_baseline_seed_checkpoint_and_policy_remain_fixed(tmp_path, monkeypatch, change):
    p, _, sha = baseline(tmp_path, monkeypatch); change(p)
    with pytest.raises(ValueError):
        health.require_early_health_baseline(p, sha)


def test_baseline_byte_tamper_and_wrong_start_are_rejected(tmp_path, monkeypatch):
    p, _, sha = baseline(tmp_path, monkeypatch)
    path = tmp_path / "campaign/early-health-baseline.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="changed"):
        health.require_early_health_baseline(p, sha)
    p["campaign"] = str(tmp_path / "other")
    with pytest.raises(ValueError, match="original completed"):
        health.materialize_early_health_baseline(p, observation(), completed_epoch=37,
                                               updates=237456, model_sha256="1" * 64)
    assert not (tmp_path / "other").exists()


def assess(current):
    before = observation()
    return continuation.assess_early_health(current, before["population"], before["reconstruction"],
        before["population"], before["reconstruction"], protocol(),
        starting_classification=before["classification"])


def test_chemical_and_schema_validity_have_independent_gates():
    assert assess(observation())["status"] == "passed"
    changed = assess(observation(strict=610, raw=641))
    assert changed["checks"]["failure_start_raw_prior_validity"]
    assert not changed["checks"]["failure_start_strict_teacher_validity"]
    assert changed["status"] == "failed"
    changed = assess(observation(strict=600, raw=600))
    assert not changed["checks"]["failure_start_raw_prior_validity"]
    assert not changed["checks"]["failure_start_strict_teacher_validity"]


@pytest.mark.parametrize("section,field,value,gate", [
    ("population", "expected_size", 40., "size_retention"),
    ("population", "nonempty_fraction", .5, "nonempty_retention"),
    ("population", "sulfur_frequency", .1, "sulfur_no_inflation"),
    ("population", "element_js", .1, "composition_no_deterioration"),
    ("population", "charge_js", .1, "charge_no_deterioration"),
    ("population", "expected_size", float("nan"), "finite"),
    ("reconstruction", "exact_reconstruction", .8, "reconstruction_retention"),
    ("reconstruction", "validity", .8, "valid_reconstruction_retention"),
])
def test_all_original_health_gates_remain_hard(section, field, value, gate):
    current = observation(); current[section][field] = value
    result = assess(current)
    assert result["status"] == "failed" and not result["checks"][gate]
    assert not result["checks"]["failure_start_" + gate]


@pytest.mark.parametrize("raises", [False, True])
def test_observer_restores_every_rng_and_mixed_module_modes_even_on_failure(raises):
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(.5)).train()
    model[0].eval()
    generators = {"train": torch.Generator().manual_seed(7), "fresh_prior": torch.Generator().manual_seed(9)}
    before = capture_rng_state(generators, False)
    modes = [m.training for m in model.modules()]
    try:
        with health.preserve_observation_state(model, generators):
            assert not any(m.training for m in model.modules())
            random.random(); np.random.random(); torch.rand(3)
            for generator in generators.values(): torch.rand(3, generator=generator)
            if raises: raise RuntimeError("fixture observation failure")
    except RuntimeError:
        assert raises
    assert continuation.nested_equal(before, capture_rng_state(generators, False))
    assert [m.training for m in model.modules()] == modes


def test_native_observation_uses_same_fixed_panels_and_restores_modes_rng(monkeypatch):
    model = torch.nn.Linear(2, 2).train(); generators = {"train": torch.Generator().manual_seed(4)}
    before = capture_rng_state(generators, False); expected = observation(); calls = []
    def population(actual, count, seed, batch_size, composition, *, classification):
        assert actual is model and not actual.training
        calls.append(("prior", count, seed, batch_size)); torch.rand(4)
        classification.summary = lambda: expected["classification"]
        return expected["population"]
    def reconstruction(actual, data, config, count, seed):
        calls.append(("reconstruction", count, seed)); torch.rand(3)
        return expected["reconstruction"]
    monkeypatch.setattr(continuation, "population_panel", population)
    monkeypatch.setattr(continuation, "validation", reconstruction)
    result = continuation.observe_early_health(model, {"val": object()}, None, protocol(), {}, generators)
    assert result["classification"]["attempts"] == 1024
    assert calls == [("prior", 1024, 904017, 32), ("reconstruction", 256, 904017)]
    assert continuation.nested_equal(before, capture_rng_state(generators, False)) and model.training


def test_bounded_examples_retain_actual_graph_labels_and_probability_arrays():
    spec, decoded = output_for([9, 9], [(0, 1, "TRIPLE")])
    hard = decoded.harden(spec)
    probabilities = tuple(v[0].softmax(-1) for v in (decoded.atoms, decoded.charges, decoded.symmetric_bond_logits))
    collector = health.EarlyPanelCollector(2)
    for i in range(7):
        collector.add(hard.atoms[0], hard.charges[0], hard.bonds[0], spec,
                      probabilities=probabilities, candidate_id=i)
    result = collector.summary()
    assert result["attempts"] == 7 and result["counts"]["chemical_failures"] == 7
    examples = result["failure_examples"]["chemical"]
    assert len(examples) == 2 and [e["candidate_id"] for e in examples] == [0, 1]
    assert examples[0]["labels"]["bonds"] == hard.bonds[0].tolist()
    assert examples[0]["categorical_probabilities"]["atoms"] == probabilities[0].detach().tolist()
    json.dumps(result, allow_nan=False)


def test_failed_snapshot_saves_exact_state_but_cannot_be_a_resume(tmp_path):
    model, _, config, _, _ = _tiny_models(); optimizer = torch.optim.AdamW(model.parameters())
    for parameter in model.parameters(): parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    generators = {"train": torch.Generator().manual_seed(8)}
    p = plan(tmp_path); path = tmp_path / "diagnostic.pt"; before = capture_rng_state(generators, False)
    sha = health.save_failed_early_health_snapshot(path, model, optimizer, config, p,
        SimpleNamespace(run_id="fixture"), completed_epoch=36, partial_epoch=37, updates=237488,
        generators=generators, health_sha256="2" * 64)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert sha == hashlib.sha256(path.read_bytes()).hexdigest()
    assert payload["epoch_complete"] is False and payload["promotable"] is False
    assert "epoch" not in payload and "chemistry_phase" not in payload and payload.get("format_version") != 1
    assert state_digest(payload["model_state"]) == state_digest(model.state_dict())
    assert continuation.nested_equal(payload["optimizer_state"], optimizer.state_dict())
    assert continuation.nested_equal(payload["rng"], before)
    with pytest.raises(ValueError, match="nonpromotable"):
        health.reject_diagnostic_snapshot(payload)
    assert health.reject_diagnostic_snapshot({"format_version": 1, "epoch": 37}) is None


def test_failure_report_precedes_snapshot_and_no_normal_checkpoint_exists(tmp_path, monkeypatch):
    p, b, _ = baseline(tmp_path, monkeypatch); model, _, config, _, _ = _tiny_models()
    optimizer = torch.optim.AdamW(model.parameters()); run = SimpleNamespace(path=tmp_path / "run", run_id="fixture", log=lambda value: None)
    current = observation(strict=610, raw=641)
    monkeypatch.setattr(continuation, "observe_early_health", lambda *args: current)
    before = observation()
    with pytest.raises(RuntimeError, match="cannot resume"):
        continuation.run_early_health_check(p, run, model, optimizer, config, {}, {}, {},
            completed_epoch=36, epoch=37, updates=237488, reference_population=before["population"],
            reference_reconstruction=before["reconstruction"], starting_population=before["population"],
            starting_reconstruction=before["reconstruction"], baseline=b)
    report = run.path / "early_health/update_000237488.json"
    receipt = json.loads(report.with_suffix(".diagnostic.json").read_text())
    assert json.loads(report.read_text())["status"] == "failed"
    assert receipt["health_record_sha256"] == hashlib.sha256(report.read_bytes()).hexdigest()
    payload = torch.load(report.with_suffix(".diagnostic.pt"), weights_only=True)
    assert payload["health_record_sha256"] == receipt["health_record_sha256"]
    assert not list(run.path.glob("epoch_*.pt")) and not (run.path / "latest_epoch.json").exists()


@pytest.mark.parametrize("outcome", ["passed", "observer_exception", "snapshot_exception"])
def test_early_check_passes_without_checkpoint_or_fails_closed_with_durable_evidence(tmp_path, monkeypatch, outcome):
    p, b, _ = baseline(tmp_path, monkeypatch); model, _, config, _, _ = _tiny_models()
    optimizer = torch.optim.AdamW(model.parameters())
    run = SimpleNamespace(path=tmp_path / "run", run_id="fixture", log=lambda value: None)
    before = observation()
    def observe(*args):
        if outcome == "observer_exception": raise ArithmeticError("fixture native observation failed")
        return before if outcome == "passed" else observation(strict=0, raw=641)
    monkeypatch.setattr(continuation, "observe_early_health", observe)
    def broken_snapshot(*args, **kwargs): raise OSError("fixture quota exhaustion")
    if outcome == "snapshot_exception":
        monkeypatch.setattr(continuation, "save_failed_early_health_snapshot", broken_snapshot)
    args = dict(completed_epoch=36, epoch=37, updates=237488, reference_population=before["population"],
        reference_reconstruction=before["reconstruction"], starting_population=before["population"],
        starting_reconstruction=before["reconstruction"], baseline=b)
    if outcome == "passed":
        assert continuation.run_early_health_check(p, run, model, optimizer, config, {}, {}, {}, **args)["status"] == "passed"
        assert not list(run.path.rglob("*.pt"))
    else:
        with pytest.raises(RuntimeError):
            continuation.run_early_health_check(p, run, model, optimizer, config, {}, {}, {}, **args)
    report = run.path / "early_health/update_000237488.json"
    value = json.loads(report.read_text())
    assert value["status"] == ("passed" if outcome == "passed" else "failed")
    if outcome == "observer_exception":
        assert value["checks"] == {"observation_completed": False}
        assert "ArithmeticError" in value["error"]
        assert report.with_suffix(".diagnostic.pt").exists()
    elif outcome == "snapshot_exception":
        assert report.with_suffix(".snapshot-error.json").exists()
        assert not report.with_suffix(".diagnostic.json").exists()
    assert not list(run.path.glob("epoch_*.pt")) and not (run.path / "latest_epoch.json").exists()


def test_handoff_observer_metadata_is_validated_before_excluding_from_old_protocol(tmp_path, monkeypatch):
    from test_failure_guidance_training import _handoff_fixture
    p, base, indices, runtime, _ = _handoff_fixture(tmp_path, monkeypatch)
    p["protocol"]["early_health"] = protocol()["early_health"]
    # The tiny preexisting lineage fixture has no 1024/256 monitoring fields.
    # Policy validation is tested against the full schema above; here record
    # that this boundary invokes it before excluding monitoring metadata, while
    # every original scientific field still has to match the original source.
    calls = []
    def validate(actual):
        calls.append(actual["early_health"])
        if actual["early_health"]["version"] != health.VERSION:
            raise ValueError("Unsupported early-health policy")
        return actual["early_health"]
    monkeypatch.setattr(continuation, "validate_early_health_policy", validate)
    payload = continuation.authenticate_chemistry_handoff(p, None, base, indices, runtime)
    assert payload["epoch"] == 36
    assert calls == [p["protocol"]["early_health"]]
    p["protocol"]["early_health"]["version"] = "unknown"
    with pytest.raises(ValueError, match="early-health policy"):
        continuation.authenticate_chemistry_handoff(p, None, base, indices, runtime)
