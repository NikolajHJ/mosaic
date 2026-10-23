"""Allocated fixtures for matched unfiltered sampling and training RNG isolation."""
import gzip
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from latentmol import chemistry_generation_comparison as comparison
from latentmol import chemistry_short_continuation as policy
from latentmol.chemistry_execution_validation import _snapshot
from latentmol.train import capture_rng_state
from latentmol.chemistry_continuation import nested_equal


def test_matching_coordinates_unfiltered_results_and_exact_rng_mode_restoration(tmp_path, monkeypatch):
    model = torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.Dropout(.2))
    model.config = SimpleNamespace(latent=4)
    model.train()
    generators = {"fresh_prior":torch.Generator().manual_seed(109)}
    monkeypatch.setattr(policy,"validate_short_continuation",lambda p:{"purpose":"baseline_training"})
    calls=[]
    def evaluate(current,codes,path,protocol):
        assert current.training is False
        calls.append((codes.clone(),dict(protocol)))
        torch.randn(10)  # A future evaluator may consume the global stream.
        with gzip.open(str(path)+".gz","wt") as stream:
            stream.write('{"valid":false,"raw_graph":"retained"}\n')
        return {"categorical":{"attempts":4096,"valid":0},"argmax":{"attempts":1024,"valid":1}}
    monkeypatch.setattr(comparison,"evaluate_codes",evaluate)
    plan={"protocol":{"evaluation_seed":904017},"guarded_restart":{"validation_updates":2500},
        "_plan_sha256":"a"*64,"archive_sha256":"b"*64,"manifest_sha256":"c"*64,
        "starting_checkpoint":{"sha256":"d"*64}}
    before=capture_rng_state(generators,torch.cuda.is_available())
    first=comparison.observe_generation(plan,model,generators,tmp_path/"start",completed_updates=0)
    second=comparison.observe_generation(plan,model,generators,tmp_path/"end",completed_updates=2500)
    assert nested_equal(before,capture_rng_state(generators,torch.cuda.is_available()))
    assert model.training and model[1].training
    assert torch.equal(calls[0][0],calls[1][0]) and calls[0][1]==calls[1][1]
    assert calls[0][0].shape==(1024,4)
    assert calls[0][1]["categorical_samples_per_latent"]==4
    import json
    record=json.loads(Path(second["path"]).read_text())
    assert record["metrics"]["categorical"]["attempts"]==4096
    with gzip.open(record["raw_graphs"]["path"],"rt") as stream:
        assert json.loads(stream.readline())["valid"] is False
    with pytest.raises(FileExistsError):
        comparison.observe_generation(plan,model,generators,tmp_path/"start",completed_updates=0)


@pytest.mark.parametrize("updates",[-1,1,2499,True])
def test_partial_or_ambiguous_generation_endpoint_cannot_be_reported(tmp_path,monkeypatch,updates):
    monkeypatch.setattr(policy,"validate_short_continuation",lambda p:{"purpose":"baseline_training"})
    with pytest.raises(ValueError,match="exact proof endpoint"):
        comparison.observe_generation({"guarded_restart":{"validation_updates":2500}},
                                      None,{},tmp_path/"invalid",completed_updates=updates)
    assert not (tmp_path/"invalid").exists()


def _failed_fixture(tmp_path, monkeypatch, *, purpose="baseline_training", updates=100):
    import json
    from latentmol.experiments import file_sha256, write_json
    from test_chemistry_short_continuation import baseline_plan
    plan = baseline_plan(tmp_path / "campaign", purpose)
    root = Path(plan["campaign"])
    run = root / "stability" / "fixture"
    run.mkdir(parents=True)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(.2))
    model.config = SimpleNamespace(latent=4)
    model.train()
    generators = {"fresh_prior": torch.Generator().manual_seed(109)}
    gate = dict(status="failed", error="Early fixed-panel chemistry health failed", run_dir=str(run), job_id="123",
        plan_sha256=plan["_plan_sha256"], archive_sha256=plan["archive_sha256"],
        manifest_sha256=plan["manifest_sha256"], starting_checkpoint_sha256=plan["starting_checkpoint"]["sha256"],
        completed_updates=updates, last_update=plan["starting_checkpoint"]["updates"] + updates,
        epoch_complete=False, validation_weights_discarded=True, generation_start={"fixture": "saved start"})
    write_json(run / "gate.json", gate)
    write_json(root / "stability_gate.json", gate)
    write_json(run / "run.json", dict(status="running", environment=dict(job_id="123"),
        source_archive_sha256=plan["archive_sha256"], source=dict(sha256=plan["manifest_sha256"])))
    calls = []
    def evaluate(current, codes, path, protocol):
        calls.append((codes.clone(), dict(protocol)))
        torch.randn(7)
        torch.randn(3, generator=generators["fresh_prior"])
        with gzip.open(str(path) + ".gz", "wt") as stream:
            stream.write(json.dumps(dict(valid=False, raw_graph=dict(atoms=[6], bonds=[[0]]))) + "\n")
        return dict(categorical=dict(attempts=4096, valid=0), argmax=dict(attempts=1024, valid=1))
    monkeypatch.setattr(comparison, "evaluate_codes", evaluate)
    binding = dict(path=str(run / "gate.json"), sha256=file_sha256(run / "gate.json"))
    return plan, model, generators, run, gate, binding, calls


@pytest.mark.parametrize("purpose", ["baseline_training", "self_guidance_test"])
def test_failed_early_endpoint_retains_same_categorical_budget_raw_invalid_graphs_and_primary_failure(tmp_path, monkeypatch, purpose):
    import json
    from latentmol.experiments import file_sha256
    plan, model, generators, run, _, binding, calls = _failed_fixture(tmp_path, monkeypatch, purpose=purpose)
    before = capture_rng_state(generators, torch.cuda.is_available())
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    comparison.observe_generation(plan, model, generators, run / "generation-start", completed_updates=0)
    receipt = comparison.observe_failed_generation(plan, model, generators, run / "generation-failed-end",
        completed_updates=100, failure_gate=binding)
    assert nested_equal(before, capture_rng_state(generators, torch.cuda.is_available()))
    assert nested_equal(weights, model.state_dict()) and model.training and model[1].training
    assert torch.equal(calls[0][0], calls[1][0]) and calls[0][1] == calls[1][1]
    value = json.loads(Path(receipt["path"]).read_text())
    assert value["version"] == comparison.FAILED_VERSION
    assert value["completed_disposable_updates"] == 100
    assert value["failure_proof"]["endpoint_status"] == "failed_partial"
    assert value["failure_proof"]["sha256"] == binding["sha256"]
    assert value["failure_proof"]["run_id"] == run.name
    assert value["promotable"] is False and value["epoch_complete"] is False
    assert value["metrics"]["categorical"]["attempts"] == 4096
    with gzip.open(value["raw_graphs"]["path"], "rt") as stream:
        assert json.loads(stream.readline())["valid"] is False
    assert file_sha256(Path(binding["path"])) == binding["sha256"]
    assert json.loads(Path(binding["path"]).read_text())["status"] == "failed"


@pytest.mark.parametrize("updates", [-1, 0, 2501, True])
def test_failed_endpoint_requires_an_actual_bounded_update_count(tmp_path, monkeypatch, updates):
    plan, model, generators, run, _, binding, _ = _failed_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="actual applied"):
        comparison.observe_failed_generation(plan, model, generators, run / "invalid", completed_updates=updates,
                                               failure_gate=binding)
    assert not (run / "invalid").exists()


@pytest.mark.parametrize("key,value", [("status", "passed"), ("completed_updates", 2500),
    ("last_update", 237557), ("plan_sha256", "9" * 64), ("starting_checkpoint_sha256", "9" * 64),
    ("epoch_complete", True), ("validation_weights_discarded", False)])
def test_failed_generation_cannot_relabel_another_point_or_success_even_with_rebound_hash(tmp_path, monkeypatch, key, value):
    from latentmol.experiments import file_sha256, write_json
    plan, model, generators, run, gate, binding, _ = _failed_fixture(tmp_path, monkeypatch)
    gate[key] = value
    write_json(run / "gate.json", gate)
    binding["sha256"] = file_sha256(run / "gate.json")
    with pytest.raises(ValueError, match="proof identity"):
        comparison.observe_failed_generation(plan, model, generators, run / "invalid", completed_updates=100,
                                               failure_gate=binding)
    assert not (run / "invalid").exists()


def test_failed_endpoint_observer_failure_still_restores_training_rng_and_modes(tmp_path, monkeypatch):
    plan, model, generators, run, _, binding, _ = _failed_fixture(tmp_path, monkeypatch)
    before = capture_rng_state(generators, torch.cuda.is_available())
    def broken(*args):
        torch.randn(11)
        torch.randn(3, generator=generators["fresh_prior"])
        raise RuntimeError("Observer fixture failure")
    monkeypatch.setattr(comparison, "evaluate_codes", broken)
    with pytest.raises(RuntimeError, match="Observer fixture"):
        comparison.observe_failed_generation(plan, model, generators, run / "generation-failed-end",
                                               completed_updates=100, failure_gate=binding)
    assert nested_equal(before, capture_rng_state(generators, torch.cuda.is_available()))
    assert model.training and model[1].training


def test_guard_hook_does_not_mask_or_overwrite_primary_failure_when_generation_or_reporting_fails(tmp_path, monkeypatch, capsys):
    import json
    from latentmol import chemistry_guarded_validation as validation
    plan, model, generators, run, gate, binding, _ = _failed_fixture(tmp_path, monkeypatch)
    primary = (run / "gate.json").read_bytes()
    def fail_observer(*args, **kwargs):
        raise RuntimeError("Secondary generation fixture failure")
    monkeypatch.setattr(comparison, "observe_failed_generation", fail_observer)
    with pytest.raises(ValueError, match="Original health fixture failure"):
        try:
            raise ValueError("Original health fixture failure")
        except ValueError:
            validation._observe_failed_generation_safely(plan, model, generators, SimpleNamespace(path=run), gate)
            raise
    sidecar = Path(plan["campaign"]) / "failed-generation-end.json"
    result = json.loads(sidecar.read_text())
    assert result["status"] == "failed_observation" and result["primary_failure_preserved"]
    assert (run / "gate.json").read_bytes() == primary
    monkeypatch.setattr(validation, "write_json", lambda *args: (_ for _ in ()).throw(OSError("Quota fixture")))
    validation._observe_failed_generation_safely(plan, model, generators, SimpleNamespace(path=run), gate)
    assert "could not be saved" in capsys.readouterr().out
    assert (run / "gate.json").read_bytes() == primary


def test_failed_full_budget_stays_failed_and_does_not_claim_a_passed_endpoint(tmp_path, monkeypatch):
    import json
    plan, model, generators, run, _, binding, _ = _failed_fixture(tmp_path, monkeypatch, updates=2500)
    result = comparison.observe_failed_generation(plan, model, generators, run / "generation-failed-end",
        completed_updates=2500, failure_gate=binding)
    value = json.loads(Path(result["path"]).read_text())
    assert value["failure_proof"]["endpoint_status"] == "failed_full_budget"
    assert value["promotable"] is False and value["epoch_complete"] is False


@pytest.mark.parametrize("recorded_updates", [0, 100])
def test_incomplete_step_never_observes_live_weights_as_the_previous_recorded_point(tmp_path, monkeypatch, recorded_updates):
    import json
    from latentmol import chemistry_guarded_validation as validation
    plan, model, generators, run, gate, _, _ = _failed_fixture(tmp_path, monkeypatch, updates=recorded_updates)
    primary = (run / "gate.json").read_bytes()
    before = capture_rng_state(generators, torch.cuda.is_available())
    def forbidden(*args, **kwargs):
        raise AssertionError("An incomplete optimizer update is not an observable recorded endpoint")
    monkeypatch.setattr(comparison, "observe_failed_generation", forbidden)
    validation._observe_failed_generation_safely(plan, model, generators, SimpleNamespace(path=run), gate,
                                               live_state_observable=False)
    value = json.loads((Path(plan["campaign"]) / "failed-generation-end.json").read_text())
    assert value["status"] == "unavailable_observation"
    assert value["completed_disposable_updates"] == recorded_updates
    assert value["last_recorded_update"] == plan["starting_checkpoint"]["updates"] + recorded_updates
    assert value["model_observation_performed"] is False and value["primary_failure_preserved"]
    assert value["promotable"] is False
    assert nested_equal(before, capture_rng_state(generators, torch.cuda.is_available()))
    assert not (run / "generation-failed-end").exists()
    assert (run / "gate.json").read_bytes() == primary
