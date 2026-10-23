"""Allocated-only fixtures for frozen-target finite-step mechanism evidence."""
from copy import deepcopy

import pytest
import torch

from latentmol import chemistry_collapse_diagnostic as diagnostic
from latentmol import chemistry_continuation as continuation
from latentmol.chemistry_execution_validation import _snapshot, _restore
from latentmol.train import capture_rng_state
from test_chemistry_continuation import _hard, _captured_gradients
from test_chemistry_valid_self_training import _models, _teacher
from test_chemistry_valid_self_guidance import device


def _plan():
    return dict(starting_checkpoint=dict(epoch=36), protocol={}, collapse_diagnostic=dict(
        version=diagnostic.VERSION, updates=100, instrument_updates=[39,96],
        source_rows=dict(path="/work3/example/stability-updates.jsonl", sha256="a"*64)))


def test_closed_original_objective_replay_scope():
    plan = _plan()
    value = diagnostic.validate_policy(plan)
    value["instrument_updates"].append(97)
    assert plan["collapse_diagnostic"]["instrument_updates"] == [39,96]


@pytest.mark.parametrize("key,value", [("updates", 8), ("updates", True),
    ("instrument_updates", [38,95]), ("version", "new"),
    ("source_rows", dict(path="relative", sha256="a"*64))])
def test_reject_changed_budget_or_unauthenticated_history(key,value):
    plan = _plan(); plan["collapse_diagnostic"][key] = value
    with pytest.raises(ValueError):
        diagnostic.validate_policy(plan)


@pytest.mark.parametrize("key", ["failure_guidance", "valid_prior_self_guidance"])
def test_original_replay_rejects_other_loss(key):
    plan = _plan(); plan["protocol"][key] = {"version":"x"}
    with pytest.raises(ValueError):
        diagnostic.validate_policy(plan)


def test_capture_leaves_production_backward_and_optimizer_proposal_unchanged(device):
    model, reference, config, protocol, composition = _models(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    generators = {"fresh_prior":torch.Generator().manual_seed(901)}
    initial = _snapshot(model, reference, optimizer, generators)
    results = []
    with _teacher(model) as teacher:
        for instrument in (False,True):
            _restore(initial,model,reference,optimizer,generators)
            context, observer = {}, diagnostic.ComponentGradients(model)
            kwargs = dict(capture_context=context, component_observer=observer) if instrument else {}
            result = continuation.chemistry_update(model,reference,optimizer,_hard(model.spec),config,protocol,
                composition,teacher,generators,7,37,35,do_step=False,microbatch_size=2,**kwargs)
            gradients = _captured_gradients(model)
            if instrument:
                before_rng = capture_rng_state(generators, device=="cuda")
                finite = diagnostic.frozen_objective(model,reference,config,protocol,composition,context)
                assert finite["total"] == pytest.approx(result["total"],rel=1e-5,abs=2e-7)
                assert continuation.nested_equal(before_rng,capture_rng_state(generators,device=="cuda"))
            optimizer.step()
            if instrument:
                delta = {name:p.detach().double()-initial["model"][name].double()
                         for name,p in model.named_parameters()}
                proof = observer.report(delta,gradients)
                assert proof["maximum_gradient_reconstruction_error"] < 2e-7
                assert "prior/repair" in proof["terms"]
            results.append((result,gradients,_snapshot(model,reference,optimizer,generators)))
    assert results[0][0]["total"] == results[1][0]["total"]
    assert continuation.nested_equal(results[0][1],results[1][1])
    assert continuation.nested_equal(results[0][2],results[1][2])


def test_finite_counterfactual_reuses_targets_and_draws_without_research(device,monkeypatch):
    model, reference, config, protocol, composition = _models(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001)
    generators = {"fresh_prior":torch.Generator().manual_seed(901)}
    context = {}
    with _teacher(model) as teacher:
        continuation.chemistry_update(model,reference,optimizer,_hard(model.spec),config,protocol,
            composition,teacher,generators,7,37,35,do_step=False,microbatch_size=2,capture_context=context)
        def forbidden(*args,**kwargs):
            raise AssertionError("Frozen objective must never draw or research targets")
        monkeypatch.setattr(teacher,"build_targets",forbidden)
        monkeypatch.setattr(torch,"randn",forbidden)
        before = diagnostic.frozen_objective(model,reference,config,protocol,composition,context)
        retained = deepcopy(model.state_dict())
        with torch.no_grad():
            model.decoder.atom_head.bias[model.spec.atomic_numbers.index(6)] += .2
        after = diagnostic.frozen_objective(model,reference,config,protocol,composition,context)
        assert after["total"] != before["total"]
        model.load_state_dict(retained)
        assert diagnostic.frozen_objective(model,reference,config,protocol,composition,context) == before
