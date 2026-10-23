"""Allocated actual-trainer contracts for the frozen valid-reference branch.

These tiny native encoder/decoder fixtures are engineering counterfactuals, not
the authentic epoch-36 scientific evaluation. No custom forward closure is
copied into the frozen anchor. The allocated full-model native gates separately
authenticate the real checkpoint, 256/512 logical batch and scientific health.
"""
from copy import deepcopy
from contextlib import nullcontext
from dataclasses import replace

import pytest
import torch

from latentmol import chemistry_continuation as c
from latentmol.atom_preserving_chemistry import AtomPreservingRepairTeacher
from latentmol.chemistry_classification import VERSION as CLASSIFICATION_VERSION
from latentmol.chemistry_execution_validation import _restore, _snapshot
from latentmol.chemistry_objective_gradient_diagnostics import ObjectiveGradientDiagnostics
from latentmol.chemistry_valid_reference_anchor import (
    FrozenValidReferenceAnchor, START_SHA, latent_identity, validate_anchor_identity,
)
from latentmol.chemistry_valid_reference_guidance import VERSION
from latentmol.probability_repair import ProbabilityRepairConfig
from test_chemistry_continuation import _captured_gradients, _hard
from test_chemistry_valid_self_guidance import device
from test_chemistry_valid_self_training import _models, _teacher


POLICY = dict(version=VERSION, weight=.1)
STEP = dict(version="adamw-displacement-scale-v1", factor=.25)


def _fixture(device):
    model, reference, config, protocol, composition = _models(device)
    protocol.pop("valid_prior_self_guidance", None)
    config = replace(config, clip_grad=1e6)
    original_keys = tuple(model.state_dict())
    original_parameters = tuple(model.parameters())
    anchor = FrozenValidReferenceAnchor(model, reference_epoch=36, checkpoint_sha256=START_SHA)
    object.__setattr__(model, "_valid_reference_anchor", anchor)
    assert tuple(model.state_dict()) == original_keys
    assert tuple(map(id, model.parameters())) == tuple(map(id, original_parameters))
    assert {id(p) for p in model.parameters()}.isdisjoint(id(p) for p in anchor._model.parameters())
    return model, reference, config, protocol, composition, anchor


def _run(model, reference, config, protocol, composition, generators, optimizer, *,
         size=2, overlap=False, step=True, observer=None):
    with _teacher(model) as teacher:
        return c.chemistry_update(model, reference, optimizer, _hard(model.spec), config, protocol,
            composition, teacher, generators, 7, 37, 35, microbatch_size=size, overlap=overlap,
            do_step=step, step_control=STEP,
            **({"component_observer": observer} if observer is not None else {}))


def _self_protocol(protocol):
    result = deepcopy(protocol)
    result["valid_prior_self_guidance"] = dict(version="valid-prior-self-nll-v1", weight=.1)
    return result


def _reference_protocol(protocol):
    result = deepcopy(protocol)
    result["valid_prior_reference_guidance"] = dict(POLICY)
    return result


@pytest.mark.parametrize("step", [False, True])
def test_same_initial_reference_matches_current_self_full_gradient_native_adamw_and_rng_exactly(device, step):
    model, reference, config, protocol, composition, anchor = _fixture(device)
    frozen = anchor.assert_unchanged()
    # Establish genuine native AdamW moments with an actual logical update at
    # zero LR. The anchor was created first; the live weights stay identical to
    # its initial valid K4 graph, while the next compared step inherits moments.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.)
    generators = {"fresh_prior": torch.Generator().manual_seed(901)}
    weights = deepcopy(model.state_dict())
    _run(model, reference, config, _self_protocol(protocol), composition, generators, optimizer)
    assert c.nested_equal(weights, model.state_dict())
    assert optimizer.state and any(float(state["step"]) > 0 for state in optimizer.state.values())
    for group in optimizer.param_groups:
        group["lr"] = .001
    initial = _snapshot(model, reference, optimizer, generators)
    observations = []
    for current in (_self_protocol(protocol), _reference_protocol(protocol)):
        _restore(initial, model, reference, optimizer, generators)
        metrics = _run(model, reference, config, current, composition, generators, optimizer, step=step)
        observations.append((metrics, _captured_gradients(model), _snapshot(model, reference, optimizer, generators)))
        assert anchor.assert_unchanged() == frozen
    old, new = observations
    assert old[0]["valid_prior_self_guidance"]["eligible"] == new[0]["valid_prior_reference_guidance"]["eligible"] == 6
    assert new[0]["valid_prior_reference_guidance"]["reference_valid_live_invalid"] == 0
    assert old[0]["valid_prior_self_nll"] == new[0]["valid_prior_reference_nll"]
    assert old[0]["valid_prior_self_weighted"] == new[0]["valid_prior_reference_weighted"]
    assert old[0]["total"] == new[0]["total"]
    for key in ("repair_targets", "repair_statuses", "teacher_classification", "search_checks"):
        assert c.nested_equal(old[0][key], new[0][key])
    assert c.nested_equal(old[1], new[1])
    assert c.nested_equal(old[2], new[2])
    if step:
        assert old[0]["optimizer_step_control"] == new[0]["optimizer_step_control"]
    _restore(initial, model, reference, optimizer, generators)


def test_live_invalid_prior_retains_reference_targets_and_decoder_vjp_after_current_self_disappears(device):
    model, reference, config, protocol, composition, anchor = _fixture(device)
    frozen = anchor.assert_unchanged()
    with torch.no_grad():
        # The native constant decoder now gives every occupied C4 pair a triple
        # bond. Each carbon has valence9, while the frozen K4 single-bond target
        # remains valid with valence3. A one-edit live repair is not fabricated.
        model.decoder.bond_head[-1].bias[model.spec.bond_names.index("TRIPLE")] = 12.
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    generators = {"fresh_prior": torch.Generator().manual_seed(901)}
    initial = _snapshot(model, reference, optimizer, generators)
    observations = []
    for current in (_self_protocol(protocol), _reference_protocol(protocol)):
        _restore(initial, model, reference, optimizer, generators)
        metrics = _run(model, reference, config, current, composition, generators, optimizer, step=False)
        observations.append((metrics, _captured_gradients(model), _snapshot(model, reference, optimizer, generators)))
        assert anchor.assert_unchanged() == frozen
    old, new = observations
    assert old[0]["valid_prior_self_guidance"]["eligible"] == 0
    assert old[0]["valid_prior_self_weighted"] == 0.
    assert new[0]["valid_prior_reference_guidance"]["eligible"] == 6
    assert new[0]["valid_prior_reference_guidance"]["reference_valid_live_invalid"] == 6
    assert new[0]["valid_prior_reference_nll"] > 0.
    assert new[0]["valid_prior_reference_weighted"] == pytest.approx(.1*new[0]["valid_prior_reference_nll"])
    assert new[0]["valid_prior_reference_decoder_gradient_norm"] > 0.
    assert new[0]["total"]-old[0]["total"] == pytest.approx(new[0]["valid_prior_reference_weighted"])
    assert old[0]["repair_statuses"] == new[0]["repair_statuses"]
    assert c.nested_equal(old[2]["rng"], new[2]["rng"])
    assert all(c.nested_equal(old[1][name], value) for name, value in new[1].items() if name.startswith("encoder."))
    assert any(not c.nested_equal(old[1][name], value) for name, value in new[1].items() if name.startswith("decoder."))
    _restore(initial, model, reference, optimizer, generators)


def test_full_vs_streamed_reference_means_actual_targets_gradient_and_rng(device):
    model, reference, config, protocol, composition, anchor = _fixture(device)
    protocol = _reference_protocol(protocol)
    frozen = anchor.assert_unchanged()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    generators = {"fresh_prior": torch.Generator().manual_seed(901)}
    initial = _snapshot(model, reference, optimizer, generators)
    observations = []
    for size in (2, 6):
        _restore(initial, model, reference, optimizer, generators)
        metrics = _run(model, reference, config, protocol, composition, generators, optimizer, size=size, step=False)
        observations.append((metrics, _captured_gradients(model), _snapshot(model, reference, optimizer, generators)))
        assert anchor.assert_unchanged() == frozen
    left, right = observations
    for key in ("total", "valid_prior_reference_nll", "valid_prior_reference_weighted", "repair_real", "repair_prior"):
        assert left[0][key] == pytest.approx(right[0][key], rel=2e-5, abs=2e-7)
    for metrics in (left[0], right[0]):
        assert metrics["valid_prior_reference_guidance"]["attempted"] == 6
        assert metrics["valid_prior_reference_guidance"]["eligible"] == 6
        assert metrics["valid_prior_reference_guidance"]["reference_valid_live_invalid"] == 0
    assert c.nested_equal(left[2]["rng"], right[2]["rng"])
    for name, gradient in left[1].items():
        if gradient is None:
            assert right[1][name] is None
        else:
            torch.testing.assert_close(gradient, right[1][name], rtol=3e-4, atol=3e-7)
    _restore(initial, model, reference, optimizer, generators)


def test_sync_async_reference_targets_full_steps_optimizer_and_native_states_match_exactly(device):
    model, reference, config, protocol, composition, anchor = _fixture(device)
    protocol = _reference_protocol(protocol)
    frozen = anchor.assert_unchanged()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    generators = {"fresh_prior": torch.Generator().manual_seed(901)}
    initial = _snapshot(model, reference, optimizer, generators)
    observations = []
    for overlap in (False, True):
        _restore(initial, model, reference, optimizer, generators)
        metrics = _run(model, reference, config, protocol, composition, generators, optimizer, overlap=overlap)
        observations.append((metrics, _snapshot(model, reference, optimizer, generators)))
        assert anchor.assert_unchanged() == frozen
    assert c.nested_equal(observations[0][1], observations[1][1])
    ignored = ("search_seconds", "prepass_seconds", "teacher_phase_seconds")
    assert {k:v for k,v in observations[0][0].items() if k not in ignored} == {
        k:v for k,v in observations[1][0].items() if k not in ignored}
    _restore(initial, model, reference, optimizer, generators)


def test_reference_component_observer_preserves_actual_full_update_all_eight_states_and_rng(device):
    """Observe the actual reference-guided trainer, including native backward.

    The tail real chunk yields two real and three prior callbacks. Diagnostic
    VJPs must reconstruct the unmodified full native .grad without another
    forward, optimizer step or random draw, including inherited AdamW moments.
    """
    from latentmol.chemistry_execution_validation import EXCLUDED_TIMINGS
    model, reference, config, protocol, composition, anchor = _fixture(device)
    protocol = _reference_protocol(protocol)
    frozen = anchor.assert_unchanged()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    generators = {"fresh_prior": torch.Generator().manual_seed(901)}
    _run(model, reference, config, protocol, composition, generators, optimizer)
    assert optimizer.state and any(bool(state["exp_avg"].count_nonzero()) for state in optimizer.state.values())
    initial = _snapshot(model, reference, optimizer, generators)
    results, receipt = [], None
    for observed in (False, True):
        _restore(initial, model, reference, optimizer, generators)
        assert c.nested_equal(initial, _snapshot(model, reference, optimizer, generators))
        observer = ObjectiveGradientDiagnostics(model, real_points=3, prior_points=6, microbatch_size=2)
        with observer if observed else nullcontext():
            metrics = _run(model, reference, config, protocol, composition, generators, optimizer,
                           observer=observer if observed else None)
        if observed:
            receipt = observer.receipt()
        results.append(({key:value for key,value in metrics.items() if key not in EXCLUDED_TIMINGS},
                        _snapshot(model, reference, optimizer, generators)))
        assert anchor.assert_unchanged() == frozen
    assert c.nested_equal(results[0], results[1])
    assert set(results[1][1]) == {"model","reference","optimizer","gradients","reference_gradients",
                                  "model_modes","reference_modes","rng"}
    assert results[1][0]["valid_prior_reference_guidance"]["eligible"] == 6
    assert results[1][0]["valid_prior_reference_weighted"] > 0
    assert results[1][0]["optimizer_step_control"]["factor"] == .25
    assert receipt["real_points"] == 3 and receipt["prior_points"] == 6 and receipt["microbatch_size"] == 2
    assert receipt["observed_microbatches"] == {"real":2,"prior":3}
    assert receipt["weighted_terms"] == ["composition","kl","occupancy","reconstruction","repair","valid_reference"]
    assert receipt["complete_logical_update"] is True and receipt["component_sum_matches_native_backward"] is True
    assert receipt["metrics"]["valid_reference"]["gradient_norm"] > 0
    assert receipt["observer_model_forwards"] == receipt["observer_optimizer_steps"] == receipt["observer_random_samples"] == 0
    _restore(initial, model, reference, optimizer, generators)


def test_reference_teacher_operations_use_one_fifo_pool_without_search_or_cache_writes(device, monkeypatch):
    model, _, _, _, _, anchor = _fixture(device)
    z = torch.zeros((2, model.config.latent), device=device)
    decoded = anchor.decode(z)
    events = []
    with AtomPreservingRepairTeacher(model.spec, ProbabilityRepairConfig(max_edits=1, checks_per_depth=1),
            backend="optimized", workers=1, classification_version=CLASSIFICATION_VERSION,
            classification_execution=dict(version="repair-classification-pool-v1")) as teacher:
        def forbidden(*args, **kwargs):
            raise AssertionError("Reference-only classification called live repair search")
        monkeypatch.setattr(teacher, "_search", forbidden)
        monkeypatch.setattr(teacher, "_parallel_misses", forbidden)
        native = teacher._build_valid_reference_snapshot
        def observe(payload):
            result = native(payload)
            events.append(result.candidate_ids)
            return result
        monkeypatch.setattr(teacher, "_build_valid_reference_snapshot", observe)
        first = teacher.build_valid_reference_targets(decoded, candidate_ids=[1,2], update=7,
            reference_identity=anchor.reference_identity, latent_sha256=latent_identity(z))
        pool = teacher._pool
        assert pool is not None and len(teacher._cache) == 0
        second = teacher.build_valid_reference_targets_async(decoded, candidate_ids=[3,4], update=7,
            reference_identity=anchor.reference_identity, latent_sha256=latent_identity(z))
        third = teacher.build_valid_reference_targets_async(decoded, candidate_ids=[5,6], update=7,
            reference_identity=anchor.reference_identity, latent_sha256=latent_identity(z))
        # Consuming handles in reverse order cannot reorder native execution.
        results = (first, third.result(), second.result())
        teacher.drain()
        assert events == [(1,2), (3,4), (5,6)]
        assert teacher._pool is pool and len(teacher._cache) == 0
        assert all(result.eligible_rows == (0,1) for result in results)
        assert all(result.receipt["reference_searches"] == 0 for result in results)
        assert all(result.receipt["accepted_repair_witnesses_created"] == 0 for result in results)


def test_anchor_stays_outside_parameter_registry_optimizer_checkpoint_and_module_mode_walk(device):
    model, _, _, _, _, anchor = _fixture(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    registered = {id(p) for p in model.parameters()}
    frozen = {id(p) for p in anchor._model.parameters()}
    assert registered.isdisjoint(frozen)
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == registered
    assert not any("valid_reference" in key for key in model.state_dict())
    model.train()
    assert all(not module.training for module in anchor._model.modules())
    assert all(not p.requires_grad and p.grad is None for p in anchor._model.parameters())
    anchor.assert_unchanged()


def test_anchor_metadata_requires_matching_original36_model_digest_and_defensive_copy():
    value = dict(reference_epoch=36, reference_sha256=START_SHA, model_sha256="1"*64, frozen=True, eval_mode=True)
    result = validate_anchor_identity(value, expected_model_sha="1"*64)
    result["model_sha256"] = "2"*64
    assert value["model_sha256"] == "1"*64
    with pytest.raises(ValueError):
        validate_anchor_identity(value, expected_model_sha="2"*64)


@pytest.mark.parametrize("change", [lambda x:x.update(reference_epoch=35),
    lambda x:x.update(reference_epoch=36.), lambda x:x.update(reference_epoch=True),
    lambda x:x.update(reference_sha256="2"*64), lambda x:x.update(model_sha256="1"*63),
    lambda x:x.update(model_sha256="A"*64), lambda x:x.update(frozen=1),
    lambda x:x.update(eval_mode=False), lambda x:x.update(extra=True)])
def test_anchor_metadata_rejects_changed_epoch_checkpoint_model_or_frozen_scope(change):
    value = dict(reference_epoch=36, reference_sha256=START_SHA, model_sha256="1"*64, frozen=True, eval_mode=True)
    change(value)
    with pytest.raises(ValueError):
        validate_anchor_identity(value, expected_model_sha="1"*64)
