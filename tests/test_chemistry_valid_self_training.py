"""Allocated-only actual trainer integration for separate prior self-NLL."""
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from latentmol import chemistry_continuation as continuation
from latentmol.atom_preserving_chemistry import AtomPreservingRepairTeacher
from latentmol.chemistry_classification import VERSION as CLASSIFICATION_VERSION
from latentmol.chemistry_execution_validation import _restore, _snapshot
from latentmol.chemistry_valid_self_guidance import VERSION
from latentmol.probability_repair import ProbabilityRepairConfig
from latentmol.train import capture_rng_state
from test_chemistry_continuation import _captured_gradients, _hard
from test_chemistry_valid_self_guidance import device
from test_failure_guidance_training import _configured_models


def _models(device):
    model, reference, config, protocol, composition = _configured_models()
    del protocol["failure_guidance"]
    # Constant, nondegenerate supported C4 complete graph: each carbon has
    # valence3. This supplies actual already-valid rows without fake teacher
    # receipts, while head biases retain genuine trainable confidence gradients.
    with torch.no_grad():
        for parameter in model.decoder.parameters():
            parameter.zero_()
        model.decoder.atom_head.bias[model.spec.atomic_numbers.index(6)] = 4.
        model.decoder.charge_head.bias[model.spec.formal_charges.index(0)] = 4.
        model.decoder.bond_head[-1].bias[model.spec.bond_names.index("SINGLE")] = 4.
    return model.to(device), reference.to(device), config, protocol, composition


def _teacher(model):
    return AtomPreservingRepairTeacher(model.spec, ProbabilityRepairConfig(max_edits=1, checks_per_depth=1),
                                      classification_version=CLASSIFICATION_VERSION)


def _run(model, reference, config, protocol, composition, generators, optimizer, *, overlap=False, size=2):
    with _teacher(model) as teacher:
        metrics = continuation.chemistry_update(model, reference, optimizer, _hard(model.spec), config, protocol,
            composition, teacher, generators, 7, 37, 35, microbatch_size=size, overlap=overlap)
    return metrics


def test_zero_weight_preserves_actual_original_loss_gradients_adam_model_and_all_rng(device):
    model, reference, config, protocol, composition = _models(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    generators = {"fresh_prior": torch.Generator().manual_seed(901)}
    initial = _snapshot(model, reference, optimizer, generators)
    observations = []
    for enabled in (False, True):
        _restore(initial, model, reference, optimizer, generators)
        current = deepcopy(protocol)
        if enabled:
            current["valid_prior_self_guidance"] = dict(version=VERSION, weight=0.)
        metrics = _run(model, reference, config, current, composition, generators, optimizer)
        observations.append((metrics, _snapshot(model, reference, optimizer, generators)))
    assert observations[0][0]["total"] == observations[1][0]["total"]
    assert observations[1][0]["valid_prior_self_guidance"]["eligible"] == 6
    assert observations[1][0]["valid_prior_self_weighted"] == 0.
    assert continuation.nested_equal(observations[0][1], observations[1][1])


def test_nonzero_self_weight_changes_decoder_gradient_but_not_encoder_or_draws(device):
    model, reference, config, protocol, composition = _models(device)
    # Isolate direct gradient scope; a common global clipping factor could
    # otherwise change encoder .grad despite a strictly decoder-only extra VJP.
    config = replace(config, clip_grad=1e6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    generators = {"fresh_prior": torch.Generator().manual_seed(901)}
    initial = _snapshot(model, reference, optimizer, generators)
    observations = []
    for weight in (0., .01):
        _restore(initial, model, reference, optimizer, generators)
        current = dict(protocol, valid_prior_self_guidance=dict(version=VERSION, weight=weight))
        metrics = _run(model, reference, config, current, composition, generators, optimizer)
        observations.append((metrics, _captured_gradients(model), capture_rng_state(generators, device == "cuda")))
    left, right = observations
    assert right[0]["valid_prior_self_guidance"]["eligible"] == 6
    assert right[0]["valid_prior_self_nll"] > 0 and right[0]["valid_prior_self_weighted"] > 0
    assert right[0]["valid_prior_self_decoder_gradient_norm"] > 0
    assert right[0]["valid_prior_self_weighted"] == pytest.approx(.01 * right[0]["valid_prior_self_nll"])
    assert continuation.nested_equal(left[2], right[2])
    assert all(continuation.nested_equal(left[1][name], value) for name, value in right[1].items()
               if name.startswith("encoder."))
    assert any(not continuation.nested_equal(left[1][name], value) for name, value in right[1].items()
               if name.startswith("decoder."))


def test_sync_async_complete_updates_keep_semantics_moments_gradients_and_rng_exact(device):
    model, reference, config, protocol, composition = _models(device)
    protocol["valid_prior_self_guidance"] = dict(version=VERSION, weight=.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    generators = {"fresh_prior": torch.Generator().manual_seed(901)}
    initial = _snapshot(model, reference, optimizer, generators)
    observed = []
    for overlap in (False, True):
        _restore(initial, model, reference, optimizer, generators)
        metrics = _run(model, reference, config, protocol, composition, generators, optimizer, overlap=overlap)
        observed.append((metrics, _snapshot(model, reference, optimizer, generators)))
    assert continuation.nested_equal(observed[0][1], observed[1][1])
    omitted = ("search_seconds", "prepass_seconds", "teacher_phase_seconds")
    assert {key: value for key, value in observed[0][0].items() if key not in omitted} == {
        key: value for key, value in observed[1][0].items() if key not in omitted}


def test_full_and_streamed_self_mean_keep_same_joint_gradient_and_sampling(device):
    model, reference, config, protocol, composition = _models(device)
    protocol["valid_prior_self_guidance"] = dict(version=VERSION, weight=.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    generators = {"fresh_prior": torch.Generator().manual_seed(901)}
    initial = _snapshot(model, reference, optimizer, generators)
    observed = []
    for size in (2, 6):
        _restore(initial, model, reference, optimizer, generators)
        with _teacher(model) as teacher:
            metrics = continuation.chemistry_update(model, reference, optimizer, _hard(model.spec), config, protocol,
                composition, teacher, generators, 7, 37, 35, microbatch_size=size, do_step=False)
        observed.append((metrics, _captured_gradients(model), capture_rng_state(generators, device == "cuda")))
    left, right = observed
    assert left[0]["real_points"] == right[0]["real_points"] == 3
    assert left[0]["fresh_prior_points"] == right[0]["fresh_prior_points"] == 6
    assert left[0]["valid_prior_self_guidance"]["attempted"] == right[0]["valid_prior_self_guidance"]["attempted"] == 6
    assert left[0]["valid_prior_self_nll"] == pytest.approx(right[0]["valid_prior_self_nll"], rel=1e-6)
    assert continuation.nested_equal(left[2], right[2])
    for name, value in left[1].items():
        assert (value is None) == (right[1][name] is None)
        if value is not None:
            torch.testing.assert_close(value, right[1][name], rtol=3e-4, atol=3e-6)
