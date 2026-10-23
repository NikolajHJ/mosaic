"""HPC-only arithmetic fixtures; native saved-state validation is separate.

These do not establish a remedy for native-model/decoder kinks or biochemical
performance. The low-signal fixture isolates rounding in the loss reduction.
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from latentmol.boltz.dbmol_objective import (
    CONTACT64_ARITHMETIC, NATIVE_ARITHMETIC, CONFIDENCE_WEIGHTS,
    DBMolObjectiveConfig, contact_loss, native_bin_centers, reduce_objective,
    reduce_profile,
)
from test_dbmol_objective import profile


def context():
    return {'token_pad_mask': torch.ones(1, 2), 'mol_type': torch.tensor([[0, 3]]),
            'asym_id': torch.tensor([[0, 1]]), 'residue_index': torch.tensor([[2, 0]]),
            'affinity_token_mask': torch.tensor([[0, 1]])}


def primitives():
    affinity = {key: torch.tensor(value, requires_grad=True) for key, value in
                (('affinity_pred_value', 1.234567), ('affinity_logits_binary', .3456789))}
    structure = {key: torch.tensor((i + 1) * .1234567, requires_grad=True)
                 for i, key in enumerate(CONFIDENCE_WEIGHTS)}
    return affinity, structure


def test_absent_arithmetic_keeps_exact_legacy_config_record(tmp_path):
    cfg = profile(tmp_path)
    legacy = cfg.record()
    assert 'objective_arithmetic' not in legacy
    assert cfg.objective_arithmetic == NATIVE_ARITHMETIC
    assert DBMolObjectiveConfig.from_mapping(legacy, tmp_path).record() == legacy
    assert replace(cfg, objective_arithmetic=NATIVE_ARITHMETIC).record() == legacy
    assert replace(cfg, objective_arithmetic=CONTACT64_ARITHMETIC).record() == {
        **legacy, 'objective_arithmetic': CONTACT64_ARITHMETIC}
    with pytest.raises(ValueError, match='arithmetic'):
        profile(tmp_path, objective_arithmetic='unknown')


def test_native_entrypoint_keeps_exact_legacy_values_and_gradients(tmp_path):
    cfg = profile(tmp_path)
    affinity, structure = primitives()
    logits = torch.linspace(-2., 3., 256).reshape(1, 2, 2, 64).requires_grad_()
    masks = (torch.tensor([True, False]), torch.tensor([False, True]))
    old = reduce_objective({**affinity, **structure, **contact_loss(logits, *masks)}, cfg)
    actual = reduce_profile(affinity, structure, logits, context(), cfg)
    assert old.keys() == actual.keys()
    for key in old:
        assert torch.equal(old[key], actual[key])
    inputs = (logits, *affinity.values(), *structure.values())
    old_grad = torch.autograd.grad(old['dbmol_total_loss'], inputs, retain_graph=True)
    new_grad = torch.autograd.grad(actual['dbmol_total_loss'], inputs)
    assert all(torch.equal(a, b) for a, b in zip(old_grad, new_grad))
    # Historical helper fixtures/immutable config classes have no new field.
    historical = SimpleNamespace(**{k: v for k, v in cfg.__dict__.items()
                                    if k != 'objective_arithmetic'})
    assert torch.equal(reduce_objective(old, historical)['dbmol_total_loss'], old['dbmol_total_loss'])


def test_precision_entrypoint_matches_cached_control_preserving_native_primitives(tmp_path):
    cfg = profile(tmp_path, objective_arithmetic=CONTACT64_ARITHMETIC)
    affinity, structure = primitives()
    logits = torch.linspace(-2., 3., 256).reshape(1, 2, 2, 64).requires_grad_()
    masks = (torch.tensor([True, False]), torch.tensor([False, True]))
    contact = contact_loss(logits.double(), *masks, bin_centers=native_bin_centers(logits).double())
    expected = reduce_objective({**{k: v.double() for k, v in {**affinity, **structure}.items()},
                                 **contact}, replace(cfg, objective_arithmetic=NATIVE_ARITHMETIC))
    actual = reduce_profile(affinity, structure, logits, context(), cfg)
    for key in ('dbmol_contact_loss', 'dbmol_affinity_loss', 'dbmol_confidence_loss', 'dbmol_total_loss'):
        assert actual[key].dtype == torch.float64
        assert torch.equal(actual[key], expected[key])
    for key, value in {**affinity, **structure}.items():
        assert actual[key] is value and actual[key].dtype == torch.float32
    inputs = (logits, *affinity.values(), *structure.values())
    ga = torch.autograd.grad(actual['dbmol_total_loss'], inputs, retain_graph=True)
    ge = torch.autograd.grad(expected['dbmol_total_loss'], inputs)
    assert all(torch.equal(a, b) and a.dtype == torch.float32 for a, b in zip(ga, ge))


def test_bin_partition_uses_original_dtype_at_rounded_boundary(tmp_path):
    logits = torch.linspace(-2., 3., 256).reshape(1, 2, 2, 64)
    centers = native_bin_centers(logits)
    cfg = profile(tmp_path, objective_arithmetic=CONTACT64_ARITHMETIC,
                  distance_cutoff=float(centers[19]))
    affinity, structure = primitives()
    actual = reduce_profile(affinity, structure, logits, context(), cfg)
    masks = (torch.tensor([True, False]), torch.tensor([False, True]))
    expected = contact_loss(logits.double(), *masks, cutoff=cfg.distance_cutoff,
                            bin_centers=centers.double())
    assert torch.equal(actual['dbmol_contact_loss'], expected['dbmol_contact_loss'])
    assert not bool((centers.double() < cfg.distance_cutoff)[19])


def test_widened_contact_retains_small_native_logit_change_lost_by_fp32_reduction(tmp_path):
    affinity, structure = primitives()
    masks = (torch.tensor([True, False]), torch.tensor([False, True]))
    template = torch.zeros(1, 2, 2, 64)
    near = native_bin_centers(template) < 8.
    template[..., ~near] = 2.
    def score(point, mode):
        logits = template.clone()
        logits[0, 0, 1, 0] = .001 * point[0]
        return reduce_profile(affinity, structure, logits, context(),
                              profile(tmp_path, objective_arithmetic=mode))['dbmol_contact_loss']
    z = torch.zeros(1, requires_grad=True)
    loss = score(z, CONTACT64_ARITHMETIC)
    gradient, = torch.autograd.grad(loss, z)
    epsilon = 1e-4
    native_difference = float(score(z.detach() + epsilon, NATIVE_ARITHMETIC)
                              - score(z.detach() - epsilon, NATIVE_ARITHMETIC)) / (2 * epsilon)
    widened_difference = float(score(z.detach() + epsilon, CONTACT64_ARITHMETIC)
                               - score(z.detach() - epsilon, CONTACT64_ARITHMETIC)) / (2 * epsilon)
    assert gradient.item() != 0 and native_difference == 0
    assert widened_difference == pytest.approx(gradient.item(), rel=1e-5, abs=1e-10)
