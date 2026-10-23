"""Allocated arithmetic fixtures, independent of actual pretrained validation."""

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mosaic.small_molecule.boltz2.dbmol import (
    assemble_components,
    contact_distogram_float64,
    contact_loss,
    native_bin_centers,
)
from mosaic.small_molecule.boltz2.native import heads


CONFIDENCE = {
    "iptm": -1.0,
    "complex_iplddt": -0.3,
    "complex_ipde": 0.3,
    "complex_pde": 0.1,
    "confidence_score": -0.3,
    "ptm": -0.1,
}


def primitive_fixture():
    values = {k: torch.tensor(float(i + 1), requires_grad=True)
              for i, k in enumerate(CONFIDENCE)}
    values.update(
        affinity_pred_value=torch.tensor(2.0, requires_grad=True),
        affinity_logits_binary=torch.tensor(100.0, requires_grad=True),
        affinity_probability_binary=torch.tensor(0.75, requires_grad=True),
        dbmol_contact_loss=torch.tensor(3.0, dtype=torch.float64,
                                        requires_grad=True),
    )
    return values


def test_exact_signs_and_weights_apply_once_without_probability_saturation():
    values = primitive_fixture()
    actual = assemble_components(values)
    confidence = sum(coef * values[name].double()
                     for name, coef in CONFIDENCE.items())
    affinity = values["affinity_pred_value"].double() - values["affinity_logits_binary"].double()
    expected = affinity + 2.5 * values["dbmol_contact_loss"] + 2.5 * confidence
    assert actual["dbmol_total_loss"].dtype == torch.float64
    torch.testing.assert_close(actual["dbmol_total_loss"], expected, atol=0, rtol=0)
    torch.testing.assert_close(actual["dbmol_affinity_loss"], affinity, atol=0, rtol=0)
    torch.testing.assert_close(actual["dbmol_confidence_loss"], confidence, atol=0, rtol=0)
    actual["dbmol_total_loss"].backward()
    assert values["affinity_pred_value"].grad.item() == 1.0
    assert values["affinity_logits_binary"].grad.item() == -1.0
    assert values["affinity_probability_binary"].grad is None
    assert values["dbmol_contact_loss"].grad.item() == 2.5
    for key, coefficient in CONFIDENCE.items():
        assert values[key].grad.item() == pytest.approx(2.5 * coefficient)


@pytest.mark.parametrize("key", tuple(CONFIDENCE) + (
    "affinity_pred_value", "affinity_logits_binary", "dbmol_contact_loss"))
@pytest.mark.parametrize("fault", ("missing", "nonscalar", "nonfinite"))
def test_missing_malformed_nonfinite_component_is_not_silently_omitted(key, fault):
    values = primitive_fixture()
    if fault == "missing":
        del values[key]
    elif fault == "nonscalar":
        values[key] = torch.ones(2)
    else:
        values[key] = torch.tensor(float("nan"))
    with pytest.raises((ValueError, FloatingPointError)):
        assemble_components(values)


def contact_fixture():
    probabilities = torch.full((1, 4, 4, 4), 0.25, dtype=torch.float64)
    probabilities[0, 0, 2] = torch.tensor([.8, .1, .05, .05])
    probabilities[0, 0, 3] = torch.tensor([.1, .1, .4, .4])
    probabilities[0, 1, 2] = torch.tensor([.2, .1, .3, .4])
    probabilities[0, 1, 3] = torch.tensor([.3, .3, .2, .2])
    probabilities = torch.cat((probabilities[..., :2],
                               probabilities[..., 2:3].expand(-1, -1, -1, 31) / 31,
                               probabilities[..., 3:4].expand(-1, -1, -1, 31) / 31), -1)
    return (probabilities.log().requires_grad_(),
            torch.tensor([True, True, False, False]),
            torch.tensor([False, False, True, True]),
            torch.cat((torch.tensor([2., 6.]), torch.arange(10., 72.))).double())


def test_contact_reduces_protein_to_ligand_best_per_residue_and_global_anchor():
    logits, pocket, ligand, centers = contact_fixture()
    result = contact_loss(logits, pocket, ligand, bin_centers=centers)
    # Compute from the actual fixture probabilities, avoiding implicit FP32
    # constants being interpreted as exact decimal FP64 input.
    probs = logits.softmax(-1)
    first = -torch.log(probs[0, 0, 2, :2].sum() + 1e-8)
    second = -torch.log(probs[0, 1, 3, :2].sum() + 1e-8)
    expected = ((first + second) / 2 + first) / 2
    torch.testing.assert_close(result["dbmol_contact_loss"], expected,
                               atol=1e-14, rtol=1e-14)
    result["dbmol_contact_loss"].backward()
    assert logits.grad[0, 0, 2].abs().max() > 0
    assert logits.grad[0, 1, 3].abs().max() > 0
    assert logits.grad[0, 0, 3].abs().max() == 0
    assert logits.grad[0, 1, 2].abs().max() == 0
    assert logits.grad[0, 2:].abs().max() == 0  # opposite orientation excluded


def test_contact_epsilon_is_additive_logspace_not_clamping_or_minimum_probability():
    logits = torch.cat((torch.tensor([-500., -501.], dtype=torch.float64),
                        torch.linspace(0., 1., 62, dtype=torch.float64)))
    logits = logits.expand(1, 2, 2, 64).clone().requires_grad_()
    result = contact_loss(logits, torch.tensor([True, False]),
                          torch.tensor([False, True]),
                          bin_centers=torch.cat((torch.tensor([2., 6.]), torch.arange(10., 72.))))
    result["dbmol_contact_loss"].backward()
    assert torch.isfinite(result["dbmol_contact_loss"])
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().max() > 0  # a probability clamp would erase this
    assert float(result["dbmol_contact_loss"]) == pytest.approx(-math.log(1e-8))


@pytest.mark.parametrize("fault", ("overlap", "empty_pocket", "empty_ligand", "mask_dtype", "mask_size", "bad_centers"))
def test_contact_rejects_unsupported_masks_without_silent_fallback(fault):
    logits, pocket, ligand, centers = contact_fixture()
    if fault == "overlap":
        ligand = pocket
    elif fault == "empty_pocket":
        pocket[:] = False
    elif fault == "empty_ligand":
        ligand[:] = False
    elif fault == "mask_dtype":
        pocket = pocket.float()
    elif fault == "mask_size":
        pocket = pocket[:-1]
    else:
        centers = centers[:-1]
    with pytest.raises(ValueError):
        contact_loss(logits, pocket, ligand, bin_centers=centers)


def test_contact_single_distogram_axis_matches_native_shape():
    logits, pocket, ligand, centers = contact_fixture()
    a = contact_loss(logits, pocket, ligand, bin_centers=centers)
    b = contact_loss(logits.unsqueeze(3), pocket, ligand, bin_centers=centers)
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)


def test_native_bin_representatives_construct_before_widening_and_strict_cutoff():
    logits = torch.zeros(1, 2, 2, 64, dtype=torch.float32)
    centers = native_bin_centers(logits)
    boundaries = torch.linspace(2., 22., 63, dtype=torch.float32)
    half = (boundaries[1] - boundaries[0]) / 2
    expected = torch.cat((boundaries[:1] - half,
                          (boundaries[:-1] + boundaries[1:]) / 2,
                          boundaries[-1:] + half))
    assert centers.dtype == torch.float32 and centers.shape == (64,)
    torch.testing.assert_close(centers, expected, atol=0, rtol=0)
    cutoff = float(centers[19])
    assert not bool((centers.double() < cutoff)[19])
    assert bool((centers.double() < cutoff)[18])
    with pytest.raises(ValueError):
        native_bin_centers(torch.zeros(1, 2, 2, 63))


class Distogram(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_bins, self.num_distograms = 64, 1
        self.distogram = nn.Linear(3, 64)
        with torch.no_grad():
            self.distogram.weight.copy_(torch.arange(192).reshape(64, 3) / 311)
            self.distogram.bias.copy_(torch.arange(64) / 137)
        self.eval().requires_grad_(False)


def test_contact_head_widens_before_symmetric_affine_preserving_input_vjp_and_frozen_state():
    module = Distogram()
    pair = torch.arange(12, dtype=torch.float32).reshape(1, 2, 2, 3).requires_grad_()
    state_before = {k: v.clone() for k, v in module.state_dict().items()}
    result = contact_distogram_float64(module, pair)
    widened = pair.double()
    expected = torch.nn.functional.linear(widened + widened.transpose(1, 2),
                                          module.distogram.weight.double(),
                                          module.distogram.bias.double())
    expected = expected.reshape(1, 2, 2, 1, 64)
    assert result.dtype == torch.float64
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
    actual_grad, = torch.autograd.grad(result.sum(), pair, retain_graph=True)
    expected_grad, = torch.autograd.grad(expected.sum(), pair)
    torch.testing.assert_close(actual_grad, expected_grad, atol=0, rtol=0)
    assert actual_grad.abs().max() > 0
    for name, value in module.state_dict().items():
        torch.testing.assert_close(value, state_before[name], atol=0, rtol=0)
    assert all(not p.requires_grad and p.grad is None for p in module.parameters())


@pytest.mark.parametrize("fault", ("training", "trainable", "bins", "multiple", "pair_dtype", "weight_dtype", "shape"))
def test_contact_head_rejects_unvalidated_layout_precision_or_state(fault):
    module, pair = Distogram(), torch.zeros(1, 2, 2, 3)
    if fault == "training": module.train()
    elif fault == "trainable": module.requires_grad_(True)
    elif fault == "bins": module.num_bins = 63
    elif fault == "multiple": module.num_distograms = 2
    elif fault == "pair_dtype": pair = pair.double()
    elif fault == "weight_dtype": module.double()
    else: pair = torch.zeros(1, 2, 3, 3)
    with pytest.raises(ValueError):
        contact_distogram_float64(module, pair)


class EnsembleHead(nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.bias = bias

    def forward(self, *, z, **_):
        return {"affinity_pred_value": z.sum().reshape(1),
                "affinity_logits_binary": (z.sum() * 0 + self.bias).reshape(1)}


def test_native_ensemble_mean_raw_logit_is_distinct_from_probability_reconstructions():
    model = SimpleNamespace(
        input_embedder=lambda f, **_: f["inputs"], affinity_ensemble=True,
        affinity_module1=EnsembleHead(-2.), affinity_module2=EnsembleHead(4.))
    features = {"token_pad_mask": torch.ones(1, 2),
                "mol_type": torch.tensor([[0, 3]]),
                "affinity_token_mask": torch.tensor([[0, 1]]),
                "inputs": torch.ones(1, 2, 1)}
    values = heads(model, features, torch.ones(1, 2, 2, 1), torch.zeros(1, 2, 3))
    expected_probability = (torch.sigmoid(torch.tensor(-2.)) + torch.sigmoid(torch.tensor(4.))) / 2
    assert float(values[2]) == 1.0
    torch.testing.assert_close(values[1], expected_probability, atol=0, rtol=0)
    assert not torch.allclose(values[1], values[2].sigmoid())
    assert not torch.allclose(torch.logit(values[1]), values[2])


def test_total_and_live_primitive_recomposition_retain_gradients_into_both_graph_leaves():
    x = torch.linspace(.1, .9, 150).reshape(25, 6).requires_grad_()
    e = torch.linspace(.2, .8, 1500).reshape(300, 5).requires_grad_()
    primitive = {name: (i + 1) * x[:, i % 6].sum() + e[:, i % 5].sum()
                 for i, name in enumerate(CONFIDENCE)}
    primitive.update(affinity_pred_value=x[:, 1].sum() + e[:, 1].sum(),
                     affinity_logits_binary=2 * x[:, 2].sum() - e[:, 2].sum(),
                     dbmol_contact_loss=x[:, 3].double().sum() + e[:, 3].double().sum())
    actual = assemble_components(primitive)
    independent = primitive["affinity_pred_value"].double() - primitive["affinity_logits_binary"].double()
    independent = independent + 2.5 * primitive["dbmol_contact_loss"].double()
    independent = independent + 2.5 * sum(coef * primitive[key].double()
                                         for key, coef in CONFIDENCE.items())
    torch.testing.assert_close(actual["dbmol_total_loss"], independent, atol=0, rtol=0)
    ga = torch.autograd.grad(actual["dbmol_total_loss"], (x, e), retain_graph=True)
    gi = torch.autograd.grad(independent, (x, e))
    for left, right in zip(ga, gi):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
        assert torch.isfinite(left).all() and left.abs().max() > 0
