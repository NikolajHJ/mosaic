"""Population size and frozen bond supervision for the compact VAE."""

import itertools

import pytest
import torch
from torch.nn import functional as F

from latentmol.representation import DecoderOutput, GraphSpec
from latentmol.training.compact_prior import (
    CompactPriorGuidance, argmax_count_loss, atom_count_probabilities, count_kl_per_slot,
    log_atom_count_probabilities,
)


def test_count_distribution_keeps_uncertainty_even_at_equal_mean():
    # Both molecules have expected size 1, but different probabilities of 0, 1 and 2.
    occupancy = torch.tensor([[.25, .75], [.5, .5]], dtype=torch.float64)
    logits = torch.stack([1 - occupancy, occupancy], dim=-1).log().requires_grad_()
    pmf = atom_count_probabilities(logits)
    torch.testing.assert_close(pmf, logits.new_tensor([[.1875, .625, .1875], [.25, .5, .25]]))
    pmf[0, 1].backward()
    assert logits.grad[0, 0, 1] < 0 < logits.grad[0, 1, 1]


def test_log_count_probability_and_gradient_match_exhaustive_categorical_enumeration():
    logits = torch.tensor([[[1., 2., 3.], [2., 1., -1.], [3., -2., 1.]]],
                          dtype=torch.float64, requires_grad=True)
    p = logits.softmax(dim=-1)[0]
    entries = [[] for _ in range(4)]
    for labels in itertools.product(range(3), repeat=3):
        entries[sum(label != 0 for label in labels)].append(p[range(3), list(labels)].prod())
    ref_log = torch.stack([sum(e) for e in entries]).log()
    result = log_atom_count_probabilities(logits)[0]
    torch.testing.assert_close(result, ref_log)
    weights = logits.new_tensor([0.13, 0.7, 0.26, 0.05])
    expected_grad, = torch.autograd.grad(ref_log @ weights, logits)
    actual_grad, = torch.autograd.grad(result @ weights, logits)
    torch.testing.assert_close(expected_grad, actual_grad)
    torch.testing.assert_close(log_atom_count_probabilities(logits.flip(1))[0], result)


@pytest.mark.parametrize("real_bias", [100., -100.])
def test_count_kl_recovers_a_gradient_from_saturated_empty_or_full_slots(real_bias):
    # A shared parameter controlling all 50 slots is a direct test of the collapse direction.
    bias = torch.tensor(real_bias, requires_grad=True)
    real = bias.expand(4, 50)
    logits = torch.stack([torch.zeros_like(real), real], dim=-1)
    log_pmfs = log_atom_count_probabilities(logits)
    assert log_pmfs.isfinite().all()
    target = F.one_hot(torch.tensor(25), 51).float()
    loss = count_kl_per_slot(log_pmfs, target)
    grad, = torch.autograd.grad(loss, bias)
    assert float(grad) == pytest.approx(0.5 if real_bias > 0 else -0.5, abs=2e-4)
    old = (atom_count_probabilities(logits).mean(0).cumsum(0)[:-1] - target.cumsum(0)[:-1]).square().mean()
    old_grad, = torch.autograd.grad(old, bias)
    assert abs(float(old_grad)) < 1e-9


def test_count_kl_allows_latents_to_specialize_and_legitimate_full_graphs():
    # One latent has one atom; another occupies both slots. This matches the reference.
    real = torch.tensor([[50., -50.], [50., 50.]])
    logits = torch.stack([torch.zeros_like(real), real], dim=-1)
    target = torch.tensor([0., 0.5, 0.5])
    log_pmfs = log_atom_count_probabilities(logits)
    assert abs(float(count_kl_per_slot(log_pmfs, target))) < 1e-5
    assert float(count_kl_per_slot(log_pmfs[:1], target)) > 10
    assert abs(float(count_kl_per_slot(log_pmfs[1:], torch.tensor([0., 0., 1.])))) < 1e-5


def test_argmax_auxiliary_trains_the_winning_element_and_selects_the_right_number_of_slots():
    # Every slot assigns a majority of total mass to elements, but NO_ATOM wins the argmax.
    logits = torch.tensor([[[0.0, -0.1, -0.2], [0.0, -0.2, -0.3], [0.0, -0.3, -0.4]],
                           [[0.0, -0.4, -0.6], [0.0, -0.5, -0.6], [0.0, -0.6, -0.7]]],
                          requires_grad=True)
    assert (logits.softmax(-1)[..., 0] < .5).all() and (logits.argmax(-1) == 0).all()
    # Empirical sizes 1 and 3. The stronger graph receives size 3.
    ref = torch.tensor([0., .5, .5])
    loss = argmax_count_loss(logits, ref, update=0)
    grad, = torch.autograd.grad(loss, logits)
    assert ((grad[..., 1] < 0).sum(-1) == torch.tensor([3, 1])).all()
    torch.testing.assert_close(grad[..., 0], -grad[..., 1])
    assert not grad[..., 2].any()
    torch.testing.assert_close(loss, argmax_count_loss(logits.flip(0).flip(1), ref, update=0))
    # Raising the trained slot winners improves the actual hard output as well.
    updated = logits.detach() - 20 * grad
    assert (updated.argmax(-1).ne(0).sum(-1) == torch.tensor([3, 1])).all()


def test_argmax_quantiles_include_rare_full_counts_across_updates():
    logits = torch.zeros(1, 2, 2, requires_grad=True)
    # Reference is 99% one atom, 1% two atoms, with one prior per update.
    ref = torch.tensor([0., .99])
    assigned = []
    for update in range(150):
        grad, = torch.autograd.grad(argmax_count_loss(logits, ref, update), logits)
        assigned.append(int((grad[0, :, 1] < 0).sum()))
    assert set(assigned) == {1, 2}
    assert 0 < assigned.count(2) < 5


@pytest.mark.parametrize("count_kl_weight, argmax_weight", [(0., 0.), (1., 0.), (1., 1.)])
def test_real_structural_repair_supervises_bonds_and_population_size_separately(count_kl_weight, argmax_weight):
    # Each draw is two disconnected carbons. The only available bridge joins them.
    spec = GraphSpec(max_nodes=3)
    atom = torch.tensor([[spec.atomic_numbers.index(6)] * 2 + [0]])
    charge = torch.full_like(atom, spec.formal_charges.index(0))
    bond = torch.zeros((1, 3, 3), dtype=torch.long)
    output = DecoderOutput(*[(F.one_hot(labels, classes).float() * 5).requires_grad_()
                             for labels, classes in zip((atom, charge, bond),
                                 (len(spec.atomic_numbers), len(spec.formal_charges), len(spec.bond_names)))])

    class Model:
        def decode(self, z):
            assert len(z) == 1
            return output

    config = dict(bond_weight=.375, count_weight=1., workers=1, count_window=2, count_histogram={"1": 7})
    if count_kl_weight:
        config["count_kl_weight"] = count_kl_weight
    if argmax_weight:
        config["argmax_count_weight"] = argmax_weight
    guide = CompactPriorGuidance(config, spec, "cpu")
    try:
        for index, expected_window in enumerate((1, 2, 2)):
            loss, metrics = guide.loss(Model(), torch.zeros(1, 4))
            assert metrics["accepted"] == 1 and metrics["count_window_points"] == expected_window
            assert ("count_kl_per_slot" in metrics) == bool(count_kl_weight)
            assert ("argmax_count_loss" in metrics) == bool(argmax_weight)
            assert float(loss.detach()) == pytest.approx(.375 * metrics["bond_loss"] + metrics["count_cdf_loss"]
                                                          + count_kl_weight * metrics.get("count_kl_per_slot", 0.)
                                                          + argmax_weight * metrics.get("argmax_count_loss", 0.))
            if index == 0:
                # A one-atom training prior gives reference P(N<=k) = [0,1,1].
                pmf = atom_count_probabilities(output.atoms).squeeze(0)
                expected = (pmf.cumsum(0)[:-1] - pmf.new_tensor([0, 1, 1])).square().mean()
                assert metrics["count_cdf_loss"] == pytest.approx(float(expected.detach()))
            loss.backward()
        assert output.charges.grad is None
        assert output.atoms.grad.isfinite().all() and output.atoms.grad.abs().sum() > 0
        assert output.bonds.grad[0, 0, 1, spec.bond_names.index("SINGLE")] < 0
        assert all(not histogram.requires_grad for histogram, _ in guide.history)
    finally:
        guide.close()
