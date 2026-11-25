"""Independent event enumeration and invariance checks; execute on allocated HPC.

The reference below does not read prepared capacities or call the production
predicate. It derives a small, independently written conditional event directly
from hard fixture labels and enumerates every incident categorical assignment.
"""
from __future__ import annotations

from dataclasses import replace
import itertools
import math

import pytest
import torch

from latentmol.chemical_environment import (
    neighbor_environment_loss,
    prepare_neighbor_environment,
)
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch


_POLICY = {"version": "common-organic-neighbor-v1", "weight": 0.3}
# Intentionally independent of chemical_environment_policy.VALENCES. Only the
# supported element/charge states actually used by the enumeration fixtures.
_CAP = {(6, 0): 4, (7, 0): 3, (7, 1): 4, (8, 0): 2, (9, 0): 1, (16, 0): 6}
_ORDER = {"NONE": 0, "SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 1}


def _fixture(elements, edges=(), charges=None, *, seed=23):
    """Return finite hard-context logits and a separate soft live bond tensor."""
    spec = GraphSpec(max_nodes=len(elements))
    labels = torch.tensor([[spec.atomic_numbers.index(value) for value in elements]])
    q = charges if charges is not None else [0] * len(elements)
    charge = torch.tensor([[spec.formal_charges.index(value) for value in q]])
    bonds = torch.zeros((1, spec.max_nodes, spec.max_nodes), dtype=torch.long)
    for i, j, name in edges:
        bonds[0, i, j] = bonds[0, j, i] = spec.bond_names.index(name)
    hard = HardGraphBatch(labels, charge, bonds)
    probabilities = hard.to_relaxed(spec)
    initial = DecoderOutput(*(value.double() * 40 - 20 for value in
                              (probabilities.atoms, probabilities.charges, probabilities.bonds)))
    generator = torch.Generator().manual_seed(seed)
    live_bonds = torch.randn(initial.bonds.shape, dtype=torch.float64, generator=generator)
    live = DecoderOutput(initial.atoms.clone().requires_grad_(),
                         initial.charges.clone().requires_grad_(), live_bonds.requires_grad_())
    return spec, hard, initial, live


def _enumerated_risks(decoded, hard, spec):
    """Exact local event using direct Python tuples, not the production DP."""
    p = decoded.symmetric_bond_logits.softmax(-1)
    hard_atoms = hard.atoms[0].tolist()
    elements = [spec.atomic_numbers[label] for label in hard_atoms]
    charges = [spec.formal_charges[label] for label in hard.charges[0].tolist()]
    active = [i for i, element in enumerate(elements) if element]
    orders = [_ORDER[name] for name in spec.bond_names]
    hard_orders = [[orders[label] for label in line] for line in hard.bonds[0].tolist()]
    risks = []
    for i in active:
        neighbors = [j for j in active if j != i]
        cap = _CAP[elements[i], charges[i]]
        mass = p.new_zeros(())
        for labels in itertools.product(range(len(orders)), repeat=len(neighbors)):
            edge_orders = [orders[label] for label in labels]
            if sum(edge_orders) > cap:
                continue
            if elements[i] == 16:
                count = sum(int(order > 0 and elements[j] != 9)
                            for j, order in zip(neighbors, edge_orders))
                if count > 4:
                    continue
            allowed = True
            for j, order in zip(neighbors, edge_orders):
                other = sum(hard_orders[j][k] for k in active if k != i)
                residual = max(_CAP[elements[j], charges[j]] - other, 0)
                if order and order > residual:
                    allowed = False
                    break
            if allowed:
                probability = p.new_ones(())
                for j, label in zip(neighbors, labels):
                    probability = probability * p[0, i, j, label]
                mass = mass + probability
        risks.append(-mass.clamp_max(1).log())
    return torch.stack(risks)


@pytest.mark.parametrize("elements,charges,edges", [
    ([16, 7, 6, 8], [0, 0, 0, 0], [(0, 1, "SINGLE"), (1, 2, "SINGLE")]),
    ([16, 7, 6, 8], [0, 1, 0, 0], [(0, 1, "SINGLE"), (1, 2, "SINGLE")]),
    ([16, 9, 6, 8], [0, 0, 0, 0], [(0, 2, "SINGLE"), (0, 3, "DOUBLE")]),
])
def test_conditional_event_matches_independent_enumeration_and_vjp(elements, charges, edges):
    spec, hard, initial, live = _fixture(elements, edges, charges)
    prepared = prepare_neighbor_environment(initial, spec, _POLICY)
    result = neighbor_environment_loss(live, prepared, spec)
    reference = _enumerated_risks(live, hard, spec)
    torch.testing.assert_close(result["per_center_risk"][0], reference, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(result["loss"], reference.mean(), rtol=2e-12, atol=2e-12)
    actual = torch.autograd.grad(result["loss"], (live.atoms, live.charges, live.bonds),
                                 retain_graph=True)
    expected = torch.autograd.grad(reference.mean(), live.bonds)[0]
    torch.testing.assert_close(actual[2], expected, rtol=2e-11, atol=2e-12)
    assert torch.count_nonzero(actual[2]) > 0
    assert torch.count_nonzero(actual[0]) == 0
    assert torch.count_nonzero(actual[1]) == 0


def test_permutation_does_not_change_event_law():
    spec, _, initial, live = _fixture([16, 7, 6, 8], [(0, 1, "SINGLE"), (1, 2, "SINGLE")])
    original = neighbor_environment_loss(live, prepare_neighbor_environment(initial, spec, _POLICY), spec)
    permutation = torch.tensor([2, 0, 3, 1])

    def permute(decoded):
        return DecoderOutput(decoded.atoms[:, permutation], decoded.charges[:, permutation],
                             decoded.bonds[:, permutation][:, :, permutation])

    candidate = neighbor_environment_loss(permute(live),
        prepare_neighbor_environment(permute(initial), spec, _POLICY), spec)
    torch.testing.assert_close(candidate["loss"], original["loss"], rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(candidate["per_center_risk"],
                               original["per_center_risk"][:, permutation], rtol=2e-12, atol=2e-12)


def test_padding_and_pairwise_logit_offsets_do_not_change_event_law():
    spec, _, initial, live = _fixture([16, 7, 6, 8], [(0, 1, "SINGLE"), (1, 2, "SINGLE")])
    original = neighbor_environment_loss(live, prepare_neighbor_environment(initial, spec, _POLICY), spec)
    padded_spec = GraphSpec(max_nodes=6)

    def pad(decoded):
        # Hard occupancy chooses NONE for the extra slots. Their arbitrary bond
        # logits must never leak into an occupied center's conditional event.
        atoms = torch.full((1, 6, len(spec.atomic_numbers)), -20., dtype=torch.float64)
        atoms[:, 4:, 0] = 20.
        atoms[:, :4] = decoded.atoms
        charges = torch.full((1, 6, len(spec.formal_charges)), -20., dtype=torch.float64)
        charges[:, 4:, spec.formal_charges.index(0)] = 20.
        charges[:, :4] = decoded.charges
        bonds = torch.full((1, 6, 6, len(spec.bond_names)), 0.7, dtype=torch.float64)
        bonds[:, :4, :4] = decoded.bonds
        return DecoderOutput(atoms, charges, bonds)

    padded = neighbor_environment_loss(pad(live),
        prepare_neighbor_environment(pad(initial), padded_spec, _POLICY), padded_spec)
    torch.testing.assert_close(padded["loss"], original["loss"], rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(padded["per_center_risk"][:, :4], original["per_center_risk"],
                               rtol=2e-12, atol=2e-12)
    assert torch.count_nonzero(padded["per_center_risk"][:, 4:]) == 0
    offsets = torch.arange(16, dtype=torch.float64).reshape(1, 4, 4, 1) / 7
    shifted = neighbor_environment_loss(replace(live, bonds=live.bonds + offsets),
        prepare_neighbor_environment(initial, spec, _POLICY), spec)
    torch.testing.assert_close(shifted["loss"], original["loss"], rtol=2e-12, atol=2e-12)


def test_sulfur_nonfluorine_rule_has_signal_below_generic_valence_ceiling():
    # This engineering context is accepted by the declared coordination profile;
    # it is not asserted to be an ordinary drug motif or experimentally stable.
    # A fifth SINGLE carbon edge has order sum5<=6, isolating the NEW rule.
    spec, _, initial, _ = _fixture([16, 6, 6, 6, 6, 6],
        [(0, j, "SINGLE") for j in range(1, 5)])
    bonds = initial.bonds.detach().clone()
    bonds[:, 0, 5] = bonds[:, 5, 0] = torch.tensor(
        [math.log(.7), math.log(.3), -60., -60., -60.], dtype=torch.float64)
    live = replace(initial, bonds=bonds.requires_grad_())
    result = neighbor_environment_loss(live, prepare_neighbor_environment(initial, spec, _POLICY), spec)
    torch.testing.assert_close(result["per_center_risk"][0, 0],
                               torch.tensor(-math.log(.7), dtype=torch.float64), rtol=2e-12, atol=2e-12)
    gradient = torch.autograd.grad(result["loss"], live.bonds)[0]
    assert gradient[0, 0, 5, spec.bond_names.index("SINGLE")] > 0
    assert gradient[0, 0, 5, spec.bond_names.index("NONE")] < 0


def test_fluorine_exemption_removes_only_coordination_count_not_valence():
    spec, _, initial, _ = _fixture([16, 6, 6, 6, 6, 9],
        [(0, j, "SINGLE") for j in range(1, 5)])
    bonds = initial.bonds.detach().clone()
    bonds[:, 0, 5] = bonds[:, 5, 0] = torch.tensor(
        [math.log(.7), math.log(.3), -60., -60., -60.], dtype=torch.float64)
    live = replace(initial, bonds=bonds.requires_grad_())
    result = neighbor_environment_loss(live, prepare_neighbor_environment(initial, spec, _POLICY), spec)
    assert result["per_center_risk"][0, 0] < 1e-12
