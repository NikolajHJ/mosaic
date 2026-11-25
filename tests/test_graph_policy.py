"""Shared policy retains old Boltz import and original probability semantics."""

import torch
import pytest

from latentmol.boltz.adapter import fixed_occupancy_graph as legacy_fixed_graph
from latentmol.graph_policy import fixed_occupancy_graph, graph_from_decoder, graph_from_mosaic
from latentmol.representation import DecoderOutput, GraphSpec


def test_mosaic_graph_remaps_sulfur_fluorine_slots_bonds_and_derivatives():
    spec = GraphSpec(max_nodes=5, atomic_numbers=(0, 17, 16, 9, 8, 7, 6),
                     formal_charges=(1, 0, -1), bond_names=("NONE", "AROMATIC", "TRIPLE", "DOUBLE", "SINGLE"))
    x = torch.softmax(torch.arange(36, dtype=torch.float64).reshape(2, 3, 6) / 7, -1).requires_grad_()
    e = torch.softmax(torch.arange(30, dtype=torch.float64).reshape(2, 3, 5) / 4, -1).requires_grad_()
    slots = [4, 1, 2]
    g = graph_from_mosaic(x, e, spec, active_slots=slots)
    assert torch.equal(g.atoms[:, slots, spec.atomic_numbers.index(16)], x[:, :, 3])
    assert torch.equal(g.atoms[:, slots, spec.atomic_numbers.index(9)], x[:, :, 4])
    assert torch.equal(g.occupancy[0], torch.tensor([0., 1., 1., 0., 1.], dtype=x.dtype))
    assert torch.all(g.charges[..., spec.formal_charges.index(0)] == 1)
    assert torch.equal(g.bonds, g.bonds.transpose(1, 2))
    assert torch.all(g.bonds[:, range(5), range(5), 0] == 1)
    assert torch.equal(g.bonds[:, 1, 2, spec.bond_names.index("DOUBLE")], e[:, 2, 2])
    assert torch.all(g.bonds[:, 0, :, 0] == 1)
    value = g.atoms[:, 4, spec.atomic_numbers.index(16)].sum() + g.bonds[:, 2, 1, spec.bond_names.index("DOUBLE")].sum()
    dx, de = torch.autograd.grad(value, (x, e))
    assert torch.all(dx[:, 0, 3] == 1) and dx.count_nonzero() == 2
    assert torch.all(de[:, 2, 2] == 1) and de.count_nonzero() == 2
    with pytest.raises(ValueError, match="distinct"):
        graph_from_mosaic(x, e, spec, active_slots=[1, 1, 2])

def test_legacy_import_is_same_function_and_shared_policy_chooses_explicitly():
    assert legacy_fixed_graph is fixed_occupancy_graph
    spec = GraphSpec(max_nodes=3)
    decoded = DecoderOutput(torch.randn(1, 3, 10), torch.randn(1, 3, 5), torch.randn(1, 3, 3, 5))
    active = torch.tensor([True, False, True])
    soft = graph_from_decoder(decoded, spec, temperature=0.7)
    expected_soft = decoded.probabilities(0.7)
    fixed = graph_from_decoder(decoded, spec, temperature=0.7, fixed_active=active)
    expected_fixed = legacy_fixed_graph(decoded, active, spec, 0.7)
    for name in ("atoms", "charges", "bonds"):
        torch.testing.assert_close(getattr(soft, name), getattr(expected_soft, name), rtol=0, atol=0)
        torch.testing.assert_close(getattr(fixed, name), getattr(expected_fixed, name), rtol=0, atol=0)
    assert not torch.allclose(soft.occupancy, active.float()[None])
    torch.testing.assert_close(fixed.occupancy, active.float()[None], rtol=0, atol=0)


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("field,shape", [
    ("atoms", (1, 2, 10)), ("atoms", (1, 3, 9)),
    ("charges", (2, 3, 5)), ("bonds", (1, 3, 3, 4)),
])
def test_graph_policy_rejects_wrong_schema_or_batch_in_both_paths(fixed, field, shape):
    spec = GraphSpec(max_nodes=3)
    decoded = DecoderOutput(torch.zeros(1, 3, 10), torch.zeros(1, 3, 5), torch.zeros(1, 3, 3, 5))
    setattr(decoded, field, torch.zeros(shape))
    with pytest.raises(ValueError, match="GraphSpec"):
        graph_from_decoder(decoded, spec, fixed_active=torch.ones(3, dtype=torch.bool) if fixed else None)


@pytest.mark.parametrize("mask", [
    torch.ones(3), torch.ones(3, dtype=torch.long),
    torch.ones(1, 3, dtype=torch.bool), torch.zeros(3, dtype=torch.bool),
])
def test_fixed_policy_requires_exact_nonempty_boolean_mask(mask):
    spec = GraphSpec(max_nodes=3)
    decoded = DecoderOutput(torch.zeros(1, 3, 10), torch.zeros(1, 3, 5), torch.zeros(1, 3, 3, 5))
    with pytest.raises(ValueError, match="nonempty boolean mask"):
        graph_from_decoder(decoded, spec, fixed_active=mask)


def test_fixed_policy_rejects_unsupported_batch_before_legacy_tensor_operations():
    spec = GraphSpec(max_nodes=3)
    decoded = DecoderOutput(torch.zeros(2, 3, 10), torch.zeros(2, 3, 5), torch.zeros(2, 3, 3, 5))
    assert graph_from_decoder(decoded, spec).atoms.shape == (2, 3, 10)
    with pytest.raises(ValueError, match="batch size one"):
        graph_from_decoder(decoded, spec, fixed_active=torch.ones(3, dtype=torch.bool))
