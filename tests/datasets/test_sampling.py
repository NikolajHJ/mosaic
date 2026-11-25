"""Categorical graph law checks; all numerical execution belongs on DTU HPC."""

import itertools
import math

import pytest
import torch

from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch, RelaxedGraph
from latentmol.sampling import hard_graph_log_prob, sample_fixed_relaxed_graph, sample_hard_graph


def _law(batch=1, nodes=2):
    spec = GraphSpec(max_nodes=nodes, atomic_numbers=(0, 6, 8),
                     formal_charges=(0, 1), bond_names=("NONE", "SINGLE"))
    atoms = torch.tensor([0.25, 0.50, 0.25], dtype=torch.float64).log().expand(batch, nodes, -1).clone()
    charges = torch.tensor([0.7, 0.3], dtype=torch.float64).log().expand(batch, nodes, -1).clone()
    bonds = torch.tensor([0.6, 0.4], dtype=torch.float64).log().expand(batch, nodes, nodes, -1).clone()
    return spec, DecoderOutput(atoms, charges, bonds)


def test_seeded_sampling_is_symmetric_masked_and_preserves_inputs():
    spec, decoded = _law(batch=128, nodes=5)
    original = tuple(value.clone() for value in (decoded.atoms, decoded.charges, decoded.bonds))
    first = sample_hard_graph(decoded, spec, generator=torch.Generator().manual_seed(37))
    second = sample_hard_graph(decoded, spec, generator=torch.Generator().manual_seed(37))
    for name in ("atoms", "charges", "bonds"):
        assert torch.equal(getattr(first, name), getattr(second, name))
        assert not getattr(first, name).requires_grad
    active = first.atoms != 0
    assert (~active).any() and active.any()
    assert torch.equal(first.bonds, first.bonds.transpose(1, 2))
    assert not first.bonds.diagonal(dim1=1, dim2=2).any()
    assert not first.bonds[~(active[:, :, None] & active[:, None, :])].any()
    assert (first.charges[~active] == spec.formal_charges.index(0)).all()
    for before, after in zip(original, (decoded.atoms, decoded.charges, decoded.bonds)):
        assert torch.equal(before, after)


def test_fixed_occupancy_conditions_atoms_and_preserves_frequency():
    spec, decoded = _law(batch=8000, nodes=3)
    mask = torch.tensor([True, True, False])
    graph = sample_hard_graph(decoded, spec, generator=torch.Generator().manual_seed(51),
                              fixed_active=mask)
    assert (graph.atoms[:, :2] != 0).all()
    assert not graph.atoms[:, 2].any()
    assert not graph.bonds[:, 2].any()
    assert not graph.charges[:, 2].any()
    # Conditioning excludes NO_ATOM, so carbon has probability .5/.75.
    assert abs(float((graph.atoms[:, 0] == 1).double().mean()) - 2 / 3) < 0.025
    assert abs(float(graph.bonds[:, 0, 1].double().mean()) - 0.4) < 0.025
    assert abs(float(graph.charges[:, 0].double().mean()) - 0.3) < 0.025


def test_variable_occupancy_does_not_double_gate_bond_probability():
    spec, decoded = _law(batch=12000)
    graph = sample_hard_graph(decoded, spec, generator=torch.Generator().manual_seed(73))
    both = (graph.atoms[:, 0] != 0) & (graph.atoms[:, 1] != 0)
    assert abs(float((graph.atoms == 0).double().mean()) - 0.25) < 0.02
    assert abs(float(graph.bonds[both, 0, 1].double().mean()) - 0.4) < 0.025
    # Unconditional P(edge) is P(both present)*P(edge|present), once only.
    assert abs(float(graph.bonds[:, 0, 1].double().mean()) - 0.75**2 * 0.4) < 0.02


def test_log_probability_matches_enumerated_graph_law_and_ignores_absent_draws():
    spec, decoded = _law()
    graphs = []
    for atoms in itertools.product(range(3), repeat=2):
        active = [value != 0 for value in atoms]
        for charges in itertools.product(*(range(2) if present else [0] for present in active)):
            for bond in (range(2) if all(active) else [0]):
                graphs.append((atoms, charges, [[0, bond], [bond, 0]]))
    graph = HardGraphBatch(torch.tensor([item[0] for item in graphs]),
                          torch.tensor([item[1] for item in graphs]),
                          torch.tensor([item[2] for item in graphs]))
    expanded = DecoderOutput(*(value.expand(len(graphs), *value.shape[1:]).clone()
                               for value in (decoded.atoms, decoded.charges, decoded.bonds)))
    log_prob = hard_graph_log_prob(expanded, graph, spec)
    assert torch.allclose(log_prob.exp().sum(), torch.tensor(1.0, dtype=torch.float64), atol=1e-12)
    # Both absent: charges and potential edge draws have been marginalized away.
    assert float(log_prob[0]) == pytest.approx(2 * math.log(0.25))
    # One occupied carbon, positive charge: no bond probability factor remains.
    index = graphs.index(((0, 1), (0, 1), [[0, 0], [0, 0]]))
    assert float(log_prob[index]) == pytest.approx(math.log(0.25 * 0.50 * 0.3))


def test_fixed_graph_score_gradient_has_no_absent_or_no_atom_terms():
    spec, decoded = _law(nodes=3)
    for value in (decoded.atoms, decoded.charges, decoded.bonds):
        value.requires_grad_(True)
    mask = torch.tensor([True, True, False])
    graph = HardGraphBatch(torch.tensor([[1, 2, 0]]), torch.tensor([[1, 0, 0]]),
                          torch.tensor([[[0, 1, 0], [1, 0, 0], [0, 0, 0]]]))
    score = hard_graph_log_prob(decoded, graph, spec, fixed_active=mask)
    assert float(score) == pytest.approx(math.log((2 / 3) * (1 / 3) * 0.3 * 0.7 * 0.4))
    atoms, charges, bonds = torch.autograd.grad(score.sum(),
                                               (decoded.atoms, decoded.charges, decoded.bonds))
    assert torch.isfinite(atoms).all() and torch.isfinite(charges).all() and torch.isfinite(bonds).all()
    assert not atoms[..., 0].any() and not atoms[:, 2].any()
    assert not charges[:, 2].any() and not bonds[:, 2].any()
    assert not bonds.diagonal(dim1=1, dim2=2).any()
    assert atoms[:, :2, 1:].abs().sum() > 0
    assert bonds[:, 0, 1].abs().sum() > 0


def test_impossible_tensor_graphs_have_zero_probability_without_chemical_filtering():
    spec, decoded = _law(batch=3)
    graph = HardGraphBatch(torch.tensor([[1, 1], [0, 1], [1, 1]]),
                          torch.zeros((3, 2), dtype=torch.long),
                          torch.tensor([[[0, 1], [0, 0]], [[0, 1], [1, 0]], [[1, 0], [0, 0]]]))
    assert torch.isneginf(hard_graph_log_prob(decoded, graph, spec)).all()


def test_single_slot_has_no_bond_draw_and_mask_may_vary_by_batch():
    spec, decoded = _law(batch=2, nodes=1)
    mask = torch.tensor([[True], [False]])
    graph = sample_hard_graph(decoded, spec, generator=torch.Generator().manual_seed(91),
                              fixed_active=mask)
    assert torch.equal(graph.atoms != 0, mask)
    assert not graph.bonds.any()
    assert torch.isfinite(hard_graph_log_prob(decoded, graph, spec, fixed_active=mask)).all()


def test_absent_charge_uses_schema_neutral_category_instead_of_zero_index():
    spec = GraphSpec(max_nodes=1)
    decoded = DecoderOutput(torch.zeros((1, 1, len(spec.atomic_numbers))),
                            torch.zeros((1, 1, len(spec.formal_charges))),
                            torch.zeros((1, 1, 1, len(spec.bond_names))))
    mask = torch.tensor([False])
    graph = sample_hard_graph(decoded, spec, generator=torch.Generator().manual_seed(1),
                              fixed_active=mask)
    assert int(graph.charges[0, 0]) == spec.formal_charges.index(0) == 2
    assert float(hard_graph_log_prob(decoded, graph, spec, fixed_active=mask)) == 0.0


def _saved_fixed_graph(batch=1):
    spec, decoded = _law(batch=batch, nodes=3)
    atoms = decoded.atoms.softmax(-1)
    atoms[..., 0] = 0
    atoms[..., 1:] = torch.tensor([0.75, 0.25], dtype=atoms.dtype)
    atoms[:, 2] = torch.tensor([1.0, 0.0, 0.0], dtype=atoms.dtype)
    charges = decoded.charges.softmax(-1)
    charges[:, 2] = torch.tensor([1.0, 0.0], dtype=charges.dtype)
    bonds = torch.zeros_like(decoded.bonds)
    bonds[..., 0] = 1
    bonds[:, 0, 1] = bonds[:, 1, 0] = torch.tensor([0.6, 0.4], dtype=bonds.dtype)
    return spec, RelaxedGraph(atoms, charges, bonds), torch.tensor([True, True, False])


def test_saved_fixed_graph_uses_original_probabilities_without_mutation():
    spec, graph, mask = _saved_fixed_graph(batch=8000)
    before = [value.clone() for value in (graph.atoms, graph.charges, graph.bonds)]
    sampled = sample_fixed_relaxed_graph(graph, spec, generator=torch.Generator().manual_seed(19),
                                        fixed_active=mask)
    repeated = sample_fixed_relaxed_graph(graph, spec, generator=torch.Generator().manual_seed(19),
                                         fixed_active=mask)
    assert torch.equal(sampled.atoms, repeated.atoms)
    assert torch.equal(sampled.bonds, repeated.bonds)
    assert (sampled.atoms[:, :2] != 0).all() and not sampled.atoms[:, 2].any()
    assert torch.equal(sampled.bonds, sampled.bonds.transpose(1, 2))
    assert not sampled.bonds.diagonal(dim1=1, dim2=2).any()
    assert not sampled.bonds[:, 2].any() and not sampled.charges[:, 2].any()
    assert abs(float((sampled.atoms[:, 0] == 1).double().mean()) - 0.75) < 0.025
    assert abs(float(sampled.bonds[:, 0, 1].double().mean()) - 0.4) < 0.025
    for expected, actual in zip(before, (graph.atoms, graph.charges, graph.bonds)):
        assert torch.equal(expected, actual)


def test_saved_sampler_rejects_soft_occupancy_and_malformed_probability_graphs():
    spec, decoded = _law()
    with pytest.raises(ValueError, match="fixed occupancy"):
        sample_fixed_relaxed_graph(decoded.probabilities(), spec,
                                    generator=torch.Generator().manual_seed(2),
                                    fixed_active=torch.ones(2, dtype=torch.bool))
    for defect, match in (("normalization", "normalized"), ("asymmetry", "symmetric"),
                          ("diagonal", "NONE"), ("absent_charge", "neutral")):
        spec, graph, mask = _saved_fixed_graph()
        if defect == "normalization":
            graph.atoms[:, 0, 1] += 0.1
        elif defect == "asymmetry":
            graph.bonds[:, 0, 1] = torch.tensor([0.5, 0.5], dtype=graph.bonds.dtype)
        elif defect == "diagonal":
            graph.bonds[:, 0, 0] = torch.tensor([0.5, 0.5], dtype=graph.bonds.dtype)
        else:
            graph.charges[:, 2] = torch.tensor([0.5, 0.5], dtype=graph.charges.dtype)
        with pytest.raises(ValueError, match=match):
            sample_fixed_relaxed_graph(graph, spec, generator=torch.Generator().manual_seed(2),
                                        fixed_active=mask)


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_temperatures_are_rejected(temperature):
    spec, decoded = _law()
    with pytest.raises(ValueError, match="temperature"):
        sample_hard_graph(decoded, spec, generator=torch.Generator().manual_seed(1),
                          temperature=temperature)
