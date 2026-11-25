"""HPC-only component and expected score-gradient engineering fixtures.

Exact enumeration below validates the expectation of the discrete estimator;
finite differences of one sampled draw would test the wrong gradient object.
"""

import itertools

import pytest
import torch

from latentmol.connectivity_regularization import (
    ConnectivityRegularizationConfig, component_costs, connectivity_regularization_losses,
    leave_one_out_connectivity_surrogate, presence_log_probability,
)
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch


def _graph(nodes, edges=(), *, active=None, device="cpu"):
    mask = [True] * nodes if active is None else active
    atoms = torch.tensor([[1 if item else 0 for item in mask]], dtype=torch.long, device=device)
    charges = torch.zeros_like(atoms)
    bonds = torch.zeros(1, nodes, nodes, dtype=torch.long, device=device)
    for left, right in edges:
        bonds[:, left, right] = bonds[:, right, left] = 1
    return HardGraphBatch(atoms, charges, bonds)


def _output(nodes=3, *, device="cpu", dtype=torch.float64):
    generator = torch.Generator(device=device).manual_seed(271)
    values = (torch.randn(2, nodes, 3, generator=generator, dtype=dtype, device=device),
              torch.randn(2, nodes, 3, generator=generator, dtype=dtype, device=device),
              torch.randn(2, nodes, nodes, 3, generator=generator, dtype=dtype, device=device))
    return DecoderOutput(*(value.requires_grad_() for value in values))


@pytest.mark.parametrize("nodes,active,edges,components,added,cost", [
    (1, [False], [], 0, 0, 0), (1, [True], [], 1, 0, 0),
    (4, None, [(0, 1), (2, 3)], 2, 1, 1 / 3),
    (4, None, [], 4, 3, 1), (4, None, [(0, 1), (1, 2), (2, 3)], 1, 0, 0),
    (4, [True, False, True, False], [], 2, 1, 1),
])
def test_exact_component_cost_including_disconnected_bonded_components(nodes, active, edges, components, added, cost):
    result = component_costs(_graph(nodes, edges, active=active))
    assert result["components"].item() == components
    assert result["minimum_unrestricted_added_edges"].item() == added
    assert result["normalized_cost"].item() == pytest.approx(cost)
    assert not result["normalized_cost"].requires_grad


def test_label_propagation_reaches_long_paths_and_is_slot_permutation_invariant():
    graph = _graph(50, [(i, i + 1) for i in range(49)])
    assert component_costs(graph)["components"].item() == 1
    graph = _graph(6, [(0, 1), (1, 2), (3, 4)])
    permutation = torch.tensor([4, 1, 5, 0, 2, 3])
    permuted = HardGraphBatch(graph.atoms[:, permutation], graph.charges[:, permutation],
                              graph.bonds[:, permutation][:, :, permutation])
    for name, actual in component_costs(graph).items():
        torch.testing.assert_close(actual, component_costs(permuted)[name])


@pytest.mark.parametrize("kind", ["self", "padding", "asymmetric"])
def test_component_input_structural_failures_are_not_repaired(kind):
    graph = _graph(2, active=[True, False] if kind == "padding" else None)
    if kind == "self":
        graph.bonds[:, 0, 0] = 1
    else:
        graph.bonds[:, 0, 1] = 1
        if kind == "padding":
            graph.bonds[:, 1, 0] = 1
    with pytest.raises(ValueError, match="symmetric edges"):
        component_costs(graph)


def test_leave_one_out_excludes_current_cost_stops_cost_gradient_and_displays_risk():
    costs = torch.tensor([[0., .2], [1., .4], [.5, .8], [.1, .6]], requires_grad=True)
    logp = torch.tensor([[-1., -2.], [-3., -1.], [-2., -4.], [-5., -1.]], requires_grad=True)
    result = leave_one_out_connectivity_surrogate(costs, logp)
    expected_baselines = torch.stack([torch.cat((costs[:index], costs[index+1:])).detach().mean(0) for index in range(4)])
    torch.testing.assert_close(result["baselines"], expected_baselines)
    torch.testing.assert_close(result["surrogate"], costs.detach().mean(0))
    gradients = torch.autograd.grad(result["surrogate"].sum(), (costs, logp), allow_unused=True)
    assert gradients[0] is None
    torch.testing.assert_close(gradients[1], (costs.detach() - expected_baselines) / 4)


def _two_node_presence_states(output, spec):
    atom = output.atoms.softmax(-1)
    occupancy = atom[..., 1:].sum(-1)[0]
    edges = output.symmetric_bond_logits.softmax(-1)[0, 0, 1]
    result = []
    for mask in itertools.product((False, True), repeat=2):
        possible_edges = (False, True) if all(mask) else (False,)
        for connected in possible_edges:
            probability = torch.stack([occupancy[i] if mask[i] else atom[0, i, 0] for i in range(2)]).prod()
            if all(mask):
                probability = probability * (edges[1:].sum() if connected else edges[0])
            graph = _graph(2, [(0, 1)] if connected else [], active=mask)
            result.append((probability, component_costs(graph)["normalized_cost"].to(output.atoms.dtype),
                           presence_log_probability(output, graph, spec)))
    return result


def test_exact_two_node_expectation_and_rao_blackwell_gradient_match_analytic_risk():
    spec = GraphSpec(max_nodes=2, atomic_numbers=(0, 6, 7), formal_charges=(0, 1, -1),
                     bond_names=("NONE", "SINGLE", "DOUBLE"))
    output_batch = _output(2)
    output = DecoderOutput(*(value[:1].detach().requires_grad_() for value in
                             (output_batch.atoms, output_batch.charges, output_batch.bonds)))
    states = _two_node_presence_states(output, spec)
    exact = sum(probability * cost for probability, cost, _ in states)
    occupancy = output.atoms.softmax(-1)[0, :, 1:].sum(-1)
    none_edge = output.symmetric_bond_logits.softmax(-1)[0, 0, 1, 0]
    analytic = occupancy.prod() * none_edge
    torch.testing.assert_close(exact.squeeze(0), analytic)
    # Enumerate two independent draws. Each other's cost is the LOO baseline.
    # Outcome probabilities are stopped when averaging estimator gradients.
    estimator = sum(
        first[0].detach() * second[0].detach() * leave_one_out_connectivity_surrogate(
            torch.stack((first[1], second[1])), torch.stack((first[2], second[2])))["surrogate"]
        for first in states for second in states)
    fields = (output.atoms, output.charges, output.bonds)
    actual = torch.autograd.grad(estimator.sum(), fields, retain_graph=True)
    expected = torch.autograd.grad(analytic, fields, allow_unused=True)
    torch.testing.assert_close(actual[0], expected[0])
    assert torch.count_nonzero(actual[1]) == 0 and expected[1] is None
    torch.testing.assert_close(actual[2], expected[2])
    # Within the real-atom and non-NONE bond groups, redistributing categories
    # at fixed occupancy/edge probability does not change presence risk.
    atom_probabilities = output.atoms.softmax(-1)
    torch.testing.assert_close(actual[0][..., 1] / atom_probabilities[..., 1],
                               actual[0][..., 2] / atom_probabilities[..., 2])
    bond_probabilities = output.symmetric_bond_logits.softmax(-1)
    torch.testing.assert_close(actual[2][..., 1] / bond_probabilities[..., 1],
                               actual[2][..., 2] / bond_probabilities[..., 2])


def test_presence_probability_marginalizes_labels_and_fixed_mask_has_zero_atom_charge_gradient():
    spec = GraphSpec(max_nodes=3, atomic_numbers=(0, 6, 7), formal_charges=(0, 1, -1),
                     bond_names=("NONE", "SINGLE", "DOUBLE"))
    full = _output(3)
    output = DecoderOutput(*(value[:1].detach().requires_grad_() for value in (full.atoms, full.charges, full.bonds)))
    graph = _graph(3, [(0, 1)], active=[True, True, False])
    changed = HardGraphBatch(graph.atoms.clone(), graph.charges.clone(), graph.bonds.clone())
    changed.atoms[0, 0] = 2
    changed.charges[0, 0] = 1
    changed.bonds[0, 0, 1] = changed.bonds[0, 1, 0] = 2
    first = presence_log_probability(output, graph, spec)
    torch.testing.assert_close(first, presence_log_probability(output, changed, spec))
    logp = presence_log_probability(output, graph, spec, fixed_active=graph.atoms[0] != 0)
    gradients = torch.autograd.grad(logp.sum(), (output.atoms, output.charges, output.bonds))
    assert torch.count_nonzero(gradients[0]) == torch.count_nonzero(gradients[1]) == 0
    assert gradients[2].abs().sum() > 0


def test_draw_budget_and_generator_state_replay_without_rejection(monkeypatch):
    spec = GraphSpec(max_nodes=3, atomic_numbers=(0, 6, 7), formal_charges=(0, 1, -1),
                     bond_names=("NONE", "SINGLE", "DOUBLE"))
    output = _output(3)
    generator = torch.Generator().manual_seed(337)
    state = generator.get_state()
    cfg = ConnectivityRegularizationConfig(draws=4)
    first = connectivity_regularization_losses(output, spec, cfg, generator=generator)
    generator.set_state(state)
    second = connectivity_regularization_losses(output, spec, cfg, generator=generator)
    assert first["diagnostics"]["attempted_draws"] == 4
    for name in ("sampled_atoms", "sampled_bonds", "components", "costs"):
        torch.testing.assert_close(first["diagnostics"][name], second["diagnostics"][name])
    assert not first["scope"]["pathwise_derivative"] and not first["scope"]["higher_derivatives_supported"]
    gradients = torch.autograd.grad(first["total"], (output.atoms, output.charges, output.bonds))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert torch.count_nonzero(gradients[1]) == 0
    disabled_state = generator.get_state()
    disabled = connectivity_regularization_losses(output, spec, ConnectivityRegularizationConfig(weight=0), generator=generator)
    assert disabled["diagnostics"]["attempted_draws"] == 0
    assert torch.equal(generator.get_state(), disabled_state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires allocated HPC GPU")
def test_cuda_graph_counting_and_seeded_replay_match_device_contract():
    cpu_graph = _graph(7, [(0, 1), (1, 2), (3, 4), (4, 5)])
    gpu_graph = cpu_graph.to("cuda")
    for name, expected in component_costs(cpu_graph).items():
        torch.testing.assert_close(component_costs(gpu_graph)[name].cpu(), expected)
    spec = GraphSpec(max_nodes=3, atomic_numbers=(0, 6, 7), formal_charges=(0, 1, -1),
                     bond_names=("NONE", "SINGLE", "DOUBLE"))
    output = _output(3, device="cuda", dtype=torch.float32)
    generator = torch.Generator(device="cuda").manual_seed(491)
    state = generator.get_state()
    first = connectivity_regularization_losses(output, spec, generator=generator)
    generator.set_state(state)
    second = connectivity_regularization_losses(output, spec, generator=generator)
    torch.testing.assert_close(first["diagnostics"]["sampled_atoms"], second["diagnostics"]["sampled_atoms"])
    torch.testing.assert_close(first["diagnostics"]["sampled_bonds"], second["diagnostics"]["sampled_bonds"])
    # CPU and CUDA RNG streams need not produce the same categorical draws.
    assert first["total"].is_cuda


@pytest.mark.parametrize("raw", [{"draws": 1}, {"draws": True}, {"weight": -1}, {"version": "new"}, {"typo": 1}])
def test_strict_connectivity_config(raw):
    with pytest.raises(ValueError):
        ConnectivityRegularizationConfig.from_mapping(raw)
