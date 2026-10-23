"""Probability-law and gradient fixtures; numerical execution belongs on DTU HPC."""

import itertools

import pytest
import torch
from torch.nn import functional as F

from latentmol.chemistry_regularization import (
    ALLOWED_VALENCES, ChemistryRegularizationConfig, capped_bond_order_distribution,
    chemistry_regularization_losses, rule_record,
)
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch


def _hard_output(spec, atom_numbers, edges=(), charges=None):
    atoms = torch.tensor([[spec.atomic_numbers.index(element) for element in atom_numbers]])
    formal = charges if charges is not None else [0] * spec.max_nodes
    charge = torch.tensor([[spec.formal_charges.index(value) for value in formal]])
    bonds = torch.zeros(1, spec.max_nodes, spec.max_nodes, dtype=torch.long)
    for i, j, name in edges:
        bonds[:, i, j] = bonds[:, j, i] = spec.bond_names.index(name)
    hard = HardGraphBatch(atoms, charge, bonds)
    graph = hard.to_relaxed(spec)
    # Zero category probability is exact in these forward/RDKit audits. The
    # numerical gradient fixtures below instead use strictly finite logits.
    return DecoderOutput(*(value.double().log() for value in (graph.atoms, graph.charges, graph.bonds))), hard


def _loss(output, spec, **kwargs):
    return chemistry_regularization_losses(output, spec,
        ChemistryRegularizationConfig(isolated_weight=0), **kwargs)


def test_capped_distribution_matches_full_categorical_enumeration_and_moments():
    probabilities = torch.tensor([[.2, .3, .1, .4], [.1, .2, .5, .2], [.3, .1, .4, .2]],
                                 dtype=torch.float64, requires_grad=True)
    low, mass, moment = capped_bond_order_distribution(probabilities, 3)
    exact = torch.zeros(10, dtype=torch.float64)
    for labels in itertools.product(range(4), repeat=3):
        probability = torch.stack([probabilities[i, label] for i, label in enumerate(labels)]).prod()
        exact = exact + F.one_hot(torch.tensor(sum(labels)), 10).double() * probability
    torch.testing.assert_close(low, exact[:4])
    torch.testing.assert_close(mass, exact[4:].sum())
    torch.testing.assert_close(moment, (exact[4:] * torch.arange(4, 10)).sum())
    torch.testing.assert_close(low.sum() + mass, torch.ones_like(mass))
    torch.testing.assert_close((low * torch.arange(4)).sum() + moment,
                               (probabilities * torch.arange(4)).sum())
    actual_excess = (low * (torch.arange(4) - 2).clamp_min(0)).sum() + moment - 2 * mass
    exact_excess = (exact * (torch.arange(10) - 2).clamp_min(0)).sum()
    torch.testing.assert_close(actual_excess, exact_excess)
    actual_gradient = torch.autograd.grad(actual_excess, probabilities, retain_graph=True)[0]
    exact_gradient = torch.autograd.grad(exact_excess, probabilities)[0]
    torch.testing.assert_close(actual_gradient, exact_gradient)


@pytest.mark.parametrize("cap", [0, 1, 2, 6])
def test_dp_mass_and_first_moment_conservation_with_large_overflow(cap):
    generator = torch.Generator().manual_seed(59)
    probabilities = torch.randn(2, 3, 20, 4, generator=generator, dtype=torch.float64).softmax(-1)
    low, mass, moment = capped_bond_order_distribution(probabilities, cap)
    torch.testing.assert_close(low.sum(-1) + mass, torch.ones_like(mass))
    torch.testing.assert_close((low * torch.arange(cap + 1)).sum(-1) + moment,
                               (probabilities * torch.arange(4)).sum(dim=(-1, -2)))


def test_distributional_tail_is_penalized_even_when_expected_valence_is_allowed():
    spec = GraphSpec(max_nodes=2)
    output, _ = _hard_output(spec, [8, 6])
    bonds = output.bonds.exp()
    # Oxygen has expected valence 1.5<=2, but half of draws have excess one.
    bonds[:, 0, 1] = bonds[:, 1, 0] = torch.tensor([.5, 0, 0, .5, 0])
    output.bonds = bonds.log()
    result = _loss(output, spec)
    torch.testing.assert_close(result["diagnostics"]["expected_local_bond_order_excess"], torch.tensor([.5], dtype=torch.float64))
    torch.testing.assert_close(result["valence_excess"], torch.tensor(.5 / 2 / 3, dtype=torch.float64))


def test_focal_occupancy_is_gated_once_and_neighbor_occupancy_is_conditional():
    spec = GraphSpec(max_nodes=2)
    output, _ = _hard_output(spec, [8, 6], [(0, 1, "TRIPLE")])
    atoms = output.atoms.exp()
    atoms[:, 0] = 0
    atoms[:, 0, 0] = .5
    atoms[:, 0, spec.atomic_numbers.index(8)] = .5
    output.atoms = atoms.log()
    result = _loss(output, spec)
    torch.testing.assert_close(result["diagnostics"]["expected_local_bond_order_excess"], torch.tensor([.5], dtype=torch.float64))
    torch.testing.assert_close(result["diagnostics"]["expected_size"], torch.tensor([1.5], dtype=torch.float64))


def test_charge_and_atom_categories_mix_full_supported_rules():
    spec = GraphSpec(max_nodes=3)
    output, _ = _hard_output(spec, [7, 6, 6], [(0, 1, "DOUBLE"), (0, 2, "DOUBLE")])
    charges = output.charges.exp()
    charges[:, 0] = 0
    charges[:, 0, spec.formal_charges.index(0)] = .25  # cap3, excess1
    charges[:, 0, spec.formal_charges.index(1)] = .75  # cap4, excess0
    output.charges = charges.log()
    result = _loss(output, spec)
    torch.testing.assert_close(result["diagnostics"]["expected_local_bond_order_excess"], torch.tensor([.25], dtype=torch.float64))
    assert result["unsupported_charge"] == 0


def test_undercoordination_and_supported_hypervalence_remain_unpenalized():
    spec = GraphSpec(max_nodes=4)
    output, _ = _hard_output(spec, [15, 8, 8, 6], [(0, 1, "DOUBLE"), (0, 2, "DOUBLE"), (0, 3, "SINGLE")])
    assert _loss(output, spec)["valence_excess"] == 0
    assert ALLOWED_VALENCES[15][-2] == (1, 3, 5)
    assert ALLOWED_VALENCES[16][-1] == (1, 3, 5)
    assert ALLOWED_VALENCES[16][-2] == (0, 2, 4)


def test_unsupported_charge_is_reported_separately_without_inventing_cap():
    spec = GraphSpec(max_nodes=2)
    output, _ = _hard_output(spec, [9, 6], [(0, 1, "TRIPLE")], [-2, 0])
    result = _loss(output, spec)
    assert result["diagnostics"]["unsupported_element_charge_mass"] == 1
    assert result["unsupported_charge"] == .5
    assert result["valence_excess"] == 0  # Unknown cap is not guessed as zero.
    assert result["scope"]["element_charge_allowed_valences"]["9"]["-2"] is None


def test_aromatic_policy_is_explicit_lower_bound_and_reports_unscored_scope():
    spec = GraphSpec(max_nodes=3)
    output, _ = _hard_output(spec, [7, 6, 6], [(0, 1, "AROMATIC"), (0, 2, "AROMATIC")])
    result = _loss(output, spec)
    assert result["valence_excess"] == 0
    assert result["diagnostics"]["aromatic_affected_atom_fraction"] == 1
    assert result["diagnostics"]["expected_aromatic_edges"] == 2
    assert "lower_bound" in result["scope"]["aromatic_order_policy"]
    assert not result["scope"]["validity_guarantee"]


def test_empty_guard_singleton_isolation_and_two_component_limitation():
    spec = GraphSpec(max_nodes=4)
    empty, _ = _hard_output(spec, [0, 0, 0, 0])
    assert _loss(empty, spec)["empty_graph"] == 1
    singleton, _ = _hard_output(spec, [6, 0, 0, 0])
    assert _loss(singleton, spec)["isolated_multi"] == 0
    disconnected, _ = _hard_output(spec, [6, 6, 6, 6], [(0, 1, "SINGLE"), (2, 3, "SINGLE")])
    # A deliberately retained limitation: isolated-atom risk alone misses this.
    assert _loss(disconnected, spec)["isolated_multi"] == 0
    isolated, _ = _hard_output(spec, [6, 6, 0, 0])
    assert _loss(isolated, spec)["isolated_multi"] == 1


def test_fractional_isolation_has_no_small_eigenvalue_or_fake_singleton_escape():
    spec = GraphSpec(max_nodes=2)
    output, _ = _hard_output(spec, [6, 6])
    atoms = output.atoms.exp()
    atoms[:, 1] = 0
    atoms[:, 1, 0] = .99
    atoms[:, 1, spec.atomic_numbers.index(6)] = .01
    output.atoms = atoms.log()
    result = _loss(output, spec)
    torch.testing.assert_close(result["isolated_multi"], torch.tensor(.02 / 1.01, dtype=torch.float64))


def test_ghost_pressure_reads_conditional_pairs_not_already_gated_graph():
    spec = GraphSpec(max_nodes=2)
    output, _ = _hard_output(spec, [6, 0])
    bonds = output.bonds.exp()
    bonds[:, 0, 1] = bonds[:, 1, 0] = torch.tensor([0., 1., 0., 0., 0.])
    output.bonds = bonds.log()
    result = _loss(output, spec)
    assert result["ghost_bonds"] == 1
    assert result["valence_excess"] == 0


def test_reference_size_is_explicit_detached_and_normalized_relative_to_reference():
    spec = GraphSpec(max_nodes=4)
    output, _ = _hard_output(spec, [6, 6, 0, 0], [(0, 1, "SINGLE")])
    cfg = ChemistryRegularizationConfig(size_weight=1)
    with pytest.raises(ValueError, match="explicit detached"):
        chemistry_regularization_losses(output, spec, cfg)
    reference = torch.tensor([4.], dtype=torch.float64)
    result = chemistry_regularization_losses(output, spec, cfg, reference_size=reference)
    assert result["reference_size"] == .25
    with pytest.raises(ValueError, match="detached"):
        chemistry_regularization_losses(output, spec, cfg, reference_size=reference.requires_grad_())


def test_all_logit_fields_have_finite_nonzero_gradients_and_double_gradcheck():
    spec = GraphSpec(max_nodes=3)
    generator = torch.Generator().manual_seed(73)
    shapes = ((1, 3, 10), (1, 3, 5), (1, 3, 3, 5))
    tensors = tuple(torch.randn(shape, generator=generator, dtype=torch.float64, requires_grad=True) for shape in shapes)
    cfg = ChemistryRegularizationConfig(ghost_weight=.2, size_weight=.3)
    reference = torch.tensor([2.], dtype=torch.float64)
    def forward(*values):
        return chemistry_regularization_losses(DecoderOutput(*values), spec, cfg, reference_size=reference)["total"]
    assert torch.autograd.gradcheck(forward, tensors, eps=1e-6, atol=2e-5, rtol=2e-4)
    gradients = torch.autograd.grad(forward(*tensors), tensors)
    assert all(torch.isfinite(gradient).all() and gradient.abs().sum() > 0 for gradient in gradients)


def test_slot_permutation_and_batch_partition_preserve_loss_values():
    spec = GraphSpec(max_nodes=4)
    generator = torch.Generator().manual_seed(85)
    output = DecoderOutput(torch.randn(2, 4, 10, generator=generator, dtype=torch.float64),
                           torch.randn(2, 4, 5, generator=generator, dtype=torch.float64),
                           torch.randn(2, 4, 4, 5, generator=generator, dtype=torch.float64))
    permutation = torch.tensor([3, 0, 2, 1])
    permuted = DecoderOutput(output.atoms[:, permutation], output.charges[:, permutation],
                             output.bonds[:, permutation][:, :, permutation])
    first, second = _loss(output, spec), _loss(permuted, spec)
    for name in first["per_graph"]:
        torch.testing.assert_close(first[name], second[name])
        parts = [_loss(DecoderOutput(output.atoms[i:i+1], output.charges[i:i+1], output.bonds[i:i+1]), spec)[name]
                 for i in range(2)]
        torch.testing.assert_close(first[name], torch.stack(parts).mean())


@pytest.mark.parametrize("kwargs", [{"version": "v2"}, {"valence_weight": -1}, {"ghost_weight": True}, {"empty_weight": float("nan")}, {"unknown": 1}])
def test_config_rejects_unversioned_unknown_or_invalid_weights(kwargs):
    with pytest.raises(ValueError):
        ChemistryRegularizationConfig.from_mapping(kwargs)


def test_rule_table_and_hard_nonaromatic_examples_against_installed_rdkit():
    from rdkit import Chem
    # This checks the installed HPC RDKit without importing it in forward.
    spec = GraphSpec(max_nodes=6)
    for element in spec.atomic_numbers[1:]:
        assert tuple(Chem.GetPeriodicTable().GetValenceList(element)) == ALLOWED_VALENCES[element][0]
    examples = [(6, 0, 4, False), (6, 0, 5, True), (7, 0, 4, True),
                (7, 1, 4, False), (8, 0, 3, True), (8, 1, 3, False)]
    for element, charge, degree, rejected in examples:
        edges = [(0, i, "SINGLE") for i in range(1, degree + 1)]
        numbers = [element] + [6] * degree + [0] * (5 - degree)
        charges = [charge] + [0] * 5
        output, _ = _hard_output(spec, numbers, edges, charges)
        molecule = Chem.RWMol()
        center = Chem.Atom(element)
        center.SetFormalCharge(charge)
        molecule.AddAtom(center)
        for i in range(degree):
            molecule.AddAtom(Chem.Atom(6))
            molecule.AddBond(0, i + 1, Chem.BondType.SINGLE)
        try:
            molecule.UpdatePropertyCache(strict=True)
            actual_rejected = False
        except (ValueError, RuntimeError):
            actual_rejected = True
        assert actual_rejected == rejected
        assert bool(_loss(output, spec)["valence_excess"] > 0) == rejected


def test_unsupported_vocabularies_fail_closed_and_rules_receipt_roundtrips():
    with pytest.raises(ValueError, match="heavy-element"):
        rule_record(GraphSpec(max_nodes=2, atomic_numbers=(0, 5)))
    with pytest.raises(ValueError, match="formal charges"):
        rule_record(GraphSpec(max_nodes=2, formal_charges=(0, 3)))
    cfg = ChemistryRegularizationConfig(ghost_weight=.3)
    assert ChemistryRegularizationConfig.from_mapping(cfg.record()).record() == cfg.record()


def test_disjoint_hard_logit_support_after_symmetrization_is_rejected():
    spec = GraphSpec(max_nodes=2)
    output, _ = _hard_output(spec, [6, 6])
    output.bonds[:, 0, 1] = -torch.inf
    output.bonds[:, 0, 1, 1] = 0
    with pytest.raises(ValueError, match="Symmetric bond rows"):
        _loss(output, spec)
