"""Allocated-HPC chemical coverage and actual logit/decoder gradient contracts."""
from copy import deepcopy
import math

import pytest
import torch
from rdkit import Chem

from latentmol.chemical_environment import (neighbor_environment_loss,
    prepare_neighbor_environment, repair_with_neighbor_environment)
from latentmol.chemical_environment_policy import VERSION, VALENCES, assess_environment
from latentmol.chemistry_regularization import ALLOWED_VALENCES
from latentmol.representation import DecoderOutput, GraphSpec

POLICY = dict(version=VERSION, weight=.1)
# Deliberately include resonance, aromatic, charged and less frequent legitimate
# families. Chemical acceptance by RDKit and common-profile acceptance are
# independently checked; strict schema reconstruction is a separate question.
LEGITIMATE = [
    "S", "CS", "CSC", "CSSC", "C=S", "S=C=S", "c1ccsc1",
    "CS(C)=O", "C[S+](C)[O-]", "CS(C)(=O)=O", "C[S+2](C)([O-])[O-]",
    "CS(=O)O", "CS(=O)(=O)O", "CS(=O)(=O)[O-]", "CS(=O)(=O)N",
    "COS(=O)(=O)N", "COS(=O)(=O)OC", "O=S(=O)([O-])[O-]",
    "CS(C)=N", "CS(C)(=N)=O", "CS(C)(=N)=N", "CS(=O)(=N)N",
    "C[S+](C)C", "C[S+](C)(C)=O", "C[S](C)=C", "C[S](C)(C)C",
    "[S-]C", "S(F)(F)(F)F", "S(F)(F)(F)(F)(F)F",
    "CS(F)(F)(F)(F)F", "CS(F)(F)(F)(F)C", "O=S(F)(F)(F)F",
    "FS(F)(F)(F)(F)c1ccccc1", "P(C)(C)C", "CP(C)(C)=O",
    "CP(C)(C)=S", "OP(=O)(O)O", "CP(=O)(O)O", "C[P+](C)(C)C",
    "[CH2-][P+](C)(C)C", "CP(C)(C)=C", "P(C)(C)(C)(C)C",
    "CN", "C[N+](C)(C)C", "C[N+](=O)[O-]", "CN=[N+]=[N-]",
    "C[N+]#[C-]", "C#N", "[O-][n+]1ccccc1", "c1cc[nH]c1",
    "c1ccncc1", "[nH+]1ccccc1", "COOC", "CC(=O)[O-]", "[NH4+]",
    "C[O+](C)C", "CF", "CCl", "CBr", "CI", "[Cl-]",
]


def labels_from_smiles(smiles, capacity=None):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, "Positive reference was not RDKit accepted: " + smiles
    n = mol.GetNumAtoms()
    spec = GraphSpec(max_nodes=capacity or n)
    atoms = [0] * spec.max_nodes
    charges = [spec.formal_charges.index(0)] * spec.max_nodes
    bonds = [[0] * spec.max_nodes for _ in atoms]
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        atoms[i] = spec.atomic_numbers.index(atom.GetAtomicNum())
        charges[i] = spec.formal_charges.index(atom.GetFormalCharge())
    names = {Chem.BondType.SINGLE: "SINGLE", Chem.BondType.DOUBLE: "DOUBLE",
             Chem.BondType.TRIPLE: "TRIPLE", Chem.BondType.AROMATIC: "AROMATIC"}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bonds[i][j] = bonds[j][i] = spec.bond_names.index(names[bond.GetBondType()])
    return spec, atoms, charges, bonds


def output_from_labels(spec, atoms, charges, bonds, strength=25.):
    fields = []
    for values, categories in ((atoms, len(spec.atomic_numbers)),
                               (charges, len(spec.formal_charges)),
                               (bonds, len(spec.bond_names))):
        labels = torch.tensor(values)[None]
        value = torch.nn.functional.one_hot(labels, categories).double() * strength
        fields.append(value.detach().requires_grad_())
    return DecoderOutput(*fields)


@pytest.mark.parametrize("smiles", LEGITIMATE)
def test_legitimate_motif_is_retained_and_concentrated_risk_is_small(smiles):
    spec, a, q, b = labels_from_smiles(smiles)
    assessment = assess_environment([a], [q], [b], spec)[0]
    assert assessment["local_valence_ok"] and assessment["within_common_organic_scope"]
    assert assessment["fully_assessed"]
    output = output_from_labels(spec, a, q, b)
    target = prepare_neighbor_environment(output, spec, POLICY)
    risk = neighbor_environment_loss(output, target, spec)
    assert 0 <= float(risk["loss"]) < 1e-6


def test_rule_table_retains_existing_charge_and_phosphorus_semantics():
    assert VALENCES == ALLOWED_VALENCES


def test_six_carbon_sulfur_is_explicit_scope_exclusion_not_claimed_impossible():
    spec, a, q, b = labels_from_smiles("CS(C)(C)(C)(C)C")
    row = assess_environment([a], [q], [b], spec)[0]
    assert row["local_valence_ok"] and not row["within_common_organic_scope"]
    assert len(row["sulfur_scope_exclusions"]) == 1
    assert row["native_chemistry_validation_performed"] is False


def test_valid_boundary_has_a_new_gradient_without_a_valence_excess():
    spec, a, q, b = labels_from_smiles("C[S](C)(C)C", capacity=6)
    # A separate carbon slot supplies a fifth uncertain edge. It is deliberately
    # an engineering graph, not a claim that a disconnected molecule is valid.
    a[-1] = spec.atomic_numbers.index(6)
    output = output_from_labels(spec, a, q, b, 8.)
    sulfur = a.index(spec.atomic_numbers.index(16))
    with torch.no_grad():
        output.bonds[0, sulfur, -1] = -8.
        output.bonds[0, -1, sulfur] = -8.
        for i, j in ((sulfur, 5), (5, sulfur)):
            output.bonds[0, i, j, 0] = 0.
            output.bonds[0, i, j, 1] = -.2
    target = prepare_neighbor_environment(output, spec, POLICY)
    assert target.context["assessments"][0]["within_common_organic_scope"]
    result = neighbor_environment_loss(output, target, spec)
    result["loss"].backward()
    assert float(result["per_center_risk"][0, sulfur]) > .1
    assert output.bonds.grad[0, sulfur, 5, 1] > 0
    assert torch.count_nonzero(output.atoms.grad) == 0
    assert torch.count_nonzero(output.charges.grad) == 0


def test_neighbor_capacity_blocks_extra_bond_to_saturated_carbon():
    spec, a, q, b = labels_from_smiles("CSC(C)(C)C")
    output = output_from_labels(spec, a, q, b, 12.)
    s = a.index(spec.atomic_numbers.index(16))
    c = 2
    target = prepare_neighbor_environment(output, spec, POLICY)
    assert target.context["neighbor_remaining_caps"][0][s][c] == 1
    with torch.no_grad():
        for i, j in ((s, c), (c, s)):
            output.bonds[0, i, j, 1] = 1.
            output.bonds[0, i, j, 2] = .5
    loss = neighbor_environment_loss(output, target, spec)["loss"]
    loss.backward()
    assert output.bonds.grad[0, s, c, 2] > 0


def test_float64_fixed_context_directional_derivative_and_symmetry():
    spec, a, q, b = labels_from_smiles("CS(=O)(=O)N")
    torch.manual_seed(91)
    output = output_from_labels(spec, a, q, b, 2.)
    target = prepare_neighbor_environment(output, spec, POLICY)
    result = neighbor_environment_loss(output, target, spec)
    gradient, = torch.autograd.grad(result["loss"], (output.bonds,))
    direction = torch.randn_like(output.bonds)
    epsilon = 1e-5
    def value(step):
        one = DecoderOutput(output.atoms, output.charges, output.bonds + step * direction)
        return neighbor_environment_loss(one, target, spec)["loss"]
    fd = (value(epsilon) - value(-epsilon)) / (2 * epsilon)
    assert torch.allclose(fd, (gradient * direction).sum(), atol=2e-8, rtol=2e-6)
    assert torch.allclose(gradient, gradient.transpose(1, 2), atol=1e-14, rtol=0)


def test_confident_invalid_graph_keeps_finite_gradient_and_bounded_step_improves():
    spec, a, q, b = labels_from_smiles("CS(C)(C)(C)(C)C")
    output = output_from_labels(spec, a, q, b, 1000.)
    target = prepare_neighbor_environment(output, spec, POLICY)
    loss = neighbor_environment_loss(output, target, spec)["loss"]
    gradient, = torch.autograd.grad(loss, (output.bonds,))
    assert torch.isfinite(loss) and torch.isfinite(gradient).all() and gradient.norm() > 0
    changed = DecoderOutput(output.atoms, output.charges, output.bonds - .1 * gradient)
    assert neighbor_environment_loss(changed, target, spec)["loss"] < loss


def test_engineering_frozen_decoder_latent_vjp_is_finite_and_nonzero():
    # Small real decoder graph, not a learned surrogate. VJP must cross its bond
    # projection and latent; atom/charge rule selection remains detached.
    torch.manual_seed(112)
    spec, a, q, b = labels_from_smiles("CSC")
    base = output_from_labels(spec, a, q, b, 2.)
    decoder = torch.nn.Sequential(torch.nn.Linear(7, 13), torch.nn.Tanh(),
                                  torch.nn.Linear(13, spec.max_nodes**2 * len(spec.bond_names))).double()
    decoder.requires_grad_(False)
    z = torch.randn(1, 7, dtype=torch.double, requires_grad=True)
    live = DecoderOutput(base.atoms, base.charges, decoder(z).reshape_as(base.bonds))
    target = prepare_neighbor_environment(base, spec, POLICY)
    gradient, = torch.autograd.grad(neighbor_environment_loss(live, target, spec)["loss"], (z,))
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
    assert all(parameter.grad is None for parameter in decoder.parameters())


def test_context_is_fixed_and_tampering_fails_closed():
    spec, a, q, b = labels_from_smiles("CSC")
    output = output_from_labels(spec, a, q, b)
    target = prepare_neighbor_environment(output, spec, POLICY)
    identity = target.identity
    with torch.no_grad():
        output.atoms[0, 0] = 0.
    neighbor_environment_loss(output, target, spec)
    assert target.identity == identity
    target.context["caps"][0][0] = 99
    with pytest.raises(ValueError, match="changed"):
        neighbor_environment_loss(output, target, spec)


def test_unsupported_charge_is_counted_as_unassessed_not_a_zero_capacity():
    spec = GraphSpec(max_nodes=1)
    a, q, b = [spec.atomic_numbers.index(9)], [spec.formal_charges.index(-2)], [[0]]
    output = output_from_labels(spec, a, q, b)
    result = neighbor_environment_loss(output, prepare_neighbor_environment(output, spec, POLICY), spec)
    assert result["unassessed_centers"] == 1 and result["assessed_centers"] == 0
    assert float(result["loss"]) == 0
