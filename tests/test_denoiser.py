"""Small graph-schema and derivative checks; execute on DTU HPC."""

import torch

from latentmol.denoiser import (DenoiserLossConfig, SchemaDenoiser,
                               corrupt_graph, denoising_reconstruction_losses)
from latentmol.models import ModelConfig
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch


def _target(spec):
    atoms = torch.zeros((2, spec.max_nodes), dtype=torch.long)
    atoms[:, :3] = torch.tensor([1, 1, 3])
    charges = torch.full_like(atoms, spec.formal_charges.index(0))
    bonds = torch.zeros((2, spec.max_nodes, spec.max_nodes), dtype=torch.long)
    bonds[:, 0, 1] = bonds[:, 1, 0] = 1
    bonds[:, 1, 2] = bonds[:, 2, 1] = 1
    return HardGraphBatch(atoms, charges, bonds)


def test_corruption_is_seeded_symmetric_and_zero_time_identity():
    spec = GraphSpec(max_nodes=8)
    target = _target(spec)
    zero = corrupt_graph(target, spec, torch.zeros(2), torch.Generator().manual_seed(3))
    assert torch.equal(zero.atoms, target.atoms)
    assert torch.equal(zero.charges, target.charges)
    assert torch.equal(zero.bonds, target.bonds)
    first = corrupt_graph(target, spec, torch.ones(2), torch.Generator().manual_seed(7))
    second = corrupt_graph(target, spec, torch.ones(2), torch.Generator().manual_seed(7))
    assert torch.equal(first.atoms, second.atoms)
    assert torch.equal(first.bonds, first.bonds.transpose(1, 2))
    assert torch.equal(first.bonds.diagonal(dim1=1, dim2=2), torch.zeros((2, 8), dtype=torch.long))
    assert (first.atoms[:, 7] == 0).all()
    present = first.atoms != 0
    assert (first.bonds[~(present[:, :, None] & present[:, None, :])] == 0).all()
    assert (first.charges[~present] == spec.formal_charges.index(0)).all()


def test_soft_graph_has_gradient_through_denoiser():
    spec = GraphSpec(max_nodes=8)
    model = SchemaDenoiser(spec, ModelConfig(hidden=16, latent=12, layers=1))
    graph = _target(spec).to_relaxed(spec)
    atoms = graph.atoms.clone().requires_grad_(True)
    graph.atoms = atoms
    output = model(graph, torch.full((2,), 0.2))
    gradient = torch.autograd.grad(output.atoms.square().mean(), atoms)[0]
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_repair_loss_penalizes_false_bond_to_true_padding_slot():
    spec = GraphSpec(max_nodes=8)
    target = _target(spec)
    atom_logits = torch.zeros((2, 8, len(spec.atomic_numbers)))
    charge_logits = torch.zeros((2, 8, len(spec.formal_charges)))
    bond_logits = torch.zeros((2, 8, 8, len(spec.bond_names)))
    bond_logits[..., 0] = 2.0
    baseline = denoising_reconstruction_losses(
        DecoderOutput(atom_logits, charge_logits, bond_logits),
        target, DenoiserLossConfig())["bond_ce"]
    bond_logits[:, 0, 3, 1] = bond_logits[:, 3, 0, 1] = 9.0
    extra = denoising_reconstruction_losses(
        DecoderOutput(atom_logits, charge_logits, bond_logits),
        target, DenoiserLossConfig())["bond_ce"]
    assert extra > baseline
