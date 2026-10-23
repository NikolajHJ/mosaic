"""Loss-kernel equivalence regressions; numerical execution belongs on DTU HPC."""

import torch
from torch.nn import functional as F

from latentmol.losses import LossConfig, reconstruction_losses
from latentmol.representation import DecoderOutput, HardGraphBatch


def test_flat_node_cross_entropy_preserves_weighted_spatial_values_and_gradients():
    """Compare to the previous CPU formula with unequal occupancy and nonneutral charges.

    The old spatial kernel remains a useful CPU reference, but is unsupported
    under strict CUDA determinism. Nonuniform padding weights ensure a mistaken
    denominator change is observable; absent charges must remain masked out.
    """
    generator = torch.Generator().manual_seed(73)
    atoms = torch.randn(2, 4, 5, generator=generator, dtype=torch.float64, requires_grad=True)
    charges = torch.randn(2, 4, 3, generator=generator, dtype=torch.float64, requires_grad=True)
    bonds = torch.randn(2, 4, 4, 5, generator=generator, dtype=torch.float64, requires_grad=True)
    predicted = DecoderOutput(atoms, charges, bonds)
    target_atoms = torch.tensor([[1, 2, 0, 0], [3, 1, 4, 0]])
    target_charges = torch.tensor([[0, 2, 1, 0], [2, 1, 0, 2]])
    target_bonds = torch.zeros(2, 4, 4, dtype=torch.long)
    target_bonds[:, 0, 1] = target_bonds[:, 1, 0] = 1
    target_bonds[1, 1, 2] = target_bonds[1, 2, 1] = 2
    target = HardGraphBatch(target_atoms, target_charges, target_bonds)
    config = LossConfig(padding_weight=0.17, charge_weight=1.9)
    actual = reconstruction_losses(predicted, target, config)
    weights = torch.ones(5, dtype=torch.float64)
    weights[0] = config.padding_weight
    old_atom = F.cross_entropy(atoms.transpose(1, 2), target_atoms, weight=weights)
    old_charge_errors = F.cross_entropy(charges.transpose(1, 2), target_charges, reduction="none")
    active = target_atoms != 0
    old_charge = (old_charge_errors * active).sum() / active.sum()
    torch.testing.assert_close(actual["atom_ce"], old_atom, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual["charge_ce"], old_charge, rtol=1e-12, atol=1e-12)
    actual_gradients = torch.autograd.grad(actual["atom_ce"] + config.charge_weight * actual["charge_ce"],
                                         (atoms, charges), retain_graph=True)
    previous_gradients = torch.autograd.grad(old_atom + config.charge_weight * old_charge, (atoms, charges))
    for flat, spatial in zip(actual_gradients, previous_gradients):
        torch.testing.assert_close(flat, spatial, rtol=1e-12, atol=1e-12)
    assert actual_gradients[1][~active].count_nonzero() == 0
