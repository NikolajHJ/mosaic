"""Named losses independent of training orchestration and property scorers."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from latentmol.representation import DecoderOutput, HardGraphBatch


@dataclass(frozen=True)
class LossConfig:
    padding_weight: float = 0.2
    no_bond_weight: float = 0.1
    charge_weight: float = 1.0
    bond_weight: float = 2.0
    size_weight: float = 0.1


def reconstruction_losses(predicted: DecoderOutput, target: HardGraphBatch, config: LossConfig) -> dict:
    """Score atom slots, present-atom charges, and upper-triangle real-atom pairs.

    Padding must not dominate atom CE; absent bonds must not dominate bond CE.
    Size MSE is normalized by maximum capacity. Bond weighting remains explicit
    in saved configs because it changes the relative importance of connectivity.

    Node CE flattens [batch, slot, class] to [batch*slot, class]. This preserves
    the same global class-weighted mean while avoiding PyTorch 2.5.1's spatial
    NLL CUDA kernel, which rejects strict deterministic execution for 3D logits.
    """
    active = target.atoms != 0
    atom_weights = predicted.atoms.new_ones(predicted.atoms.shape[-1])
    atom_weights[0] = config.padding_weight
    atom = F.cross_entropy(predicted.atoms.reshape(-1, predicted.atoms.shape[-1]),
                           target.atoms.reshape(-1), weight=atom_weights)
    charge_errors = F.cross_entropy(predicted.charges.reshape(-1, predicted.charges.shape[-1]),
                                   target.charges.reshape(-1), reduction="none").reshape_as(target.charges)
    charge = (charge_errors * active).sum() / active.sum().clamp_min(1)
    upper = torch.ones(target.bonds.shape[-2:], dtype=torch.bool, device=active.device).triu(1)
    pair_mask = active[:, :, None] & active[:, None, :] & upper
    bond_weights = predicted.bonds.new_ones(predicted.bonds.shape[-1])
    bond_weights[0] = config.no_bond_weight
    if pair_mask.any():
        bond = F.cross_entropy(predicted.symmetric_bond_logits[pair_mask], target.bonds[pair_mask], weight=bond_weights)
    else:
        bond = predicted.bonds.sum() * 0.0
    expected_size = (1.0 - predicted.atoms.softmax(-1)[..., 0]).sum(-1)
    size = ((expected_size - active.sum(-1)) / target.atoms.shape[-1]).square().mean()
    reconstruction = atom + config.charge_weight * charge + config.bond_weight * bond + config.size_weight * size
    return {"atom_ce": atom, "charge_ce": charge, "bond_ce": bond, "size_mse": size, "reconstruction": reconstruction}


def latent_kl(mu, logvar):
    """Gaussian KL to N(0,I), summed over latent dimensions and averaged over molecules."""
    return -0.5 * (1.0 + logvar - mu.square() - logvar.exp()).sum(-1).mean()
