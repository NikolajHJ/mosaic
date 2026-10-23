"""A small noise-conditioned graph repair model on the project's exact schema.

This is a separate denoising autoencoder feasibility model, not a DeFoG
checkpoint, exact graph likelihood, or synthesis/chemical-stability oracle.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from latentmol.losses import LossConfig, reconstruction_losses
from latentmol.models import GraphAutoencoder, ModelConfig
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch, RelaxedGraph


@dataclass(frozen=True)
class DenoiserLossConfig:
    """One frozen repair intervention; not an independently tuned property score."""

    padding_weight: float = 0.5
    no_bond_weight: float = 1.0
    charge_weight: float = 1.0
    bond_weight: float = 2.0
    size_weight: float = 0.1
    padding_margin: int = 4
    clean_retention_weight: float = 0.5


def denoising_reconstruction_losses(predicted: DecoderOutput, target: HardGraphBatch,
                                   config: DenoiserLossConfig) -> dict:
    """Supervise absent bonds near the true molecular boundary as well as real pairs.

    The mask comes only from clean target occupancy, never predicted occupancy.
    This prevents a decoder from lowering its bond penalty by deleting atoms.
    Each unordered pair contributes once via symmetric logits.
    """
    base = reconstruction_losses(predicted, target, LossConfig(
        padding_weight=config.padding_weight,
        no_bond_weight=config.no_bond_weight,
        charge_weight=config.charge_weight,
        bond_weight=config.bond_weight,
        size_weight=config.size_weight))
    batch, nodes = target.atoms.shape
    size = (target.atoms != 0).sum(-1)
    eligible = torch.arange(nodes, device=target.atoms.device)[None, :] < (
        size + config.padding_margin).clamp(max=nodes)[:, None]
    upper = torch.ones((nodes, nodes), device=target.atoms.device, dtype=torch.bool).triu(1)
    pair_mask = eligible[:, :, None] & eligible[:, None, :] & upper
    weights = predicted.bonds.new_ones(predicted.bonds.shape[-1])
    weights[0] = config.no_bond_weight
    bond = F.cross_entropy(predicted.symmetric_bond_logits[pair_mask],
                           target.bonds[pair_mask], weight=weights)
    total = (base["atom_ce"] + config.charge_weight * base["charge_ce"]
             + config.bond_weight * bond + config.size_weight * base["size_mse"])
    return {"atom_ce": base["atom_ce"], "charge_ce": base["charge_ce"],
            "bond_ce": bond, "size_mse": base["size_mse"],
            "reconstruction": total,
            "target_active_bond_ce": base["bond_ce"]}


class SchemaDenoiser(nn.Module):
    """Reuse the proven continuous encoder/decoder boundary with time conditioning."""

    def __init__(self, spec: GraphSpec, config: ModelConfig):
        super().__init__()
        self.spec = spec
        self.backbone = GraphAutoencoder(spec, config)
        self.time_shift = nn.Sequential(nn.Linear(1, config.latent), nn.SiLU(),
                                        nn.Linear(config.latent, config.latent))
        nn.init.zeros_(self.time_shift[-1].weight)
        nn.init.zeros_(self.time_shift[-1].bias)

    def forward(self, graph: RelaxedGraph, time: torch.Tensor) -> DecoderOutput:
        if time.ndim != 1 or time.shape[0] != graph.atoms.shape[0]:
            raise ValueError("Expected one corruption time per graph")
        mu, _ = self.backbone.encode(graph)
        return self.backbone.decode(mu + self.time_shift(time[:, None]))


def corrupt_graph(target: HardGraphBatch, spec: GraphSpec, time: torch.Tensor,
                  generator: torch.Generator) -> HardGraphBatch:
    """Replace local categorical slots and undirected bonds with seeded draws.

    Only true occupied slots plus four following padding slots are eligible for
    replacement; far padding stays absent. This bounds false-node noise and
    explicitly tests NO_ATOM recovery near the molecular boundary. Bond noise
    samples each unordered eligible pair once and preserves graph symmetry.
    """
    atoms, charges, bonds = target.atoms, target.charges, target.bonds
    batch, nodes = atoms.shape
    if time.shape != (batch,) or not bool(((time >= 0) & (time <= 1)).all()):
        raise ValueError("Corruption times must have shape [batch] and lie in [0,1]")
    if charges.shape != atoms.shape or bonds.shape != (batch, nodes, nodes):
        raise ValueError("Hard graph shapes disagree")
    device = atoms.device
    active = (atoms != 0).sum(-1)
    eligible = torch.arange(nodes, device=device)[None, :] < (active + 4).clamp(max=nodes)[:, None]
    atom_mask = eligible & (torch.rand((batch, nodes), device=device, generator=generator)
                            < time[:, None])
    charge_mask = eligible & (torch.rand((batch, nodes), device=device, generator=generator)
                              < time[:, None])
    noisy_atoms = torch.where(atom_mask, torch.randint(len(spec.atomic_numbers),
                              atoms.shape, device=device, generator=generator), atoms)
    noisy_charges = torch.where(charge_mask, torch.randint(len(spec.formal_charges),
                                charges.shape, device=device, generator=generator), charges)
    noisy_charges = noisy_charges.masked_fill(
        noisy_atoms == 0, spec.formal_charges.index(0))
    upper = torch.ones((nodes, nodes), device=device, dtype=torch.bool).triu(1)
    eligible_pairs = eligible[:, :, None] & eligible[:, None, :] & upper
    bond_mask = eligible_pairs & (torch.rand((batch, nodes, nodes),
                                   device=device, generator=generator) < time[:, None, None])
    replacement = torch.randint(len(spec.bond_names), bonds.shape,
                                device=device, generator=generator)
    upper_bonds = torch.where(bond_mask, replacement, bonds).triu(1)
    noisy_bonds = upper_bonds + upper_bonds.transpose(1, 2)
    present = noisy_atoms != 0
    noisy_bonds = noisy_bonds.masked_fill(
        ~(present[:, :, None] & present[:, None, :]), 0)
    return HardGraphBatch(noisy_atoms, noisy_charges, noisy_bonds)
