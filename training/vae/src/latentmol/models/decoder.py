"""One-shot slot decoder with a symmetric categorical head for each atom pair."""

import torch
from torch import nn

from latentmol.representation import DecoderOutput, GraphSpec


class GraphDecoder(nn.Module):
    """Decode a latent vector directly to differentiable graph logits.

    Pair features are symmetric by construction. Softmax is across bond types,
    never across neighboring atoms; bond probabilities therefore do not compete
    with unrelated pairs as ordinary attention weights would.
    """

    def __init__(self, spec: GraphSpec, hidden: int, latent: int):
        super().__init__()
        self.spec = spec
        self.hidden = hidden
        self.nodes = nn.Sequential(
            nn.Linear(latent, 2 * hidden), nn.SiLU(),
            nn.Linear(2 * hidden, spec.max_nodes * hidden),
        )
        self.refine = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU())
        self.atom_head = nn.Linear(hidden, len(spec.atomic_numbers))
        self.charge_head = nn.Linear(hidden, len(spec.formal_charges))
        self.bond_head = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, len(spec.bond_names)),
        )

    def forward(self, z) -> DecoderOutput:
        h = self.refine(self.nodes(z).reshape(-1, self.spec.max_nodes, self.hidden))
        left, right = h[:, :, None, :], h[:, None, :, :]
        pair_features = torch.cat((left + right, (left - right).abs(), left * right), dim=-1)
        return DecoderOutput(self.atom_head(h), self.charge_head(h), self.bond_head(pair_features))
