"""An explicitly ordered graph encoder for canonical-slot reconstruction."""

import torch
from torch import nn

from latentmol.representation import GraphSpec, RelaxedGraph


class MessageLayer(nn.Module):
    """Bond-specific messages with occupancy-aware degree normalization."""

    def __init__(self, hidden: int, bond_types: int):
        super().__init__()
        self.messages = nn.ModuleList(nn.Linear(hidden, hidden, bias=False) for _ in range(bond_types - 1))
        self.update = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, graph: RelaxedGraph):
        messages = torch.zeros_like(h)
        for bond_type, transform in enumerate(self.messages, start=1):
            messages = messages + torch.bmm(graph.bonds[..., bond_type], transform(h))
        degree = graph.bonds[..., 1:].sum(dim=(-1, -2)).clamp_min(1.0)
        messages = messages / degree[..., None]
        return self.norm(h + self.update(torch.cat((h, messages), dim=-1))) * graph.occupancy[..., None]


class GraphEncoder(nn.Module):
    """Encode canonical node slots; this baseline is not permutation invariant.

    Positions and flattened node features preserve the supervised canonical
    ordering. A future invariant encoder requires a matching-aware decoder/loss.
    """

    def __init__(self, spec: GraphSpec, hidden: int, latent: int, layers: int,
                 pool_hidden: int | None = None, residual_layers: int = 0):
        super().__init__()
        # The old checkpoints used 2 * hidden. A separate pooled width lets a
        # wider molecular latent coexist with narrower per-atom states.
        pool_hidden = 2 * hidden if pool_hidden is None else pool_hidden
        self.atom_projection = nn.Linear(len(spec.atomic_numbers), hidden, bias=False)
        self.charge_projection = nn.Linear(len(spec.formal_charges), hidden, bias=False)
        self.position = nn.Parameter(torch.randn(spec.max_nodes, hidden) * 0.02)
        self.layers = nn.ModuleList(MessageLayer(hidden, len(spec.bond_names)) for _ in range(layers))
        self.pool = nn.Sequential(nn.Flatten(1), nn.Linear(spec.max_nodes * hidden, pool_hidden), nn.SiLU())
        # Empty for all historical checkpoints: adding the new option does not
        # introduce missing keys when loading their state dictionaries strictly.
        self.pool_updates = nn.ModuleList(PooledResidualBlock(pool_hidden) for _ in range(residual_layers))
        self.mu = nn.Linear(pool_hidden, latent)
        self.logvar = nn.Linear(pool_hidden, latent)

    def forward(self, graph: RelaxedGraph):
        h = self.atom_projection(graph.atoms) + self.charge_projection(graph.charges) + self.position
        h = h * graph.occupancy[..., None]
        for layer in self.layers:
            h = layer(h, graph)
        pooled = self.pool(h)
        for update in self.pool_updates:
            pooled = update(pooled)
        return self.mu(pooled), self.logvar(pooled).clamp(-10.0, 8.0)


class PooledResidualBlock(nn.Module):
    """One width-preserving affine update and an identity shortcut.

    Two such blocks after pooling implement the requested 320→320→320 MLP.
    The affine pre-norm is trainable; the non-affine post-norm bounds scale
    without an extra learned gain. A residual path remains through each block.
    """

    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.update = nn.Linear(width, width)
        self.postnorm = nn.LayerNorm(width, elementwise_affine=False)

    def forward(self, value):
        return self.postnorm(value + torch.nn.functional.silu(self.update(self.norm(value))))
