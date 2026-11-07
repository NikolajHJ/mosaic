"""Ordered encoding, learned set pooling, and discrete canonical input ordering."""

import torch
from torch import nn

from latentmol.representation import GraphSpec, HardGraphBatch, RelaxedGraph


class MessageLayer(nn.Module):
    """Bond-specific messages with an explicit aggregation policy.

    Historical checkpoints use a degree mean. Without positional features that
    loses neighbor multiplicity (identical carbons have identical means at every
    degree). The opt-in fixed-scale sum retains it without tying features to slots.
    """

    def __init__(self, hidden: int, bond_types: int, aggregation: str = "mean", sum_scale: float = 3.):
        super().__init__()
        self.aggregation, self.sum_scale = aggregation, sum_scale
        self.messages = nn.ModuleList(nn.Linear(hidden, hidden, bias=False) for _ in range(bond_types - 1))
        self.update = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, graph: RelaxedGraph):
        messages = torch.zeros_like(h)
        for bond_type, transform in enumerate(self.messages, start=1):
            messages = messages + torch.bmm(graph.bonds[..., bond_type], transform(h))
        if self.aggregation == "mean":
            degree = graph.bonds[..., 1:].sum(dim=(-1, -2)).clamp_min(1.0)
            messages = messages / degree[..., None]
        else:
            messages = messages / self.sum_scale
        return self.norm(h + self.update(torch.cat((h, messages), dim=-1))) * graph.occupancy[..., None]


class GraphEncoder(nn.Module):
    """The historical ordered baseline and a separately selected invariant mode.

    Positions and flattened node features preserve the supervised canonical
    ordering in the default mode. Invariant mode removes positional features and
    uses occupancy-aware sum/mean/count pooling. Canonical mode retains the full
    ordered encoder behind lossless hard-graph preprocessing; it does not accept
    soft graphs or supply input gradients. Matching-aware reconstruction is
    provided separately in ``latentmol.training.matching``.
    """

    def __init__(self, spec: GraphSpec, hidden: int, latent: int, layers: int,
                 pool_hidden: int | None = None, residual_layers: int = 0,
                 pooling: str = "ordered", aggregation: str = "mean",
                 degree_features: bool = False, sum_scale: float = 3.):
        super().__init__()
        # The old checkpoints used 2 * hidden. A separate pooled width lets a
        # wider molecular latent coexist with narrower per-atom states.
        pool_hidden = 2 * hidden if pool_hidden is None else pool_hidden
        if pooling not in ("ordered", "invariant", "canonical"):
            raise ValueError("Unknown encoder pooling")
        self.pooling, self.capacity, self.spec = pooling, spec.max_nodes, spec
        self.atom_projection = nn.Linear(len(spec.atomic_numbers), hidden, bias=False)
        self.charge_projection = nn.Linear(len(spec.formal_charges), hidden, bias=False)
        self.position = (nn.Parameter(torch.randn(spec.max_nodes, hidden) * 0.02)
                         if pooling != "invariant" else None)
        # Opt-in tensors only: old ordered/invariant states still load strictly.
        self.degree_projection = (nn.Linear(len(spec.bond_names)-1, hidden, bias=False)
                                  if degree_features else None)
        self.sum_scale = sum_scale
        self.layers = nn.ModuleList(MessageLayer(hidden, len(spec.bond_names), aggregation, sum_scale)
                                    for _ in range(layers))
        # Keep the historical pool.1 parameter names and default state exactly.
        self.pool = nn.Sequential(nn.Flatten(1) if pooling != "invariant" else nn.Identity(),
            nn.Linear(spec.max_nodes * hidden if pooling != "invariant" else 2*hidden+1, pool_hidden), nn.SiLU())
        # Empty for all historical checkpoints: adding the new option does not
        # introduce missing keys when loading their state dictionaries strictly.
        self.pool_updates = nn.ModuleList(PooledResidualBlock(pool_hidden) for _ in range(residual_layers))
        self.mu = nn.Linear(pool_hidden, latent)
        self.logvar = nn.Linear(pool_hidden, latent)

    def forward(self, graph: RelaxedGraph | HardGraphBatch, *, from_canonical_cache: bool = False):
        if from_canonical_cache:
            if self.pooling != "canonical" or not isinstance(graph, HardGraphBatch):
                raise ValueError("The canonical cache path requires hard labels and a canonical encoder")
            graph = graph.to_relaxed(self.spec)
        elif self.pooling == "canonical":
            # Explicit hard-input boundary, never part of the decoder/scorer AD.
            from latentmol.datasets.ordering import canonical_encoder_input
            graph = canonical_encoder_input(graph, self.spec)
        h = self.atom_projection(graph.atoms) + self.charge_projection(graph.charges)
        if self.position is not None:
            h = h + self.position
        if self.degree_projection is not None:
            degree = (graph.bonds[..., 1:] * graph.occupancy[:, None, :, None]).sum(-2)
            h = h + self.degree_projection(degree/self.sum_scale)
        h = h * graph.occupancy[..., None]
        for layer in self.layers:
            h = layer(h, graph)
        if self.pooling == "invariant":
            # h is already occupancy-gated by the input/each message layer.
            # Do not gate it again for soft input occupancies. Count is explicit
            # so mean pooling cannot discard molecular size.
            count = graph.occupancy.sum(1, keepdim=True)
            total = h.sum(1)
            h = torch.cat((total/self.capacity, total/count.clamp_min(1.), count/self.capacity), dim=-1)
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
