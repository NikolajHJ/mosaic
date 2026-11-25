"""Ordered encoding, learned set pooling, and discrete canonical input ordering."""

import torch
from torch import nn

from latentmol.representation import GraphSpec, HardGraphBatch, RelaxedGraph


def _graph_attention_inputs(graph: RelaxedGraph):
    """Build shared shortest-path buckets once for all blocks on a hard graph.

    Buckets are self, 1, 2, 3, 4 and 5+ (including unreachable pairs). Adding
    self-loops makes each Boolean matrix power represent reachability in *at
    most* that many hops. Occupancy masks exclude padding from every path.
    """
    occupied = graph.occupancy.bool()
    bond_type = graph.bonds.argmax(dim=-1)
    live_pairs = occupied[:, :, None] & occupied[:, None, :]
    direct = (bond_type != 0) & live_pairs
    diagonal = torch.eye(direct.size(-1), device=direct.device, dtype=torch.bool)
    self_pair = diagonal & live_pairs
    one = (direct | self_pair).to(graph.atoms.dtype)
    two = (one @ one > 0).to(one.dtype)
    three = two @ one > 0
    four = two @ two > 0
    # Count missed reachability thresholds to get the clipped distance.
    distance = ((~self_pair).long() + (one == 0) + (two == 0) + (~three) + (~four))
    return occupied, bond_type, distance


class GraphAttentionBlock(nn.Module):
    """Two pre-normalized residual updates with independent structural biases."""

    def __init__(self, hidden: int, heads: int, bond_types: int,
                 attention_head_width: int | None = None):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads if attention_head_width is None else attention_head_width
        self.inner_width = heads * self.head_dim
        self.attention_norm = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * self.inner_width)
        self.attention_output = nn.Linear(self.inner_width, hidden)
        self.distance_bias = nn.Embedding(6, heads)
        self.bond_bias = nn.Embedding(bond_types, heads, padding_idx=0)
        self.feedforward_norm = nn.LayerNorm(hidden)
        self.feedforward = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.SiLU(),
                                         nn.Linear(2 * hidden, hidden))
        nn.init.normal_(self.distance_bias.weight, std=0.02)
        with torch.no_grad():
            nn.init.normal_(self.bond_bias.weight[1:], std=0.02)
        # The initial graft reproduces the parent. Both output projections
        # receive gradients on the very first update.
        for projection in (self.attention_output, self.feedforward[-1]):
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(self, h, occupied, bond_type, distance):
        batch, slots, _ = h.shape
        qkv = self.qkv(self.attention_norm(h)).reshape(batch, slots, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        bias = (self.distance_bias(distance) + self.bond_bias(bond_type)).permute(0, 3, 1, 2).to(q.dtype)
        # Finite masking also handles an entirely empty graph. Its query outputs
        # are cleared below, so it cannot contribute any learned atom state.
        scores = (q @ k.transpose(-1, -2)) * (self.head_dim ** -0.5) + bias
        scores = scores.masked_fill(~occupied[:, None, None, :], torch.finfo(scores.dtype).min)
        context = scores.softmax(dim=-1) @ v
        h = h + self.attention_output(context.transpose(1, 2).reshape(batch, slots, self.inner_width))
        return (h + self.feedforward(self.feedforward_norm(h))) * occupied[..., None]


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
                 degree_features: bool = False, sum_scale: float = 3.,
                 graph_attention_layers: int = 0, attention_heads: int = 4,
                 attention_head_width: int | None = None):
        super().__init__()
        # The old checkpoints used 2 * hidden. A separate pooled width lets a
        # wider molecular latent coexist with narrower per-atom states.
        pool_hidden = 2 * hidden if pool_hidden is None else pool_hidden
        if pooling not in ("ordered", "invariant", "canonical"):
            raise ValueError("Unknown encoder pooling")
        if (type(graph_attention_layers) is not int or graph_attention_layers < 0
                or (graph_attention_layers and (pooling != "canonical" or attention_heads < 1
                                                or (attention_head_width is None and hidden % attention_heads)))):
            raise ValueError("Graph attention requires canonical input and a divisible or explicitly specified head width")
        if attention_head_width is not None and (type(attention_head_width) is not int or attention_head_width < 1):
            raise ValueError("Explicit attention head width must be a positive integer")
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
        self.graph_attention = nn.ModuleList(GraphAttentionBlock(hidden, attention_heads, len(spec.bond_names),
                                                                attention_head_width)
                                             for _ in range(graph_attention_layers))
        if self.graph_attention and self.degree_projection is not None:
            nn.init.zeros_(self.degree_projection.weight)
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
        if self.graph_attention:
            structural_inputs = _graph_attention_inputs(graph)
            for block in self.graph_attention:
                h = block(h, *structural_inputs)
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
