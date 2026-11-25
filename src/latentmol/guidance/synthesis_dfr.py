# DFRscore network/feature portions: Copyright (c) 2022 Hyeongwoo Kim.
# MIT License: Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including without
# limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom
# the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Frozen DFRscore with an exploratory continuous graph adapter.

The network and released weights are those of Hwoo-Kim/DFRscore (MIT).
This module changes its binary attention mask to probability-weighted
attention.  It does *not* claim that soft chemistry features are native RDKit
features: hydrogen/degree/hybridization are local valence approximations and
ring flags are bounded non-backtracking closed-walk indicators, not SSSR.
There is no RDKit, argmax, graph discretization, or model training in this path.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from latentmol.representation import GraphSpec, RelaxedGraph


DFR_WEIGHTS_SHA256 = "56e6c58c8baf3770ac5eab7dbae81fd7f9622ab9ec6b9a453ed5dc93029faffd"
DFR_SOURCE = "https://github.com/Hwoo-Kim/DFRscore"


def _categorical_linear(value: torch.Tensor, largest: int) -> torch.Tensor:
    """Linear interpolation between integer categories plus one overflow bin."""
    bins = torch.arange(largest + 1, device=value.device, dtype=value.dtype)
    regular = (1 - (value[..., None] - bins).abs()).clamp_min(0)
    overflow = (1 - regular.sum(-1, keepdim=True)).clamp(0, 1)
    return torch.cat((regular, overflow), -1)


def _hydrogen_table(spec: GraphSpec) -> torch.Tensor:
    """Static local closed-shell valence heuristic; no live molecular decisions."""
    defaults = {5: (3,), 6: (4,), 7: (3,), 8: (2,), 9: (1,), 15: (3, 5),
                16: (2, 4, 6), 17: (1,), 35: (1,), 53: (1,)}
    rows = torch.zeros(len(spec.atomic_numbers), len(spec.formal_charges), 9)
    for a, number in enumerate(spec.atomic_numbers):
        for q, charge in enumerate(spec.formal_charges):
            allowed = defaults.get(number, (0,))
            if number == 6:
                allowed = (max(0, 4 - abs(charge)),)
            elif number in (7, 8):
                allowed = (max(0, defaults[number][0] + charge),)
            elif number in (9, 17, 35, 53) and charge < 0:
                allowed = (0,)
            elif number == 5:
                allowed = (max(0, 3 - charge),)
            for valence in range(9):
                candidates = [v for v in allowed if v >= valence]
                rows[a, q, valence] = min(candidates) - valence if candidates else 0
    return rows


def _ring_walk_features(edge: torch.Tensor) -> torch.Tensor:
    """Six smooth, bounded ring proxies for lengths 3..7 and 8+.

    Track non-backtracking walks with [start, first, previous, current] state. This
    is exact for simple isolated rings but does not reproduce RDKit SSSR on
    fused/caged graphs. Repeated traversal of an isolated 3/4-cycle is removed
    from its 6/8-cycle proxy; other self-intersecting walks remain a limitation.
    """
    batch, nodes, _ = edge.shape
    eye = torch.eye(nodes, device=edge.device, dtype=edge.dtype)
    start_previous = eye[:, None, :, None]
    first_current = eye[None, :, None, :]
    walk = edge[..., None, None] * start_previous[None] * first_current[None]
    closure = eye[None, :, None, None, :] * (1 - eye)[None, None, :, :, None]
    closed = {}
    for length in range(2, 9):
        # Retaining the first neighbor rejects a reverse step across the
        # closing edge too; otherwise ring-plus-tail walks create false rings.
        incoming = walk.sum(3)
        reverse = walk.transpose(3, 4)
        walk = edge[:, None, None] * (incoming[..., None] - reverse)
        if length >= 3:
            counts = (walk * closure).sum((2, 3, 4)) * 0.5
            closed[length] = counts.clamp_min(0)
    values = []
    for length in range(3, 9):
        count = closed[length]
        if length == 6:
            count = count - closed[3]
        elif length == 8:
            count = count - closed[4]
        values.append(count.clamp(0, 1))
    return torch.stack(values, -1)


class _WeightedAttention(nn.Module):
    """DFRscore's attention with edge probability in the attention measure."""
    def __init__(self, dim=256, heads=8):
        super().__init__()
        self.dim, self.heads = dim, heads
        self.W = nn.Linear(dim, dim)
        self.a = nn.Linear(dim, 2 * heads)

    def forward(self, x, adjacency):
        projected = self.W(x)
        shape = projected.shape
        pair = self.a(projected).reshape(*shape[:-1], self.heads, 2).transpose(1, 2)
        logits = F.leaky_relu(pair[..., 0].unsqueeze(-1) + pair[..., 1].unsqueeze(-2), 0.2)
        # The edge weight multiplies exp(logit), avoiding log(0) and retaining
        # a useful one-sided derivative for a currently absent candidate edge.
        supported = adjacency[:, None] > 0
        scale = torch.where(supported, logits, -torch.inf).amax(-1, keepdim=True)
        # A zero-weight absent edge must not set the stabilization maximum:
        # otherwise valid hard-graph messages can all underflow. The cap only
        # affects boundary derivatives of absent edges with extreme logits;
        # positive-edge exponents are never above zero before this cap.
        weight = (logits - scale).clamp_max(20).exp() * adjacency[:, None]
        attention = weight / weight.sum(-1, keepdim=True).clamp_min(torch.finfo(x.dtype).tiny)
        value = projected.reshape(*shape[:-1], self.heads, self.dim // self.heads).transpose(1, 2)
        output = (attention @ value).transpose(1, 2).reshape(shape)
        return F.relu(output)


class _FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.fcs = nn.ModuleList((nn.Linear(256, 128), nn.Linear(128, 128), nn.Linear(128, 1)))

    def forward(self, x):
        for layer in self.fcs[:-1]:
            x = F.elu(layer(x))
        return self.fcs[-1](x)


class FrozenDFRscore(nn.Module):
    """Released DFRscore weights with an explicitly approximate soft adapter.

    ``difficulty`` and ``raw_score`` are the same scalar, lower is easier.
    DFRscore was trained to approximate a bounded retrosynthetic-depth label;
    it is neither an experimental synthesis cost nor a route certificate.
    """
    def __init__(self, spec: GraphSpec, weights_path: str | Path):
        super().__init__()
        weights_path = Path(weights_path)
        digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        if digest != DFR_WEIGHTS_SHA256:
            raise ValueError("DFRscore checkpoint does not match the recorded public weights")
        self.spec = spec
        self.embedding = nn.Linear(49, 256, bias=False)
        self.GAT_layers = nn.ModuleList(_WeightedAttention() for _ in range(6))
        self.dense = _FeedForward()
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.load_state_dict(state, strict=True)
        self.register_buffer("hydrogens", _hydrogen_table(spec), persistent=False)
        numbers = (6, 7, 8, 9, 16, 17, 35, 53, 5, 15)
        element_map = torch.zeros(len(spec.atomic_numbers), 11)
        for i, number in enumerate(spec.atomic_numbers):
            element_map[i, numbers.index(number) if number in numbers else 10] = 1
        self.register_buffer("element_map", element_map, persistent=False)
        charge_map = torch.zeros(len(spec.formal_charges), 6)
        for i, charge in enumerate(spec.formal_charges):
            charge_map[i, charge + 2 if -2 <= charge <= 2 else 5] = 1
        self.register_buffer("charge_map", charge_map, persistent=False)
        orders = {"NONE": 0, "SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 1.5}
        if any(name not in orders for name in spec.bond_names):
            raise ValueError("DFRscore adapter encountered an unsupported bond vocabulary")
        self.register_buffer("bond_orders", torch.tensor([orders[name] for name in spec.bond_names]), persistent=False)
        self.requires_grad_(False)
        self.eval()
        self.adapter_metadata = dict(
            model="DFRscore", source=DFR_SOURCE, weights_sha256=digest,
            difficulty_direction="lower_is_easier", policy="soft-dfr-v2",
            feature_approximations=[
                "hydrogens and total degree: expected local closed-shell valence table",
                "hybridization: multiple-bond and conjugated neutral/nonpositive N/O heuristic",
                "chirality: unspecified; the graph schema carries no stereochemistry",
                "ring flags: bounded non-backtracking closed walks, not RDKit SSSR",
                "size8 flag also proxies larger rings, which are not separately enumerated",
            ],
            native_parity_scope="weighted attention equals stock attention for binary adjacency and identical features",
            boundary_gradient_note="absent-edge attention exponent capped at20 above supported-edge maximum for numerical stability",
        )

    def _validate(self, graph, active):
        n = self.spec.max_nodes
        if (not isinstance(active, torch.Tensor) or active.dtype != torch.bool
                or active.shape != (n,) or not active.any()):
            raise ValueError("DFRscore requires a nonempty fixed active-slot mask")
        tensors = (graph.atoms, graph.charges, graph.bonds)
        if graph.atoms.ndim != 3 or graph.atoms.shape[0] < 1:
            raise ValueError("DFRscore atoms must have shape [batch,nodes,categories]")
        batch = graph.atoms.shape[0]
        expected = ((batch, n, len(self.spec.atomic_numbers)),
                    (batch, n, len(self.spec.formal_charges)),
                    (batch, n, n, len(self.spec.bond_names)))
        for tensor, shape in zip(tensors, expected):
            if tuple(tensor.shape) != shape or not tensor.is_floating_point():
                raise ValueError("DFRscore graph does not match its probability schema")
            if (not torch.isfinite(tensor).all() or bool((tensor < -1e-6).any())
                    or not torch.allclose(tensor.sum(-1), torch.ones_like(tensor[..., 0]), atol=2e-5, rtol=0)):
                raise ValueError("DFRscore graph must contain finite simplex probabilities")
        if len({x.device for x in tensors}) != 1 or len({x.dtype for x in tensors}) != 1:
            raise ValueError("DFRscore graph tensors must share device and dtype")
        if not torch.allclose(graph.bonds, graph.bonds.transpose(1, 2), atol=2e-5, rtol=0):
            raise ValueError("DFRscore requires symmetric bond probabilities")
        mask = active.to(graph.atoms.device)
        if (not torch.allclose(graph.atoms[:, mask, 0], torch.zeros_like(graph.atoms[:, mask, 0]), atol=1e-6, rtol=0)
                or not torch.allclose(graph.atoms[:, ~mask, 0], torch.ones_like(graph.atoms[:, ~mask, 0]), atol=1e-6, rtol=0)):
            raise ValueError("DFRscore currently requires the shared fixed-occupancy graph policy")

    def features(self, graph: RelaxedGraph, active: torch.Tensor):
        """Return 49 soft atom features and weighted adjacency, with live inputs."""
        self._validate(graph, active)
        mask = active.to(graph.atoms.device)
        atom = graph.atoms[:, mask]
        charge = graph.charges[:, mask]
        bonds = graph.bonds[:, mask][:, :, mask]
        nodes = atom.shape[1]
        eye = torch.eye(nodes, device=atom.device, dtype=atom.dtype)[None]
        # Sum real-bond categories rather than subtracting the no-bond
        # probability: float32 can round NONE to1 while retaining positive
        # real-bond mass, spuriously moving such pairs to the boundary branch.
        edge = bonds[..., 1:].sum(-1) * (1 - eye)
        adjacency = edge + eye
        valence = (bonds @ self.bond_orders.to(bonds)).sum(-1)
        interpolation = _categorical_linear(valence, 8)[..., :9]
        atom_charge = atom[..., :, None] * charge[..., None, :]
        hydrogen_by_valence = torch.einsum("bnaq,aqv->bnv", atom_charge, self.hydrogens.to(atom))
        hydrogen = (hydrogen_by_valence * interpolation).sum(-1)
        degree = edge.sum(-1) + hydrogen

        def bond_probability(name):
            return bonds[..., self.spec.bond_names.index(name)] if name in self.spec.bond_names else edge * 0

        triple = bond_probability("TRIPLE") * (1 - eye)
        double = bond_probability("DOUBLE") * (1 - eye)
        aromatic = bond_probability("AROMATIC") * (1 - eye)
        any_triple = 1 - (1 - triple).prod(-1)
        any_double_or_aromatic = 1 - (1 - double - aromatic).clamp(0, 1).prod(-1)
        single = bond_probability("SINGLE") * (1 - eye)
        neighbor_unsaturated = 1 - (1 - single * any_double_or_aromatic[:, None, :]).prod(-1)
        nitrogen_oxygen = sum(atom[..., i] for i, number in enumerate(self.spec.atomic_numbers) if number in (7, 8))
        nonpositive_charge = sum(charge[..., i] for i, value in enumerate(self.spec.formal_charges) if value <= 0)
        conjugated = nitrogen_oxygen * nonpositive_charge * neighbor_unsaturated
        unsaturated = 1 - (1 - any_double_or_aromatic) * (1 - conjugated)
        double_total = double.sum(-1)
        sp = 1 - (1 - any_triple) * (1 - (double_total - 1).clamp(0, 1))
        sp2 = (1 - sp) * unsaturated
        sp3 = (1 - sp) * (1 - unsaturated)
        zero = sp * 0
        hybrid = torch.stack((zero, sp, sp2, sp3, zero, zero, zero), -1)
        chirality = torch.stack((torch.ones_like(sp), zero, zero, zero), -1)
        aromatic_atom = 1 - (1 - aromatic).prod(-1)
        rings = _ring_walk_features(edge)
        features = torch.cat((atom @ self.element_map.to(atom), _categorical_linear(degree, 6),
                              charge @ self.charge_map.to(charge), _categorical_linear(hydrogen, 4),
                              hybrid, chirality, aromatic_atom[..., None], rings), -1)
        if features.shape[-1] != 49 or not torch.isfinite(features).all():
            raise ValueError("DFRscore soft feature computation is incompatible or nonfinite")
        return features, adjacency

    def score_features(self, features, adjacency):
        """Score supplied features; useful for native-network parity checks."""
        x = self.embedding(features)
        for layer in self.GAT_layers:
            x = layer(x, adjacency)
        x = x.mean(1)
        return F.elu(self.dense(x)).squeeze(-1) + 1.5

    def reference_score(self, mol) -> float:
        """Original hard RDKit features, strictly outside the gradient path.

        This is the released network's discrete-molecule score, useful for
        measuring the feature-adapter discrepancy during posthoc evaluation.
        """
        from rdkit import Chem

        if mol is None or not mol.GetNumAtoms():
            raise ValueError("DFRscore reference requires a nonempty RDKit molecule")

        def one_hot(value, choices):
            value = value if value in choices else choices[-1]
            return [value == choice for choice in choices]

        features = []
        hybrid = [Chem.rdchem.HybridizationType.UNSPECIFIED, Chem.rdchem.HybridizationType.SP,
                  Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3,
                  Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2,
                  Chem.rdchem.HybridizationType.OTHER]
        chirality = [Chem.rdchem.ChiralType.CHI_UNSPECIFIED, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                     Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, Chem.rdchem.ChiralType.CHI_OTHER]
        for atom in mol.GetAtoms():
            features.append(one_hot(atom.GetSymbol(), ["C", "N", "O", "F", "S", "Cl", "Br", "I", "B", "P", "ELSE"])
                            + one_hot(atom.GetTotalDegree(), [0, 1, 2, 3, 4, 5, 6, "ELSE"])
                            + one_hot(atom.GetFormalCharge(), [-2, -1, 0, 1, 2, "ELSE"])
                            + one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, "ELSE"])
                            + one_hot(atom.GetHybridization(), hybrid)
                            + one_hot(atom.GetChiralTag(), chirality)
                            + [atom.GetIsAromatic()] + [False] * 6)
        for ring in Chem.GetSymmSSSR(mol):
            slot = min(len(ring) - 3, 5)
            for index in ring:
                features[index][43 + slot] = True
        dtype, device = self.embedding.weight.dtype, self.embedding.weight.device
        x = torch.tensor(features, dtype=dtype, device=device)[None]
        a = torch.tensor(Chem.GetAdjacencyMatrix(mol), dtype=dtype, device=device)[None]
        a = a + torch.eye(a.shape[-1], dtype=dtype, device=device)[None]
        with torch.no_grad():
            score = self.score_features(x, a)
        if not torch.isfinite(score).all():
            raise ValueError("DFRscore native reference produced a nonfinite score")
        return float(score.item())

    def forward(self, graph: RelaxedGraph, active: torch.Tensor):
        features, adjacency = self.features(graph, active)
        score = self.score_features(features, adjacency)
        if not torch.isfinite(score).all():
            raise ValueError("DFRscore produced a nonfinite synthesis score")
        return dict(difficulty=score, raw_score=score)
