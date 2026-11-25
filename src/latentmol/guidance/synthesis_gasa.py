# SPDX-License-Identifier: GPL-2.0-only
"""Frozen GASA with an explicitly experimental continuous graph extension.

Network algebra follows cadd-synthetic/GASA (GPL-2.0), with released weights.
The original DGL graph has discrete connectivity; here expected degree and
probability-weighted incoming messages extend it to the shared relaxed graph.
For binary adjacency and identical edge features the layer algebra is the
original graph network. This statement does not establish native chemistry
feature parity: conjugation/ring membership reuse the documented SynFrag soft
proxies. Stereochemistry is unspecified because our schema has no stereo field.

GASA's released forward starts from degree, not its prepared atom descriptors.
Element/charge can affect our conjugation proxy but have no direct embedding.
No RDKit, graph hardening, DGL dependency, or model training is on this path.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from latentmol.representation import GraphSpec, RelaxedGraph
from .synthesis_synfrag import FEATURE_POLICY, soft_features


GASA_SOURCE = "https://github.com/cadd-synthetic/GASA"
GASA_WEIGHTS_SHA256 = "7b6dcff15afc787d7a5b78befb5eae487c885819c048c206f0d9e42b5391d4bb"


def reference_features(mol):
    """Evaluation-only RDKit bond features in the released GASA ordering.

    This reproduces the feature definitions from GASA's ``BondF`` without a
    DGL dependency. It is never called from the differentiable graph path.
    """
    count = mol.GetNumAtoms()
    edge = torch.zeros(1, count, count, 11)
    adjacency = torch.eye(count)[None]
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        features = ([float(str(bond.GetBondType()) == kind) for kind in
                     ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")]
                    + [float(bond.GetIsConjugated()), float(bond.IsInRing())]
                    + [float(str(bond.GetStereo()) == kind) for kind in
                       ("STEREONONE", "STEREOANY", "STEREOZ", "STEREOE")]
                    + [0.])
        edge[:, i, j] = edge[:, j, i] = torch.tensor(features)
        adjacency[:, i, j] = adjacency[:, j, i] = 1
    edge[:, torch.arange(count), torch.arange(count), -1] = 1
    return edge, adjacency


def _incoming_attention(logits: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Normalize [batch,source,destination,head] over incoming sources.

    Multiplying exp(logit) by edge existence avoids log(0). Supported edges
    set the numerical scale; only absent-edge boundary exponents can hit the
    cap. Every node has a unit self loop, including isolated active atoms.
    """
    weight = weights[..., None]
    scale = torch.where(weight > 0, logits, -torch.inf).amax(1, keepdim=True)
    cap = min(20., math.log(torch.finfo(logits.dtype).max) / 4)
    unnormalized = (logits - scale).clamp_max(cap).exp() * weight
    return unnormalized / unnormalized.sum(1, keepdim=True).clamp_min(torch.finfo(logits.dtype).tiny)


class FrozenGASA(nn.Module):
    """Frozen easy/hard classifier; minimize negative log P(easy).

    ``raw_score`` is P(hard), matching the native class-1 direction. Neither
    score is a route certificate or a calibrated synthesis-success probability.
    Checkpoint tensors are immutable buffers, allowing input gradients only.
    """

    def __init__(self, spec: GraphSpec, weights_path: str | Path):
        super().__init__()
        self.spec = spec
        weights_path = Path(weights_path)
        digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()
        if digest != GASA_WEIGHTS_SHA256:
            raise ValueError("GASA checkpoint does not match the recorded public weights")
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        if set(checkpoint) != {"model_state_dict"}:
            raise ValueError("Incompatible GASA checkpoint envelope")
        state = checkpoint["model_state_dict"]
        expected = {"gnn.fc.weight": (1152, 1), "gnn.linears1.weight": (192, 203),
                    "gnn.linears2.weight": (128, 384), "gat1.fc.weight": (384, 128),
                    "gat2.fc.weight": (64, 384), "gat3.fc.weight": (64, 64),
                    "predict.1.weight": (2, 32)}
        if any(k not in state or tuple(state[k].shape) != shape for k, shape in expected.items()):
            raise ValueError("Incompatible GASA checkpoint architecture")
        self._keys = {}
        for key, tensor in state.items():
            if not isinstance(tensor, torch.Tensor) or not bool(torch.isfinite(tensor).all()):
                raise ValueError("Nonfinite or invalid GASA checkpoint tensor")
            name = key.replace(".", "__")
            self.register_buffer(name, tensor.clone())
            self._keys[key] = name
        self.eval().requires_grad_(False)
        self.adapter_metadata = dict(model="GASA", source=GASA_SOURCE, license="GPL-2.0",
            weights_sha256=digest, policy="soft-gasa-v1", difficulty_direction="lower_is_easier",
            difficulty="negative_log_probability_easy", raw_score="probability_hard",
            chemistry_feature_policy=FEATURE_POLICY,
            feature_approximations=["expected degree including unit self loop",
                "edge-existence-weighted attention and incoming mean",
                "ring and conjugation proxies shared with SynFrag; not native RDKit features",
                "stereochemistry unspecified; no direct atom/charge embeddings in released GASA"],
            native_parity_scope="binary adjacency and identical edge features: original layer algebra",
            boundary_gradient_note="absent-edge attention exponent capped relative to supported maximum")

    def _weight(self, key):
        return getattr(self, self._keys[key])

    def _linear(self, value, prefix, *, bias=True):
        return F.linear(value, self._weight(prefix + ".weight"),
                        self._weight(prefix + ".bias") if bias else None)

    def features(self, graph: RelaxedGraph, active: torch.Tensor):
        """Return conditional GASA edge11 features and weighted adjacency.

        Shared bond marginals already include endpoint occupancy; do not gate
        them a second time. This pilot uses a fixed Boolean occupied-slot mask.
        """
        n = self.spec.max_nodes
        tensors = (graph.atoms, graph.charges, graph.bonds)
        if (not isinstance(active, torch.Tensor) or active.dtype != torch.bool
                or active.shape != (n,) or not bool(active.any())):
            raise ValueError("GASA requires a nonempty fixed active-slot mask")
        batch = graph.atoms.shape[0] if graph.atoms.ndim == 3 else 0
        shapes = ((batch, n, len(self.spec.atomic_numbers)),
                  (batch, n, len(self.spec.formal_charges)),
                  (batch, n, n, len(self.spec.bond_names)))
        if batch < 1 or any(tuple(t.shape) != shape or not t.is_floating_point()
                            for t, shape in zip(tensors, shapes)):
            raise ValueError("GASA graph does not match its probability schema")
        if len({t.device for t in tensors}) != 1 or len({t.dtype for t in tensors}) != 1:
            raise ValueError("GASA graph tensors must share device and dtype")
        for tensor in tensors:
            if (not bool(torch.isfinite(tensor).all()) or bool((tensor < -1e-6).any())
                    or not torch.allclose(tensor.sum(-1), torch.ones_like(tensor[..., 0]), atol=2e-5, rtol=0)):
                raise ValueError("GASA requires finite simplex probabilities")
        if not torch.allclose(graph.bonds, graph.bonds.transpose(1, 2), atol=2e-5, rtol=0):
            raise ValueError("GASA requires symmetric bond probabilities")
        mask = active.to(graph.atoms.device)
        if (not torch.allclose(graph.atoms[:, mask, 0], torch.zeros_like(graph.atoms[:, mask, 0]), atol=1e-6, rtol=0)
                or not torch.allclose(graph.atoms[:, ~mask, 0], torch.ones_like(graph.atoms[:, ~mask, 0]), atol=1e-6, rtol=0)):
            raise ValueError("GASA requires the shared fixed-occupancy graph policy")
        _, synfrag_edges, adjacency = soft_features(graph, mask, self.spec)
        # SynFrag: type4, ring, conjugated, stereo5. GASA: type4,
        # conjugated, ring, stereo4, self-loop flag. Unknown stereo is unused.
        edge = synfrag_edges[..., [0, 1, 2, 3, 5, 4, 6, 7, 8, 9]]
        eye = torch.eye(adjacency.shape[-1], device=edge.device, dtype=edge.dtype)[None]
        edge = torch.cat((edge, eye.expand_as(adjacency)[..., None]), -1)
        return edge, adjacency

    def _gat(self, h, adjacency, prefix):
        heads, out_dim = self._weight(prefix + ".attn_l").shape[1:]
        value = self._linear(h, prefix + ".fc", bias=False).reshape(*h.shape[:2], heads, out_dim)
        left = (value * self._weight(prefix + ".attn_l")).sum(-1)
        right = (value * self._weight(prefix + ".attn_r")).sum(-1)
        logits = F.leaky_relu(left[:, :, None] + right[:, None, :], .2)
        attention = _incoming_attention(logits, adjacency)
        result = torch.einsum("bsdh,bshf->bdhf", attention, value)
        return result + self._weight(prefix + ".bias").reshape(heads, out_dim)

    def forward_features(self, edge, adjacency):
        """Score supplied edge features; binary reference and soft-input entry.

        Adjacency has [batch,source,destination] orientation. The live molecular
        adapter supplies symmetric edges, but asymmetric reference graphs are
        useful for catching accidental reversal of DGL's incoming reductions.
        """
        if (adjacency.ndim != 3 or adjacency.shape[1] < 1
                or adjacency.shape[1] != adjacency.shape[2]
                or edge.shape != (*adjacency.shape, 11)
                or not edge.is_floating_point() or not adjacency.is_floating_point()
                or edge.device != adjacency.device or edge.dtype != adjacency.dtype
                or not bool(torch.isfinite(edge).all()) or not bool(torch.isfinite(adjacency).all())
                or bool((adjacency < 0).any()) or bool((adjacency.sum(1) <= 0).any())):
            raise ValueError("GASA needs finite edge[B,N,N,11] and positive incoming adjacency")
        degree = adjacency.sum(1)[..., None]
        value = self._linear(degree, "gnn.fc", bias=False).reshape(*degree.shape[:2], 6, 192)
        left = (value * self._weight("gnn.attn_l")).sum(-1)
        right = (value * self._weight("gnn.attn_r")).sum(-1)
        attention = _incoming_attention(F.leaky_relu(left[:, :, None] + right[:, None, :], .2), adjacency)
        h = torch.einsum("bsdh,bshf->bdhf", attention, value).flatten(-2)
        h = self._linear(F.leaky_relu(h), "gnn.full")
        nodes = h.shape[1]
        source = h[:, :, None].expand(-1, -1, nodes, -1)
        destination = h[:, None].expand(-1, nodes, -1, -1)
        message = torch.tanh(self._linear(torch.cat((source, edge), -1), "gnn.linears1"))
        message = torch.cat((destination, message), -1)
        incoming = (message * adjacency[..., None]).sum(1) / adjacency.sum(1)[..., None]
        h = torch.tanh(self._linear(incoming, "gnn.linears2"))
        h = F.elu(self._gat(h, adjacency, "gat1")).flatten(-2)
        h = F.elu(self._gat(h, adjacency, "gat2")).flatten(-2)
        h = F.elu(self._gat(h, adjacency, "gat3")).mean(-2)
        node_weight = self._linear(h, "readout.weight_and_sum.atom_weighting.0").sigmoid()
        pooled = torch.cat(((h * node_weight).sum(1), h.amax(1)), -1)
        logits = self._linear(self._linear(pooled, "predict.0"), "predict.1")
        log_probabilities = logits.log_softmax(-1)
        if not bool(torch.isfinite(log_probabilities).all()):
            raise ValueError("Nonfinite GASA output")
        return {"difficulty": -log_probabilities[:, 0], "raw_score": log_probabilities[:, 1].exp()}

    def forward(self, graph: RelaxedGraph, active: torch.Tensor):
        return self.forward_features(*self.features(graph, active))
