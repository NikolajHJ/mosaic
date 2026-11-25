"""Frozen SynFrag weights with an experimental, continuous graph-input bridge.

The dense network follows the released AttentiveFP layer algebra; no DGL or
DeepChem imports are needed. Given the same node/edge features and a binary edge
mask, it computes the original hard-graph network. ``soft_features`` is a NEW
extension of DeepChem's 30/11 features, not a claim of exact chemistry-feature
parity. Hydrogen counts, hybridization, donor/acceptor, conjugation and ring
membership are explicit smooth proxies. No hardening/chemistry conversion occurs
in that live path. ``reference_features`` is an evaluation-only RDKit boundary.

Source: simmzx/SynFrag, commit 2c6cacec198c2746cade22643330d38cc19c62a4.
The official output is 1-pred (higher means easier). Its linear regression head
is not a calibrated probability; this adapter minimizes pred without clipping.
"""
# AttentiveFP layer adaptation from SynFrag, Copyright (c) 2025 Xiang Zhang.
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
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


FEATURE_POLICY = "synfrag-soft-features-v1-effective-resistance-rings"
ELEMENTS = (6, 7, 8, 9, 15, 16, 17, 35, 53)


def _soft_categories(value, maximum):
    """Piecewise-linear integer bins plus an overflow bin; no hardening."""
    support = torch.arange(maximum + 1, dtype=value.dtype, device=value.device)
    bins = F.relu(1 - (value[..., None] - support).abs())
    overflow = ((value - maximum).clamp(0, 1) + (-value).clamp(0, 1))[..., None]
    return torch.cat((bins, overflow), -1)


def _nonzero_probability(value):
    return value.clamp(0, 1)


def soft_features(graph, active, spec):
    """Return node30, edge11, and bond-existence weights (with unit self loops).

    Inputs use the shared graph schema. Bonds already include occupancy gating;
    it is NOT applied again. This pilot requires fixed occupied slots. Ring
    evidence uses effective resistance: bridge edges have wR≈1 and cycle edges
    wR<1. A 1e-5 Laplacian regularizer keeps disconnected/near-disconnected soft
    graphs finite. Multiplication by N and clipping extends the hard cycle flag.
    These and the chemical-feature proxies are recorded experimental choices.
    """
    n = graph.atoms.shape[1]
    if active.shape != (n,) or active.dtype != torch.bool or not bool(active.any()):
        raise ValueError("SynFrag requires a nonempty fixed Boolean active mask")
    a = graph.atoms[:, active]
    q = graph.charges[:, active]
    b = graph.bonds[:, active][:, :, active]
    if a.shape[-1] != len(spec.atomic_numbers) or q.shape[-1] != len(spec.formal_charges):
        raise ValueError("SynFrag graph categories do not match its schema")
    if b.shape[-1] != len(spec.bond_names):
        raise ValueError("SynFrag bond categories do not match its schema")
    if any(not bool(torch.isfinite(t).all()) for t in (a, q, b)):
        raise ValueError("Nonfinite SynFrag molecular probabilities")
    if bool((a[..., 0] > 1e-5).any()):
        raise ValueError("SynFrag fixed-slot pilot cannot drop NO_ATOM probability")
    unknown = set(spec.bond_names) - {"NONE", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"}
    if unknown:
        raise ValueError(f"Unsupported SynFrag bond categories: {unknown}")
    count = a.shape[1]
    eye = torch.eye(count, device=a.device, dtype=a.dtype)[None]
    # Sum occupied bond channels: 1-NONE catastrophically cancels when NONE
    # rounds to one despite tiny positive bond channels in float32.
    exists = b[..., 1:].sum(-1) * (1 - eye)
    by_name = {name: b[..., k] for k, name in enumerate(spec.bond_names)}
    zero = torch.zeros_like(exists)
    single = by_name.get("SINGLE", zero)
    double = by_name.get("DOUBLE", zero)
    triple = by_name.get("TRIPLE", zero)
    aromatic = by_name.get("AROMATIC", zero)
    order = single + 2 * double + 3 * triple + 1.5 * aromatic
    degree, valence = exists.sum(-1), order.sum(-1)
    charge = q @ q.new_tensor(spec.formal_charges)
    index = {z: k for k, z in enumerate(spec.atomic_numbers)}
    atom = {z: a[..., index[z]] if z in index else a.new_zeros(a.shape[:2]) for z in ELEMENTS}
    element = torch.stack([atom[z] for z in ELEMENTS], -1)
    element = torch.cat((element, (a[..., 1:].sum(-1) - element.sum(-1))[..., None]), -1)

    # Default valences, with a continuous extension of common hypervalent sulfur.
    default = {6: 4., 7: 3., 8: 2., 9: 1., 15: 3., 16: 2., 17: 1., 35: 1., 53: 1.}
    target = sum(atom[z] * default[z] for z in ELEMENTS)
    target = target + (atom[7] + atom[8]) * charge
    target = target + atom[16] * (2 * (valence - 2).clamp(0, 1) + 2 * (valence - 4).clamp(0, 1))
    hydrogens = (target - valence).clamp(0, 5)
    has_h = _nonzero_probability(hydrogens)
    atom_aromatic = 1 - (1 - aromatic).prod(-1)
    has_multiple = 1 - (1 - (double + aromatic).clamp(0, 1)).prod(-1)
    sp = (1 - (1 - triple).prod(-1) + (double.sum(-1) - 1).clamp(0, 1)).clamp(0, 1)
    # A carbonyl-adjacent lone pair extends local hybridization/conjugation.
    carbonyl_c = atom[6] * (1 - (1 - double * atom[8][:, None, :]).prod(-1))
    adjacent_carbonyl = (single * carbonyl_c[:, None, :]).sum(-1).clamp(0, 1)
    nonpositive = 1 - charge.clamp(0, 1)
    lone_pair_sp2 = (atom[7] + atom[8]) * adjacent_carbonyl * nonpositive
    sp2 = (1 - sp) * (1 - (1 - has_multiple) * (1 - lone_pair_sp2))
    sp3 = (1 - sp - sp2).clamp_min(0)
    hypervalent = (atom[16] * (valence - 2).clamp(0, 1)
                  + atom[15] * (valence - 3).clamp(0, 1)).clamp(0, 1)
    hybrid = torch.stack((sp, sp2, sp3), -1) * (1 - hypervalent[..., None])

    # Approximate the actual RDKit BaseFeatures donor/acceptor SMARTS used by
    # DeepChem, rather than substituting Lipinski's different HBA/HBD rules.
    aromatic_neighbor = (single * atom_aromatic[:, None, :]).sum(-1).clamp(0, 1)
    nitrile = (triple * atom[6][:, None, :]).sum(-1).clamp(0, 1)
    n_acceptor = atom[7] * (1 - has_h) * (1 - (1 - atom_aromatic) * (1 - nitrile) * (1 - aromatic_neighbor))
    halogen = atom[9] + atom[17] + atom[35] + atom[53]
    halogen_neighbors = (exists * halogen[:, None, :]).sum(-1)
    fluorine_context = (exists * atom[6][:, None, :] * (1 - (halogen_neighbors[:, None, :] - 1).clamp(0, 1))).sum(-1).clamp(0, 1)
    acceptor = (n_acceptor + atom[8] + atom[9] * fluorine_context) * nonpositive
    carbon_neighbors = (1 - exists * (1 - atom[6][:, None, :] * (1 - atom_aromatic[:, None, :]))).prod(-1)
    tertiary_donor = (1 - (degree - 3).abs()).clamp(0, 1) * carbon_neighbors
    donor = atom[7] * (has_h + (1 - has_h) * tertiary_donor) + (atom[8] + atom[16]) * has_h
    node = torch.cat((element, charge[..., None], hybrid, donor[..., None],
                      acceptor[..., None], atom_aromatic[..., None],
                      _soft_categories(degree + hydrogens, 5),
                      _soft_categories(hydrogens, 4)), -1)

    laplacian = torch.diag_embed(degree) - exists
    regularized = laplacian + torch.ones_like(laplacian) / count + 1e-5 * eye
    inv = torch.linalg.inv(regularized)
    diagonal = inv.diagonal(dim1=-2, dim2=-1)
    resistance = diagonal[..., :, None] + diagonal[..., None, :] - 2 * inv
    in_ring = (count * (1 - exists * resistance)).clamp(0, 1) * (1 - eye)
    unsaturated = (sp2 + sp).clamp(0, 1)
    unsaturated_neighbors = (exists * unsaturated[:, None, :]).sum(-1)
    conjugated_neighbors = (torch.maximum(unsaturated_neighbors[:, :, None],
                                          unsaturated_neighbors[:, None, :]) - 1).clamp(0, 1)
    conjugated = unsaturated[:, :, None] * unsaturated[:, None, :] * conjugated_neighbors
    stereo = torch.zeros((*exists.shape, 5), device=a.device, dtype=a.dtype)
    stereo[..., 0] = 1  # Saved graph schema has no stereochemistry channel.
    # Bond features are conditional on existence; diagonal self-loop features are zero.
    conditional = torch.stack((single, double, triple, aromatic), -1) / exists[..., None].clamp_min(1e-8)
    edge = torch.cat((conditional, in_ring[..., None], conjugated[..., None], stereo), -1)
    edge = edge * (1 - eye[..., None])
    return node, edge, exists + eye


def reference_features(mol):
    """Evaluation-only DeepChem-style hard features, using RDKit outside AD.

    This reproduces the documented MolGraphConvFeaturizer feature ordering. It
    does not call DeepChem itself, so version-specific native parity is not claimed.
    Returns float tensors and a binary edge mask, including zero-feature self loops.
    """
    from rdkit import Chem, RDConfig
    from rdkit.Chem import ChemicalFeatures

    factory = ChemicalFeatures.BuildFeatureFactory(str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef"))
    families = {"Donor": set(), "Acceptor": set()}
    for feature in factory.GetFeaturesForMol(mol):
        if feature.GetFamily() in families:
            families[feature.GetFamily()].update(feature.GetAtomIds())

    def one(value, support, unknown=False):
        out = [float(value == item) for item in support]
        return out + ([float(value not in support)] if unknown else [])

    nodes = []
    for atom in mol.GetAtoms():
        nodes.append(one(atom.GetAtomicNum(), ELEMENTS, True)
                     + [float(atom.GetFormalCharge())]
                     + one(str(atom.GetHybridization()), ("SP", "SP2", "SP3"))
                     + [float(atom.GetIdx() in families["Donor"]), float(atom.GetIdx() in families["Acceptor"])]
                     + [float(atom.GetIsAromatic())]
                     + one(atom.GetTotalDegree(), tuple(range(6)), True)
                     + one(atom.GetTotalNumHs(), tuple(range(5)), True))
    n = len(nodes)
    edge, adjacency = torch.zeros(n, n, 11), torch.eye(n)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        features = (one(str(bond.GetBondType()), ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"))
                    + [float(bond.IsInRing()), float(bond.GetIsConjugated())]
                    + one(str(bond.GetStereo()), ("STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"), True))
        edge[i, j] = edge[j, i] = torch.tensor(features)
        adjacency[i, j] = adjacency[j, i] = 1
    return torch.tensor(nodes)[None], edge[None], adjacency[None]


class FrozenSynFrag(nn.Module):
    """Pretrained scalar difficulty with a new differentiable graph extension."""

    feature_policy = FEATURE_POLICY

    def __init__(self, spec, weights_path):
        super().__init__()
        self.spec = spec
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        required = {"gnn.init_context.project_node.0.weight": (300, 30),
                    "gnn.init_context.project_edge1.0.weight": (300, 41),
                    "predict.1.weight": (1, 300)}
        if any(key not in state or tuple(state[key].shape) != shape for key, shape in required.items()):
            raise ValueError("Incompatible SynFrag checkpoint architecture")
        self._keys = {}
        for key, value in state.items():
            if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
                raise ValueError("Invalid or nonfinite SynFrag checkpoint tensor")
            name = key.replace(".", "__")
            self.register_buffer(name, value.clone())
            self._keys[key] = name
        self.eval()

    def _weight(self, key):
        return getattr(self, self._keys[key])

    def _linear(self, x, prefix):
        return F.linear(x, self._weight(prefix + ".weight"), self._weight(prefix + ".bias"))

    def _gru(self, x, hidden, prefix):
        i = F.linear(x, self._weight(prefix + ".weight_ih"), self._weight(prefix + ".bias_ih"))
        h = F.linear(hidden, self._weight(prefix + ".weight_hh"), self._weight(prefix + ".bias_hh"))
        ir, iz, inn = i.chunk(3, -1)
        hr, hz, hn = h.chunk(3, -1)
        reset, update = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
        new = torch.tanh(inn + reset * hn)
        return new + (hidden - new) * update

    @staticmethod
    def _attention(logits, weights):
        # Only supported edges set the numerical scale: a large logit on an
        # absent edge must not underflow every present message. Multiplication
        # preserves an edge-creation derivative at zero. Its boundary exponent
        # is bounded (20 in float32/64, lower in float16); positive-edge logits
        # never exceed the supported maximum, so their forward law is unchanged.
        supported = weights > 0
        scale = torch.where(supported, logits, -torch.inf).amax(-1, keepdim=True)
        cap = min(20., math.log(torch.finfo(logits.dtype).max) / 4)
        values = (logits - scale).clamp_max(cap).exp() * weights
        return values / values.sum(-1, keepdim=True).clamp_min(torch.finfo(values.dtype).tiny)

    def forward_features(self, node, edge, weights):
        """Dense original layer algebra, useful for hard-feature reference scoring."""
        if node.ndim != 3 or edge.shape != (*node.shape[:2], node.shape[1], 11) or node.shape[-1] != 30:
            raise ValueError("SynFrag expects node[B,N,30], edge[B,N,N,11]")
        if weights.shape != edge.shape[:3] or bool((weights.sum(-1) <= 0).any()):
            raise ValueError("SynFrag requires positive incoming weight per active node")
        if any(not bool(torch.isfinite(t).all()) for t in (node, edge, weights)):
            raise ValueError("Nonfinite SynFrag feature input")
        count = node.shape[1]
        src = node[:, None].expand(-1, count, -1, -1)
        prefix = "gnn.init_context."
        hidden = F.leaky_relu(self._linear(node, prefix + "project_node.0"))
        edge_hidden = F.leaky_relu(self._linear(torch.cat((src, edge), -1), prefix + "project_edge1.0"))
        dst = hidden[:, :, None].expand(-1, -1, count, -1)
        logits = F.leaky_relu(self._linear(torch.cat((dst, edge_hidden), -1), prefix + "project_edge2.1")).squeeze(-1)
        attention = self._attention(logits, weights)
        messages = self._linear(edge_hidden, prefix + "attentive_gru.edge_transform.1")
        context = F.elu((attention[..., None] * messages).sum(2))
        hidden = F.relu(self._gru(context, hidden, prefix + "attentive_gru.gru"))
        for layer in range(3):
            prefix = f"gnn.gnn_layers.{layer}."
            src = hidden[:, None].expand(-1, count, -1, -1)
            dst = hidden[:, :, None].expand(-1, -1, count, -1)
            logits = F.leaky_relu(self._linear(torch.cat((dst, src), -1), prefix + "project_edge.1")).squeeze(-1)
            attention = self._attention(logits, weights)
            projected = self._linear(hidden, prefix + "attentive_gru.project_node.1")
            context = F.elu((attention[..., None] * projected[:, None]).sum(2))
            hidden = F.relu(self._gru(context, hidden, prefix + "attentive_gru.gru"))
        graph = hidden.sum(1)
        prefix = "readout.readouts.0."
        logits = F.leaky_relu(self._linear(torch.cat((F.relu(graph)[:, None].expand_as(hidden), hidden), -1),
                                            prefix + "compute_logits.0")).squeeze(-1)
        attention = logits.softmax(-1)
        context = F.elu((attention[..., None] * self._linear(hidden, prefix + "project_nodes.1")).sum(1))
        graph = self._gru(context, graph, prefix + "gru")
        difficulty = self._linear(graph, "predict.1").squeeze(-1)
        if not bool(torch.isfinite(difficulty).all()):
            raise ValueError("Nonfinite frozen SynFrag score")
        return {"difficulty": difficulty, "raw_score": 1 - difficulty}

    def forward(self, graph, active):
        return self.forward_features(*soft_features(graph, active, self.spec))

    @torch.no_grad()
    def reference_score(self, mol):
        """Detached hard evaluation with reference features; never a live loss."""
        dtype, device = self._weight("predict.1.weight").dtype, self._weight("predict.1.weight").device
        features = [t.to(device=device, dtype=dtype) for t in reference_features(mol)]
        return self.forward_features(*features)
