"""Frozen single-ligand TacoGFN QVina head on precomputed PharmacoNet pocket features.

Uses the layer names and single-graph operations of PharmacoNet's released
`pmnet_appl/tacogfn_reward/proxy.py` (Seo et al.; Shen et al., TacoGFN).
Input edge directions and order come from its Open Babel featurizer. Open
Babel is used when preparing a new hard graph. The differentiable path uses
PyTorch alone.
"""

# PharmacoNet: Copyright (c) 2023 Seonghwan Seo.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import torch
from torch import Tensor, nn


def topology_from_smiles(canonical_smiles: str) -> dict:
    """The released featurization, preserving Open Babel's original edge direction."""
    from openbabel import pybel
    from openbabel.pybel import ob

    molecule = pybel.readstring("smi", canonical_smiles)
    atom_ids = {6: 0, 7: 1, 8: 2, 9: 3, 15: 4, 16: 5, 17: 6, 35: 7, 53: 8}
    atoms = [atom_ids.get(atom.atomicnum, 9) for atom in molecule.atoms]
    if not atoms:
        raise ValueError("Open Babel read an empty TacoGFN ligand")
    sources, destinations, bonds = [], [], []
    for bond in ob.OBMolBondIter(molecule.OBMol):
        sources.append(bond.GetBeginAtomIdx() - 1)
        destinations.append(bond.GetEndAtomIdx() - 1)
        bonds.append(3 if bond.IsAromatic() else {1: 0, 2: 1, 3: 2}.get(bond.GetBondOrder(), 4))
    return {"x": atoms, "edge_index": [sources, destinations], "edge_attr": bonds}


class _GINELayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(128, 128)
        self.nn = nn.Sequential(nn.Linear(128, 128), nn.LeakyReLU())
        self.register_buffer("eps", torch.zeros(1))

    def forward(self, nodes, edges, source, dest):
        messages = (nodes[source] + self.lin(edges)).relu()
        summed = torch.zeros_like(nodes).index_add(0, dest, messages)
        return self.nn(summed + (1 + self.eps) * nodes)


class _LigandEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.node_layer = nn.Embedding(10, 128)
        self.edge_layer = nn.Embedding(5, 128)
        self.conv_list = nn.ModuleList(_GINELayer() for _ in range(4))
        self.head = nn.Sequential(nn.Linear(128, 256), nn.LayerNorm(256))
        self.readout_layer = nn.Linear(256, 256)
        self.readout_gate = nn.Linear(256, 256)

    def forward(self, atom_probs, bond_probs, edge_index):
        base = atom_probs @ self.node_layer.weight
        edges = bond_probs @ self.edge_layer.weight
        source, dest = edge_index.unbind(0)
        nodes = base
        for layer in self.conv_list:
            nodes = layer(nodes, edges, source, dest)
        nodes = nodes + base
        pooled = torch.cat((nodes.sum(0), nodes.mean(0))).unsqueeze(0)
        graph = self.readout_gate(pooled) * self.readout_layer(pooled)
        return self.head(nodes), graph


class TacoGFNHead(nn.Module):
    """One frozen pocket, accepting soft ligand atom and bond embeddings."""

    def __init__(self, pocket: Tensor, hotspots: Tensor):
        super().__init__()
        if (pocket.shape != (256,) or hotspots.ndim != 2 or hotspots.shape[1] != 256
                or pocket.dtype != torch.float32 or hotspots.dtype != torch.float32
                or not torch.isfinite(pocket).all() or not torch.isfinite(hotspots).all()):
            raise ValueError("Invalid TacoGFN pocket or hotspot features")
        self.register_buffer("pocket", pocket.detach(), persistent=False)
        self.register_buffer("hotspots", hotspots.detach(), persistent=False)
        self.ligand_encoder = _LigandEncoder()
        self.energy_bias_mlp = nn.Sequential(nn.Linear(512, 256), nn.LeakyReLU(), nn.Linear(256, 1))
        self.interaction_mlp = nn.Sequential(nn.Linear(256, 256), nn.LeakyReLU())
        self.pair_energy_layer = nn.Linear(256, 1)
        self.pair_energy_gate = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())

    @classmethod
    def from_checkpoint(cls, checkpoint, pocket_file, device="cpu"):
        pocket, hotspots = torch.load(pocket_file, map_location="cpu", weights_only=True)
        # Allocating linear layers consumes RNG. Preserve the caller's state so
        # loading the frozen guidance head cannot shift a paired trajectory's RNG.
        with torch.random.fork_rng(devices=[]):
            head = cls(pocket, hotspots)
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
        # The excluded layers produced the already-cached pocket/hotspot inputs.
        pocket_layers = ("token_mlp.", "pocket_mlp_list.", "pocket_mlp.", "concat_layer.", "concat_gate.")
        head.load_state_dict({k: v for k, v in weights.items() if not k.startswith(pocket_layers)}, strict=True)
        if any(not value.isfinite().all() for value in head.state_dict().values()):
            raise ValueError("Nonfinite TacoGFN scoring weights")
        return head.eval().requires_grad_(False).to(device)

    def forward(self, atom_probs: Tensor, bond_probs: Tensor, edge_index: Tensor) -> Tensor:
        if (atom_probs.ndim != 2 or atom_probs.shape[0] == 0 or atom_probs.shape[1] != 10
                or bond_probs.ndim != 2 or bond_probs.shape[1] != 5
                or edge_index.shape != (2, bond_probs.shape[0]) or edge_index.dtype != torch.long):
            raise ValueError("TacoGFN expects [N,10] atoms, [E,5] bonds and [2,E] directed edges")
        atoms, graph = self.ligand_encoder(atom_probs, bond_probs, edge_index)
        site_pairs = torch.einsum("ik,jk->ijk", atoms, self.hotspots)
        interactions = self.interaction_mlp(site_pairs)
        pair_energy = self.pair_energy_layer(interactions) * self.pair_energy_gate(interactions)
        bias = self.energy_bias_mlp(torch.cat((self.pocket.unsqueeze(0), graph), dim=1))
        return pair_energy.sum() + bias.squeeze()
