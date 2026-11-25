"""Frozen plain ZINC PIG-VAE from the Graph LDA release, projected onto a fixed graph.

Upstream commit 591abff0efddb5dc75dfdb63db37a9c2ac39be69. The released
training and preprocessing notebooks fix 26 slots, 43 node features and 13
edge features. RDKit is used only to prepare an encoded start. Decoder calls
are ordinary differentiable Torch operations on the supplied latent.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import math
from pathlib import Path
import sys

from rdkit import Chem
import torch
from torch.nn import functional as F

from latentmol.chemistry import encode_smiles
from latentmol.representation import GraphSpec, RelaxedGraph


MAX_ATOMS = 26
CHECKPOINT_SHA256 = "01043a07aff4789dd3cc16100f6c15587962d78b978dc53824db33b7afae1a98"
# Includes __init__ because pickle resolves classes by the upstream module name.
_SOURCE_DIGESTS = {
    "__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "models.py": "b78182a8891a80e9522c727cb022dd5fd6be06c919ed3281b4baf7b13d84f079",
    "graph_transformer.py": "4bf5e0aef0162964bacf3497fbc53befed92624095b65c71a2ddbc98f45fe14b",
}
_ELEMENTS = ("Si", "C", "N", "S", "P", "O", "F", "Cl", "Br", "I", "B", "Unknown")
_ELEMENT_CHANNELS = (1, 2, 5, 6, 4, 3, 7, 8, 9)  # shared schema C,N,O,F,P,S,Cl,Br,I
_HYBRID = ("SP", "SP2", "SP3", "SP3D", "SP3D2")


def _one_hot(value, choices):
    # Upstream one_of_k_encoding_unk maps an unknown label to the final entry.
    value = value if value in choices else choices[-1]
    return [value == item for item in choices]


def _input_graph(smiles):
    """Match the training notebook's RDKit and shortest-path features, including padding."""
    record = encode_smiles(smiles, GraphSpec(max_nodes=MAX_ATOMS))
    mol = Chem.MolFromSmiles(record["smiles"])
    count = mol.GetNumAtoms()
    node_rows = []
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0:
            raise ValueError("The fixed neutral-atom pilot excludes each individually charged atom")
        degree = atom.GetDegree()
        if degree > 10:
            raise ValueError("Atom degree outside pretrained vocabulary")
        node_rows.append(
            _one_hot(atom.GetSymbol(), _ELEMENTS)
            + _one_hot(degree, range(11))
            + _one_hot(atom.GetImplicitValence(), range(7))
            + [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
            + _one_hot(atom.GetHybridization().name, _HYBRID)
            + [atom.GetIsAromatic()]
            + _one_hot(atom.GetTotalNumHs(), range(5))
        )
    nodes = torch.zeros(1, MAX_ATOMS, 43)
    nodes[0, :count] = torch.tensor(node_rows)
    mask = torch.arange(MAX_ATOMS).unsqueeze(0) < count
    distance = torch.zeros(MAX_ATOMS, MAX_ATOMS, dtype=torch.long)
    # Isolated padding distances were inf→long→clamp(0,5) = 0 in the notebook.
    distance[:count, :count] = torch.as_tensor(Chem.GetDistanceMatrix(mol)).long().clamp_max(5)
    edges = torch.zeros(1, MAX_ATOMS, MAX_ATOMS, 13)
    edges[0, :, :, :6] = F.one_hot(distance, 6)
    edges[..., 6] = 1  # explicit no bond; distance features alone are insufficient
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        label = int(record["bond_types"][i, j])
        for a, b in ((i, j), (j, i)):
            edges[0, a, b, 6] = 0
            edges[0, a, b, 6 + label] = 1
            edges[0, a, b, 11] = bond.GetIsConjugated()
            edges[0, a, b, 12] = bond.IsInRing()
    return record["smiles"], nodes, edges, mask


@dataclass(frozen=True)
class FixedZincStart:
    smiles: str
    count: int
    mask: torch.Tensor
    permutation: torch.Tensor


class GraphLDAZincVAE(torch.nn.Module):
    """A fixed-count, neutral-only adapter for one identified published checkpoint."""

    def __init__(self, network):
        super().__init__()
        self.network = network.eval().requires_grad_(False)

    @classmethod
    def load(cls, source_root, checkpoint_path, device="cpu"):
        """Load the authors' full-model archive using pinned code and explicit allowed classes."""
        package = (Path(source_root) / "pigvae_all" / "pigvae").resolve()
        for filename, digest in _SOURCE_DIGESTS.items():
            content = (package / filename).read_bytes().replace(b"\r\n", b"\n")
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError(f"Unexpected upstream PIG-VAE source: {filename}")
        with Path(checkpoint_path).open("rb") as fh:
            digest = hashlib.sha256()
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
            if digest.hexdigest() != CHECKPOINT_SHA256:
                raise ValueError("Checkpoint SHA256 differs from the published plain ZINC VAE")
            sys.path.insert(0, str(package.parent))
            try:
                modules = [importlib.import_module(f"pigvae.{name}") for name in ("models", "graph_transformer")]
                if any(Path(m.__file__).resolve().parent != package for m in modules):
                    raise ValueError("Conflicting PIG-VAE source already imported")
                trusted = [c for m in modules for c in vars(m).values()
                           if isinstance(c, type) and c.__module__ == m.__name__]
                trusted += [torch.nn.Linear, torch.nn.LayerNorm, torch.nn.ModuleList, torch.nn.Dropout, set]
                fh.seek(0)
                with torch.serialization.safe_globals(trusted):
                    network = torch.load(fh, map_location="cpu", weights_only=True)
            finally:
                sys.path.pop(0)
        return cls(network).to(device).eval()

    @staticmethod
    def spec_for(state):
        return GraphSpec(max_nodes=state.count)

    @torch.no_grad()
    def encode_start(self, smiles):
        """Use posterior mean and fixed hard ranks, breaking identical scores by input order.

        The paper's train-time soft permutation depends on the input molecule and
        can carry information besides the latent. Here only a discrete reordering
        of slots accompanies z. Both experiment arms retain that same order.
        """
        canonical, nodes, edges, mask = _input_graph(smiles)
        device = next(self.network.parameters()).device
        nodes, edges, mask = nodes.to(device), edges.to(device), mask.to(device)
        hidden, per_atom = self.network.encoder(nodes, edges, mask)
        bottle = self.network.bottle_neck_encoder
        mean = bottle.w(F.relu(hidden))[:, :bottle.d_out]
        scores = self.network.permuter.score(per_atom, mask)
        count = int(mask.sum())
        order = scores[0, :count, 0].argsort(descending=True, stable=True)
        perm = torch.eye(MAX_ATOMS, device=device).unsqueeze(0)
        perm[0, :count, :count] = F.one_hot(order.argsort(), count).to(perm)
        if mean.shape != (1, 90) or not torch.isfinite(mean).all() or not torch.isfinite(scores).all():
            raise ValueError("Invalid pretrained encoder output")
        return mean, FixedZincStart(canonical, count, mask, perm)

    def decode(self, latent, state, temperature=1.0):
        """Emit exactly N neutral atoms and symmetric typed bond distributions.

        BCE element logits model independent Bernoulli labels. Conditional on
        exactly one supported element being on, p(i) is proportional to its
        odds, exp(logit_i). Softmax thus implements this declared conditioning.
        Bonds use the *trained* five-way categorical slice, including aromatic.
        """
        if latent.shape != (1, 90) or not torch.isfinite(latent).all():
            raise ValueError("Require one finite 90-dimensional latent")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Temperature must be finite and positive")
        count = state.count
        node, edge, _ = self.network.decode(latent, state.permutation, state.mask)
        if not torch.isfinite(node).all() or not torch.isfinite(edge).all():
            raise FloatingPointError("Nonfinite pretrained decoder output")
        elements = (node[:, :count, _ELEMENT_CHANNELS] / temperature).softmax(-1)
        atoms = F.pad(elements, (1, 0))  # occupied slots have exactly zero NO_ATOM
        spec = self.spec_for(state)
        charges = atoms.new_zeros(1, count, len(spec.formal_charges))
        charges[..., spec.formal_charges.index(0)] = 1
        bonds = (edge[:, :count, :count, 6:11] / temperature).softmax(-1)
        diag = torch.eye(count, device=latent.device, dtype=torch.bool)[None, :, :, None]
        absent = bonds.new_tensor([1., 0., 0., 0., 0.])
        return RelaxedGraph(atoms, charges, torch.where(diag, absent, bonds))
