"""Decode a frozen TorchScript VAE into Mosaic's fixed-count, neutral graph.

An interchangeable archive carries its graph vocabulary in vae.json. Its
encode method takes canonical padded integer labels for atoms, charges and
bonds, and returns one posterior mean. Its decode method takes a latent and
returns atom logits and symmetric conditional bond logits, before occupancy
gating. Both methods use a batch of one. Only the latent requires gradients.
"""

import hashlib
import json
import math
from pathlib import Path

import torch

from mosaic.small_molecule.core.state import BOND_STATES, ELEMENTS


_BONDS = tuple("NONE" if name == "NO_BOND" else name for name in BOND_STATES)
_NUMBERS = {"C": 6, "N": 7, "O": 8, "S": 16, "F": 9, "Cl": 17}
_ELEMENTS = tuple(_NUMBERS[element] for element in ELEMENTS)


class GraphVAE:
    def __init__(self, module, metadata, digest):
        self.module = module.eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)
        if metadata.get("format") != "mosaic-graph-vae-v1":
            raise ValueError("Expected a Mosaic graph-VAE runtime with mosaic-graph-vae-v1 metadata")
        self.capacity, self.latent_dim = metadata["max_nodes"], metadata["latent_dim"]
        if any(type(n) is not int or n < 1 for n in (self.capacity, self.latent_dim)):
            raise ValueError("Invalid VAE capacity or latent dimension")
        self.atoms, self.charges, self.bonds = (
            tuple(metadata[key]) for key in ("atomic_numbers", "formal_charges", "bond_names")
        )
        if (any(len(set(v)) != len(v) for v in (self.atoms, self.charges, self.bonds))
                or self.atoms[:1] != (0,) or not set(_ELEMENTS).issubset(self.atoms)
                or 0 not in self.charges or not set(_BONDS).issubset(self.bonds)):
            raise ValueError("VAE vocabulary must cover Mosaic elements, neutral charge and five bonds")
        if not callable(getattr(module, "decode", None)) or not callable(getattr(module, "encode", None)):
            raise ValueError("VAE runtime must expose encode and decode")
        self.element_indices = [self.atoms.index(x) for x in _ELEMENTS]
        self.bond_indices = [self.bonds.index(x) for x in _BONDS]
        self.identity = {"format": metadata["format"], "sha256": digest,
                         "latent_dim": self.latent_dim, "max_nodes": self.capacity}

    def _logits(self, z):
        if z.shape != (1, self.latent_dim) or z.dtype != torch.float32 or not torch.isfinite(z).all():
            raise ValueError("VAE latent must be one finite float32 vector of the configured dimension")
        atoms, bonds = self.module.decode(z)
        n = self.capacity
        if (atoms.shape != (1, n, len(self.atoms)) or bonds.shape != (1, n, n, len(self.bonds))
                or not torch.isfinite(atoms).all() or not torch.isfinite(bonds).all()
                or not torch.allclose(bonds, bonds.transpose(1, 2), atol=1e-5, rtol=1e-5)):
            raise ValueError("VAE returned incompatible, nonfinite, or asymmetric graph logits")
        return atoms, bonds

    @torch.no_grad()
    def initialize(self, n_atoms, seed, smiles=None):
        """Create the latent and fix the native slot mapping for the whole run."""
        if type(n_atoms) is not int or not 1 <= n_atoms <= self.capacity:
            raise ValueError(f"VAE n_atoms must be within 1..{self.capacity}")
        if smiles is not None:
            z, slots = self._encode_smiles(smiles, n_atoms)
            self._logits(z)
            return z, slots
        z = torch.randn(1, self.latent_dim, generator=torch.Generator().manual_seed(seed))
        atoms, _ = self._logits(z)
        # Log odds retain the ranking even when the probabilities round to 1.
        occupancy = atoms[0, :, 1:].logsumexp(-1) - atoms[0, :, 0]
        slots = occupancy.argsort(descending=True, stable=True)[:n_atoms].sort().values
        return z, slots

    def _encode_smiles(self, smiles, n_atoms):
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() != n_atoms or len(Chem.GetMolFrags(mol)) != 1:
            raise ValueError("VAE initialization needs one connected SMILES matching n_atoms")
        for atom in mol.GetAtoms():
            if (atom.GetAtomicNum() not in _ELEMENTS or atom.GetFormalCharge() != 0
                    or atom.GetIsotope() or atom.GetNumRadicalElectrons()):
                raise ValueError("VAE initialization supports neutral, non-isotopic Mosaic elements without radicals")
        Chem.RemoveStereochemistry(mol)
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        # canonical_position -> input_atom_index, recorded by the SMILES writer.
        order = json.loads(mol.GetProp("_smilesAtomOutputOrder"))
        ordered = Chem.MolFromSmiles(canonical)
        if sorted(order) != list(range(n_atoms)) or ordered is None:
            raise ValueError("Could not map canonical VAE slots to input atom order")
        atoms = torch.zeros((1, self.capacity), dtype=torch.long)
        charges = torch.full_like(atoms, self.charges.index(0))
        bonds = torch.full((1, self.capacity, self.capacity), self.bonds.index("NONE"), dtype=torch.long)
        # Rebuild from just the representable labels to catch explicit-H cases
        # where dropping that extra information would encode a different molecule.
        rebuilt = Chem.RWMol()
        for atom in ordered.GetAtoms():
            atoms[0, atom.GetIdx()] = self.atoms.index(atom.GetAtomicNum())
            rebuilt.AddAtom(Chem.Atom(atom.GetAtomicNum()))
        for bond in ordered.GetBonds():
            i, j, label = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), str(bond.GetBondType())
            if label not in _BONDS:
                raise ValueError(f"Unsupported Mosaic bond {label}")
            bonds[0, i, j] = bonds[0, j, i] = self.bonds.index(label)
            rebuilt.AddBond(i, j, bond.GetBondType())
        Chem.SanitizeMol(rebuilt)
        if Chem.MolToSmiles(rebuilt, canonical=True, isomericSmiles=False) != canonical:
            raise ValueError("VAE input cannot round-trip through the neutral 2D graph schema")
        return self.module.encode(atoms, charges, bonds), torch.tensor(order).argsort()

    def decode(self, z, slots, temperature=1.0):
        """Condition on occupied native slots and supported elements and bonds."""
        if (slots.ndim != 1 or slots.dtype != torch.int64 or not slots.numel()
                or slots.min() < 0 or slots.max() >= self.capacity or len(slots.unique()) != len(slots)):
            raise ValueError("VAE slot map must contain distinct, in-range integer slots")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("VAE temperature must be finite and positive")
        atoms, bonds = self._logits(z)
        i, j = torch.triu_indices(len(slots), len(slots), 1)
        x = atoms[0, slots][:, self.element_indices]
        e = bonds[0, slots[i], slots[j]][:, self.bond_indices]
        return (x / temperature).softmax(-1), (e / temperature).softmax(-1)


def load_provider(checkpoint=None):
    """Load the bundled CPU model, or a compatible trusted archive from YAML."""
    if checkpoint is None:
        package = Path(__file__).resolve()
        checkpoint = package.with_name("epoch45-runtime.pt")
        if not checkpoint.is_file():
            checkpoint = package.parents[4] / "models" / "dbmol-vae" / "epoch45-runtime.pt"
    checkpoint = Path(checkpoint)
    extra = {"vae.json": ""}
    with checkpoint.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        module = torch.jit.load(stream, map_location="cpu", _extra_files=extra)
    if not extra["vae.json"]:
        raise ValueError("VAE checkpoint must be a Mosaic TorchScript runtime containing vae.json")
    return GraphVAE(module, json.loads(extra["vae.json"]), digest)
