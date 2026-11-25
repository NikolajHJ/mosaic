"""Versioned tensor contracts shared by datasets, decoders, and future scorers.

Atoms use heavy-atom slots with NO_ATOM at index zero. Bond index zero means no
bond. Formal charges are categorical during reconstruction; their expected
numeric value is available for a continuous Boltz-style input adapter.
"""

from dataclasses import asdict, dataclass
import hashlib
import json

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class GraphSpec:
    """The ordered vocabularies are part of checkpoint identity, not defaults to guess."""

    max_nodes: int = 50
    atomic_numbers: tuple[int, ...] = (0, 6, 7, 8, 9, 15, 16, 17, 35, 53)
    formal_charges: tuple[int, ...] = (-2, -1, 0, 1, 2)
    bond_names: tuple[str, ...] = ("NONE", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
    version: int = 1

    def __post_init__(self):
        if not self.atomic_numbers or not self.formal_charges or not self.bond_names:
            raise ValueError("Vocabularies cannot be empty")
        if self.max_nodes < 1 or self.atomic_numbers[0] != 0 or self.bond_names[0] != "NONE":
            raise ValueError("Require positive capacity and index-zero NO_ATOM/NO_BOND")
        if 0 not in self.formal_charges:
            raise ValueError("A neutral charge category is required")
        for vocabulary in (self.atomic_numbers, self.formal_charges, self.bond_names):
            if len(set(vocabulary)) != len(vocabulary):
                raise ValueError("Vocabulary entries must be unique")
        if self.version != 1:
            raise ValueError(f"Unsupported graph schema version: {self.version}")

    def to_dict(self) -> dict:
        """Return a JSON-compatible schema, including vocabulary order."""
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def from_dict(cls, value: dict) -> "GraphSpec":
        fields = dict(value)
        for key in ("atomic_numbers", "formal_charges", "bond_names"):
            if key in fields:
                fields[key] = tuple(fields[key])
        return cls(**fields)

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass
class HardGraphBatch:
    """Discrete reconstruction targets: categorical indices, with batch axis first."""

    atoms: Tensor  # [B, N]
    charges: Tensor  # [B, N]
    bonds: Tensor  # [B, N, N]

    def to(self, device) -> "HardGraphBatch":
        return HardGraphBatch(*(x.to(device) for x in (self.atoms, self.charges, self.bonds)))

    def to_relaxed(self, spec: GraphSpec) -> "RelaxedGraph":
        """Convert labels before the differentiable model boundary."""
        return RelaxedGraph(
            F.one_hot(self.atoms, len(spec.atomic_numbers)).float(),
            F.one_hot(self.charges, len(spec.formal_charges)).float(),
            F.one_hot(self.bonds, len(spec.bond_names)).float(),
        )


@dataclass
class RelaxedGraph:
    """Continuous graph with occupancy-gated marginal bond probabilities.

    Bond probabilities include endpoint occupancy. They are not conditional on
    both endpoints existing. Hard input graphs must have NONE bonds at padding.
    """

    atoms: Tensor  # [B, N, A]
    charges: Tensor  # [B, N, Q]
    bonds: Tensor  # [B, N, N, E], symmetric, no-bond diagonal

    @property
    def occupancy(self) -> Tensor:
        return 1.0 - self.atoms[..., 0]

    @property
    def expected_size(self) -> Tensor:
        return self.occupancy.sum(-1)

    def expected_charge(self, spec: GraphSpec) -> Tensor:
        values = self.charges.new_tensor(spec.formal_charges)
        return (self.charges @ values) * self.occupancy


@dataclass
class DecoderOutput:
    """Raw logits for loss computation, with a separate differentiable graph view."""

    atoms: Tensor
    charges: Tensor
    bonds: Tensor

    @property
    def symmetric_bond_logits(self) -> Tensor:
        """Single undirected interpretation shared by loss and inference."""
        return (self.bonds + self.bonds.transpose(1, 2)) * 0.5

    def probabilities(self, temperature: float = 1.0) -> RelaxedGraph:
        """Enforce symmetry and softly suppress bonds to unoccupied slots.

        Gating is applied once here. Scorer adapters must not apply the same bond
        occupancy factor a second time. Score pooling still needs its own mask.
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        atoms = (self.atoms / temperature).softmax(-1)
        charges = (self.charges / temperature).softmax(-1)
        bond_logits = self.symmetric_bond_logits
        bonds = (bond_logits / temperature).softmax(-1)
        occ = 1.0 - atoms[..., 0]
        present_pair = occ[:, :, None] * occ[:, None, :]
        diagonal = torch.eye(occ.shape[1], device=occ.device, dtype=occ.dtype)
        present_pair = present_pair * (1.0 - diagonal)
        bonds = bonds * present_pair[..., None]
        no_bond = torch.zeros_like(bonds)
        no_bond[..., 0] = 1.0
        bonds = bonds + (1.0 - present_pair[..., None]) * no_bond
        return RelaxedGraph(atoms, charges, bonds)

    def harden(self, spec: GraphSpec) -> HardGraphBatch:
        """Argmax for evaluation only; never use this in a latent gradient path.

        Choose conditional bonds before applying the hard occupancy mask. Using
        softly gated bond probabilities here would bias hardening toward no-bond.
        """
        atoms = self.atoms.argmax(-1)
        charges = self.charges.argmax(-1)
        bonds = self.symmetric_bond_logits.argmax(-1)
        active = atoms != 0
        pair = active[:, :, None] & active[:, None, :]
        diagonal = torch.eye(atoms.shape[1], device=atoms.device, dtype=torch.bool)
        bonds = bonds.masked_fill(~pair | diagonal, 0)
        charges = charges.masked_fill(~active, spec.formal_charges.index(0))
        return HardGraphBatch(atoms, charges, bonds)
