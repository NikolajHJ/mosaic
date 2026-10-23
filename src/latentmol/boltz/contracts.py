"""Tensor-only snapshot of an officially featurized, single-ligand complex."""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from latentmol.representation import GraphSpec, HardGraphBatch

BOLTZ_COMMIT = "b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc"
HF_REVISION = "6fdef46d763fee7fbb83ca5501ccceff43b85607"
AFFINITY_SHA256 = "dcc5cd3722b1c9eaa34267e4ae32f55cbbf1963f4c19319381ccfa30fdd2ca9e"
NATIVE_BOND_IDS = {"NONE": 0, "SINGLE": 2, "DOUBLE": 3, "TRIPLE": 4, "AROMATIC": 5}


@dataclass
class PreparedComplex:
    """Immutable correspondence, not an inferred match of each decoded molecule.

    Maps have one entry per graph slot, with -1 for absent slots. Active slots
    map injectively onto all affinity ligand atom/token rows. This prototype
    permits one atom per ligand token, batch size one, and a single fixed pose.
    """

    features: dict[str, Tensor]
    coordinates: Tensor  # [1, 1, A, 3], detached bound/predicted reference pose
    slot_to_atom: Tensor  # [N], long
    slot_to_token: Tensor  # [N], long
    seed: HardGraphBatch
    spec: GraphSpec
    metadata: dict

    @property
    def active(self) -> Tensor:
        """Fixed occupancy; unoccupied slots cannot be born in this adapter."""
        return self.slot_to_atom >= 0

    def validate(self) -> None:
        """Reject silent remapping, incomplete ligand coverage and shape mistakes."""
        f, n = self.features, self.spec.max_nodes
        if self.slot_to_atom.shape != (n,) or self.slot_to_token.shape != (n,):
            raise ValueError("Slot maps must have GraphSpec.max_nodes entries")
        if self.slot_to_atom.dtype != torch.long or self.slot_to_token.dtype != torch.long:
            raise ValueError("Slot maps must be long tensors")
        if (self.slot_to_atom < -1).any() or (self.slot_to_token < -1).any():
            raise ValueError("Only -1 is allowed as an absent-slot sentinel")
        for value, shape, categories in ((self.seed.atoms, (1, n), len(self.spec.atomic_numbers)),
                                         (self.seed.charges, (1, n), len(self.spec.formal_charges)),
                                         (self.seed.bonds, (1, n, n), len(self.spec.bond_names))):
            if value.shape != shape or value.dtype != torch.long or (value < 0).any() or (value >= categories).any():
                raise ValueError("Seed labels have a malformed shape, dtype or categorical index")
        if not torch.equal(self.seed.bonds, self.seed.bonds.transpose(1, 2)):
            raise ValueError("Seed bonds must be symmetric")
        if not torch.equal(self.active, self.slot_to_token >= 0):
            raise ValueError("Atom and token slot masks differ")
        if not self.active.any() or not torch.equal(self.seed.atoms[0] != 0, self.active):
            raise ValueError("Nonempty seed occupancy must equal the immutable slot map")
        if self.seed.atoms.shape != (1, n):
            raise ValueError("Only one seed per prepared complex is supported")
        a, t = f["ref_element"].shape[1], f["token_pad_mask"].shape[1]
        if f["ref_element"].shape != (1, a, 128):
            raise ValueError("Expected native [1,A,128] element features")
        if self.coordinates.shape != (1, 1, a, 3):
            raise ValueError("Expected exactly one [1,1,A,3] reference pose")
        if self.coordinates.requires_grad or not torch.isfinite(self.coordinates).all():
            raise ValueError("Reference coordinates must be finite and detached")
        ai, ti = self.slot_to_atom[self.active], self.slot_to_token[self.active]
        if ai.max() >= a or ti.max() >= t or ai.unique().numel() != ai.numel() or ti.unique().numel() != ti.numel():
            raise ValueError("Slot maps must be bounded and injective")
        ligand = f["affinity_token_mask"][0].bool() & f["token_pad_mask"][0].bool()
        if not torch.equal(ti.sort().values, ligand.nonzero().flatten()):
            raise ValueError("Slot map must cover exactly the affinity ligand tokens")
        if not f["atom_pad_mask"][0, ai].bool().all() or (f["mol_type"][0, ti] == 0).any():
            raise ValueError("Ligand map addresses padding or protein tokens")
        pool = f["atom_to_token"][0]
        if not torch.all(pool[ai, ti] == 1) or not torch.all(pool[:, ti].sum(0) == 1):
            raise ValueError("Require one atom per mapped ligand token")
        if not torch.all(f["token_to_rep_atom"][0, ti, ai] == 1):
            raise ValueError("Representative atom and slot map disagree")
        if self.metadata.get("boltz_commit") != BOLTZ_COMMIT:
            raise ValueError("Prepared features have a different Boltz source identity")
        # The mapping is a scientific identity, not just an index table. Repeat
        # this on load, so a swapped map/seed cannot pass structural checks alone.
        from .adapter import adapt_graph
        mapped, native_types = adapt_graph(self.seed.to_relaxed(self.spec), self)
        for key in ("ref_element", "ref_charge", "token_bonds"):
            if not torch.equal(mapped[key], f[key].to(mapped[key])):
                raise ValueError(f"Seed and mapped native {key} differ")
        expected_types = torch.nn.functional.one_hot(f["type_bonds"].long(), 7).to(native_types)
        if not torch.equal(native_types, expected_types):
            raise ValueError("Seed and mapped native typed bonds differ")

    def to(self, device) -> "PreparedComplex":
        """Move tensor data together; preserve provenance and categorical identity."""
        return PreparedComplex({k: v.to(device) for k, v in self.features.items()},
                               self.coordinates.to(device), self.slot_to_atom.to(device),
                               self.slot_to_token.to(device), self.seed.to(device),
                               self.spec, self.metadata)

    def save(self, path: Path) -> None:
        """Write only tensors and plain metadata; never serialize Boltz/RDKit objects."""
        self.validate()
        if path.exists():
            raise FileExistsError(path)
        payload = {"format_version": 1, "features": self.features,
                   "coordinates": self.coordinates, "slot_to_atom": self.slot_to_atom,
                   "slot_to_token": self.slot_to_token,
                   "seed": {"atoms": self.seed.atoms, "charges": self.seed.charges, "bonds": self.seed.bonds},
                   "graph_spec": self.spec.to_dict(), "metadata": self.metadata}
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path, device="cpu") -> "PreparedComplex":
        """Load a local tensor snapshot and validate before any scorer evaluation."""
        raw = torch.load(path, map_location=device, weights_only=True)
        if raw.get("format_version") != 1:
            raise ValueError("Unsupported prepared-complex format")
        value = cls(raw["features"], raw["coordinates"], raw["slot_to_atom"],
                    raw["slot_to_token"], HardGraphBatch(**raw["seed"]),
                    GraphSpec.from_dict(raw["graph_spec"]), raw["metadata"])
        value.validate()
        return value
