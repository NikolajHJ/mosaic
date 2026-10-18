"""Detached reference extraction; see source_manifest.json. Do not edit manually."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json


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
