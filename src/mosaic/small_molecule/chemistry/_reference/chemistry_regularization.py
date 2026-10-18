"""Detached reference extraction; see source_manifest.json. Do not edit manually."""
from __future__ import annotations


RULE_VERSION = "rdkit_book_2024_09_1_explicit_charge_v1"


RULE_SOURCES = (
    "https://www.rdkit.org/docs/RDKit_Book.html#valence-calculation-and-allowed-valences",
    "https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/Atom.cpp",
)


ALLOWED_VALENCES: dict[int, dict[int, tuple[int, ...] | None]] = {
    6: {-2: (2,), -1: (3,), 0: (4,), 1: (3,), 2: (2,)},
    7: {-2: (1,), -1: (2,), 0: (3,), 1: (4,), 2: (3,)},
    8: {-2: (0,), -1: (1,), 0: (2,), 1: (3,), 2: (4,)},
    9: {-2: None, -1: (0,), 0: (1,), 1: (2,), 2: (3,)},
    15: {-2: (1, 3, 5), -1: (2, 4, 6), 0: (3, 5), 1: (4,), 2: (3,)},
    16: {-2: (0, 2, 4), -1: (1, 3, 5), 0: (2, 4, 6), 1: (3, 5), 2: (4,)},
    17: {-2: None, -1: (0,), 0: (1,), 1: (2, 4, 6), 2: (3, 5)},
    35: {-2: None, -1: (0,), 0: (1,), 1: (2, 4, 6), 2: (3, 5)},
    53: {-2: None, -1: (0, 2, 4, 6), 0: (1, 3, 5), 1: (2, 4, 6), 2: (3, 5)},
}
