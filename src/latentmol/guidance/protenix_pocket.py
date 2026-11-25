"""Explicit one-chain pocket metadata and token mapping for new targets."""
from __future__ import annotations

import math
import re

VERSION = "protenix-explicit-pocket-v1"
THREE_TO_ONE = dict(zip(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
    "ARNDCQEGHILKMFPSTWYV",
))


def validate_pocket(pocket):
    if pocket.get("version") != VERSION or pocket.get("status") != "defined":
        raise ValueError("Unknown explicit Protenix pocket schema")
    sequence = pocket["protein_sequence"]
    numbering = pocket["protein_author_residues"]
    if (not sequence or any(c not in THREE_TO_ONE.values() for c in sequence)
            or len(numbering) != len(sequence) or pocket.get("protein_chain") != "A"
            or not re.fullmatch(r"[0-9A-Za-z]{4}", pocket.get("pdb_id", ""))
            or not re.fullmatch(r"[0-9a-f]{64}", pocket.get("pdb_sha256", ""))):
        raise ValueError("Invalid protein identity/author numbering")
    ligand = pocket["reference_ligand_locator"]
    if (ligand["chain"] != "A" or type(ligand["residue"]) is not int
            or not ligand["component"] or pocket["reference_ligand_heavy_atoms"] < 1):
        raise ValueError("Invalid reference ligand locator")
    positions = []
    for row in pocket["residues"]:
        pos = row["sequence_position"]
        if (type(pos) is not int or not 1 <= pos <= len(sequence)
                or THREE_TO_ONE.get(row["residue_name"]) != sequence[pos - 1]
                or row["author_residue"] != numbering[pos - 1]
                or not math.isfinite(row["minimum_reference_distance_angstrom"])
                or row["minimum_reference_distance_angstrom"] < 0):
            raise ValueError("Pocket author/sequence/residue mapping differs")
        positions.append(pos)
    if not positions or positions != sorted(set(positions)):
        raise ValueError("Empty, unordered or duplicated pocket residue positions")
    return pocket


def map_explicit_pocket(context, pocket):
    import numpy as np

    validate_pocket(pocket)
    sequence = pocket["protein_sequence"]
    if context.identities.get("protein_sequence") != sequence:
        raise ValueError("Protenix context and target pocket sequences differ")
    protein = np.asarray(context.protein_token_indices)
    features = context.base_features
    residues = np.asarray(features["residue_index"])[protein]
    chains = np.asarray(features["asym_id"])[protein]
    if (len(protein) != len(sequence) or len(set(chains.tolist())) != 1
            or not np.array_equal(np.sort(residues), np.arange(1, len(sequence) + 1))):
        raise ValueError("Native Protenix protein token/residue mapping differs")
    mapping = dict(zip(residues.tolist(), protein.tolist()))
    return tuple(int(mapping[row["sequence_position"]]) for row in pocket["residues"])
