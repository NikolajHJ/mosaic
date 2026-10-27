"""Authenticated BRD4 crystallographic pocket in the receptor's original frame.

Experiment-owned extraction follows Mosaic's published heavy-neighborhood
definition, without importing a changing sibling checkout. Geometry extraction
and torch tensor construction execute only in allocated HPC stages.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence


PDB_SHA256 = "534dcb8ea9fa29956249b5133b22187c3867363b16a6b98dd8219c18c5d3250f"
VERSION = "brd4-jq1-heavy-neighborhood-v1"
THREE_TO_ONE = dict(zip(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
    "ARNDCQEGHILKMFPSTWYV"))


@dataclass(frozen=True)
class CrystalAtom:
    record: str
    chain: str
    author_residue: int
    insertion: str
    residue_name: str
    atom_name: str
    element: str
    coordinates: tuple[float, float, float]
    serial: int
    altloc: str
    occupancy: float


def _pdb_atoms(pdb_path) -> tuple[list[CrystalAtom], dict]:
    raw = Path(pdb_path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != PDB_SHA256:
        raise ValueError("Experimental 3MXF PDB bytes differ from the sealed source")
    selected, altloc_count, hydrogen_count = {}, 0, 0
    for line in raw.decode("ascii").splitlines():
        if line.startswith("ENDMDL"):
            break
        if line[:6].strip() not in {"ATOM", "HETATM"}:
            continue
        record, chain = line[:6].strip(), line[21:22]
        residue_name = line[17:20].strip()
        if chain != "A":
            continue
        author = int(line[22:26])
        if not ((record == "ATOM" and 42 <= author <= 168)
                or (record == "HETATM" and residue_name == "JQ1")):
            continue
        insertion = line[26:27].strip()
        if insertion:
            raise ValueError("Unexpected insertion code in the authenticated BRD4 construct")
        element = line[76:78].strip().capitalize()
        if not element:
            raise ValueError("Pinned PDB contains an atom with no explicit element")
        if element in {"H", "D"}:
            hydrogen_count += 1
            continue
        xyz = tuple(float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54)))
        occupancy = float(line[54:60])
        if not all(math.isfinite(value) for value in (*xyz, occupancy)):
            raise ValueError("Nonfinite crystallographic coordinate/occupancy")
        altloc = line[16:17]
        atom = CrystalAtom(record, chain, author, insertion, residue_name,
                           line[12:16].strip(), element, xyz, int(line[6:11]),
                           altloc, occupancy)
        key = (record, chain, author, insertion, residue_name, atom.atom_name)
        rank = (occupancy, altloc == " ", altloc == "A")
        if key in selected:
            altloc_count += 1
        if key not in selected or rank > selected[key][0]:
            selected[key] = (rank, atom)
    atoms = [value[1] for value in selected.values()]
    return atoms, {"pdb_sha256": actual, "model_policy": "first model only",
                   "altloc_policy": "maximum occupancy, then blank, then A",
                   "alternate_records_removed": altloc_count,
                   "hydrogen_or_deuterium_records_excluded": hydrogen_count}


def validate_definition(definition: dict) -> dict:
    if (definition.get("version") != VERSION
            or definition.get("pdb_sha256") != PDB_SHA256
            or definition.get("protein_chain") != "A"
            or definition.get("radius_angstrom") != 6.0
            or definition.get("reference_ligand") != "JQ1"
            or definition.get("reference_ligand_heavy_atoms") != 31
            or definition.get("anchor_author_residues") != [81, 140]
            or definition.get("anchor_sequence_positions") != [40, 99]):
        raise ValueError("Changed BRD4/JQ1 pocket protocol")
    sequence = definition["protein_sequence"]
    rows = definition["residues"]
    positions = [row["sequence_position"] for row in rows]
    if len(sequence) != 127 or positions != sorted(set(positions)) or not {40, 56, 99} <= set(positions):
        raise ValueError("Incomplete pocket residue/sequence definition")
    for row in rows:
        index = row["sequence_position"] - 1
        if (not 0 <= index < 127 or row["author_residue"] != index + 42
                or row.get("insertion_code", "") != ""
                or THREE_TO_ONE.get(row["residue_name"]) != sequence[index]
                or not math.isfinite(row["minimum_jq1_distance_angstrom"])
                or not 0 <= row["minimum_jq1_distance_angstrom"] <= 6.0):
            raise ValueError("Pocket author/sequence/amino-acid identity mismatch")
    return definition


def derive_definition(pdb_path, fixture_path=None) -> dict:
    atoms, parse_receipt = _pdb_atoms(pdb_path)
    ligand = [atom for atom in atoms if atom.record == "HETATM"]
    identities = {(atom.chain, atom.author_residue, atom.insertion, atom.residue_name)
                  for atom in ligand}
    if len(identities) != 1 or len(ligand) != 31:
        raise ValueError("Require one unique crystallographic 31-heavy-atom JQ1")
    protein = [atom for atom in atoms if atom.record == "ATOM"]
    by_author = {author: [atom for atom in protein if atom.author_residue == author]
                 for author in range(42, 169)}
    sequence, residues = [], []
    for position, author in enumerate(range(42, 169), 1):
        rows = by_author[author]
        names = {atom.residue_name for atom in rows}
        if len(names) != 1 or next(iter(names)) not in THREE_TO_ONE:
            raise ValueError("Missing/nonstandard residue in authenticated 127-residue construct")
        name = next(iter(names))
        sequence.append(THREE_TO_ONE[name])
        distance = min(math.dist(atom.coordinates, lig.coordinates)
                       for atom in rows for lig in ligand)
        if distance <= 6.0:
            residues.append({"author_residue": author, "sequence_position": position,
                             "insertion_code": "", "residue_name": name,
                             "minimum_jq1_distance_angstrom": distance})
    sequence = "".join(sequence)
    fixture_receipt = None
    if fixture_path is not None:
        raw_fixture = Path(fixture_path).read_bytes()
        fixture = json.loads(raw_fixture)
        if (fixture.get("protein_sequence") != sequence
                or fixture.get("protein_author_residues") != list(range(42, 169))
                or fixture.get("pdb_sha256") != PDB_SHA256):
            raise ValueError("Pocket and independently authenticated protein fixture differ")
        fixture_receipt = hashlib.sha256(raw_fixture).hexdigest()
    definition = {
        "version": VERSION, "status": "defined", "pdb_id": "3MXF",
        "pdb_sha256": PDB_SHA256, "fixture_sha256": fixture_receipt,
        "protein_chain": "A", "protein_sequence": sequence,
        "protein_author_residues": list(range(42, 169)),
        "reference_ligand": "JQ1", "reference_ligand_chain": "A",
        "reference_ligand_author_residue": ligand[0].author_residue,
        "reference_ligand_insertion_code": ligand[0].insertion,
        "reference_ligand_heavy_atoms": len(ligand), "radius_angstrom": 6.0,
        "selection": "all protein heavy atoms in residues with any heavy atom within6A ofJQ1; no solvent/ions",
        "anchor_author_residues": [81, 140], "anchor_sequence_positions": [40, 99],
        "required_pocket_sequence_positions": [40, 56, 99],
        "pose_requirements": {"anchor_max_distance_angstrom": 6.0,
                              "pocket_max_distance_angstrom": 6.0,
                              "minimum_ligand_fraction": 0.5},
        "residues": residues, "parse_receipt": parse_receipt,
        "coordinate_frame": "original crystallographic receptor frame; no centering here",
    }
    return validate_definition(definition)


def prepare_pocket(pdb_path, native_atom_decoder: Sequence[str], *,
                   fixture_path=None, device="cpu") -> tuple[dict, dict]:
    """Full-atom native tensor over the exact biological residue selection."""
    import torch

    definition = derive_definition(pdb_path, fixture_path)
    atoms, _ = _pdb_atoms(pdb_path)
    residue_ids = {row["author_residue"] for row in definition["residues"]}
    encoder = {symbol: index for index, symbol in enumerate(native_atom_decoder)}
    if len(encoder) != len(native_atom_decoder) or not encoder:
        raise ValueError("Native atom vocabulary must be unique/nonempty")
    candidates = [atom for atom in atoms
                  if atom.record == "ATOM" and atom.author_residue in residue_ids]
    included = [atom for atom in candidates if atom.element in encoder]
    excluded = [atom for atom in candidates if atom.element not in encoder]
    if not included or {atom.author_residue for atom in included} != residue_ids:
        raise ValueError("Native vocabulary filtering lost an entire selected pocket residue")
    xyz = torch.tensor([atom.coordinates for atom in included], device=device,
                        dtype=torch.float32)
    labels = torch.tensor([encoder[atom.element] for atom in included],
                           device=device, dtype=torch.long)
    pocket = {"x": xyz, "one_hot": torch.nn.functional.one_hot(
        labels, len(native_atom_decoder)).float(),
        "mask": torch.zeros(len(included), dtype=torch.long, device=device),
        "size": torch.tensor([len(included)], dtype=torch.long, device=device)}
    definition["native_pocket"] = {
        "representation": "full-atom", "atom_decoder": list(native_atom_decoder),
        "atom_count": len(included),
        "atom_rows": [{"native_row": index, "serial": atom.serial,
                       "chain": atom.chain, "author_residue": atom.author_residue,
                       "sequence_position": atom.author_residue - 41,
                       "insertion_code": atom.insertion, "residue_name": atom.residue_name,
                       "atom_name": atom.atom_name, "element": atom.element,
                       "coordinates": list(atom.coordinates)}
                      for index, atom in enumerate(included)],
        "excluded_unsupported_atoms": [atom.__dict__ for atom in excluded],
        "selected_residues_before_vocabulary_filter": sorted(residue_ids),
        "ligand_reference_coordinates": [list(atom.coordinates) for atom in atoms
                                          if atom.record == "HETATM"],
    }
    return pocket, definition


def audit_pose(positions, definition: dict) -> dict:
    """Check supplied candidate pose against the crystallographic biological pocket."""
    validate_definition(definition)
    if hasattr(positions, "detach"):
        positions = positions.detach().cpu().tolist()
    ligand = [tuple(float(value) for value in xyz) for xyz in positions]
    if not ligand or any(len(xyz) != 3 or not all(math.isfinite(x) for x in xyz)
                         for xyz in ligand):
        raise ValueError("Nonfinite/malformed candidate pose")
    rows = definition["native_pocket"]["atom_rows"]
    distances = {}
    for residue in definition["residues"]:
        sequence_position = residue["sequence_position"]
        pocket_atoms = [row["coordinates"] for row in rows
                        if row["sequence_position"] == sequence_position]
        if not pocket_atoms:
            raise ValueError("Selected native pocket residue has no coordinates")
        distances[sequence_position] = [min(math.dist(xyz, protein) for protein in pocket_atoms)
                                        for xyz in ligand]
    anchors = {str(position): min(distances[position])
               for position in definition["anchor_sequence_positions"]}
    closest = [min(distance[i] for distance in distances.values())
               for i in range(len(ligand))]
    fraction = sum(distance <= 6.0 for distance in closest) / len(closest)
    passed = all(distance <= 6.0 for distance in anchors.values()) and fraction >= 0.5
    return {"schema": "diffsbdd-crystal-pocket-pose-audit-v1", "passed": passed,
            "anchor_minimum_distances_angstrom": anchors,
            "ligand_minimum_pocket_distances_angstrom": closest,
            "ligand_fraction_within_6A_of_pocket": fraction,
            "scope": "supplied pose in original crystal receptor frame; no redocking"}
