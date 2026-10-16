"""Prepare the authenticated 181L sequence/contact fixture on allocated DTU HPC.

The crystal selects residue IDs only. Coordinates, bound ligand and solvents
are not model inputs; the graph adapter still uses sequence-only proteins.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path


PDB_SHA256 = "77018feaaa65bb22dea47c784e8c059b0ccc09cd6dc7442b79cce83f3170985f"
AMINO_ACIDS = dict(
    zip(
        "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
        "ARNDCQEGHILKMFPSTWYV",
        strict=True,
    )
)


def prepare(text, *, pocket_distance=5.0):
    """Map PDB author residues to SEQRES; reject mismatches or ambiguous ligand."""
    sequence, protein, ligand = [], [], []
    ligand_ids = set()
    for line in text.splitlines():
        record = line[:6].strip()
        if record == "SEQRES" and line[11] == "A":
            sequence.extend(AMINO_ACIDS[name] for name in line[19:70].split())
        if record not in ("ATOM", "HETATM") or line[21] != "A":
            continue
        if line[16] not in (" ", "A") or float(line[54:60]) <= 0:
            continue
        if line[76:78].strip() in ("H", "D"):
            continue
        xyz = tuple(float(line[start : start + 8]) for start in (30, 38, 46))
        number, insertion = int(line[22:26]), line[26].strip()
        if record == "ATOM":
            protein.append((number, insertion, AMINO_ACIDS[line[17:20]], xyz))
        elif line[17:20] == "BNZ":
            ligand_ids.add((number, insertion))
            ligand.append(xyz)
    sequence = "".join(sequence)
    if len(sequence) != 164 or len(ligand) != 6 or len(ligand_ids) != 1:
        raise ValueError("Expected the full 164-residue chain A and one six-atom BNZ")
    contacts = set()
    for number, insertion, letter, xyz in protein:
        if (
            insertion
            or not 1 <= number <= len(sequence)
            or sequence[number - 1] != letter
        ):
            raise ValueError("Author/SEQRES mapping is not the declared direct mapping")
        if any(
            sum((a - b) ** 2 for a, b in zip(xyz, point, strict=True))
            <= pocket_distance**2
            for point in ligand
        ):
            contacts.add(number)
    if not contacts or sequence[98] != "A":
        raise ValueError("Missing ligand contacts or L99A mutation")
    return {
        "contract": "small_molecule_v0.1",
        "smiles": "CCCCCC",
        "n_heavy_atoms": 6,
        "element_order": ["C"] * 6,
        "bond_assignments": [[i, i + 1, "SINGLE"] for i in range(5)],
        "protein_sequence": sequence,
        "protein_name": "T4 lysozyme L99A (181L construct, including C54T/C97A)",
        "reference_smiles": "c1ccccc1",
        "reference_name": "benzene (BNZ A 400)",
        "reference_bond_assignments": [[i, i + 1, "AROMATIC"] for i in range(5)]
        + [[0, 5, "AROMATIC"]],
        "pocket_protein_ordinals": [number - 1 for number in sorted(contacts)],
        "pocket_author_residues": sorted(contacts),
        "crystal_contact_distance_angstrom": pocket_distance,
        "pdb_id": "181L",
        "pdb_sha256": PDB_SHA256,
        "source_url": "https://files.rcsb.org/download/181L.pdb",
        "interpretation": "real construct sequence; crystal-derived residue selection only; no structural template, MSA or bound ligand in optimization context; waters, ions and HED excluded",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Preprocessing requires allocated DTU HPC")
    data = args.pdb.read_bytes()
    if hashlib.sha256(data).hexdigest() != PDB_SHA256:
        raise ValueError("Unapproved 181L input bytes")
    fixture = prepare(data.decode("ascii"))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "fixture.json").write_text(json.dumps(fixture, indent=2) + "\n")
    print(
        json.dumps(
            {
                "job_id": os.environ["LSB_JOBID"],
                "pdb_sha256": PDB_SHA256,
                "residues": len(fixture["protein_sequence"]),
                "pocket_author_residues": fixture["pocket_author_residues"],
            }
        )
    )


if __name__ == "__main__":
    main()
