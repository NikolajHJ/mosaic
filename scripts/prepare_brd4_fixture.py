"""Authenticate the complete 3MXF BRD4-BD1 sequence on allocated DTU HPC.

Only the deposited protein sequence enters the model. JQ1, other HETATM
records and every coordinate are excluded; no pocket is selected.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

from prepare_known_ligand_fixture import AMINO_ACIDS


PDB_SHA256 = "534dcb8ea9fa29956249b5133b22187c3867363b16a6b98dd8219c18c5d3250f"


def prepare(text):
    """Check all 127 deposited residues against author numbering 42..168."""
    sequence, observed = [], {}
    for line in text.splitlines():
        if line.startswith("SEQRES") and line[11] == "A":
            sequence.extend(AMINO_ACIDS[name] for name in line[19:70].split())
        if not line.startswith("ATOM  ") or line[21] != "A":
            continue
        number, insertion = int(line[22:26]), line[26].strip()
        letter = AMINO_ACIDS[line[17:20]]
        if insertion or number in observed and observed[number] != letter:
            raise ValueError("Ambiguous 3MXF author residue mapping")
        observed[number] = letter
    sequence = "".join(sequence)
    if len(sequence) != 127 or set(observed) != set(range(42, 169)):
        raise ValueError("Expected complete 127-residue BRD4-BD1 chain A")
    if any(sequence[number - 42] != letter for number, letter in observed.items()):
        raise ValueError("3MXF author/SEQRES identity mismatch")
    if sequence[:2] != "SM":
        raise ValueError("Preserve deposited S42/M43 construct, including SEQADV")
    return {
        "contract": "small_molecule_v0.1",
        "smiles": "CCCCCC",
        "n_heavy_atoms": 6,
        "element_order": ["C"] * 6,
        "bond_assignments": [[i, i + 1, "SINGLE"] for i in range(5)],
        "protein_sequence": sequence,
        "protein_name": "human BRD4-BD1 (3MXF deposited construct)",
        "protein_author_residues": list(range(42, 169)),
        "contact_partner_scope": "whole_target",
        "pdb_id": "3MXF",
        "pdb_sha256": PDB_SHA256,
        "source_url": "https://files.rcsb.org/download/3MXF.pdb",
        "known_ligand": "JQ1; excluded, no neutralization or native reference score",
        "interpretation": "complete deposited sequence, author residues 42..168, including deposited M43; all target residues eligible; no pocket, MSA, structural template, JQ1, solvent, ions or crystal coordinates in model context",
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
        raise ValueError("Unapproved 3MXF input bytes")
    fixture = prepare(data.decode("ascii"))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "fixture.json").write_text(json.dumps(fixture, indent=2) + "\n")
    print(json.dumps({"job_id": os.environ["LSB_JOBID"], "fixture": fixture}))


if __name__ == "__main__":
    main()
