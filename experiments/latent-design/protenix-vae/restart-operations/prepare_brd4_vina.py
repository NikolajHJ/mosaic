"""Prepare the 3MXF chain-A/JQ1 docking pocket and redock JQ1 as a control.

Use the deposited https://files.rcsb.org/download/3MXF.pdb with the prepare command.
For redock, the crystal-frame SDF comes from
https://models.rcsb.org/v1/3mxf/ligand?auth_asym_id=A&auth_seq_id=1&encoding=sdf .
The pilot used PDBFixer 1.12.0, OpenMM 8.6.1, Meeko 0.7.1, Gemmi 0.7.4,
RDKit 2025.09.6, and AutoDock Vina 1.2.7. Only preparation requires PDBFixer
and OpenMM. Use the repo src on PYTHONPATH and VINA_EXECUTABLE if needed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile


def prepare(pdb, output):
    import numpy as np
    from openmm import Platform
    from openmm.app import PDBFile
    from pdbfixer import PDBFixer

    pdb_hash = hashlib.sha256(pdb.read_bytes()).hexdigest()
    lines = pdb.read_text().splitlines()
    protein = [line for line in lines if line.startswith("ATOM  ") and line[21] == "A"
               and line[16] in " A"]
    ligand = [line for line in lines if line.startswith("HETATM") and line[17:20] == "JQ1"
              and line[21] == "A" and int(line[22:26]) == 1 and line[76:78].strip() != "H"]
    if len(ligand) != 31 or len({line[22:27] for line in protein}) != 127:
        raise ValueError("Expected 3MXF chain A with 127 observed residues and JQ1 A1 with 31 heavy atoms")
    xyz = np.array([[float(line[i:i+8]) for i in (30, 38, 46)] for line in ligand])
    low, high = xyz.min(axis=0), xyz.max(axis=0)
    # Bounds are centered on the native heavy atoms, with 6 Å clearance on each face.
    box = {"center": ((low + high) / 2).round(4).tolist(),
           "size": (high - low + 12).round(4).tolist(),
           "source": "RCSB PDB 3MXF, JQ1 A1", "pdb_sha256": pdb_hash,
           "ligand_heavy_atoms": len(ligand), "padding_per_face_angstrom": 6}
    if output.exists():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "chain.pdb"
        source.write_text("\n".join(line[:16] + " " + line[17:] for line in protein) + "\nEND\n")
        fixer = PDBFixer(filename=str(source), platform=Platform.getPlatformByName("CPU"))
        fixer.findMissingResidues()
        fixer.missingResidues = {}  # Keep exactly the observed residues.
        fixer.findMissingAtoms()
        box["rebuilt_heavy_atoms"] = {
            f"{res.name} {res.chain.id}{res.id}": [a.name for a in atoms]
            for res, atoms in fixer.missingAtoms.items()
        }
        box["added_terminal_atoms"] = {f"{r.name} {r.chain.id}{r.id}": list(atoms)
                                       for r, atoms in fixer.missingTerminals.items()}
        fixer.addMissingAtoms(seed=1701)
        random.seed(1701)
        fixer.addMissingHydrogens(pH=7.4)
        repaired = Path(tmp) / "repaired.pdb"
        with repaired.open("w") as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)
        out = Path(tmp) / "receptor.pdbqt"
        subprocess.run([sys.executable, "-m", "meeko.cli.mk_prepare_receptor",
                        "--read_pdb", str(repaired), "-p", str(out)], check=True)
        atoms = [line for line in out.read_text().splitlines() if line.startswith("ATOM")]
        if len({line[21:27] for line in atoms}) != 127:
            raise ValueError("A receptor residue was lost during preparation")
        output.mkdir(parents=True)
        (output / "receptor.pdbqt").write_bytes(out.read_bytes())
        (output / "box.json").write_text(json.dumps(box, indent=2) + "\n")


def redock(native_sdf, pocket, output, seed, exhaustiveness):
    from meeko import PDBQTMolecule, RDKitMolCreate
    from rdkit import Chem
    from rdkit.Chem import rdMolAlign
    from latentmol.evaluation.vina_docking import VinaDockingScorer

    native = Chem.SDMolSupplier(str(native_sdf))[0]
    if native is None or native.GetNumHeavyAtoms() != 31:
        raise ValueError("Expected deposited 31-heavy-atom JQ1 SDF in the 3MXF frame")
    # The RCSB model-server SDF contains the +1 CCD JQ1 microstate.
    if Chem.GetFormalCharge(native) != 1:
        raise ValueError("Unexpected native JQ1 microstate")
    scorer = VinaDockingScorer(pocket / "receptor.pdbqt", pocket / "box.json",
                                seed=seed, exhaustiveness=exhaustiveness, cpu=1)
    output.mkdir(parents=True, exist_ok=False)
    pose_path = output / "jq1-top.pdbqt"
    result = scorer.score(Chem.MolToSmiles(native), pose_path=pose_path)
    result["reference"] = "3MXF JQ1 A1, RCSB CCD +1 microstate"
    result["ligand_input"] = "canonical isomeric SMILES, ETKDGv3 from scratch"
    if result["status"] == "ok":
        pose = PDBQTMolecule.from_file(str(pose_path), skip_typing=True)
        docked = Chem.RemoveHs(RDKitMolCreate.from_pdbqt_mol(pose)[0])
        if Chem.MolToSmiles(docked) != Chem.MolToSmiles(native):
            raise ValueError("JQ1 graph changed in docking pose export")
        result["native_frame_heavy_atom_rmsd_angstrom"] = float(rdMolAlign.CalcRMS(docked, native))
        result["rmsd_method"] = "symmetry-aware RDKit CalcRMS; no superposition"
    for name in ("box.json", "receptor.pdbqt"):
        result[name + "_sha256"] = hashlib.sha256((pocket / name).read_bytes()).hexdigest()
    (output / "control.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result["status"] == "ok"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--pdb", required=True, type=Path)
    prep.add_argument("--output", required=True, type=Path)
    ctrl = sub.add_parser("redock")
    ctrl.add_argument("--native-sdf", required=True, type=Path)
    ctrl.add_argument("--pocket", required=True, type=Path)
    ctrl.add_argument("--output", required=True, type=Path)
    ctrl.add_argument("--seed", type=int, default=1701)
    ctrl.add_argument("--exhaustiveness", type=int, default=4)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.pdb, args.output)
    else:
        sys.exit(0 if redock(args.native_sdf, args.pocket, args.output, args.seed, args.exhaustiveness) else 1)
