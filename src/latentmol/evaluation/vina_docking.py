"""Rigid-receptor docking of a discrete molecule with the native Vina executable."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


class VinaDockingScorer:
    """Dock one fixed input microstate. More negative affinities rank first."""

    def __init__(self, receptor_path, box_path, *, seed=17, exhaustiveness=4, cpu=1):
        self.receptor = Path(receptor_path).resolve(strict=True)
        box = json.loads(Path(box_path).read_text())
        self.center = tuple(map(float, box["center"]))
        self.size = tuple(map(float, box["size"]))
        if (len(self.center) != 3 or len(self.size) != 3
                or not all(map(math.isfinite, self.center + self.size)) or min(self.size) <= 0):
            raise ValueError("Expected finite 3D center and positive box dimensions in angstroms")
        if not (0 < seed < 2**31 and exhaustiveness >= 1 and cpu >= 1):
            raise ValueError("Vina requires a positive int32 seed, exhaustiveness and CPU count")
        self.seed, self.exhaustiveness, self.cpu = seed, exhaustiveness, cpu
        sibling = Path(sys.executable).with_name("vina.exe" if os.name == "nt" else "vina")
        self.binary = os.environ.get("VINA_EXECUTABLE") or (
            str(sibling) if sibling.is_file() else shutil.which("vina"))
        if not self.binary:
            raise FileNotFoundError("Install AutoDock Vina or set VINA_EXECUTABLE")
        self.version = subprocess.check_output([self.binary, "--version"], text=True, timeout=15).strip()
        # Fail early on deployment errors. Candidate-specific errors are reported per attempt.
        from meeko import MoleculePreparation

        self.preparer = MoleculePreparation()

    def score(self, smiles: str, *, pose_path=None) -> dict:
        """Embed a hard SMILES from scratch and search the box. Optionally retain the best pose."""
        from meeko import PDBQTWriterLegacy
        from rdkit import Chem
        from rdkit.Chem import AllChem

        if pose_path is not None and Path(pose_path).exists():
            raise FileExistsError(pose_path)
        result = {
            "smiles": smiles, "status": "failed", "affinity_kcal_mol": None,
            "vina_version": self.version, "seed": self.seed, "exhaustiveness": self.exhaustiveness,
        }
        started = time.monotonic()
        try:
            mol = Chem.MolFromSmiles(smiles)
            if (mol is None or mol.GetNumHeavyAtoms() == 0 or len(Chem.GetMolFrags(mol)) != 1
                    or any(a.GetNumRadicalElectrons() for a in mol.GetAtoms())):
                raise ValueError("Expected a connected, sanitized, radical-free molecule")
            canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
            result["canonical_smiles"] = canonical
            mol = Chem.MolFromSmiles(canonical)
            result["unspecified_stereo_count"] = sum(
                s.specified == Chem.StereoSpecified.Unspecified for s in Chem.FindPotentialStereo(mol))
            mol = Chem.AddHs(mol)
            embed = AllChem.ETKDGv3()
            embed.randomSeed = self.seed
            embed.numThreads = 1
            if AllChem.EmbedMolecule(mol, embed) != 0:
                raise ValueError("ETKDG embedding failed")
            embedded = Chem.Mol(mol)
            Chem.RemoveStereochemistry(embedded)
            Chem.AssignStereochemistryFrom3D(embedded)
            result["embedded_isomeric_smiles"] = Chem.MolToSmiles(
                Chem.RemoveHs(embedded), isomericSmiles=True)
            setups = self.preparer.prepare(mol)
            if len(setups) != 1:
                raise ValueError(f"Expected one ligand setup, received {len(setups)}")
            ligand, ok, message = PDBQTWriterLegacy.write_string(setups[0])
            if not ok:
                raise ValueError(message)
            with tempfile.TemporaryDirectory(prefix="latentmol-vina-") as temp:
                ligand_path, output = Path(temp) / "ligand.pdbqt", Path(temp) / "pose.pdbqt"
                ligand_path.write_text(ligand)
                cmd = [self.binary, "--receptor", str(self.receptor), "--ligand", str(ligand_path),
                       "--out", str(output), "--scoring", "vina", "--num_modes", "1",
                       "--seed", str(self.seed), "--exhaustiveness", str(self.exhaustiveness),
                       "--cpu", str(self.cpu)]
                for name, values in (("center", self.center), ("size", self.size)):
                    for axis, value in zip("xyz", values):
                        cmd.extend([f"--{name}_{axis}", str(value)])
                run = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                if run.returncode or not output.is_file():
                    raise ValueError(f"Vina exited {run.returncode}: {(run.stderr or run.stdout)[-1000:]}")
                pose = output.read_text()
                marker = next((line for line in pose.splitlines() if line.startswith("REMARK VINA RESULT:")), "")
                if not marker:
                    raise ValueError("Vina produced no scored pose")
                affinity = float(marker.split()[3])
                if not math.isfinite(affinity):
                    raise ValueError("Vina produced a nonfinite affinity")
        except (ValueError, RuntimeError, TypeError, IndexError, KeyError, subprocess.TimeoutExpired) as exc:
            result["reason"] = str(exc)
        else:
            if pose_path is not None:
                with Path(pose_path).open("x") as handle:
                    handle.write(pose)
            result.update(status="ok", affinity_kcal_mol=affinity)
        result["wall_seconds"] = round(time.monotonic() - started, 3)
        return result
