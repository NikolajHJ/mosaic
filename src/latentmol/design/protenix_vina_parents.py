"""Budgeted native docking of hard parent candidates in synthesis waves."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter


def _native_scores(config, smiles):
    from latentmol.evaluation.vina_docking import VinaDockingScorer

    scorer = VinaDockingScorer(config["receptor"]["path"], config["box"]["path"],
                               seed=config["seed"], exhaustiveness=config["exhaustiveness"], cpu=1)
    if "pose_output_dir" not in config:
        return {name: scorer.score(name) for name in smiles}
    from meeko import PDBQTMolecule, RDKitMolCreate
    from rdkit import Chem

    results = {}
    for name in smiles:
        stem = hashlib.sha256(name.encode()).hexdigest()
        folder = Path(config["pose_output_dir"])
        path = folder / (stem + ".pdbqt")
        result = scorer.score(name, pose_path=path)
        results[name] = result
        if result["status"] != "ok":
            continue
        result["pose_pdbqt"] = path.name
        # Meeko restores bonds, charges and heavy-atom ordering from the
        # original microstate encoded in its REMARK records.
        try:
            poses = RDKitMolCreate.from_pdbqt_mol(PDBQTMolecule.from_file(str(path), skip_typing=True))
            if len(poses) != 1 or poses[0] is None or poses[0].GetNumConformers() != 1:
                raise ValueError("Expected one complete ligand and the single best Vina pose")
            mol = Chem.RemoveHs(poses[0])
            if Chem.MolToSmiles(mol, isomericSmiles=False) != Chem.MolToSmiles(
                    Chem.MolFromSmiles(name), isomericSmiles=False):
                raise ValueError("Vina pose export changed the designed molecular graph")
            block = Chem.MolToMolBlock(mol, kekulize=False)
        except (ValueError, RuntimeError, TypeError, IndexError, KeyError) as exc:
            result["pose_export_error"] = str(exc)
            continue
        sdf = path.with_suffix(".sdf")
        with sdf.open("x") as stream:
            stream.write(block + "$$$$\n")
        result["pose_sdf"] = sdf.name
    return results


class VinaParentPanel:
    """One fixed receptor, grid and seed per instance; local cache shared by arms."""

    def __init__(self, config):
        self.config = dict(config)
        self.panel_size = config["panel_size"]
        if (type(self.panel_size) is not int or not 1 <= self.panel_size <= 32
                or type(config["seed"]) is not int or not 1 <= config["seed"] < 2**31
                or type(config["exhaustiveness"]) is not int or not 1 <= config["exhaustiveness"] <= 64):
            raise ValueError("Invalid Vina panel size, seed, or exhaustiveness")
        for key in ("receptor", "box"):
            asset = config[key]
            if hashlib.sha256(Path(asset["path"]).read_bytes()).hexdigest() != asset["sha256"]:
                raise ValueError(f"Changed Vina {key} asset")
        self.cache = {}
        self.requests = 0
        self.seconds = 0.

    def score_many(self, names):
        names = list(names)
        if len(names) > self.panel_size or len(names) != len(set(names)):
            raise ValueError("Vina request exceeds the distinct panel allowance")
        self.requests += len(names)
        missing = [name for name in names if name not in self.cache]
        began = perf_counter()
        if missing:
            if python := self.config.get("python"):
                # The docking environment can carry its own RDKit/Meeko builds.
                # Only SMILES and numeric readouts cross the process boundary.
                with TemporaryDirectory(prefix="latentmol-vina-panel-") as temp:
                    output = Path(temp) / "scores.json"
                    subprocess.run([python, "-m", __name__, str(output)],
                                   input=json.dumps([self.config, missing]), text=True, check=True)
                    result = json.loads(output.read_text(encoding="utf-8"))
            else:
                result = _native_scores(self.config, missing)
            if set(result) != set(missing):
                raise ValueError("Native Vina returned different molecule identities")
            for name, row in result.items():
                value = row.get("affinity_kcal_mol")
                if row["status"] == "ok":
                    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                        raise ValueError(f"Nonfinite native Vina score for {name}")
                elif row["status"] != "failed" or value is not None or not row.get("reason"):
                    raise ValueError(f"Inconsistent native Vina failure for {name}")
            self.cache.update(result)
        self.seconds += perf_counter() - began
        return {name: dict(self.cache[name], cache_hit=name not in missing) for name in names}

    def accounting(self):
        return dict(logical_requests=self.requests, unique_attempts=len(self.cache),
                    cache_hits=self.requests - len(self.cache),
                    successes=sum(row["status"] == "ok" for row in self.cache.values()),
                    failures=sum(row["status"] == "failed" for row in self.cache.values()),
                    wall_seconds=self.seconds, config=self.config)


if __name__ == "__main__":
    config, names = json.load(sys.stdin)
    with Path(sys.argv[1]).open("x", encoding="utf-8") as stream:
        json.dump(_native_scores(config, names), stream, allow_nan=False)
