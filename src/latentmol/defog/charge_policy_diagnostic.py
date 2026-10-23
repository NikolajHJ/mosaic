"""HPC CPU parity of the opt-in output rule against pinned native DeFoG code.

This invokes the real native molecule builder, never a model, sampling worker,
Boltz score or optimizer. Passing establishes conversion engineering only. A
separate bounded native repair/target-FD experiment must establish usefulness.
Authored without local numerical execution; run inside an allocated HPC job.
"""

import argparse
import importlib.util
import os
from pathlib import Path
import subprocess

import torch

from latentmol.defog.assets import DEFOG_COMMIT
from latentmol.defog.codec import (GUACAMOL_ATOMIC_NUMBERS, NATIVE_CHARGE_RULE,
    PROJECT_CHARGE_POLICY, native_charge_molecule)
from latentmol.experiments import ExperimentRun, file_sha256, write_json


def parity_cases():
    """Prespecified labeled graphs; no chemistry-dependent selection or fitting."""
    def row(name, numbers, edges, expected):
        x = torch.tensor([[GUACAMOL_ATOMIC_NUMBERS.index(n) for n in numbers]])
        e = torch.zeros((1, len(numbers), len(numbers)), dtype=torch.long)
        for i, j, bond in edges:
            e[0, i, j] = e[0, j, i] = bond
        return name, x, e, expected
    yield row("neutral_ethane", [6, 6], [(0, 1, 1)], [0, 0])
    yield row("native_N_plus", [7, 6, 6, 6, 6], [(0, i, 1) for i in range(1, 5)], [1, 0, 0, 0, 0])
    yield row("native_O_plus", [8, 6, 6, 6], [(0, i, 1) for i in range(1, 4)], [1, 0, 0, 0])
    # The native rule mentions S, but RDKit allows expanded sulfur valence;
    # three single bonds need not raise the exception that triggers +1.
    yield row("sulfur_not_forced_positive", [16, 6, 6, 6], [(0, i, 1) for i in range(1, 4)], [0, 0, 0, 0])
    yield row("negative_charge_not_inferred", [6, 6, 8, 8], [(0, 1, 1), (1, 2, 2), (1, 3, 1)], [0] * 4)
    yield row("hydrogen_dependent_N_charge_not_inferred", [7, 6, 6, 6], [(0, i, 1) for i in range(1, 4)], [0] * 4)
    yield row("aromatic_ring", [6] * 6, [(i, (i + 1) % 6, 4) for i in range(6)], [0] * 6)
    yield row("invalid_carbon_not_corrected", [6] * 6, [(0, i, 1) for i in range(1, 6)], [0] * 6)
    yield row("disconnected_not_cropped", [6, 6], [], [0, 0])
    yield row("native_boron_not_relabelled", [5, 6], [(0, 1, 1)], [0, 0])
    yield row("permuted_N_plus", [6, 6, 7, 6, 6], [(2, i, 1) for i in (0, 1, 3, 4)], [0, 0, 1, 0, 0])


def molecule_record(mol):
    """Compare order/labels and full-graph outcomes, with no SMILES rewriting."""
    from rdkit import Chem
    record = {"atomic_numbers": [atom.GetAtomicNum() for atom in mol.GetAtoms()],
              "formal_charges": [atom.GetFormalCharge() for atom in mol.GetAtoms()],
              "aromatic_atoms": [atom.GetIsAromatic() for atom in mol.GetAtoms()],
              "bonds": [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), str(bond.GetBondType())]
                        for bond in mol.GetBonds()]}
    copy = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(copy)
    except (ValueError, RuntimeError) as error:
        record.update(sanitized=False, sanitization_error=type(error).__name__)
    else:
        record.update(sanitized=True, connected_components=len(Chem.GetMolFrags(copy)),
                      radical_electrons=[atom.GetNumRadicalElectrons() for atom in copy.GetAtoms()])
    return record


def verify_source(source):
    source = Path(source).resolve(strict=True)
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != DEFOG_COMMIT or subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain"], text=True).strip():
        raise ValueError("Require the pinned clean native DeFoG checkout")
    relative = "src/analysis/rdkit_functions.py"
    path = source / relative
    original = subprocess.check_output(["git", "-C", str(source), "show", f"{DEFOG_COMMIT}:{relative}"])
    if path.read_bytes() != original:
        raise ValueError("Native builder file differs from the pinned commit")
    return path, {"path": str(source), "commit": commit, "clean": True,
                  "builder_relative_path": relative, "builder_sha256": file_sha256(path)}


def run(source, run_root):
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Native charge-rule parity requires an allocated DTU HPC job")
    from rdkit import Chem, rdBase
    native_path, provenance = verify_source(source)
    cfg = {"charge_policy": PROJECT_CHARGE_POLICY, "rule": NATIVE_CHARGE_RULE,
           "native_source": provenance, "rdkit_version": rdBase.rdkitVersion,
           "helper_source_sha256": file_sha256(Path(__file__).with_name("codec.py")),
           "scope": "CPU native-builder conversion parity; zero model/repair/optimization calls"}
    with ExperimentRun(Path(run_root), "defog-native-charge-rule-parity", cfg, Path.cwd()) as experiment:
        result = {"passed": False, "native_builder_parity": False, "case_count": 0,
                  "cases": [], "model_calls": 0, "sampled_repairs": 0, **cfg}
        path = experiment.path / "summary.json"
        write_json(path, result)
        spec = importlib.util.spec_from_file_location("_latentmol_pinned_charge_parity", native_path)
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        decoder = [Chem.GetPeriodicTable().GetElementSymbol(n) for n in GUACAMOL_ATOMIC_NUMBERS]
        for name, x, e, expected in parity_cases():
            row = {"name": name, "native_labels": {"atoms": x.tolist(), "bonds": e.tolist()},
                   "expected_formal_charges": expected, "passed": False}
            result["cases"].append(row)
            try:
                before_x, before_e = x.clone(), e.clone()
                observed = molecule_record(native_charge_molecule(x, e))
                reference = molecule_record(native.build_molecule_with_partial_charges(x[0].clone(), e[0].clone(), decoder))
                row.update(helper=observed, native=reference,
                    inputs_unchanged=torch.equal(x, before_x) and torch.equal(e, before_e),
                    passed=observed == reference and observed["formal_charges"] == expected
                           and torch.equal(x, before_x) and torch.equal(e, before_e))
            except Exception as error:
                row.update(error=str(error), error_type=type(error).__name__)
            result["case_count"] = len(result["cases"])
            write_json(path, result)
        verify_source(source)
        result["native_builder_parity"] = all(row["passed"] for row in result["cases"])
        result["passed"] = result["native_builder_parity"] and result["case_count"] == 11
        write_json(path, result)
        if not result["passed"]:
            raise RuntimeError("Native charge-builder parity failed; no opt-in repair run is certified")
        experiment.annotate(native_builder_parity=True, case_count=result["case_count"])
        return experiment.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.source, args.run_root))


if __name__ == "__main__":
    main()
