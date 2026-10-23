"""HPC structural audit of the frozen native-rescore candidate pools.

RDKit parsing, standardization and alerts are diagnostics. None measures a
synthetic route, chemical stability in solution, or experimental binding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from boltz.data.parse.schema import standardize
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, QED, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from latentmol.experiments import ExperimentRun, file_sha256, write_json
from .prepare import installed_boltz_provenance


def _catalog():
    params = FilterCatalogParams()
    for name in ("PAINS_A", "PAINS_B", "PAINS_C", "BRENK"):
        params.AddCatalog(getattr(FilterCatalogParams.FilterCatalogs, name))
    return FilterCatalog(params)


def describe(smiles: str, catalog) -> dict:
    """Separate input and native-standardized identities in every result row."""
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return {"input_smiles": smiles, "parse_failure": True}
    try:
        native = standardize(smiles)
        effective = Chem.MolFromSmiles(native)
        if effective is None:
            raise ValueError("Pinned Boltz standardization returned an unparsable SMILES")
        standardized = Chem.MolToSmiles(effective, isomericSmiles=False)
        standardization_failure = None
    except (TypeError, ValueError) as exc:
        standardized = None
        effective = None
        standardization_failure = str(exc)
    rings = list(molecule.GetRingInfo().AtomRings())
    oxygen_three_rings = sum(len(ring) == 3 and any(
        molecule.GetAtomWithIdx(index).GetAtomicNum() == 8 for index in ring)
        for ring in rings)
    alerts = sorted({entry.GetDescription() for entry in catalog.GetMatches(molecule)})
    return {"input_smiles": smiles, "parse_failure": False,
            "input_canonical_smiles": Chem.MolToSmiles(molecule, isomericSmiles=False),
            "native_standardized_smiles": standardized,
            "native_standardized_heavy_atoms": effective.GetNumHeavyAtoms()
            if effective is not None else None,
            "standardization_failure": standardization_failure,
            "standardization_changed_identity":
                Chem.MolToSmiles(molecule, isomericSmiles=False)
                != standardized if standardized is not None else None,
            "heavy_atoms": molecule.GetNumHeavyAtoms(),
            "net_formal_charge": sum(atom.GetFormalCharge() for atom in molecule.GetAtoms()),
            "molecular_weight": Descriptors.MolWt(molecule),
            "clogp": Crippen.MolLogP(molecule),
            "tpsa": rdMolDescriptors.CalcTPSA(molecule),
            "qed": QED.qed(molecule),
            "ring_sizes": sorted(len(ring) for ring in rings),
            "oxygen_containing_three_membered_rings": oxygen_three_rings,
            "rdkit_filter_alerts": alerts}


def run_audit(ae_selection: Path, vae_selection: Path, run_root: Path) -> Path:
    paths = {"ae": ae_selection, "vae": vae_selection}
    selections = {}
    for mode, path in paths.items():
        parent = json.loads((path.parent / "run.json").read_text())
        if parent.get("name") != f"{mode}-native-fresh-pose":
            raise ValueError("Selection file does not belong to matching native run")
        value = json.loads(path.read_text())
        if not value["baseline_smiles"] or not value["unique_smiles"]:
            raise ValueError("Native selection lacks a baseline or candidates")
        selections[mode] = value
    provenance = installed_boltz_provenance()
    config = {"ae_selection_sha256": file_sha256(ae_selection),
              "vae_selection_sha256": file_sha256(vae_selection),
              "ae_selection": str(ae_selection), "vae_selection": str(vae_selection),
              "boltz_provenance": provenance,
              "alert_catalogs": ["PAINS_A", "PAINS_B", "PAINS_C", "BRENK"]}
    with ExperimentRun(run_root, "selected-molecule-chemical-audit", config, Path.cwd()) as run:
        catalog = _catalog()
        all_smiles = sorted(set(selections["ae"]["unique_smiles"])
                            | set(selections["vae"]["unique_smiles"]))
        rows = [describe(smiles, catalog) for smiles in all_smiles]
        for row in rows:
            smiles = row["input_smiles"]
            row["selected_by"] = {
                mode: {"baseline": smiles == value["baseline_smiles"],
                       "memberships": [
                           {"arm": arm, "role": role, "repeat": chosen["repeat"],
                            "start_index": chosen["start_index"]}
                           for arm, group in value["by_arm"].items()
                           for chosen in group["chosen"]
                           for role, identity in (("endpoint", chosen["smiles"]),
                                                  ("perturbed_start", chosen["start_smiles"]))
                           if identity == smiles]}
                for mode, value in selections.items() if smiles in value["unique_smiles"]}
        write_json(run.path / "molecules.json", rows)
        effective_by_input = {row["input_smiles"]: row.get("native_standardized_smiles")
                              for row in rows}
        endpoint_rows = []
        for mode, value in selections.items():
            baseline = value["baseline_smiles"]
            for arm, group in value["by_arm"].items():
                for chosen in group["chosen"]:
                    original = chosen["smiles"]
                    start = chosen["start_smiles"]
                    endpoint_effective = effective_by_input[original]
                    start_effective = effective_by_input[start]
                    baseline_effective = effective_by_input[baseline]
                    complete = all(identity is not None for identity in
                                   (endpoint_effective, start_effective, baseline_effective))
                    endpoint_rows.append({"mode": mode, "arm": arm,
                                          "repeat": chosen["repeat"],
                                          "start_index": chosen["start_index"],
                                          "endpoint_input": original,
                                          "start_input": start,
                                          "baseline_input": baseline,
                                          "endpoint_native_identity": endpoint_effective,
                                          "start_native_identity": start_effective,
                                          "baseline_native_identity": baseline_effective,
                                          "effective_changed_from_both":
                                          endpoint_effective not in (start_effective, baseline_effective)
                                          if complete else None})
        write_json(run.path / "endpoint_identity_audit.json", endpoint_rows)
        summary = {"input_identities": len(rows),
                   "native_standardized_identities": len({
                       row["native_standardized_smiles"] for row in rows
                       if row.get("native_standardized_smiles")}),
                   "native_standardization_failures": sum(bool(row.get(
                       "standardization_failure")) for row in rows),
                   "standardization_changes": sum(row.get("standardization_changed_identity") is True
                                                   for row in rows),
                   "input_with_filter_alerts": sum(bool(row.get("rdkit_filter_alerts")) for row in rows),
                   "input_with_oxygen_three_ring": sum(bool(row.get(
                       "oxygen_containing_three_membered_rings")) for row in rows),
                   "selected_endpoint_identities": len(endpoint_rows),
                   "selected_effective_changed_from_both": sum(
                       row["effective_changed_from_both"] is True for row in endpoint_rows),
                   "selected_collapsed_to_start_or_baseline": sum(
                       row["effective_changed_from_both"] is False for row in endpoint_rows),
                   "selected_identity_unknown_due_to_preprocessing_failure": sum(
                       row["effective_changed_from_both"] is None for row in endpoint_rows),
                   "limitation": "Structural descriptors and filter catalogs do not establish stability, synthetic accessibility, route recovery, or binding."}
        write_json(run.path / "summary.json", summary)
        run.log({"stage": "structural_audit", **summary})
    return run.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ae-selection", type=Path, required=True)
    parser.add_argument("--vae-selection", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    print(run_audit(args.ae_selection, args.vae_selection, args.run_root))


if __name__ == "__main__":
    main()
