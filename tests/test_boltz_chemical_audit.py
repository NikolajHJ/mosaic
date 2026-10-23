"""Small structural checks for the selected-candidate audit; run on HPC."""

import json

import latentmol.boltz.chemical_audit as chemical_audit

from latentmol.boltz.chemical_audit import _catalog, describe, run_audit


def test_three_membered_oxygen_ring_is_reported_without_synthesis_claim():
    row = describe("C1CO1", _catalog())
    assert row["parse_failure"] is False
    assert row["oxygen_containing_three_membered_rings"] == 1
    assert row["ring_sizes"] == [3]


def test_native_affinity_standardization_collapses_carboxylate():
    catalog = _catalog()
    neutral = describe("NC(Cc1ccc(O)cc1)C(=O)O", catalog)
    charged = describe("NC(Cc1ccc(O)cc1)C(=O)[O-]", catalog)
    assert charged["native_standardized_smiles"] == neutral["native_standardized_smiles"]
    assert charged["standardization_changed_identity"]


def test_audit_writes_roles_and_groups_standardized_identities(tmp_path):
    ae = tmp_path / "ae"
    vae = tmp_path / "vae"
    for mode, folder in (("ae", ae), ("vae", vae)):
        folder.mkdir()
        (folder / "run.json").write_text(json.dumps({"name": f"{mode}-native-fresh-pose"}))
        value = {"baseline_smiles": "NC(Cc1ccc(O)cc1)C(=O)O",
                 "unique_smiles": ["NC(Cc1ccc(O)cc1)C(=O)O",
                                   "NC(Cc1ccc(O)cc1)C(=O)[O-]"],
                 "by_arm": {"target": {"chosen": [{"smiles": "NC(Cc1ccc(O)cc1)C(=O)[O-]",
                                                 "start_smiles": "NC(Cc1ccc(O)cc1)C(=O)O",
                                                 "repeat": 0, "start_index": 0}]}}}
        (folder / "frozen_selection.json").write_text(json.dumps(value))
    result = run_audit(ae / "frozen_selection.json", vae / "frozen_selection.json",
                       tmp_path / "runs")
    summary = json.loads((result / "summary.json").read_text())
    assert summary["input_identities"] == 2
    assert summary["native_standardized_identities"] == 1
    rows = json.loads((result / "molecules.json").read_text())
    charged = next(row for row in rows if row["net_formal_charge"] == -1)
    assert charged["selected_by"]["ae"]["memberships"][0]["role"] == "endpoint"
    identities = json.loads((result / "endpoint_identity_audit.json").read_text())
    assert len(identities) == 2
    assert not any(row["effective_changed_from_both"] for row in identities)


def test_failed_native_standardization_is_retained(tmp_path, monkeypatch):
    def reject(smiles):
        raise ValueError("excluded by native standardizer")

    monkeypatch.setattr(chemical_audit, "standardize", reject)
    ae = tmp_path / "ae"
    vae = tmp_path / "vae"
    for mode, folder in (("ae", ae), ("vae", vae)):
        folder.mkdir()
        (folder / "run.json").write_text(json.dumps({"name": f"{mode}-native-fresh-pose"}))
        (folder / "frozen_selection.json").write_text(json.dumps({
            "baseline_smiles": "CC", "unique_smiles": ["CC"],
            "by_arm": {"target": {"chosen": []}}}))
    result = run_audit(ae / "frozen_selection.json", vae / "frozen_selection.json",
                       tmp_path / "runs")
    summary = json.loads((result / "summary.json").read_text())
    assert summary["native_standardization_failures"] == 1
    assert summary["standardization_changes"] == 0
