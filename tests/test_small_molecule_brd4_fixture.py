"""BRD4 preparation and objective selection preserve the explicit whole target."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def _load(folder, name):
    path = Path(__file__).resolve().parents[1] / folder
    sys.path.insert(0, str(path))
    try:
        spec = importlib.util.spec_from_file_location(name, path / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def _text():
    return (Path(__file__).parent / "data/brd4_3mxf/3MXF.pdb").read_text()


def test_complete_deposited_brd4_without_pocket_or_jq1():
    fixture = _load("scripts", "prepare_brd4_fixture").prepare(_text())
    assert len(fixture["protein_sequence"]) == 127
    assert fixture["protein_sequence"][:2] == "SM"
    assert fixture["protein_author_residues"] == list(range(42, 169))
    assert fixture["contact_partner_scope"] == "whole_target"
    assert "pocket_protein_ordinals" not in fixture
    assert "reference_smiles" not in fixture
    assert fixture["smiles"] == "CCCCCC"


def test_missing_brd4_residue_is_rejected():
    text = "\n".join(
        line
        for line in _text().splitlines()
        if not (line.startswith("ATOM  ") and int(line[22:26]) == 42)
    )
    with pytest.raises(ValueError, match="complete"):
        _load("scripts", "prepare_brd4_fixture").prepare(text)


def test_whole_target_factory_does_not_select_protein_subset():
    from mosaic.small_molecule.losses import NativeContactLoss

    module = _load("examples", "protenix_relaxed_gradients")
    context = SimpleNamespace(protein_token_indices=(8, 2, 5))
    config = {"objective": "native_whole_target", "contact_cutoff_angstrom": 8.0}
    objective, pocket = module._contact_objective(config, context, {})
    assert isinstance(objective, NativeContactLoss)
    assert pocket == ()
    with pytest.raises(ValueError, match="selected pocket"):
        module._contact_objective(config, context, {"pocket_protein_ordinals": [0]})
    with pytest.raises(ValueError, match="Unknown"):
        module._contact_objective({**config, "objective": "typo"}, context, {})


def test_existing_pocket_factory_retains_explicit_native_indices():
    from mosaic.small_molecule.losses import DBMolContactLoss

    module = _load("examples", "protenix_relaxed_gradients")
    context = SimpleNamespace(protein_token_indices=(8, 2, 5))
    objective, pocket = module._contact_objective(
        {"contact_cutoff_angstrom": 8.0, "log_epsilon": 1e-8},
        context,
        {"pocket_protein_ordinals": [2, 0]},
    )
    assert isinstance(objective, DBMolContactLoss)
    assert pocket == (5, 8)
