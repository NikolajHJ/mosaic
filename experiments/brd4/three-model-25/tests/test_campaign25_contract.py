"""Allocated contract tests for explicit scope and no chemistry leakage."""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[4] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


def test_worker_final_identity_accepts_transport_metadata_with_exact_passed_digests():
    from campaign25_contract import verify_frozen_response
    reply = {"status": "passed", "id": 63, "elapsed_seconds": 2.3, "model_digest": "a" * 64, "context_digest": "b" * 64}
    assert verify_frozen_response(reply, "a" * 64, "b" * 64) == {"model_digest": "a" * 64, "context_digest": "b" * 64}


@pytest.mark.parametrize("field,value", [("model_digest", "c" * 64), ("context_digest", "c" * 64), ("status", "failed"), ("status", None)])
def test_worker_final_identity_still_rejects_mutation_or_missing_success(field, value):
    from campaign25_contract import verify_frozen_response
    reply = {"status": "passed", "model_digest": "a" * 64, "context_digest": "b" * 64, field: value}
    with pytest.raises(ValueError, match="identity/status"):
        verify_frozen_response(reply, "a" * 64, "b" * 64)


ROOT = _repository_root(__file__)
spec = importlib.util.spec_from_file_location("campaign25_contract", _layout_path(ROOT / "scripts/campaign25_contract.py"))
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)


@pytest.mark.parametrize("arm", contract.ARMS)
def test_three_inputs_have_matched_size_initialization_and_budget(arm):
    c = json.loads((_layout_path(ROOT / f"examples/configs/brd4_25_{arm}_10000.json")).read_text())
    assert contract.config25(c) == c
    assert c["n_atoms"] * (c["n_atoms"] - 1) // 2 == 300
    assert c["element_js_weight"] == 0


@pytest.mark.parametrize("key,value", [("n_atoms", 6), ("steps", 9999), ("seed", 18), ("initial_step_size", .1), ("element_js_weight", .1), ("chemistry_enabled", True), ("repair_schedule", {})])
def test_no_silent_protocol_changes(key, value):
    c = json.loads((_layout_path(ROOT / "examples/configs/brd4_25_boltz2_10000.json")).read_text())
    c[key] = value
    with pytest.raises(ValueError):
        contract.config25(c)


def test_fresh_fixture_keeps_complete_target_and_25_carbon_chain():
    original = json.loads((_layout_path(ROOT / "tests/data/brd4_3mxf/fixture.json")).read_text())
    current = json.loads((_layout_path(ROOT / "tests/data/brd4_3mxf/fixture_25.json")).read_text())
    for key in ("protein_sequence", "protein_author_residues", "pdb_sha256", "contact_partner_scope"):
        assert current[key] == original[key]
    assert current["smiles"] == "C" * 25
    assert current["bond_assignments"] == [[i, i + 1, "SINGLE"] for i in range(24)]


def test_chemistry_schedule_not_stretched_or_substituted():
    c = json.loads((_layout_path(ROOT / "examples/configs/brd4_25_boltz2_chemistry_10000.json")).read_text())
    changed = deepcopy(c)
    changed["repair_schedule"]["final"] = 5
    with pytest.raises(ValueError):
        contract.config25(changed)


@pytest.mark.parametrize("arm", contract.ARMS)
def test_new_pocket_configs_keep_the_same_budget_and_explicit_no_diffusion(arm):
    c = json.loads((_layout_path(ROOT / f"examples/configs/brd4_jq1_25_{arm}_10000.json")).read_text())
    assert contract.config25(c) == c
    assert c["pocket"] == contract.POCKET_POLICY and c["pocket"]["diffusion_gradient"] is False
    changed = deepcopy(c)
    changed["pocket"]["diffusion_gradient"] = True
    with pytest.raises(ValueError, match="pocket"):
        contract.config25(changed)
