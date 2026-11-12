"""Retained candidate rankings, identity pooling and exact discovery tensors."""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


_SPEC = importlib.util.spec_from_file_location("stage3_candidates", Path(__file__).parents[1] / "scripts/stage3_candidates.py")
archive = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(archive)


def fixture():
    # Same ethanol identity with three atom mappings, one propane and one
    # disconnected graph that has the highest score but cannot enter the top.
    graphs = {
        "a": ([0, 0, 2], [1, 0, 1], "CCO", .2, True),
        "b": ([0, 2, 0], [0, 1, 1], "CCO", .4, True),
        "c": ([2, 0, 0], [1, 0, 1], "CCO", .4, True),
        "p": ([0, 0, 0], [1, 0, 1], "CCC", .3, True),
        "d": ([0, 0, 2], [0, 0, 0], "C.C.O", .9, False),
        "m": ([0, 2, 0], [1, 0, 1], "COC", .1, True),
    }
    states, candidates, xs, es = [], {}, [], []
    for update, hard_id in enumerate(["a", "a", "b", "p", "b", "d", "c", "m"]):
        elements, bonds, smiles, score, valid = graphs[hard_id]
        mixing = .05 + update * .01
        x = (1 - mixing) * np.eye(6, dtype=np.float32)[elements] + mixing / 6
        e = (1 - mixing) * np.eye(5, dtype=np.float32)[bonds] + mixing / 5
        row = {"update": update, "hard_id": hard_id, "smiles": smiles, "hard_score": score,
               "soft_score": score + .05, "stored_soft_score": score + .03,
               "soft_hard_gap": .05, "valid": valid, "sanitized": True, "connected": valid}
        states.append(row)
        candidates.setdefault(hard_id, {**row, "first_seen": update,
                                       "element_indices": elements, "bond_upper_indices": bonds})
        xs.append(x); es.append(e)
    return {"run_id": "test", "n_atoms": 3, "seed": 17, "model_key": [17, 1]}, states, candidates, xs, es


def test_top_distinct_smiles_selects_highest_scoring_ordered_mapping(tmp_path):
    config, states, candidates, xs, es = fixture()
    result = archive.write_candidate_archive(tmp_path, config, states, candidates, xs, es, limit=2)
    payload = json.loads((tmp_path / "top_valid.json").read_text())
    rows = payload["molecules"]
    assert [(r["rank"], r["smiles"], r["hard_id"], r["hard_score"]) for r in rows] == [
        (1, "CCO", "b", .4), (2, "CCC", "p", .3)]
    assert result["saved_top_valid"] == 2
    assert payload["unique_scored_valid_molecules"] == 3
    assert "2D graph" in payload["description_scope"]
    assert "no native rescoring" in payload["description_scope"]
    assert rows[0]["is_initial_identity"] and not rows[0]["is_initial_state"]
    assert rows[0]["molecular_weight"] == pytest.approx(46.069, abs=.001)
    assert rows[0]["heavy_atoms"] == 3 and rows[0]["formal_charge"] == 0
    assert 0 <= rows[0]["qed"] <= 1 and rows[0]["rdkit_version"]
    assert rows[0]["discovery_soft_score"] == pytest.approx(.45)


def test_raw_order_and_exact_discovery_soft_arrays_are_retained(tmp_path):
    config, states, candidates, xs, es = fixture()
    archive.write_candidate_archive(tmp_path, config, states, candidates, xs, es)
    with np.load(tmp_path / "top_valid.npz", allow_pickle=False) as saved:
        assert saved["source_updates"].tolist() == [2, 3, 7]
        np.testing.assert_array_equal(saved["source_soft_X"][0], xs[2])
        np.testing.assert_array_equal(saved["source_soft_E_upper"][0], es[2])
        assert not np.array_equal(saved["source_soft_X"][0], xs[-1])
        assert saved["hard_X"][0].argmax(-1).tolist() == candidates["b"]["element_indices"]
        assert saved["hard_E_upper"][0].argmax(-1).tolist() == candidates["b"]["bond_upper_indices"]
        assert np.logical_or(saved["hard_X"] == 0, saved["hard_X"] == 1).all()
        assert saved["model_key"].tolist() == [17, 1]
        assert saved["smiles"].tolist() == ["CCO", "CCC", "COC"]
    with (tmp_path / "top_valid.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["artifact_index"] == "0"
    assert rows[0]["source_soft_artifact"] == "top_valid.npz"


def test_all_ordered_graphs_and_visit_statistics_survive_deduplication(tmp_path):
    config, states, candidates, xs, es = fixture()
    archive.write_candidate_archive(tmp_path, config, states, candidates, xs, es)
    all_payload = json.loads((tmp_path / "all_candidates.json").read_text())
    rows = {row["hard_id"]: row for row in all_payload["candidates"]}
    assert len(rows) == 6 and all_payload["observed_state_count"] == 8
    assert rows["a"]["visits"] == 2 and rows["a"]["first_seen"] == 0 and rows["a"]["last_seen"] == 1
    assert not rows["d"]["valid"] and rows["d"]["hard_score"] == .9
    assert rows["d"]["descriptor_status"] == "not_valid_connected"
    assert "molecular_weight" not in rows["d"]
    chosen = json.loads((tmp_path / "top_valid.json").read_text())["molecules"][0]
    assert chosen["visits"] == 2 and chosen["first_seen"] == 2 and chosen["last_seen"] == 4
    assert chosen["chemical_visits"] == 5 and chosen["chemical_mapping_count"] == 3
    assert chosen["chemical_first_seen"] == 0 and chosen["chemical_last_seen"] == 6


@pytest.mark.parametrize("empty_states", [False, True])
def test_no_valid_molecule_writes_loadable_empty_archives(tmp_path, empty_states):
    config, states, candidates, xs, es = fixture()
    if empty_states:
        states, candidates, xs, es = [], {}, [], []
    else:
        states = [{**states[5], "update": 0}]
        candidates = {"d": {**candidates["d"], "first_seen": 0, "update": 0}}
        xs, es = [xs[5]], [es[5]]
    config.pop("model_key")
    result = archive.write_candidate_archive(tmp_path, config, states, candidates, xs, es)
    payload = json.loads((tmp_path / "top_valid.json").read_text())
    assert payload["molecules"] == [] and result["saved_top_valid"] == 0
    assert not payload["model_key_available"]
    with np.load(tmp_path / "top_valid.npz", allow_pickle=False) as saved:
        assert saved["hard_X"].shape == (0, 3, 6)
        assert saved["hard_E_upper"].shape == (0, 3, 5)
        assert saved["source_soft_X"].shape == (0, 3, 6)
        assert saved["model_key"].shape == (0,)
    with (tmp_path / "top_valid.csv").open(newline="") as stream:
        assert list(csv.DictReader(stream)) == []


def test_source_label_mismatch_and_missing_candidate_are_explicit(tmp_path):
    config, states, candidates, xs, es = fixture()
    wrong = {k: dict(v) for k, v in candidates.items()}
    wrong["b"]["element_indices"] = [0, 0, 2]
    with pytest.raises(ValueError, match="discovery probabilities"):
        archive.write_candidate_archive(tmp_path, config, states, wrong, xs, es)
    del candidates["b"]
    with pytest.raises(ValueError, match="Every observed"):
        archive.write_candidate_archive(tmp_path, config, states, candidates, xs, es)


def test_archive_is_repeatable_without_native_calls(tmp_path):
    config, states, candidates, xs, es = fixture()
    archive.write_candidate_archive(tmp_path, config, states, candidates, xs, es)
    first = (tmp_path / "top_valid.json").read_bytes()
    archive.write_candidate_archive(tmp_path, config, states, list(candidates.values()), xs, es)
    assert (tmp_path / "top_valid.json").read_bytes() == first


def test_selected_discovery_gradients_targets_and_missing_masks(tmp_path):
    config, states, candidates, xs, es = fixture()
    gx = np.full((3, 6), .25, np.float32)
    ge = np.full((3, 5), .125, np.float32)
    details = {"b": {"update": 2, "gradient_native_X": gx, "gradient_teacher_E_upper": ge,
                     "teacher_target_X": xs[2], "teacher_target_E_upper": es[2],
                     "temperature": .5, "teacher_weight": .0625, "applied_teacher_weight": .125,
                     "teacher_time": .8}}
    archive.write_candidate_archive(tmp_path, config, states, candidates, xs, es, details_by_hard_id=details)
    with np.load(tmp_path / "top_valid.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["gradient_native_X"][0], gx)
        np.testing.assert_array_equal(saved["gradient_teacher_E_upper"][0], ge)
        np.testing.assert_array_equal(saved["teacher_target_X"][0], xs[2])
        assert saved["gradient_native_X_available"].tolist() == [True, False, False]
        assert saved["gradient_native_E_upper_available"].tolist() == [False, False, False]
        assert saved["temperature_available"].tolist() == [True, False, False]
        assert saved["applied_teacher_weight"].tolist() == [.125, 0, 0]
        assert not saved["gradient_native_X"][1:].any()
        assert all(np.isfinite(saved[key]).all() for key in saved.files if saved[key].dtype.kind in "fiu")
    rows = json.loads((tmp_path / "top_valid.json").read_text())["molecules"]
    assert rows[0]["discovery_gradient_available"] and rows[0]["discovery_teacher_target_available"]
    assert not rows[-1]["discovery_gradient_available"]


def test_wrong_update_or_nonfinite_detail_is_rejected(tmp_path):
    config, states, candidates, xs, es = fixture()
    with pytest.raises(ValueError, match="discovery update"):
        archive.write_candidate_archive(tmp_path, config, states, candidates, xs, es,
                                        details_by_hard_id={"b": {"update": 4, "temperature": .5}})
    with pytest.raises(ValueError, match="finite"):
        archive.write_candidate_archive(tmp_path, config, states, candidates, xs, es,
            details_by_hard_id={"b": {"update": 2, "gradient_native_X": np.full((3, 6), np.nan)}})


def test_optional_descriptor_failure_preserves_valid_ranked_candidates(tmp_path, monkeypatch):
    from rdkit.Chem import Crippen
    config, states, candidates, xs, es = fixture()
    def failed_logp(molecule):
        raise RuntimeError("Synthetic optional descriptor failure")
    monkeypatch.setattr(Crippen, "MolLogP", failed_logp)
    result = archive.write_candidate_archive(tmp_path, config, states, candidates, xs, es)
    rows = json.loads((tmp_path / "top_valid.json").read_text())["molecules"]
    assert result["saved_top_valid"] == 3
    assert rows[0]["smiles"] == "CCO" and rows[0]["hard_score"] == .4
    assert rows[0]["molecular_weight"] == pytest.approx(46.069, abs=.001)
    assert rows[0]["logp"] is None and rows[0]["descriptor_status"] == "partial"
    assert "Synthetic optional descriptor failure" in rows[0]["descriptor_error"]
    assert "logp" in rows[0]["descriptor_errors"]
    with np.load(tmp_path / "top_valid.npz", allow_pickle=False) as saved:
        assert saved["hard_X"].shape[0] == 3


def test_nonfinite_optional_descriptor_is_unavailable_and_graph_integrity_still_required(tmp_path, monkeypatch):
    from rdkit.Chem import Crippen
    config, states, candidates, xs, es = fixture()
    monkeypatch.setattr(Crippen, "MolLogP", lambda molecule: float("nan"))
    archive.write_candidate_archive(tmp_path, config, states, candidates, xs, es)
    chosen = json.loads((tmp_path / "top_valid.json").read_text())["molecules"][0]
    assert chosen["logp"] is None and "nonfinite" in chosen["descriptor_error"]
    candidates["a"]["smiles"] = "CCC"
    with pytest.raises(ValueError, match="canonical SMILES"):
        archive.write_candidate_archive(tmp_path, config, states, candidates, xs, es)
