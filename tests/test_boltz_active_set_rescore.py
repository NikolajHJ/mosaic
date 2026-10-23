"""Fixture-only endpoint/identity checks; no native model or HPC invocation."""

import hashlib
import json

import pytest
import torch
import yaml

from latentmol.boltz.active_set_rescore import (
    ARMS, _paired_results, _parent_target_identity, _retention_result,
    _target_identity, select_endpoints,
)
from latentmol.boltz.native_rescore import POSE_SEEDS
from latentmol.chemistry import encode_smiles, inspect_graph
from latentmol.experiments import file_sha256
from latentmol.multimodel.study import hard_digest
from latentmol.representation import GraphSpec, HardGraphBatch


def _write(path, value):
    path.write_text(json.dumps(value))


def _row(arm, step, smiles):
    spec = GraphSpec(max_nodes=4)
    encoded = encode_smiles(smiles or "CC", spec)
    hard = HardGraphBatch(
        torch.tensor(encoded["atom_types"], dtype=torch.long)[None],
        torch.tensor(encoded["charges"], dtype=torch.long)[None],
        torch.tensor(encoded["bond_types"], dtype=torch.long)[None])
    if smiles is None:
        hard.bonds.zero_()  # Retain a truly disconnected, invalid endpoint.
    inspected = inspect_graph(hard.atoms[0], hard.charges[0], hard.bonds[0], spec)
    return {"candidate_id": f"{arm}-s{step:04d}-d000", "arm": arm,
            "step": step, "draw": 0, "kind": "argmax",
            "graph_spec": spec.to_dict(), "graph_sha256": hard_digest(hard),
            "smiles": inspected.smiles, "failure": inspected.failure_reason,
            "heavy_atoms": int((hard.atoms != 0).sum()),
            "labels": {key: getattr(hard, key)[0].tolist()
                       for key in ("atoms", "charges", "bonds")}}


def _save_rows(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _seal(pilot):
    summary = json.loads((pilot / "summary.json").read_text())
    summary["candidate_file_sha256"] = {
        arm: file_sha256(pilot / arm / "candidates.jsonl") for arm in ARMS}
    _write(pilot / "summary.json", summary)


@pytest.fixture
def pilot(tmp_path):
    config = {"optimizer": {"steps": 2}, "diagnostics": {"hard_samples": 0},
              "objective": {"output": "affinity_probability_binary", "direction": "maximize"}}
    _write(tmp_path / "run.json", {
        "status": "completed", "run_id": "fixture",
        "config": config,
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "graph_spec_sha256": GraphSpec(max_nodes=4).fingerprint()})
    summaries = {}
    for arm in ARMS:
        folder = tmp_path / arm
        folder.mkdir()
        rows = [_row(arm, 0, "CC"), _row(arm, 1, "CC"),
                _row(arm, 2, "CCC" if arm == "active_set" else None)]
        _save_rows(folder / "candidates.jsonl", rows)
        summaries[arm] = {"arm": arm, "steps": 2, "gated_contexts": 1,
                          "endpoint": {"step": 2, "context_sha256": "a" * 64,
                                       "argmax_smiles": rows[-1]["smiles"]}}
        _write(folder / "summary.json", summaries[arm])
        _write(folder / "gates.json", {
            "step-0000": {"passed": True, "native_seed_parity": True,
                          "context_sha256": "a" * 64}})
    _write(tmp_path / "summary.json", {"arms": summaries, "frozen_state_unchanged": True})
    _seal(tmp_path)
    return tmp_path


def test_selection_recomputes_valid_and_invalid_endpoints_and_seals_parent(pilot):
    selected = select_endpoints(pilot)
    assert selected["arms"]["active_set"]["endpoint_changed"] is True
    assert selected["arms"]["fixed_initial_mask"]["endpoint_valid"] is False
    assert selected["arms"]["fixed_initial_mask"]["endpoint"]["failure"] == "disconnected"
    assert "active_set/gates.json" in selected["parent_artifact_sha256"]


def test_completed_legacy_count_bank_keeps_strict_endpoint_and_hash_checks(pilot):
    record = json.loads((pilot / "run.json").read_text())
    record["config"]["contexts"] = {count: {"path": f"/count-{count}", "sha256": "a" * 64}
                                     for count in range(1, 51)}
    record["config_sha256"] = hashlib.sha256(
        json.dumps(record["config"], sort_keys=True).encode()).hexdigest()
    _write(pilot / "run.json", record)
    before = (pilot / "run.json").read_bytes()
    selected = select_endpoints(pilot)
    assert selected["configuration_identity"]["historical_context_keys_reconstructed"] is True
    assert selected["arms"]["fixed_initial_mask"]["endpoint_valid"] is False
    assert (pilot / "run.json").read_bytes() == before
    changed = json.loads(before)
    changed["config"]["contexts"]["10"]["path"] = "/modified"
    _write(pilot / "run.json", changed)
    with pytest.raises(ValueError, match="hash changed"):
        select_endpoints(pilot)


def test_single_declared_arm_does_not_require_unrequested_control(pilot):
    record = json.loads((pilot / 'run.json').read_text())
    record['config']['arms'] = ['fixed_initial_mask']
    record['config_sha256'] = hashlib.sha256(json.dumps(record['config'], sort_keys=True).encode()).hexdigest()
    _write(pilot / 'run.json', record)
    summary = json.loads((pilot / 'summary.json').read_text())
    summary['arms'] = {'fixed_initial_mask': summary['arms']['fixed_initial_mask']}
    summary['candidate_file_sha256'] = {'fixed_initial_mask': summary['candidate_file_sha256']['fixed_initial_mask']}
    _write(pilot / 'summary.json', summary)
    selected = select_endpoints(pilot)
    assert set(selected['arms']) == {'fixed_initial_mask'}
    assert set(_paired_results(selected, [], record['config']['objective'])) == {'fixed_initial_mask'}


def test_changed_candidate_file_is_rejected_before_native_calls(pilot):
    path = pilot / "active_set" / "candidates.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="candidate files changed"):
        select_endpoints(pilot)


@pytest.mark.parametrize("field,value,match", [
    ("smiles", "CO", "SMILES/diagnostic"),
    ("graph_sha256", "0" * 64, "graph hash"),
    ("kind", "categorical", "id/arm/kind"),
    ("heavy_atoms", 99, "SMILES/diagnostic"),
])
def test_sealed_but_semantically_inconsistent_candidates_fail(pilot, field, value, match):
    path = pilot / "active_set" / "candidates.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1][field] = value
    _save_rows(path, rows)
    _seal(pilot)
    with pytest.raises(ValueError, match=match):
        select_endpoints(pilot)


def test_missing_intermediate_step_fails_even_with_endpoints(pilot):
    path = pilot / "active_set" / "candidates.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    _save_rows(path, [rows[0], rows[-1]])
    _seal(pilot)
    with pytest.raises(ValueError, match="candidate step/draw budget"):
        select_endpoints(pilot)


def test_failed_parent_gate_is_not_a_completed_native_candidate_source(pilot):
    _write(pilot / "active_set" / "gates.json", {
        "step-0000": {"passed": False, "native_seed_parity": True}})
    with pytest.raises(ValueError, match="gradient/native-parity gate failed"):
        select_endpoints(pilot)


def test_missing_frozen_state_verification_rejects_parent(pilot):
    path = pilot / "summary.json"
    value = json.loads(path.read_text())
    value["frozen_state_unchanged"] = False
    _write(path, value)
    with pytest.raises(ValueError, match="frozen-state"):
        select_endpoints(pilot)


def test_catalog_header_must_match_annotated_optimization_identity(tmp_path):
    catalog = tmp_path / "catalog.json"
    _write(catalog, {"target_sequence_sha256": "c" * 64, "msa_sha256": "b" * 64})
    parent = {"target_sequence_sha256": "a" * 64, "msa_sha256": "b" * 64,
              "config": {"context_manifest": {"path": str(catalog), "sha256": file_sha256(catalog)}}}
    with pytest.raises(ValueError, match="Catalog header differs"):
        _parent_target_identity(parent)
    parent["target_sequence_sha256"] = "c" * 64
    assert _parent_target_identity(parent) == ("c" * 64, "b" * 64)
    del parent["target_sequence_sha256"]
    with pytest.raises(ValueError, match="run-verified"):
        _parent_target_identity(parent)


def _template(tmp_path):
    msa = tmp_path / "target.a3m"
    msa.write_text(">target\nAA\n")
    template = {"version": 1,
                "sequences": [{"protein": {"id": "A", "sequence": "AA", "msa": "target.a3m"}},
                              {"ligand": {"id": "B", "smiles": "CC"}}],
                "properties": [{"affinity": {"binder": "B"}}]}
    path = tmp_path / "input.yaml"
    return path, template, hashlib.sha256(b"AA").hexdigest(), file_sha256(msa)


def test_relative_template_msa_is_resolved_against_yaml(tmp_path):
    path, template, target, msa = _template(tmp_path)
    path.write_text(yaml.safe_dump(template))
    checked = _target_identity(path, target, msa)
    assert checked["template"]["sequences"][0]["protein"]["msa"] == str((tmp_path / "target.a3m").resolve())


@pytest.mark.parametrize("extra", ["rna", "dna", "ccd", "modifications", "binder"])
def test_template_cannot_add_unverified_chemistry(tmp_path, extra):
    path, template, target, msa = _template(tmp_path)
    if extra in ("rna", "dna"):
        template["sequences"].append({extra: {"id": "X", "sequence": "AAA"}})
    elif extra == "ccd":
        template["sequences"][1]["ligand"]["ccd"] = "ATP"
    elif extra == "modifications":
        template["sequences"][0]["protein"]["modifications"] = []
    else:
        template["properties"][0]["affinity"]["binder"] = "A"
    path.write_text(yaml.safe_dump(template))
    with pytest.raises(ValueError, match="Native template"):
        _target_identity(path, target, msa)


def _identity(smiles="CC", changed=False):
    return {"parse_failure": False, "standardization_failure": None,
            "standardization_changed_identity": changed, "native_standardized_smiles": smiles}


def test_native_identity_change_cannot_count_as_retained_gain(pilot):
    selection = select_endpoints(pilot)
    results = []
    for seed in POSE_SEEDS:
        for smiles, binary, continuous in (("CC", 0.2, 4.0), ("CCC", 0.9, 2.0)):
            result = {"smiles": smiles, "seed": seed, "status": "completed",
                      "affinity": {"affinity_probability_binary": binary, "affinity_pred_value": continuous}}
            results.append(_retention_result(result, _identity("CC", changed=smiles == "CCC")))
    assert results[1]["retention_eligible"] is False
    assert results[1]["retention_failure"]
    comparison = _paired_results(selection, results, {"output": "affinity_pred_value", "direction": "minimize"})
    assert all(row["endpoint_minus_start_binary"] is None
               for row in comparison["active_set"]["paired_seeds"])
    for row in results:
        row["retention_eligible"] = True
    comparison = _paired_results(selection, results, {"output": "affinity_pred_value", "direction": "minimize", "scale": 2})
    pair = comparison["active_set"]["paired_seeds"][0]
    assert pair["endpoint_minus_start_binary"] == pytest.approx(0.7)
    assert pair["endpoint_minus_start_continuous"] == -2
    assert pair["endpoint_minus_start_signed_objective"] == -1


def test_native_failure_is_never_retention_eligible():
    result = _retention_result({"status": "failed", "affinity": None}, _identity())
    assert result["retention_eligible"] is False


def test_invalid_start_keeps_valid_endpoint_but_no_paired_improvement(pilot):
    for arm in ARMS:
        path = pilot / arm / "candidates.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0] = _row(arm, 0, None)
        _save_rows(path, rows)
    _seal(pilot)
    selection = select_endpoints(pilot)
    chosen = selection["arms"]["active_set"]
    assert chosen["endpoint_valid"]
    assert chosen["endpoint_changed"] is None and not chosen["identity_comparable"]
    assert chosen["hard_graph_changed"]
    endpoint = {"affinity_probability_binary": 0.4, "affinity_pred_value": 1.2}
    comparisons = _paired_results(selection, [
        {"smiles": "CCC", "seed": seed, "retention_eligible": True, "affinity": endpoint}
        for seed in POSE_SEEDS], {"output": "affinity_probability_binary", "direction": "maximize"})
    for row in comparisons["active_set"]["paired_seeds"]:
        assert row["start"] is None and row["endpoint"] == endpoint
        assert not row["paired_retention_eligible"]
        assert row["endpoint_minus_start_binary"] is None
