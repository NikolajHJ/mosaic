"""Allocated fixtures for balanced selection, native failures and pair accounting."""

from copy import deepcopy
import json
from pathlib import Path
import time

import pytest

from latentmol.boltz import embedding_comparison_rescore as subject


def members():
    return [{"case_id": "case-0", "arm": "start", "state": "baseline", "smiles": "CCO", "failure": None},
            {"case_id": "case-0", "arm": "relaxed", "state": "terminal", "smiles": "OCC", "failure": None},
            {"case_id": "case-0", "arm": "relaxed", "state": "best", "smiles": "CCC", "failure": None},
            {"case_id": "case-0", "arm": "learned", "state": "terminal", "smiles": None, "failure": "sanitization_failed"},
            {"case_id": "case-0", "arm": "learned", "state": "best", "smiles": "CCC", "failure": None}]


def selection():
    return subject.freeze_selection(members(), expected_case_count=1)


def test_selection_preserves_invalid_members_and_deduplicates_canonical_identity():
    actual = selection()
    assert len(actual["members"]) == 5 and actual["invalid_members"] == 1
    assert len(actual["requests"]) == 4
    assert {row["smiles"] for row in actual["requests"]} == {"CCO", "CCC"}
    assert {(row["smiles"], row["seed"]) for row in actual["requests"]} == {
        (smiles, seed) for smiles in ("CCO", "CCC") for seed in (17, 23)}
    assert actual["members"][3]["failure"] == "sanitization_failed"
    assert actual == selection()


@pytest.mark.parametrize("change", ["missing_role", "duplicate_role", "valid_with_failure", "invalid_no_failure", "bad_case"])
def test_selection_rejects_biased_or_inconsistent_denominators(change):
    rows = members()
    if change == "missing_role":
        rows.pop()
    elif change == "duplicate_role":
        rows[-1]["arm"], rows[-1]["state"] = "relaxed", "terminal"
    elif change == "valid_with_failure":
        rows[0]["failure"] = "bad"
    elif change == "invalid_no_failure":
        rows[3]["failure"] = None
    else:
        rows[0]["case_id"] = "../unsafe"
    with pytest.raises(ValueError):
        subject.freeze_selection(rows, expected_case_count=1)


@pytest.mark.parametrize("smiles", ["C.C", "[Na+]", "[13CH4]", "[CH3]", "C[C@H](O)C(=O)O", "F/C=C/F", "invalid"])
def test_selection_rejects_unsupported_identity_without_repair(smiles):
    rows = members()
    rows[0]["smiles"] = smiles
    with pytest.raises(ValueError):
        subject.freeze_selection(rows, expected_case_count=1)


@pytest.mark.parametrize("seeds", [(17, 17), (17,), (17, True), (-1, 23)])
def test_pose_seed_policy_is_balanced(seeds):
    with pytest.raises(ValueError):
        subject.freeze_selection(members(), expected_case_count=1, pose_seeds=seeds)


def test_changed_native_settings_require_a_new_protocol():
    settings = deepcopy(subject.NATIVE_SETTINGS)
    settings["affinity_recycles"] = 0
    with pytest.raises(ValueError):
        subject.freeze_selection(members(), expected_case_count=1, native_settings=settings)


def test_native_cli_integer_options_cannot_be_equal_float_values():
    settings = deepcopy(subject.NATIVE_SETTINGS)
    settings["structure_steps"] = 100.0
    with pytest.raises(ValueError):
        subject.freeze_selection(members(), expected_case_count=1, native_settings=settings)


def test_complete_thirty_request_bound_is_supported_without_score_ranking():
    rows = []
    index = 1
    for case in range(3):
        for original in members():
            item = {**original, "case_id": f"case-{case}", "smiles": "C" * index, "failure": None}
            index += 1
            rows.append(item)
    actual = subject.freeze_selection(rows)
    assert len(actual["requests"]) == 30
    assert len(actual["members"]) == 15


def results():
    return [{"id": request["id"], "smiles": request["smiles"], "seed": request["seed"],
             "retention_eligible": True, "status": "completed", "affinity": {"affinity_pred_value": 2.0 if request["smiles"] == "CCO" else 1.0,
                                                       "affinity_probability_binary": .1 if request["smiles"] == "CCO" else .2}}
            for request in selection()["requests"]]


def test_native_pairs_compare_same_seeds_and_reuse_shared_identity():
    report = subject.paired_report(selection(), results())
    assert report["all_requests_terminal"] is True
    same = report["cases"][0]
    assert same["identity_changed"] is False
    assert same["mean_delta"]["affinity_pred_value"] == 0
    assert report["cases"][1]["mean_delta"]["affinity_pred_value"] == -1
    assert report["cases"][2]["mean_delta"] is None
    assert report["cases"][2]["failure"] == "sanitization_failed"
    assert report["cases"][3]["mean_delta"] == report["cases"][1]["mean_delta"]


def test_missing_or_standardized_native_outcomes_cannot_be_a_complete_gain_pair():
    rows = results()
    rows[0]["retention_eligible"] = False
    report = subject.paired_report(selection(), rows)
    assert any(row["mean_delta"] is None for row in report["cases"])
    report = subject.paired_report(selection(), rows[:-1])
    assert report["all_requests_terminal"] is False
    with pytest.raises(ValueError):
        subject.paired_report(selection(), [*rows, rows[0]])


def test_native_result_outside_frozen_selection_cannot_enter_report():
    rows = results()
    rows[0]["smiles"] = "CCCC"
    with pytest.raises(ValueError, match="outside"):
        subject.paired_report(selection(), rows)


def test_nonfinite_native_outputs_do_not_become_reported_gains():
    rows = results()
    rows[0]["affinity"]["affinity_pred_value"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        subject.paired_report(selection(), rows)


def target(tmp_path):
    template, msa = tmp_path / "template.yaml", tmp_path / "query.a3m"
    template.write_text("template")
    msa.write_text("query")
    return {"msa_path": str(msa), "msa_sha256": subject.hashlib.sha256(msa.read_bytes()).hexdigest(),
            "template_path": str(template), "template_sha256": subject.hashlib.sha256(template.read_bytes()).hexdigest(),
            "template": {}, "protein_sequence": "P", "target_sequence_sha256": "a" * 64}


def test_load_target_authenticates_template_sequence_and_msa(tmp_path):
    import yaml
    msa = tmp_path / "query.a3m"
    msa.write_text(">query\nAAA\n")
    template = tmp_path / "input.yaml"
    template.write_text(yaml.safe_dump({"version": 1, "sequences": [
        {"protein": {"id": "A", "sequence": "AAA", "msa": str(msa)}},
        {"ligand": {"id": "B", "smiles": "CCC"}}], "properties": [{"affinity": {"binder": "B"}}]}))
    seq_sha = subject.hashlib.sha256(b"AAA").hexdigest()
    msa_sha = subject.hashlib.sha256(msa.read_bytes()).hexdigest()
    template_sha = subject.hashlib.sha256(template.read_bytes()).hexdigest()
    actual = subject.load_target(template, seq_sha, msa_sha, template_sha256=template_sha)
    assert actual["protein_sequence"] == "AAA"
    assert actual["msa_path"] == str(msa.resolve())
    assert actual["template_sha256"] == template_sha
    msa.write_text(">query\nBBB\n")
    with pytest.raises(ValueError, match="protein/MSA"):
        subject.load_target(template, seq_sha, msa_sha, template_sha256=template_sha)


def allow_hpc(monkeypatch):
    monkeypatch.setenv("LSB_JOBID", "fixture-123")
    monkeypatch.setenv("LATENTMOL_SOURCE_SHA256", "a" * 64)
    monkeypatch.setattr(subject, "_verify_cli_defaults", lambda: {
        "path": __file__, "sha256": subject.hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


def test_runtime_budget_records_partial_before_launch(tmp_path, monkeypatch):
    allow_hpc(monkeypatch)
    from latentmol.boltz import fresh_trajectory
    monkeypatch.setattr(fresh_trajectory, "execute_request", lambda *args: pytest.fail("Cannot launch without child headroom"))
    report = subject.run_followup(selection(), target(tmp_path), tmp_path, tmp_path / "out", deadline=time.monotonic() + 10)
    assert report["status"] == "partial_runtime_not_passed"
    assert report["terminal_native_requests"] == 0
    assert report["unattempted_requests"] == 4
    assert (tmp_path / "out" / "selection.json").is_file()


def test_all_native_failures_are_retained_and_not_retried(tmp_path, monkeypatch):
    allow_hpc(monkeypatch)
    from latentmol.boltz import fresh_trajectory
    calls = []
    def failed(request, target, cache, attempt, protocol):
        attempt.mkdir(parents=True)
        calls.append(request)
        return {"id": request["id"], "smiles": request["smiles"], "seed": request["seed"],
                "retention_eligible": False, "affinity": None, "status": "native_failed", "failure": "retained timeout"}
    monkeypatch.setattr(fresh_trajectory, "execute_request", failed)
    report = subject.run_followup(selection(), target(tmp_path), tmp_path, tmp_path / "out", deadline=time.monotonic() + 10000)
    assert report["status"] == "completed" and report["all_requests_terminal"] is True
    assert len(calls) == 4 and report["successful_native_requests"] == 0
    assert report["failed_native_requests"] == 4 and report["identity_ineligible_requests"] == 0
    assert len(json.loads((tmp_path / "out" / "native-results.json").read_text())) == 4


def test_corrupt_native_audit_stops_instead_of_becoming_chemical_failure(tmp_path, monkeypatch):
    allow_hpc(monkeypatch)
    from latentmol.boltz import fresh_trajectory
    def corrupt(*args):
        raise RuntimeError("corrupt audit")
    monkeypatch.setattr(fresh_trajectory, "execute_request", corrupt)
    with pytest.raises(RuntimeError, match="corrupt audit"):
        subject.run_followup(selection(), target(tmp_path), tmp_path, tmp_path / "out", deadline=time.monotonic() + 10000)
    assert json.loads((tmp_path / "out" / "stopped.json").read_text())["status"] == "failed_not_passed"


def test_frozen_selection_and_allocation_binding_are_mandatory(tmp_path, monkeypatch):
    monkeypatch.delenv("LSB_JOBID", raising=False)
    with pytest.raises(RuntimeError):
        subject.run_followup(selection(), target(tmp_path), tmp_path, tmp_path / "out", deadline=time.monotonic() + 10000)
    allow_hpc(monkeypatch)
    changed = selection()
    changed["members"][0]["smiles"] = "CCCC"
    with pytest.raises(ValueError, match="selection identity"):
        subject.run_followup(changed, target(tmp_path), tmp_path, tmp_path / "out", deadline=time.monotonic() + 10000)
