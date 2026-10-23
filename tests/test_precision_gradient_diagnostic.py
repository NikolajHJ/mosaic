"""HPC-only synthetic checks for candidate scope, identity and outcome accounting."""
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from latentmol.boltz.precision_gradient_diagnostic import (ARITHMETIC, candidate_reducer,
    child_complete, direct_module, planned_endpoints, require_cached_parity, selected_cases)
from test_dbmol_objective import profile


def test_direct_objective_alias_registers_dataclass_without_replacing_production_module():
    import sys
    import latentmol.boltz.dbmol_objective as original
    name = "_fixture_explicit_precision_objective"
    try:
        module = direct_module("dbmol_objective.py", name, register=True)
        assert sys.modules[name] is module
        assert module.DBMolObjectiveConfig.__module__ == name
        assert sys.modules["latentmol.boltz.dbmol_objective"] is original
    finally:
        sys.modules.pop(name, None)


def test_candidate_binding_and_manifest_restore_on_failure(monkeypatch, tmp_path):
    import latentmol.boltz.dbmol_objective as objective
    import latentmol.boltz.scoring as scoring
    original_profile = profile(tmp_path)
    @dataclass(frozen=True)
    class Config:
        dbmol: object
    class Scorer:
        def __init__(self):
            self.dbmol_config = original_profile
            self.recycling_steps = 0
            self.last_contact_selection = {"old": True}
        @property
        def manifest_config(self):
            return self.dbmol_config.record()
        def _reduce(self, *args):
            return "original"
    manifest = SimpleNamespace(adapter_source_sha256="a" * 64, validation_report_sha256=None)
    candidate_manifest = SimpleNamespace(candidate=True)
    scorer = Scorer()
    branch = SimpleNamespace(scorer=scorer, manifest=manifest, prepared=object())
    checked = []
    branch.verify_context_unchanged = lambda: checked.append(scorer.manifest_config)
    monkeypatch.setattr(scoring, "boltz_branch_manifest", lambda *args, **kwargs: candidate_manifest)
    calls = []
    monkeypatch.setattr(objective, "reduce_profile", lambda *args, **kwargs: calls.append((args, kwargs)) or "new")
    config = Config(original_profile)
    with pytest.raises(RuntimeError, match="fixture"):
        with candidate_reducer(branch, config, objective) as candidate:
            assert candidate is not config and config.dbmol is original_profile
            assert candidate.dbmol.objective_arithmetic == ARITHMETIC
            assert branch.manifest is candidate_manifest
            assert scorer._reduce(1, 2, 3, 4) == "new"
            assert calls[-1][0][-1] is candidate.dbmol
            raise RuntimeError("fixture")
    assert scorer.dbmol_config is original_profile and branch.manifest is manifest
    assert scorer._reduce() == "original" and "_reduce" not in scorer.__dict__
    assert scorer.last_contact_selection == {"old": True}
    assert checked and "objective_arithmetic" not in config.dbmol.record()


def _parity_records():
    heads = {key: {"dtype": "torch.float32", "sha256": key} for key in ("distogram", "pae_logits")}
    original = {"values": {"contact_float64": .3, "binding_contact_float64": .7, "repair_nll": 5.},
                "head_tensors": heads, "target_sha256": "target"}
    candidate = {"values": {"contact": .3, "binding": .7, "repair_nll": 5.}, "raw_heads": deepcopy(heads),
                 "dtypes": {"contact": "torch.float64", "binding": "torch.float64", "combined": "torch.float64", "repair_nll": "torch.float32"},
                 "target_sha256": "target"}
    old_gradient = {key: torch.tensor([value]) for key, value in
                    (("contact_float64", .1), ("binding_contact_float64", .2), ("repair_nll", .3))}
    gradients = {key: old_gradient[old].clone() for key, old in
                 (("contact", "contact_float64"), ("binding", "binding_contact_float64"), ("repair_nll", "repair_nll"))}
    return candidate, gradients, original, old_gradient


def test_cached_parity_is_exact_and_does_not_require_combined_repair_widening():
    candidate, gradients, original, old = _parity_records()
    require_cached_parity(candidate, gradients, original, old)
    gradients["contact"] += 1e-6
    with pytest.raises(ValueError, match="cached FP64"):
        require_cached_parity(candidate, gradients, original, old)


@pytest.mark.parametrize("mutation", [
    lambda row: row["raw_heads"]["distogram"].update(sha256="different"),
    lambda row: row["dtypes"].update(contact="torch.float32"),
    lambda row: row["dtypes"].update(repair_nll="torch.float64"),
    lambda row: row.update(target_sha256="different"),
])
def test_cached_parity_rejects_native_head_dtype_and_target_drift(mutation):
    candidate, gradients, original, old = _parity_records()
    mutation(candidate)
    with pytest.raises(ValueError):
        require_cached_parity(candidate, gradients, original, old)


def test_endpoint_union_retains_original_failed_direction_and_enforces_cap():
    from latentmol.boltz.local_fd import EPSILONS
    from latentmol.multimodel.study import tensor_digest
    point = torch.linspace(-.2, .3, 8).reshape(1, 8)
    gradients = {"contact": torch.arange(1., 9.).reshape_as(point), "combined": torch.arange(8., 0., -1).reshape_as(point)}
    old = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0.]])
    directions = {"random_2": (old, {"replayed_sha256": "old"})}
    plans, points = planned_endpoints(point, gradients, directions, seed=1729, maximum=500)
    for epsilon in EPSILONS:
        for sign in (1, -1):
            assert tensor_digest({"z": point + sign * epsilon * old}) in points
    assert all(row["selection_completed_before_endpoint_scores"] for row in plans.values())
    with pytest.raises(ValueError, match="no direction was dropped"):
        planned_endpoints(point, gradients, directions, seed=1729, maximum=1)


def test_failed_scientific_gate_can_be_complete_but_missing_or_runtime_evidence_cannot():
    evidence = {key: True for key in ("diagnostic_completed", "frozen_state_unchanged", "parent_unchanged",
        "baseline_exact_source081_scalar_parity", "original_failed_gradient_exact_source081_match", "engineering_head_cut_complete",
        "objective_context_restored", "candidate_native_seed_parity", "candidate_cached_contact_binding_exact_parity")}
    evidence.update(candidate_gate_passed=False, candidate_gates={key:
        {"status": "failed", "passed": False, "policy": {"version": "local_v1"}} for key in ("contact", "combined")})
    assert child_complete("prior-0", {"status": "completed"}, evidence)
    assert not child_complete("prior-0", {"status": "failed"}, evidence)
    evidence["candidate_gate_passed"] = True
    assert not child_complete("prior-0", {"status": "completed"}, evidence)
    evidence["candidate_gate_passed"] = False
    del evidence["candidate_gates"]["combined"]
    assert not child_complete("prior-0", {"status": "completed"}, evidence)


def test_coordinator_selects_both_original_step20_cases():
    plan = {"cases": [{"id": "prior-0", "step": 20}, {"id": "prior-3", "step": 50}, {"id": "prior-5", "step": 20}]}
    assert [case["id"] for case in selected_cases(plan)] == ["prior-0", "prior-5"]
    plan["cases"][0]["step"] = 21
    with pytest.raises(ValueError, match="step20"):
        selected_cases(plan)
