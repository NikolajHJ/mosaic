"""HPC-only controller receipt fixtures; no native model efficacy claim."""

from pathlib import Path

import pytest

import latentmol.repair_guidance.random_study as study
from latentmol.experiments import write_json


def test_trace_keeps_binding_and_repair_separate_from_weight_target_jumps(tmp_path):
    folder = tmp_path / "fixed_initial_mask"
    folder.mkdir()
    for step, weight, binding, total in ((0, .001, -2., -1.9), (1, .1, -3., 7.)):
        write_json(folder / f"metrics-{step:04d}.json", {"step": step, "update_status": "accepted",
            "score": {"loss": total, "binding_loss": binding, "components": {"affinity": binding},
                "repair_guidance": {"weight": weight, "normalized_nll": 100., "raw_nll": 100.,
                    "weighted_loss": 100 * weight, "target_sha256": "a" * 64, "status": "active"}},
            "selection": {"count": 3}, "natural_selection": {"count": 4},
            "argmax_smiles": "CCC", "valid_candidates": 2, "candidate_count": 9, "latent_displacement": .2})
    write_json(folder / "repair-weight-0001.json", {"weighted_objective_jump": 9.9, "is_optimization_improvement": False})
    write_json(folder / "repair-refresh-objective-0001.json", {"weighted_objective_jump": .3})
    summary = {"arms": {"fixed_initial_mask": {"optimization_status": "completed_budget",
        "accepted_updates": 100, "requested_steps": 100, "completed_requested_budget": True}}, "oracle_cost": {}}
    report = study.trajectory_report(tmp_path, summary)
    assert report["trace"][1]["binding_loss"] == -3
    assert report["trace"][1]["total_loss"] == 7
    assert report["trace"][1]["repair_nll"] == 100
    assert report["fixed_counts"] == [3] and report["natural_counts"] == [4, 4]
    assert report["prescribed_candidate_count"] == 18 and report["valid_candidate_count"] == 4
    assert len(report["target_objective_jump_receipts"]) == 1
    assert "net_proxy_improvement" not in report


def _controller(monkeypatch, tmp_path):
    p = {"output_root": str(tmp_path / "outputs"), "name": "fixture-ramp", "gate": {"id": "prior-0"},
         "cases": [{"id": f"prior-{i}", "native_template": {"path": "/unused"}} for i in range(10)],
         "cache": "/unused", "allocation_seconds": 79200, "gate_timeout_seconds": 7200,
         "optimization_timeout_seconds": 6000, "rescore_timeout_seconds": 900, "comparability": "descriptive"}
    identity = {"source_manifest_sha256": "a" * 64, "source_archive_sha256": "b" * 64}
    p.update(identity)
    monkeypatch.setattr(study, "require_hpc", lambda: None)
    monkeypatch.setattr(study, "load_protocol", lambda *args: p)
    monkeypatch.setattr(study, "source_manifest", lambda *args: {"sha256": identity["source_manifest_sha256"]})
    monkeypatch.setenv("LATENTMOL_SOURCE_SHA256", identity["source_archive_sha256"])
    return p


def test_completed_all_fallback_gate_blocks_all_ten_long_runs(monkeypatch, tmp_path):
    _controller(monkeypatch, tmp_path)
    calls = []
    def fallback(*args):
        calls.append(args[2])
        return {"status": "completed", "repair_evidence": {"repair_validation_passed": False}}
    monkeypatch.setattr(study, "_optimize", fallback)
    with pytest.raises(RuntimeError, match="long cohort not launched"):
        study.run(Path("/unused"), "c" * 64)
    assert calls == ["native-gate"]
    report = study.read(next((tmp_path / "outputs/controller").glob("*/campaign.json")))
    assert not report["native_gate_passed"]
    assert all(r["status"] == "not_attempted" and r["reason"] == "blocked_by_native_guidance_gate" for r in report["cases"])


def test_failed_seed_is_retained_and_does_not_replace_or_skip_remaining_seeds(monkeypatch, tmp_path):
    _controller(monkeypatch, tmp_path)
    calls = []
    def native_gate_then_fail(*args):
        calls.append(args[2])
        if args[2] == "native-gate":
            return {"status": "completed", "repair_evidence": {"repair_validation_passed": True}}
        raise RuntimeError("prescribed seed failed")
    monkeypatch.setattr(study, "_optimize", native_gate_then_fail)
    output = study.run(Path("/unused"), "c" * 64)
    report = study.read(output / "campaign.json")
    assert calls == ["native-gate", *[f"prior-{i}" for i in range(10)]]
    assert report["native_gate_passed"] and not report["completed_requested_cohort"]
    assert all(r["status"] == "failed" for r in report["cases"])


def test_allocation_boundary_preserves_unattempted_cohort(monkeypatch, tmp_path):
    _controller(monkeypatch, tmp_path)
    calls = []
    def gate(*args):
        calls.append(args[2])
        return {"status": "completed", "repair_evidence": {"repair_validation_passed": True}}
    monkeypatch.setattr(study, "_optimize", gate)
    times = iter([0., 79000., 79000.])
    monkeypatch.setattr(study.time, "monotonic", lambda: next(times))
    output = study.run(Path("/unused"), "c" * 64)
    report = study.read(output / "campaign.json")
    assert calls == ["native-gate"]
    assert all(r["status"] == "not_attempted" and "allocation_boundary" in r["reason"] for r in report["cases"])
