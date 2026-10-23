"""HPC-only engineering fixtures for local FD policy; no pretrained model load."""

from dataclasses import asdict, replace
import json

import pytest
import torch
import yaml

from latentmol.boltz.active_set import ActiveSetConfig
from latentmol.boltz.local_fd import EPSILONS, direction_plan, neighboring_passes, validate_local_fd
from latentmol.config_identity import config_sha256
from latentmol.initialization import LatentStart
from latentmol.multimodel.config import Diagnostics, Objective, Optimizer


def scalar_score(function):
    def score(point):
        value = function(point)
        return {"loss": float(value), "loss_dtype": str(value.dtype)}
    return score


def validate(point, gradient, score, **kwargs):
    return validate_local_fd(point, gradient, score, baseline=score(point)["loss"],
                             seed=1729, rtol=0.05, atol=1e-6, minimum_passes=2, **kwargs)


def test_coarse_curvature_does_not_hide_the_neighboring_local_derivative_plateau():
    point = torch.zeros((1, 4), dtype=torch.float64)
    gradient = torch.ones_like(point)
    score = scalar_score(lambda x: 1 + x.sum() + 100000 * x.sum() ** 3)
    report = validate(point, gradient, score)
    assert report["passed"]
    rows = [row for row in report["finite_differences"] if row["direction"] == "gradient"]
    assert all(row["passed"] for row in rows if row["epsilon"] in (1e-5, 3e-5, 1e-4))
    assert not next(row for row in rows if row["epsilon"] == 0.001)["passed"]
    assert report["direction_results"]["gradient"]["passing_neighbor_runs"]
    assert any(row["forward_backward_slope_difference"] is not None for row in rows)


def test_wrong_backward_is_rejected_at_every_scale_and_not_resampled():
    point = torch.zeros((1, 4), dtype=torch.float64)
    wrong_gradient = -torch.ones_like(point)
    report = validate(point, wrong_gradient, scalar_score(lambda x: 1 + x.sum()))
    assert not report["passed"]
    assert not any(row["passed"] for row in report["finite_differences"])
    assert len(report["direction_results"]) == 4
    assert report["direction_selection"]["selection_completed_before_endpoint_scores"]
    # All selected directions retain the complete grid even after the first
    # failed endpoint pair; no post-FD proposal replaces a failing direction.
    assert len(report["finite_differences"]) == 4 * len(EPSILONS)


def test_nearly_orthogonal_projection_is_rejected_before_the_first_endpoint(monkeypatch):
    proposals = iter([torch.tensor([[0., 1.]], dtype=torch.float64),
                      torch.tensor([[1., 1.]], dtype=torch.float64),
                      torch.tensor([[1., -1.]], dtype=torch.float64),
                      torch.tensor([[2., 1.]], dtype=torch.float64)])
    monkeypatch.setattr(torch, "randn", lambda *args, **kwargs: next(proposals))
    point = torch.zeros((1, 2), dtype=torch.float64)
    gradient = torch.tensor([[1., 0.]], dtype=torch.float64)
    report, endpoint_calls = {}, []
    def score(x):
        # The entire selected plan is fixed and persisted before this call.
        assert len(report["direction_selection"]["selected_labels"]) == 4
        assert report["direction_selection"]["proposals"][0]["accepted"] is False
        endpoint_calls.append(x.clone())
        return {"loss": float(1 + x[0, 0]), "loss_dtype": str(x.dtype)}
    validate_local_fd(point, gradient, score, baseline=1., seed=1729,
                      rtol=0.05, atol=1e-6, minimum_passes=2, report=report)
    assert report["passed"] and len(endpoint_calls) == 90
    assert not report["legacy_projection"]["passed"]
    assert report["direction_selection"]["proposals"][0]["rejection_reason"]


def test_seeded_plan_is_repeatable_and_proposal_receipts_preserve_rejections():
    point = torch.zeros((1, 64))
    gradient = torch.arange(64).reshape_as(point).float() + 1
    first, legacy, record = direction_plan(point, gradient, seed=1729)
    second, other_legacy, other_record = direction_plan(point, gradient, seed=1729)
    assert record == other_record and torch.equal(legacy, other_legacy)
    assert all(torch.equal(first[key], second[key]) for key in first)
    assert len(first) == 4 and len(record["proposals"]) <= 64
    assert all(abs(row["analytic_projection_float64"]) >= row["minimum_absolute_projection"]
               for row in record["proposals"] if row["accepted"])


def test_bounded_exhausted_pool_fails_before_any_model_endpoint(monkeypatch):
    monkeypatch.setattr(torch, "randn", lambda *args, **kwargs: torch.tensor([[0., 1.]]))
    point, gradient = torch.zeros(1, 2), torch.tensor([[1., 0.]])
    def forbidden_score(point):
        raise AssertionError("An exhausted analytic-only direction pool must not score endpoints")
    report = validate_local_fd(point, gradient, forbidden_score, baseline=1., seed=1729,
                               rtol=0.05, atol=1e-6, minimum_passes=2)
    assert not report["passed"] and report["status"] == "failed_direction_pool"
    assert len(report["direction_selection"]["proposals"]) == 64


def test_sparse_passes_are_not_a_neighboring_plateau():
    rows = [{"epsilon": epsilon, "passed": epsilon in (1e-5, 1e-4, 0.001)} for epsilon in EPSILONS]
    assert not neighboring_passes(rows, minimum=2)["passed"]


@pytest.mark.parametrize("analytic", [1e-8, 1.01e-6])
def test_all_zero_fd_cannot_validate_an_uninformative_derivative_via_atol(analytic):
    point = torch.zeros((1, 1), dtype=torch.float64)
    report = validate(point, torch.full_like(point, analytic), scalar_score(lambda x: torch.ones((), dtype=x.dtype)))
    assert not report["passed"]
    assert all(row["passed_tolerance"] for row in report["finite_differences"])
    assert not any(row["informative"] for row in report["finite_differences"])


def old_config(tmp_path):
    asset = {"path": str(tmp_path / "asset.pt"), "sha256": "0" * 64}
    return ActiveSetConfig(name="historical", mode="vae", decoder=asset, affinity_checkpoint=asset,
                           context_manifest=None, contexts={6: asset, 13: asset}, anchor_count=13,
                           output_root=tmp_path / "out", device="cuda", temperature=1.0, threshold=0.5,
                           recycling_steps=0, objective=Objective(output="affinity_probability_binary", direction="maximize"),
                           optimizer=Optimizer(), diagnostics=Diagnostics(), allow_lossy_charge=True)


def test_absent_policy_keeps_the_exact_old_record_fields_and_config_identity(tmp_path):
    config = old_config(tmp_path)
    asset = config.decoder
    # This is the explicit pre-local-policy record contract, including untouched
    # shared Diagnostics defaults. No new key may silently change its digest.
    expected = {"name": "historical", "mode": "vae", "decoder": asset, "affinity_checkpoint": asset,
                "context_manifest": None, "contexts": {6: asset, 13: asset}, "anchor_count": 13,
                "output_root": str(tmp_path / "out"), "device": "cuda", "temperature": 1.0,
                "threshold": 0.5, "recycling_steps": 0, "objective": asdict(config.objective),
                "optimizer": asdict(config.optimizer), "diagnostics": asdict(Diagnostics()),
                "allow_lossy_charge": True, "context_identity": None, "start": asdict(LatentStart()),
                "arms": ("active_set", "fixed_initial_mask"), "element_diagnostics": False}
    assert config.record() == expected
    assert config_sha256(config.record()) == config_sha256(expected)
    opted = replace(config, fd_policy="local_v1")
    assert opted.record()["fd_policy"] == "local_v1"
    assert config_sha256(opted.record()) != config_sha256(expected)


def test_config_loader_requires_an_explicit_known_policy(tmp_path):
    raw = {"version": 1, "name": "fixture", "mode": "vae",
           "decoder": {"path": "decoder.pt", "sha256": "0" * 64},
           "affinity_checkpoint": {"path": "scorer.pt", "sha256": "0" * 64},
           "contexts": {13: {"path": "context.pt", "sha256": "0" * 64}},
           "anchor_count": 13, "output_root": "outputs"}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    assert "fd_policy" not in ActiveSetConfig.load(path).record()
    raw["fd_policy"] = "local_v1"
    path.write_text(yaml.safe_dump(raw))
    assert ActiveSetConfig.load(path).fd_policy == "local_v1"
    raw["fd_policy"] = "guess"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="fd_policy"):
        ActiveSetConfig.load(path)


@pytest.mark.parametrize("wrong_backward", [False, True])
def test_opt_in_active_set_gate_preserves_native_boundary_and_rejects_wrong_backward(tmp_path, wrong_backward):
    from latentmol.boltz.active_set import ContextBank, make_branch, select_slots, validate_branch
    from latentmol.multimodel.engine import Counters
    from test_boltz_active_set_runtime import _config, _Decoder, _Scorer, SPEC
    config = replace(_config(tmp_path), fd_policy="local_v1")
    decoder, scorer = _Decoder(), _Scorer(bad_derivative=wrong_backward)
    point = torch.zeros(1, 1, dtype=torch.float64)
    selected = select_slots(decoder.decode(point), temperature=1, threshold=.5)
    branch = make_branch(scorer, ContextBank(config, SPEC), selected, config, "c" * 64)
    report_path = tmp_path / "local-gate.json"
    if wrong_backward:
        with pytest.raises(ValueError, match="local_v1 finite-difference gate failed"):
            validate_branch(decoder, point, branch, config, Counters(), report_path=report_path)
        saved = json.loads(report_path.read_text())
        assert saved["status"] == "failed" and saved["passed"] is False
        assert saved["native_seed_parity"] and saved["local_fd_validation"]["passed"] is False
    else:
        report = validate_branch(decoder, point, branch, config, Counters(), report_path=report_path)
        assert report["passed"] and report["native_seed_parity"]
        assert report["local_fd_validation"]["policy"]["version"] == "local_v1"
        assert len(report["finite_differences"]) == 4 * len(EPSILONS)
        assert json.loads(report_path.read_text())["passed"] is True
        assert report["measured"]["loss_dtype"] == "torch.float64"
