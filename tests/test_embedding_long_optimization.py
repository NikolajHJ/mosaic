"""Guarded scheduling/resume/native-boundary fixtures; allocated HPC only."""

import copy
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from latentmol.multimodel.study import tensor_digest
from latentmol.optimization import NonfiniteObjective
from latentmol.boltz.embedding_long_optimization import (
    EligibilityRejected, GuardedEmbeddingObjective, file_sha256, run_guarded_optimization)
from test_embedding_comparison import native_objective


class ToyGuard:
    """Known scalar and domain isolate real shared step selection/resume logic."""

    def __init__(self, *, mode="quadratic", reject=None, consume_rng=False):
        self.mode, self.reject, self.consume_rng = mode, reject, consume_rng
        self.base = SimpleNamespace(calls=dict(forward=0, backward=0, hard=0))
        self.calls = dict(assess_attempted=0, assess_completed=0, rejected=0)
        self.context_sha256 = "c" * 64
        self.comparison_graph_policy_sha256 = "d" * 64
        self.guard_record = dict(mse_caps={"head": 1.}, strict_conditioned_validity_required=True)

    def assert_unchanged(self, *, full=False):
        return dict(full_state_check=full, frozen_state_unchanged=True,
            context_sha256=self.context_sha256, frozen_state_sha256={"toy": "e" * 64})

    def output(self, z, require_gradient=False):
        loss = 1. if self.mode == "flat" else float((z - 1).square().sum())
        gradient = torch.zeros_like(z) if self.mode == "flat" else 2 * (z - 1)
        if self.mode == "wrong-gradient":
            gradient = -gradient
        return dict(loss=loss, loss_dtype=str(z.dtype), scores={"dbmol_total_loss": loss},
            gradient=gradient if require_gradient else None,
            context_sha256=self.context_sha256,
            comparison_graph_policy_sha256=self.comparison_graph_policy_sha256)

    def assess(self, z, *, include_tensors=False):
        self.calls["assess_attempted"] += 1
        label = int(float(z.sum()) >= .7)
        eligible = self.reject is None or not self.reject(z)
        hard = dict(atoms=torch.tensor([[label + 1]]), charges=torch.zeros(1, 1, dtype=torch.long),
            bonds=torch.zeros(1, 1, 1, dtype=torch.long))
        record = dict(eligible=eligible, rejection_reasons=[] if eligible else ["fixture_domain"],
            context_sha256=self.context_sha256, latent_sha256=tensor_digest({"z": z}),
            hard_graph_sha256=tensor_digest(hard),
            feature_audit={"head/mse": float(z.square().mean())},
            classifications={key: dict(strict_teacher_valid=eligible, raw_chemical_valid=eligible,
                                      reason=None if eligible else "sanitization")
                             for key in ("conditioned", "natural")},
            canonical_identities={"conditioned": str(label), "natural": str(label)})
        tensors = dict(z=z.detach().cpu().clone(), conditioned=hard, natural=copy.deepcopy(hard))
        if include_tensors:
            tensors["relaxed"] = {key: value.double() for key, value in hard.items()}
        self.calls["assess_completed"] += 1
        return dict(record=record, tensors=tensors)

    def evaluate(self, z, *, require_gradient=False, assessment=None):
        assessment = self.assess(z) if assessment is None else assessment
        if not assessment["record"]["eligible"]:
            self.calls["rejected"] += 1
            raise EligibilityRejected(assessment["record"])
        if self.mode in ("nonfinite-trial", "native-error") and not require_gradient and z.count_nonzero():
            if self.mode == "native-error":
                raise RuntimeError("native fixture failure")
            raise NonfiniteObjective("native numeric failure")
        if self.consume_rng:
            random.random()
            np.random.random()
            torch.rand(1)
        self.base.calls["forward"] += 1
        self.base.calls["backward"] += int(require_gradient)
        return self.output(z, require_gradient)

    def hard_score(self, assessment):
        self.base.calls["hard"] += 1
        label = int(assessment["tensors"]["conditioned"]["atoms"].item())
        return {"dbmol_total_loss": float(11 - label)}

    def observe(self, z, assessment, hard_scores):
        learned = self.evaluate(z, assessment=assessment)
        return dict(**assessment["record"], scores=dict(learned=learned["scores"],
            relaxed=self.output(z)["scores"], hard_fixed_context=hard_scores))


def binding(objective, start):
    observed = objective.output(start, True)
    result = {key: "a" * 64 for key in ("source_sha256", "plan_sha256", "vae_checkpoint_sha256",
        "bridge_checkpoint_sha256", "calibration_sha256", "gate_sha256")}
    return dict(**result, context_sha256=objective.context_sha256,
        comparison_graph_policy_sha256=objective.comparison_graph_policy_sha256,
        start_sha256=tensor_digest({"z": start}), baseline_scores=observed["scores"],
        gradient_sha256=tensor_digest({"gradient": observed["gradient"]}))


def run(output, objective=None, **kwargs):
    objective = objective or ToyGuard()
    start = torch.zeros(1, 1, dtype=torch.float64)
    return run_guarded_optimization(start, objective, output, case_id="case-001",
        binding=kwargs.pop("binding", binding(objective, start)),
        target_updates=kwargs.pop("target_updates", 6), segment_updates=kwargs.pop("segment_updates", 6),
        learning_rate=kwargs.pop("learning_rate", .25), max_gradient_norm=1., trust_radius=3.,
        observation_every=kwargs.pop("observation_every", 2), **kwargs)


def load_state(result):
    reference = result["current_state"]
    assert file_sha256(reference["path"]) == reference["sha256"]
    return torch.load(reference["path"], weights_only=True)


def test_domain_rejections_happen_before_native_and_are_explicit(tmp_path):
    objective = ToyGuard(reject=lambda z: bool(z.count_nonzero()))
    result = run(tmp_path / "run", objective, target_updates=2, segment_updates=2)
    assert result["counts"]["all_ineligible_no_op"] == result["counts"]["no_op"] == 2
    assert result["counts"]["applied"] == result["counts"]["improving"] == 0
    records = [json.loads(line) for line in (tmp_path / "run/segments/from-000000/calls.jsonl").read_text().splitlines()]
    rejected = [row for row in records if row["status"] == "eligibility_rejected_before_native"]
    assert len(rejected) == 32 and all(row["rejected_graphs"] for row in rejected)
    assert not [row for row in records if row.get("trial") is not None and row["status"] == "completed"]
    iterations = [json.loads(line) for line in (tmp_path / "run/segments/from-000000/updates.jsonl").read_text().splitlines()]
    assert all(row["step"]["eligibility_rejections"] == 16 and row["step"]["native_nonfinite_trials"] == 0 for row in iterations)
    assert torch.equal(load_state(result)["z"], torch.zeros(1, 1, dtype=torch.float64))


def test_all_eligible_native_nonfinite_trials_fail_instead_of_becoming_guard_noop(tmp_path):
    with pytest.raises(NonfiniteObjective, match="No finite candidate"):
        run(tmp_path / "run", ToyGuard(mode="nonfinite-trial"))
    failure = json.loads((tmp_path / "run/failed.json").read_text())
    assert failure["last_committed"]["path"].endswith("state-000000.pt")
    assert failure["error_type"] == "NonfiniteObjective"


def test_noop_stall_stops_before_target_and_cannot_automatically_resume(tmp_path):
    result = run(tmp_path / "run", ToyGuard(mode="flat"), target_updates=10,
        segment_updates=10, max_consecutive_noop=3)
    assert result["status"] == "scientific_stall_target_incomplete"
    assert result["completed_update"] == result["counts"]["consecutive_noop"] == 3
    with pytest.raises(ValueError, match="Scientific stall"):
        run(tmp_path / "run", ToyGuard(mode="flat"), target_updates=10,
            max_consecutive_noop=3, resume=result["current_state"])


def test_uphill_fallback_survives_among_eligible_trials(tmp_path):
    result = run(tmp_path / "run", ToyGuard(mode="wrong-gradient"), target_updates=1, segment_updates=1)
    assert result["counts"]["forced_fallback"] == result["counts"]["applied"] == 1
    assert result["counts"]["improving"] == result["counts"]["no_op"] == 0
    row = json.loads((tmp_path / "run/segments/from-000000/updates.jsonl").read_text())
    assert row["step"]["accepted_trial"] == 15


def test_hard_cache_and_best_valid_use_slot_graph_identity_and_common_loss(tmp_path):
    objective = ToyGuard()
    result = run(tmp_path / "run", objective)
    assert objective.base.calls["hard"] == 2
    assert result["best_valid"]["update"] == 3 and result["best_valid"]["hard_loss"] == 9.
    assert len(list((tmp_path / "run/hard-cache").glob("*.json"))) == 2
    assert set(p.name for p in (tmp_path / "run/observations").glob("*.json")) == {
        "update-000000.json", "update-000002.json", "update-000004.json", "update-000006.json"}
    for path in (tmp_path / "run/observations").glob("*.json"):
        row = json.loads(path.read_text())
        assert file_sha256(row["tensors"]["path"]) == row["tensors"]["sha256"]
        assert set(torch.load(row["tensors"]["path"], weights_only=True)) == {"z", "conditioned", "natural", "relaxed"}


def test_every_atomic_update_has_sha_commit_receipt_and_actual_current_latent(tmp_path):
    result = run(tmp_path / "run", target_updates=3, segment_updates=3)
    commits = [json.loads(line) for line in (tmp_path / "run/segments/from-000000/commits.jsonl").read_text().splitlines()]
    assert [row["update"] for row in commits] == [0, 1, 2, 3]
    assert all(file_sha256(row["current_state"]["path"]) == row["current_state"]["sha256"] for row in commits)
    state = load_state(result)
    assert state["update"] == state["counts"]["iterations"] == 3
    torch.testing.assert_close(state["z"], torch.tensor([[.75]], dtype=torch.float64), rtol=0, atol=0)
    assert state["baseline_observation"]["update"] == 0 and state["last_observation"]["update"] == 3
    for reference in (state["baseline_observation"], state["last_observation"]):
        assert file_sha256(reference["record"]["path"]) == reference["record"]["sha256"]
        observed = json.loads(Path(reference["record"]["path"]).read_text())
        assert observed["tensors"] == reference["tensors"]
        assert file_sha256(reference["tensors"]["path"]) == reference["tensors"]["sha256"]


def test_split_resume_matches_uninterrupted_latent_rng_and_counts_exactly(tmp_path):
    def seed():
        random.seed(73)
        np.random.seed(73)
        torch.manual_seed(73)
    seed()
    whole = run(tmp_path / "whole", ToyGuard(consume_rng=True))
    seed()
    first = run(tmp_path / "split", ToyGuard(consume_rng=True), segment_updates=2)
    assert first["status"] == "segment_complete_target_pending"
    second = run(tmp_path / "split", ToyGuard(consume_rng=True), resume=first["current_state"], segment_updates=4)
    left, right = load_state(whole), load_state(second)
    assert torch.equal(left["z"], right["z"]) and torch.equal(left["z0"], right["z0"])
    assert left["counts"] == right["counts"] and left["model_calls"] == right["model_calls"]
    assert left["guard_calls"] == right["guard_calls"]
    assert left["rng"]["python"] == right["rng"]["python"] and left["rng"]["numpy"] == right["rng"]["numpy"]
    assert torch.equal(left["rng"]["torch"], right["rng"]["torch"])
    assert left["best_valid"]["update"] == right["best_valid"]["update"]


def test_resume_preserves_original_global_radius_not_segment_start(tmp_path):
    first = run(tmp_path / "run", segment_updates=2, learning_rate=400.)
    second = run(tmp_path / "run", resume=first["current_state"], segment_updates=4, learning_rate=400.)
    state = load_state(second)
    assert torch.equal(state["z0"], torch.zeros_like(state["z"]))
    assert float((state["z"] - state["z0"]).norm()) <= 3. + 1e-5


def test_corrupt_cached_hard_score_cannot_select_a_best_valid_state(tmp_path):
    first = run(tmp_path / "run", ToyGuard(mode="flat"), segment_updates=2)
    cache = next((tmp_path / "run/hard-cache").glob("*.json"))
    value = json.loads(cache.read_text())
    value["scores"]["dbmol_total_loss"] = -999.
    cache.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Hard-cache bytes"):
        run(tmp_path / "run", ToyGuard(mode="flat"), resume=first["current_state"])


def test_corrupt_cache_receipt_chain_is_rejected_on_resume(tmp_path):
    first = run(tmp_path / "run", segment_updates=2)
    index = tmp_path / "run/hard-cache/index.jsonl"
    row = json.loads(index.read_text())
    row["sha256"] = "f" * 64
    index.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="receipt chain differs"):
        run(tmp_path / "run", resume=first["current_state"])


@pytest.mark.parametrize("field", ["source_sha256", "plan_sha256", "bridge_checkpoint_sha256", "calibration_sha256", "gate_sha256"])
def test_resume_rejects_changed_authenticated_input(field, tmp_path):
    first = run(tmp_path / "run", segment_updates=2)
    objective = ToyGuard()
    changed = binding(objective, torch.zeros(1, 1, dtype=torch.float64))
    changed[field] = "b" * 64
    with pytest.raises(ValueError, match="Resume source"):
        run(tmp_path / "run", objective, binding=changed, resume=first["current_state"])


def test_exact_start_score_or_gradient_gate_mismatch_prevents_updates(tmp_path):
    objective = ToyGuard()
    bad = binding(objective, torch.zeros(1, 1, dtype=torch.float64))
    bad["gradient_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="Exact learned score/gradient"):
        run(tmp_path / "run", objective, binding=bad)
    assert not list((tmp_path / "run/states").glob("*.pt"))


def test_deadline_returns_committed_baseline_without_claiming_updates(tmp_path):
    result = run(tmp_path / "run", deadline=time.monotonic() - 1)
    assert result["status"] == "segment_complete_target_pending" and result["stopped_for_deadline"]
    assert result["completed_update"] == result["counts"]["iterations"] == 0
    assert load_state(result)["gate_binding_verified"]


def test_unexpected_native_failure_keeps_last_atomic_state_and_failure_receipt(tmp_path):
    with pytest.raises(RuntimeError, match="native fixture failure"):
        run(tmp_path / "run", ToyGuard(mode="native-error"))
    failure = json.loads((tmp_path / "run/failed.json").read_text())
    assert failure["error_type"] == "RuntimeError"
    assert torch.load(failure["last_committed"]["path"], weights_only=True)["update"] == 0


def test_real_guard_rejects_before_native_and_saves_actual_relaxed_probabilities():
    base, z = native_objective()
    caps = {descriptor.key: 1e-30 for descriptor in base.bank.descriptors}
    guard = GuardedEmbeddingObjective(base, caps)
    assessment = guard.assess(z, include_tensors=True)
    assert not assessment["record"]["eligible"]
    assert any(reason.startswith("mse_cap:") for reason in assessment["record"]["rejection_reasons"])
    assert base.calls["boltz_forward_attempted"] == 0
    with pytest.raises(EligibilityRejected):
        guard.evaluate(z, assessment=assessment)
    assert base.calls["boltz_forward_attempted"] == 0
    graph = assessment["tensors"]["relaxed"]
    assert set(graph) == {"atoms", "charges", "bonds"}
    assert all(bool(value.isfinite().all()) and value.is_floating_point() for value in graph.values())
    assert not any(value.requires_grad for value in graph.values())
    from latentmol.boltz.embedding_comparison_policy import comparison_graph
    expected = comparison_graph(base.vae.decode(z), base.prepared.active,
        base.prepared.spec, policy=base.comparison_graph_policy)
    for field in graph:
        assert torch.equal(graph[field], getattr(expected, field).cpu())
    assert guard.assert_unchanged(full=True)["frozen_state_unchanged"]


@pytest.mark.parametrize("change", ["missing", "extra", "negative", "nan"])
def test_real_guard_requires_complete_finite_calibration(change):
    base, _ = native_objective()
    caps = {descriptor.key: 1. for descriptor in base.bank.descriptors}
    key = next(iter(caps))
    if change == "missing":
        del caps[key]
    elif change == "extra":
        caps["unknown.head"] = 1.
    else:
        caps[key] = -1. if change == "negative" else float("nan")
    with pytest.raises(ValueError, match="caps"):
        GuardedEmbeddingObjective(base, caps)
