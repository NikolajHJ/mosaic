"""Allocated-HPC fixtures for opt-in training integration; no efficacy claim."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from latentmol import chemistry_continuation as continuation
from latentmol.chemistry_runtime import (CHEMISTRY_HANDOFF_VERSION, FAILURE_GUIDANCE_VERSION,
                                         protocol_config_path, validate_failure_guidance,
                                         validate_handoff_policy, validate_starting_checkpoint)
from latentmol.failure_guidance_training import (FailureAwareRepairTeacher, FailureTeacherBatch,
                                                 build_repair_teacher)
from latentmol.atom_preserving_chemistry import AtomPreservingRepairTeacher
from latentmol.probability_repair import ProbabilityRepairConfig
from latentmol.repair_failure_guidance import (FailureGuidanceConfig, failure_aware_repair_loss,
                                              prepare_failure_guidance)
from latentmol.repair_progress import ProgressSearchConfig
from latentmol.repair_worker_pool import RepairWorkerPool
from latentmol.representation import DecoderOutput
from latentmol.train import capture_rng_state
from test_chemistry_continuation import _captured_gradients, _hard, _tiny_models
from test_repair_failure_guidance import output_for


def policy(depth=None):
    return {"version": FAILURE_GUIDANCE_VERSION, "loss": asdict(FailureGuidanceConfig()),
            "progress_search": None if depth is None else asdict(ProgressSearchConfig(max_edits=depth))}


def test_failure_policy_is_optional_strict_and_independently_copied():
    assert validate_failure_guidance(None) is None
    original = policy(6)
    parsed = validate_failure_guidance(original)
    parsed["loss"]["local_bond_weight"] = 100
    parsed["progress_search"]["max_edits"] = 12
    assert original["loss"]["local_bond_weight"] == .1
    assert original["progress_search"]["max_edits"] == 6
    assert protocol_config_path({"protocol_config": "configs/chemistry_failure_variants/bond6.json"}).endswith("bond6.json")


@pytest.mark.parametrize("change", [
    lambda p: p.update(extra=True),
    lambda p: p["loss"].update(extra=True),
    lambda p: p["loss"].update(local_bond_weight=0),
    lambda p: p["loss"].update(partial_weight=float("nan")),
    lambda p: p["progress_search"].update(max_edits=13),
    lambda p: p["progress_search"].update(allow_atom_edits=1),
    lambda p: p["progress_search"].update(atom_edit_penalty=0),
])
def test_invalid_failure_policies_are_not_defaulted_or_coerced(change):
    value = policy(6); change(value)
    with pytest.raises(ValueError):
        validate_failure_guidance(value)


def test_default_factory_keeps_the_legacy_teacher_and_no_new_objective():
    spec, _ = output_for([9, 9], [(0, 1, "TRIPLE")])
    with build_repair_teacher(spec, {"search": asdict(ProbabilityRepairConfig())}) as teacher:
        assert type(teacher) is AtomPreservingRepairTeacher


def _semantic(batch):
    # Full provenance includes actual elapsed seconds and remains tamper-bound.
    # Scientific target/coverage equality intentionally excludes observation time.
    return {key: value for key, value in batch.prepared.receipt.items() if key != "identity"}


@pytest.mark.parametrize("depth", [None, 2])
def test_teacher_sync_and_async_prepare_exact_original_snapshot_and_same_objective(depth):
    spec, decoded = output_for([9, 9], [(0, 1, "TRIPLE")])
    options = dict(failure_guidance=policy(depth))
    with FailureAwareRepairTeacher(spec, ProbabilityRepairConfig(max_edits=1, checks_per_depth=1), **options) as teacher:
        expected = teacher.build_targets(decoded, candidate_ids=[71], update=7)
        teacher.clear_cache()
        pending = teacher.build_targets_async(decoded, candidate_ids=[71], update=7)
        actual = pending.result(); teacher.drain()
    assert isinstance(expected, FailureTeacherBatch)
    assert expected.base.candidate_ids == actual.base.candidate_ids == (71,)
    assert expected.prepared.base is expected.base
    assert actual.prepared.base is actual.base
    assert _semantic(actual) == _semantic(expected)
    one = failure_aware_repair_loss(decoded, expected.prepared, spec)
    two = failure_aware_repair_loss(decoded, actual.prepared, spec)
    torch.testing.assert_close(one["total"], two["total"], rtol=0, atol=0)
    assert expected.preparation_seconds >= 0 and actual.preparation_seconds >= 0


def test_progress_pool_reuses_the_same_executor_and_returns_original_order():
    from latentmol.repair_progress import search_progress_repair
    spec, decoded = output_for([9, 9], [(0, 1, "TRIPLE")])
    hard = decoded.harden(spec)
    raw = tuple(v[0].numpy().copy() for v in (hard.atoms, hard.charges, hard.bonds))
    probs = tuple(v[0].detach().softmax(-1).numpy().copy() for v in
                  (decoded.atoms, decoded.charges, decoded.symmetric_bond_logits))
    task = (raw, probs, spec, ProgressSearchConfig(max_edits=2))
    expected = search_progress_repair(*task)
    with RepairWorkerPool(1) as pool:
        results = pool.map_progress([task, task])
        executor = pool._executor
        assert pool.map_progress([task]) == [expected]
        assert pool._executor is executor
    assert results == [expected, expected]


def _configured_models():
    model, reference, config, protocol, composition = _tiny_models()
    protocol.update(prior_ratio=2, composition_prior_weight=1.5,
                    failure_guidance=policy(),
                    search=asdict(ProbabilityRepairConfig(max_edits=1, checks_per_depth=1)))
    return model, reference, config, protocol, composition


def test_new_objective_full_and_microbatch_joint_gradients_preserve_ratio_and_denominators():
    model, reference, config, protocol, composition = _configured_models()
    other = deepcopy(model)
    hard = _hard(model.spec)
    original_rng = torch.get_rng_state().clone()
    observed = []
    for candidate, size in ((model, 4), (other, 2)):
        torch.set_rng_state(original_rng)
        generator = torch.Generator().manual_seed(901)
        with build_repair_teacher(candidate.spec, protocol) as teacher:
            metrics = continuation.chemistry_update(candidate, reference, torch.optim.AdamW(candidate.parameters()),
                hard, config, protocol, composition, teacher, {"fresh_prior": generator}, 7, 37, 35,
                do_step=False, microbatch_size=size)
        observed.append((metrics, _captured_gradients(candidate), capture_rng_state({"fresh_prior": generator}, False)))
    left, right = observed
    assert left[0]["real_points"] == right[0]["real_points"] == 3
    assert left[0]["fresh_prior_points"] == right[0]["fresh_prior_points"] == 6
    assert left[0]["repair_coefficient"] == right[0]["repair_coefficient"] == protocol["repair_weight"]
    assert left[0]["total"] == pytest.approx(right[0]["total"], rel=3e-5, abs=3e-6)
    assert continuation.nested_equal(left[2], right[2])
    for name, gradient in left[1].items():
        assert (gradient is None) == (right[1][name] is None), name
        if gradient is not None:
            torch.testing.assert_close(gradient, right[1][name], rtol=3e-4, atol=3e-6)
    for item in observed:
        assert item[0]["failure_guidance_coverage"]["real"]["attempted_rows"] == 3
        assert item[0]["failure_guidance_coverage"]["prior"]["attempted_rows"] == 6


def test_new_sync_and_async_updates_preserve_semantic_chain_gradients_optimizer_and_rng():
    model, reference, config, protocol, composition = _configured_models()
    other = deepcopy(model)
    hard = _hard(model.spec)
    original_rng = torch.get_rng_state().clone()
    observed = []
    for candidate, overlap in ((model, False), (other, True)):
        torch.set_rng_state(original_rng)
        generator = torch.Generator().manual_seed(901)
        optimizer = torch.optim.AdamW(candidate.parameters())
        with build_repair_teacher(candidate.spec, protocol) as teacher:
            metrics = continuation.chemistry_update(candidate, reference, optimizer, hard, config, protocol,
                composition, teacher, {"fresh_prior": generator}, 7, 37, 35, microbatch_size=2, overlap=overlap)
        observed.append((metrics, optimizer.state_dict(), _captured_gradients(candidate),
                         capture_rng_state({"fresh_prior": generator}, False)))
    left, right = observed
    for offset in (1, 2, 3):
        assert continuation.nested_equal(left[offset], right[offset])
    assert continuation.nested_equal(model.state_dict(), other.state_dict())
    assert {key: value for key, value in left[0].items() if key not in ("search_seconds", "prepass_seconds", "teacher_phase_seconds")} == {
        key: value for key, value in right[0].items() if key not in ("search_seconds", "prepass_seconds", "teacher_phase_seconds")}


def test_wrong_legacy_teacher_cannot_silently_ignore_an_enabled_new_loss():
    model, reference, config, protocol, composition = _configured_models()
    with AtomPreservingRepairTeacher(model.spec, ProbabilityRepairConfig()) as teacher:
        with pytest.raises(ValueError, match="matching versioned teacher"):
            continuation.chemistry_update(model, reference, torch.optim.AdamW(model.parameters()), _hard(model.spec),
                config, protocol, composition, teacher, {"fresh_prior": torch.Generator().manual_seed(1)}, 7, 37, 35)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("kind", ["base", "fallback", "progress"])
def test_frozen_cpu_references_accept_live_cpu_or_cuda_logits_without_identity_rewrites(device, kind):
    """Integer/provenance storage stays CPU; VJPs follow the live decoder device."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Allocated CUDA fixture required; CPU pass is not GPU validation")
    spec, output = output_for([9, 9], [(0, 1, "TRIPLE")])
    with AtomPreservingRepairTeacher(spec,
            ProbabilityRepairConfig(max_edits=1, checks_per_depth=32 if kind == "base" else 1)) as teacher:
        base = teacher.build_targets(output, candidate_ids=[71], update=7)
    prepared = prepare_failure_guidance(output, base, spec,
        progress_search=ProgressSearchConfig(max_edits=2) if kind == "progress" else None)
    identity = prepared.identity
    cpu = failure_aware_repair_loss(output, prepared, spec)["total"]
    expected = torch.autograd.grad(cpu, (output.atoms, output.charges, output.bonds))
    live = DecoderOutput(*(v.detach().to(device).clone().requires_grad_() for v in
                           (output.atoms, output.charges, output.bonds)))
    value = failure_aware_repair_loss(live, prepared, spec)["total"]
    actual = torch.autograd.grad(value, (live.atoms, live.charges, live.bonds))
    torch.testing.assert_close(value.detach().cpu(), cpu.detach(), rtol=1e-10, atol=1e-11)
    for left, right in zip(expected, actual):
        assert right.device.type == device and torch.isfinite(right).all()
        torch.testing.assert_close(right.cpu(), left, rtol=1e-9, atol=1e-10)
    assert prepared.identity == identity
    for target in base.targets:
        if target is not None:
            assert target._graph.atoms.device.type == target._raw.atoms.device.type == "cpu"


def _handoff_fixture(tmp_path, monkeypatch):
    def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True)); return sha(path)
    root = tmp_path / "latentmol"; root.mkdir()
    release = root / "releases/source121"; release.mkdir(parents=True)
    algorithm_file = release / "src/latentmol/dummy.py"
    algorithm_file.parent.mkdir(parents=True); algorithm_file.write_text("# original identity\n")
    files = {"dummy.py": sha(algorithm_file)}
    algorithm = {"files": files, "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}
    archive = release.with_suffix(".tar.gz"); archive.write_bytes(b"immutable-source121")
    campaign = root / "runs/source121"; run = campaign / "segments/001/native"; run.mkdir(parents=True)
    _, _, _, original_protocol, _ = _configured_models()
    original_protocol.pop("failure_guidance")
    source = {"release": str(release), "campaign": str(campaign), "archive_sha256": sha(archive),
        "manifest_sha256": "a" * 64, "protocol": original_protocol, "execution": None,
        "parent_checkpoint_sha256": "b" * 64, "parent_epoch": 35, "parent_updates": 100,
        "parent_archive_sha256": "c" * 64, "parent_manifest_sha256": "d" * 64,
        "dataset_sha256": "e" * 64, "split_sha256": "f" * 64, "config_sha256": "1" * 64,
        "training_molecules": 3}
    source_path = root / "submissions/source121-plan.json"; source_sha = write(source_path, source)
    gate_path = campaign / "benchmark_gate.json"
    gate_sha = write(gate_path, {"status": "passed", "plan_sha256": source_sha, "gates": {"original_gate": True}})
    checkpoint_path = run / "epoch_0036.pt"; checkpoint_path.write_bytes(b"native-completed-checkpoint")
    checkpoint_sha = sha(checkpoint_path)
    completion = {"status": "passed", "epoch_complete": True, "plan_sha256": source_sha,
        "completed_epoch": 36, "updates": 101, "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha, "run_dir": str(run)}
    completion_path = campaign / "segment-001-completion.json"; completion_sha = write(completion_path, completion)
    write(run / "run.json", {"status": "completed", "run_id": "native-source121",
        "source_archive_sha256": source["archive_sha256"], "source": {"sha256": source["manifest_sha256"]},
        "config": {"_plan_sha256": source_sha}})
    write(run / "latest_epoch.json", {"epoch": 36, "updates": 101, "checkpoint_sha256": checkpoint_sha,
        "health": {"original_health": True}, "real_points": 3, "fresh_prior_points": 6})
    base = {"train_config": {"unchanged": True}, "graph_spec": {"max_nodes": 4},
            "model_config": {"latent": 4}, "loss_config": {"base": True}}
    runtime, indices = {"torch": "native"}, {"cohort": "same"}
    payload = dict(base, format_version=1, epoch=36, dataset_sha256=source["dataset_sha256"],
        source_archive_sha256=source["archive_sha256"], source_sha256=source["manifest_sha256"], run_id="native-source121",
        continuation={"version": 1, "updates": 101, "epoch_complete": True, "scheduler_state": None,
            "runtime": runtime, "indices": indices, "algorithm": algorithm, "split_sha256": source["split_sha256"],
            "rng": {"loaders": {"fresh_prior": "saved-not-reseeded"}}},
        chemistry_phase={"version": continuation.VERSION, "plan_sha256": source_sha,
            "parent_sha256": source["parent_checkpoint_sha256"], "benchmark_gate_sha256": gate_sha,
            "execution": None, "parent_epoch": 35, "protocol": original_protocol})
    plan = dict(source, work_root=str(root), handoff_version=CHEMISTRY_HANDOFF_VERSION,
        protocol={**original_protocol, "failure_guidance": policy()},
        starting_checkpoint={"path": str(checkpoint_path), "sha256": checkpoint_sha,
            "source_plan": str(source_path), "source_plan_sha256": source_sha,
            "completion": str(completion_path), "completion_sha256": completion_sha,
            "epoch": 36, "updates": 101, "algorithm": algorithm})
    monkeypatch.setattr(continuation, "source_manifest", lambda path: {"sha256": "a" * 64})
    monkeypatch.setattr(torch, "load", lambda *a, **k: deepcopy(payload))
    return plan, base, indices, runtime, payload


def test_completed_chemistry_handoff_keeps_original_reference_and_native_prior_rng(tmp_path, monkeypatch):
    plan, base, indices, runtime, _ = _handoff_fixture(tmp_path, monkeypatch)
    parsed = validate_starting_checkpoint(plan["starting_checkpoint"])
    assert parsed["epoch"] == 36 and plan["parent_epoch"] == 35
    actual = continuation.authenticate_chemistry_handoff(plan, None, base, indices, runtime)
    assert actual["continuation"]["rng"]["loaders"]["fresh_prior"] == "saved-not-reseeded"


def test_legacy_control_can_share_native_handoff_but_cannot_train(tmp_path, monkeypatch):
    plan, base, indices, runtime, _ = _handoff_fixture(tmp_path, monkeypatch)
    plan["protocol"].pop("failure_guidance")
    assert validate_handoff_policy(plan)["epoch"] == 36
    actual = continuation.authenticate_chemistry_handoff(plan, None, base, indices, runtime)
    assert actual["epoch"] == 36
    with pytest.raises(ValueError, match="benchmark-only"):
        continuation.train_phase(plan)


def test_handoff_version_cannot_be_omitted_or_silently_defaulted(tmp_path, monkeypatch):
    plan, base, indices, runtime, _ = _handoff_fixture(tmp_path, monkeypatch)
    plan.pop("handoff_version")
    with pytest.raises(ValueError, match="handoff version"):
        continuation.authenticate_chemistry_handoff(plan, None, base, indices, runtime)


def test_old_native_gate_alone_cannot_execute_new_training_or_final(tmp_path, monkeypatch):
    plan = {"protocol": {"failure_guidance": policy()}, "campaign": str(tmp_path),
            "_benchmark_gate_sha256": "b" * 64}
    (tmp_path / "benchmark_gate.json").write_text(json.dumps({"status": "passed", "gates": {"old": True}}))
    def forbidden(*args, **kwargs):
        raise AssertionError("Unauthenticated readiness reached native model/GPU setup")
    monkeypatch.setattr(continuation, "setup", forbidden)
    with pytest.raises(ValueError, match="failure-readiness SHA"):
        continuation.train_phase(plan)
    with pytest.raises(ValueError, match="failure-readiness SHA"):
        continuation.final_evaluation(plan, tmp_path / "fake.pt", "c" * 64)


def test_new_checkpoint_does_not_write_before_readiness_authentication(tmp_path, monkeypatch):
    plan = {"protocol": {"failure_guidance": policy()}}
    calls = []
    monkeypatch.setattr(continuation, "save_checkpoint", lambda *args: calls.append(args))
    path = tmp_path / "fake.pt"
    with pytest.raises(ValueError, match="failure-readiness SHA"):
        continuation.phase_checkpoint(path, None, None, None, None, 36, None, None, plan)
    assert calls == [] and not path.exists()


def test_readiness_verifier_receives_the_exact_private_binding_without_device_operations(monkeypatch):
    plan = {"protocol": {"failure_guidance": policy()}, "_failure_readiness_sha256": "a" * 64}
    calls = []
    def authenticate(actual, expected):
        calls.append((actual, expected)); return {"status": "passed", "test_only": True}
    monkeypatch.setitem(sys.modules, "latentmol.failure_guidance_selection",
                        SimpleNamespace(authenticate_readiness=authenticate))
    assert continuation.require_failure_readiness(plan)["status"] == "passed"
    assert calls == [(plan, "a" * 64)]
    assert continuation.require_failure_readiness({"protocol": {}}) is None


@pytest.mark.parametrize("part", ["fresh_prior", "updates", "algorithm", "runtime", "protocol"])
def test_handoff_rejects_missing_rng_or_changed_native_lineage(tmp_path, monkeypatch, part):
    plan, base, indices, runtime, payload = _handoff_fixture(tmp_path, monkeypatch)
    if part == "fresh_prior": payload["continuation"]["rng"]["loaders"] = {}
    elif part == "protocol": payload["chemistry_phase"]["protocol"] = {"wrong": True}
    else: payload["continuation"][part] = {"wrong": True} if part in ("algorithm", "runtime") else 102
    with pytest.raises(ValueError, match="native checkpoint lineage"):
        continuation.authenticate_chemistry_handoff(plan, None, base, indices, runtime)
