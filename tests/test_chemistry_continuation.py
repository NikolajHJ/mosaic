"""Exact logical-batch chemistry-phase fixtures; execute only on DTU HPC.

These tests use actual small triangle encoder/decoder modules, not a replacement
scorer. The fixed-target fixture isolates streamed loss/gradient bookkeeping;
the atom-preserving teacher and its chemistry oracle have separate fixtures.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest
import torch

from latentmol.atom_preserving_chemistry import AtomPreservingRepairTeacher, RepairTargetBatch
from latentmol.atom_preserving_chemistry import composition_statistics, population_js, repair_loss
from latentmol.chemistry import encode_smiles
from latentmol.chemistry_continuation import (
    VERSION, ParentCheckpointMismatch, authenticate_parent_metadata, chemistry_update,
    execution_settings, hard_slice, load_parent_config, metadata_differences, nested_equal, read_plan,
    reconstruction_chunk, reconstruction_denominators,
)
from latentmol.losses import LossConfig, latent_kl, reconstruction_losses
from latentmol.models import GraphAutoencoder, ModelConfig
from latentmol.probability_repair import ProbabilityRepairConfig
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch
from latentmol.train import TrainConfig, capture_rng_state, restore_rng_state


def test_execution_stanza_is_separate_from_unchanged_scientific_protocol(monkeypatch):
    execution = {"backend": "optimized", "workers": 12, "torch_threads": 4, "gpu_cpus": 16}
    protocol = {"prior_ratio": 3, "repair_weight": .5}
    plan = {"execution": deepcopy(execution), "protocol": deepcopy(protocol)}
    monkeypatch.setenv("LSB_DJOB_NUMPROC", "16")
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.setenv("MKL_NUM_THREADS", "4")
    monkeypatch.setattr(torch, "get_num_threads", lambda: 4)
    assert execution_settings(plan, allocated=True) == execution
    assert plan["protocol"] == protocol
    assert execution_settings({"protocol": protocol}, allocated=True) is None
    observed = execution_settings(plan)
    observed["workers"] = 0
    assert plan["execution"] == execution


@pytest.mark.parametrize("seconds", [22 * 3600, 71 * 3600])
def test_native_plan_reader_accepts_bounded_runtime_before_authenticating_artifacts(tmp_path, monkeypatch, seconds):
    """A longer sealed plan passes duration validation, never artifact identity.

    Deliberately stop at the invalid-work-root guard: accepting time cannot
    make an unsigned or nonexistent native release ready for execution.
    """
    import latentmol.chemistry_continuation as module
    protocol = json.loads((Path(__file__).resolve().parents[1] / "configs/chemistry_triangle_1to3.json").read_text())
    protocol["segment_wall_seconds"] = seconds
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"protocol": protocol, "work_root": "/not-the-authorized-root"}))
    calls = []
    monkeypatch.setattr(module, "authenticated", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="Require declared DTU work3 root"):
        read_plan(path, "fixture-plan")
    assert calls == [(path, "fixture-plan")]


@pytest.mark.parametrize("field,invalid", [("segment_wall_seconds", True), ("segment_wall_seconds", 255601),
    ("segment_wall_seconds", 255600.0), ("benchmark_wall_seconds", 6301), ("benchmark_wall_seconds", "6300")])
def test_native_plan_reader_rejects_invalid_duration_before_artifact_access(tmp_path, monkeypatch, field, invalid):
    import latentmol.chemistry_continuation as module
    protocol = json.loads((Path(__file__).resolve().parents[1] / "configs/chemistry_triangle_1to3.json").read_text())
    protocol[field] = invalid
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"protocol": protocol}))
    monkeypatch.setattr(module, "authenticated", lambda *args: None)
    with pytest.raises(ValueError, match=field + " requires integer seconds"):
        read_plan(path, "fixture-plan")


@pytest.mark.parametrize("ratio,microbatch", [(3, 128), (3, 256), (2, 128), (2, 256)])
def test_variant_plan_reader_requires_explicit_ratio_version_and_composition_policy_before_artifacts(tmp_path, monkeypatch, ratio, microbatch):
    import latentmol.chemistry_continuation as module
    protocol = json.loads((Path(__file__).resolve().parents[1] / "configs/chemistry_triangle_1to3.json").read_text())
    protocol.update(prior_ratio=ratio, microbatch_size=microbatch, composition_prior_weight=3 / ratio,
                    version="joint-atom-preserving-chemistry-v2" if ratio == 2 else VERSION)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"protocol": protocol, "work_root": "/not-the-authorized-root"}))
    monkeypatch.setattr(module, "authenticated", lambda *args: None)
    with pytest.raises(ValueError, match="Require declared DTU work3 root"):
        read_plan(path, "fixture-plan")
    protocol["version"] = VERSION if ratio == 2 else "joint-atom-preserving-chemistry-v2"
    path.write_text(json.dumps({"protocol": protocol}))
    with pytest.raises(ValueError, match="declared joint FP32"):
        read_plan(path, "fixture-plan")


@pytest.mark.parametrize("key,value", [("backend", "reference"), ("workers", 0), ("workers", 12.),
    ("torch_threads", True), ("gpu_cpus", 4), ("surprise", 1)])
def test_optimized_execution_rejects_undeclared_resource_policies(key, value):
    execution = {"backend": "optimized", "workers": 12, "torch_threads": 4, "gpu_cpus": 16}
    execution[key] = value
    with pytest.raises(ValueError, match="explicit chemistry execution"):
        execution_settings({"execution": execution})


@pytest.mark.parametrize("slots,omp,mkl,torch_threads", [("4", "4", "4", 4), ("", "4", "4", 4),
    ("16", "16", "4", 4), ("16", "4", "16", 4), ("16", "4", "4", 16)])
def test_optimized_native_setup_requires_real_allocated_slots_and_parent_threads(monkeypatch, slots, omp, mkl, torch_threads):
    execution = {"backend": "optimized", "workers": 12, "torch_threads": 4, "gpu_cpus": 16}
    monkeypatch.setenv("LSB_DJOB_NUMPROC", slots)
    monkeypatch.setenv("OMP_NUM_THREADS", omp)
    monkeypatch.setenv("MKL_NUM_THREADS", mkl)
    monkeypatch.setattr(torch, "get_num_threads", lambda: torch_threads)
    with pytest.raises(ValueError, match="allocated CPU slots|parent runtime"):
        execution_settings({"execution": execution}, allocated=True)


def _hard(spec, molecules=("CO", "C", "CCO")):
    graphs = [encode_smiles(smiles, spec) for smiles in molecules]
    return HardGraphBatch(*(torch.tensor([graph[name].tolist() for graph in graphs], dtype=torch.long)
                            for name in ("atom_types", "charges", "bond_types")))


def _decoded_slice(decoded, start, stop):
    return DecoderOutput(*(value[start:stop] for value in (decoded.atoms, decoded.charges, decoded.bonds)))


def _repair_target(spec):
    """A valid constant witness with complete slot/charge identity provenance."""
    atom = torch.full((1, spec.max_nodes, len(spec.atomic_numbers)), -12.)
    charge = torch.full((1, spec.max_nodes, len(spec.formal_charges)), -12.)
    bond = torch.full((1, spec.max_nodes, spec.max_nodes, len(spec.bond_names)), -12.)
    atom[..., 0] = 0
    atom[:, :2, 0] = -12
    atom[:, 0, spec.atomic_numbers.index(8)] = 0
    atom[:, 1, spec.atomic_numbers.index(6)] = 0
    charge[..., spec.formal_charges.index(0)] = 0
    bond[..., 0] = 0
    bond[:, 0, 1, 0] = bond[:, 1, 0, 0] = -12
    bond[:, 0, 1, spec.bond_names.index("TRIPLE")] = 0
    bond[:, 1, 0, spec.bond_names.index("TRIPLE")] = 0
    teacher = AtomPreservingRepairTeacher(spec, ProbabilityRepairConfig(max_edits=1, checks_per_depth=16))
    result = teacher.build_targets(DecoderOutput(atom, charge, bond), candidate_ids=[0], update=0)
    assert result.targets[0] is not None
    return result.targets[0]


class _FixedTeacher:
    """Deterministic references independent of chunk boundaries, with zero option."""

    def __init__(self, spec, *, accepted=True):
        self.prototype = _repair_target(spec) if accepted else None
        self.calls = []

    def build_targets(self, decoded, *, candidate_ids, update):
        ids = tuple(candidate_ids)
        self.calls.append({"ids": ids,
                           "atoms": decoded.atoms.detach().clone(),
                           "charges": decoded.charges.detach().clone(),
                           "bonds": decoded.bonds.detach().clone()})
        targets = tuple(None if self.prototype is None else replace(self.prototype, candidate_id=value,
                                                                     source_update=update) for value in ids)
        status = "censored" if self.prototype is None else "accepted"
        return RepairTargetBatch(targets, ids, {"accepted_targets": 0 if self.prototype is None else len(ids),
                                              "status_counts": {status: len(ids)}, "search_checks": 0,
                                              "elapsed_seconds": 0., "source_update": update})


class _DeferredFixedTeacher(_FixedTeacher):
    """Observable schedule: every prepass chunk exists before any target wait."""

    def __init__(self, spec, expected_chunks, *, fail_index=None):
        super().__init__(spec)
        self.expected_chunks, self.fail_index = expected_chunks, fail_index
        self.queued, self.received, self.drains = [], [], 0

    def build_targets(self, *args, **kwargs):
        raise AssertionError("Async update used the synchronous teacher interface")

    def build_targets_async(self, decoded, *, candidate_ids, update):
        batch = super().build_targets(decoded, candidate_ids=candidate_ids, update=update)
        index = len(self.queued)
        self.queued.append(batch)
        owner = self

        class Deferred:
            def result(self):
                assert len(owner.queued) == owner.expected_chunks
                assert index == len(owner.received)
                owner.received.append(index)
                if index == owner.fail_index:
                    raise RuntimeError("Injected late detached repair failure")
                return batch

        return Deferred()

    def drain(self):
        self.drains += 1


def _tiny_models():
    torch.manual_seed(7301)
    spec = GraphSpec(max_nodes=4)
    architecture = ModelConfig(hidden=8, latent=4, layers=2, decoder_variant="triangle_refine",
                               global_hidden=8, pair_hidden=4, refinement_rounds=1, attention_heads=2,
                               encoder_residual_layers=1, triangle_hidden=3)
    reference = GraphAutoencoder(spec, architecture).eval().requires_grad_(False)
    model = deepcopy(reference).requires_grad_(True)
    # Nonzero parent-anchor error is needed to verify that streaming preserves
    # its gradient, rather than testing the trivially zero phase-boundary value.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(.015 * torch.randn_like(parameter))
    config = TrainConfig(mode="vae", epochs=100, batch_size=4, device="cpu", model=architecture,
                         beta=.02, kl_warmup_epochs=20, clip_grad=5.,
                         loss=LossConfig(padding_weight=.17, no_bond_weight=.23,
                                         charge_weight=.7, bond_weight=1.3, size_weight=.4))
    protocol = {"prior_ratio": 3, "real_batch_size": 4, "microbatch_size": 4,
                "repair_weight": .7, "repair_ramp_epochs": 2,
                "real_repair_weight": .25, "prior_repair_weight": .75,
                "element_js_weight": .3, "charge_js_weight": .2,
                "occupancy_anchor_weight": .4}
    composition = {"elements_distribution": [.55, .10, .20, .05, .01, .04, .03, .01, .01],
                   "charges_distribution": [0., .03, .90, .07, 0.]}
    return model, reference, config, protocol, composition


def _captured_gradients(model):
    return {name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()}


@pytest.mark.parametrize("ratio", [2, 3])
def test_async_order_preserves_exact_targets_joint_update_optimizer_and_rng(ratio):
    model, reference, config, protocol, composition = _tiny_models()
    if ratio == 2:
        protocol.update(prior_ratio=2, composition_prior_weight=1.5)
    other = deepcopy(model)
    hard = _hard(model.spec)
    sync = _FixedTeacher(model.spec)
    # Three real rows in chunks2+1, then 6/9 prior rows in chunks of2.
    async_teacher = _DeferredFixedTeacher(model.spec, 2 + (3 * ratio + 1) // 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    other_optimizer = torch.optim.AdamW(other.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    generator = torch.Generator().manual_seed(990)
    other_generator = torch.Generator().manual_seed(990)
    global_state = torch.get_rng_state().clone()
    expected = chemistry_update(model, reference, optimizer, hard, config, protocol, composition, sync,
        {"fresh_prior": generator}, 7, 34, 31, microbatch_size=2)
    expected_rng = capture_rng_state({"fresh_prior": generator}, False)
    torch.set_rng_state(global_state)
    actual = chemistry_update(other, reference, other_optimizer, hard, config, protocol, composition,
        async_teacher, {"fresh_prior": other_generator}, 7, 34, 31, microbatch_size=2, overlap=True)
    for name in model.state_dict():
        torch.testing.assert_close(model.state_dict()[name], other.state_dict()[name], rtol=0, atol=0)
    assert nested_equal(optimizer.state_dict(), other_optimizer.state_dict())
    assert nested_equal(_captured_gradients(model), _captured_gradients(other))
    assert nested_equal(expected_rng, capture_rng_state({"fresh_prior": other_generator}, False))
    assert {k: v for k, v in expected.items() if k not in ("prepass_seconds", "search_seconds")} == {
        k: v for k, v in actual.items() if k not in ("prepass_seconds", "search_seconds")}
    assert async_teacher.received == list(range(async_teacher.expected_chunks))
    assert async_teacher.drains >= 1
    assert [c["ids"] for c in sync.calls] == [c["ids"] for c in async_teacher.calls]
    expected_ids = [7 * ((1 + ratio) * 4) + row for row in range(3)]
    expected_ids += [7 * ((1 + ratio) * 4) + 4 + row for row in range(3 * ratio)]
    assert [v for c in async_teacher.calls for v in c["ids"]] == expected_ids


def test_late_async_error_drains_all_tasks_and_never_steps_optimizer():
    model, reference, config, protocol, composition = _tiny_models()
    hard = _hard(model.spec)
    before = deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters())
    teacher = _DeferredFixedTeacher(model.spec, 7, fail_index=4)
    with pytest.raises(RuntimeError, match="late detached repair failure"):
        chemistry_update(model, reference, optimizer, hard, config, protocol, composition, teacher,
            {"fresh_prior": torch.Generator().manual_seed(990)}, 7, 34, 31, microbatch_size=2, overlap=True)
    assert teacher.drains >= 1 and len(teacher.queued) == 7
    assert optimizer.state_dict()["state"] == {}
    assert nested_equal(before, model.state_dict())


def test_ratio2_population_value_and_joint_vjp_match_direct_importance_weighted_objective():
    """Independent full objective catches a missing1.5 factor on either VJP."""
    model, reference, config, protocol, composition = _tiny_models()
    protocol.update(prior_ratio=2, composition_prior_weight=1.5)
    direct = deepcopy(model)
    hard = _hard(model.spec)
    initial = torch.get_rng_state().clone()
    teacher = _FixedTeacher(model.spec)
    optimizer = torch.optim.AdamW(model.parameters())
    measured = chemistry_update(model, reference, optimizer, hard, config, protocol, composition, teacher,
        {"fresh_prior": torch.Generator().manual_seed(901)}, 7, 34, 31, do_step=False, microbatch_size=2)
    torch.set_rng_state(initial)
    mu, logvar = direct.encode(hard.to_relaxed(model.spec))
    epsilon = torch.randn(len(hard.atoms), direct.config.latent)
    real = direct.decode(mu + epsilon * (.5 * logvar).exp())
    prior_z = torch.randn(6, direct.config.latent, generator=torch.Generator().manual_seed(901))
    prior = direct.decode(prior_z)
    fixed = _FixedTeacher(model.spec)
    real_targets = fixed.build_targets(real, candidate_ids=[84, 85, 86], update=7)
    prior_targets = fixed.build_targets(prior, candidate_ids=list(range(88, 94)), update=7)
    real_stats, prior_stats = composition_statistics(real, model.spec), composition_statistics(prior, model.spec)
    js = {name: population_js(getattr(real_stats, name).double() + 1.5 * getattr(prior_stats, name).double(),
                              composition[name + "_distribution"])
          for name in ("elements", "charges")}
    with torch.no_grad():
        old_occupancy = reference.decode(prior_z).probabilities().occupancy
    anchor = (prior.probabilities().occupancy - old_occupancy).square().mean()
    loss = (reconstruction_losses(real, hard, config.loss)["reconstruction"] + config.beta * latent_kl(mu, logvar)
        + protocol["repair_weight"] * (.25 * repair_loss(real, real_targets, model.spec)
                                       + .75 * repair_loss(prior, prior_targets, model.spec))
        + protocol["element_js_weight"] * js["elements"] + protocol["charge_js_weight"] * js["charges"]
        + protocol["occupancy_anchor_weight"] * anchor)
    loss.backward()
    assert measured["total"] == pytest.approx(float(loss.detach()), rel=3e-5, abs=3e-6)
    assert measured["element_js"] == pytest.approx(float(js["elements"].detach()), rel=3e-5, abs=3e-6)
    _assert_gradients_match(_captured_gradients(model), _captured_gradients(direct))


def _assert_gradients_match(left, right):
    assert left.keys() == right.keys()
    for name in left:
        assert (left[name] is None) == (right[name] is None), name
        if left[name] is not None:
            torch.testing.assert_close(left[name], right[name], rtol=3e-4, atol=3e-6,
                                       msg=lambda message: f"{name}: {message}")


@pytest.mark.parametrize("slices", [[(0, 1), (1, 3)], [(0, 2), (2, 3)], [(0, 1), (1, 2), (2, 3)]])
def test_global_weighted_reconstruction_chunk_matches_full_batch_objective_and_gradient(slices):
    spec = GraphSpec(max_nodes=4)
    hard = _hard(spec, ("CCCC", "C", "CO"))
    generator = torch.Generator().manual_seed(1234)
    decoded = DecoderOutput(*(torch.randn(shape, generator=generator, dtype=torch.float64, requires_grad=True)
                              for shape in ((3, 4, 10), (3, 4, 5), (3, 4, 4, 5))))
    config = LossConfig(padding_weight=.17, no_bond_weight=.23, charge_weight=.7,
                        bond_weight=1.3, size_weight=.4)
    full = reconstruction_losses(decoded, hard, config)["reconstruction"]
    denominators = reconstruction_denominators(hard, config, dtype=decoded.atoms.dtype)
    streamed = sum(reconstruction_chunk(_decoded_slice(decoded, start, stop), hard_slice(hard, start, stop),
                                        config, denominators) for start, stop in slices)
    torch.testing.assert_close(streamed, full, rtol=1e-11, atol=1e-12)
    values = (decoded.atoms, decoded.charges, decoded.bonds)
    full_gradient = torch.autograd.grad(full, values, retain_graph=True)
    streamed_gradient = torch.autograd.grad(streamed, values)
    for left, right in zip(streamed_gradient, full_gradient):
        torch.testing.assert_close(left, right, rtol=1e-11, atol=1e-12)
    # This fixture intentionally includes unequal padding/pair denominators;
    # averaging per-chunk weighted means would produce a different objective.
    naive = sum(reconstruction_losses(_decoded_slice(decoded, start, stop), hard_slice(hard, start, stop),
                                      config)["reconstruction"] * (stop - start) / 3
                for start, stop in slices)
    assert not torch.isclose(naive, full, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("microbatch", [1, 2])
@pytest.mark.parametrize("accepted", [True, False])
def test_chemistry_update_exact_full_vs_streamed_joint_gradients_rng_and_one_to_three_tail(microbatch, accepted):
    model, reference, config, protocol, composition = _tiny_models()
    streamed_model = deepcopy(model)
    hard = _hard(model.spec)  # Tail: three real rows, nine fresh prior rows.
    teacher = _FixedTeacher(model.spec, accepted=accepted)
    streamed_teacher = _FixedTeacher(model.spec, accepted=accepted)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    streamed_optimizer = torch.optim.AdamW(streamed_model.parameters(), lr=config.learning_rate,
                                           weight_decay=config.weight_decay)
    generator = torch.Generator().manual_seed(4537)
    initial_prior = generator.get_state().clone()
    streamed_generator = torch.Generator()
    streamed_generator.set_state(initial_prior)
    initial_global = torch.get_rng_state().clone()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    full_metrics = chemistry_update(model, reference, optimizer, hard, config, protocol, composition, teacher,
                                    {"fresh_prior": generator}, update=7, epoch=34, phase_start_epoch=31,
                                    do_step=False, microbatch_size=12)
    full_gradient = _captured_gradients(model)
    full_global_after = torch.get_rng_state().clone()
    torch.set_rng_state(initial_global)
    streamed_metrics = chemistry_update(streamed_model, reference, streamed_optimizer, hard, config, protocol,
                                        composition, streamed_teacher, {"fresh_prior": streamed_generator},
                                        update=7, epoch=34, phase_start_epoch=31, do_step=False,
                                        microbatch_size=microbatch)
    _assert_gradients_match(_captured_gradients(streamed_model), full_gradient)
    for key in ("reconstruction", "kl", "repair_real", "repair_prior", "occupancy_anchor", "element_js",
                "charge_js", "total", "pre_clip_gradient_norm"):
        assert streamed_metrics[key] == pytest.approx(full_metrics[key], rel=3e-5, abs=3e-6), key
    assert full_metrics["real_points"] == streamed_metrics["real_points"] == 3
    assert full_metrics["fresh_prior_points"] == streamed_metrics["fresh_prior_points"] == 9
    assert full_metrics["repair_targets"] == streamed_metrics["repair_targets"] == {"real": 3 if accepted else 0,
                                                                                   "prior": 9 if accepted else 0}
    assert full_metrics["occupancy_anchor"] > 0
    assert full_metrics["group_gradient_norms"]["encoder"] > 0
    assert full_metrics["group_gradient_norms"]["decoder"] > 0
    assert torch.equal(generator.get_state(), streamed_generator.get_state())
    assert torch.equal(full_global_after, torch.get_rng_state())
    expected_prior = torch.Generator(); expected_prior.set_state(initial_prior)
    torch.randn(9, model.config.latent, generator=expected_prior)
    assert torch.equal(expected_prior.get_state(), generator.get_state())
    # IDs reserve the configured real region, but keep the true tail ratio.
    expected_ids = [7 * 16 + row for row in range(3)] + [7 * 16 + 4 + row for row in range(9)]
    assert [value for call in teacher.calls for value in call["ids"]] == expected_ids
    assert [value for call in streamed_teacher.calls for value in call["ids"]] == expected_ids
    def by_id(calls):
        return {identity: tuple(call[field][row] for field in ("atoms", "charges", "bonds"))
                for call in calls for row, identity in enumerate(call["ids"])}
    full_rows, streamed_rows = by_id(teacher.calls), by_id(streamed_teacher.calls)
    for identity in expected_ids:
        for left, right in zip(full_rows[identity], streamed_rows[identity]):
            torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-6)
    # A gradient-only benchmark must not silently optimize or reset moments.
    assert optimizer.state_dict()["state"] == streamed_optimizer.state_dict()["state"] == {}
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in reference.parameters())


def test_phase_scalar_retains_explicit_three_to_one_repair_weights_and_warmup():
    model, reference, config, protocol, composition = _tiny_models()
    hard = _hard(model.spec)
    teacher = _FixedTeacher(model.spec)
    optimizer = torch.optim.AdamW(model.parameters())
    metrics = chemistry_update(model, reference, optimizer, hard, config, protocol, composition, teacher,
                               {"fresh_prior": torch.Generator().manual_seed(909)}, update=2, epoch=32,
                               phase_start_epoch=31, do_step=False, microbatch_size=2)
    assert metrics["repair_coefficient"] == pytest.approx(.35)  # First phase epoch: half ramp.
    expected = (metrics["reconstruction"] + metrics["beta"] * metrics["kl"]
                + .35 * (.25 * metrics["repair_real"] + .75 * metrics["repair_prior"])
                + protocol["element_js_weight"] * metrics["element_js"]
                + protocol["charge_js_weight"] * metrics["charge_js"]
                + protocol["occupancy_anchor_weight"] * metrics["occupancy_anchor"])
    assert metrics["total"] == pytest.approx(expected)


def test_two_actual_adamw_updates_resume_exact_moments_weights_and_train_rng(tmp_path):
    model, reference, config, protocol, composition = _tiny_models()
    teacher = _FixedTeacher(model.spec)
    hard = _hard(model.spec)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    generators = {"fresh_prior": torch.Generator().manual_seed(8851)}
    chemistry_update(model, reference, optimizer, hard, config, protocol, composition, teacher, generators,
                     update=7, epoch=34, phase_start_epoch=31, do_step=True, microbatch_size=2)
    assert optimizer.state and all(int(state["step"]) == 1 for state in optimizer.state.values())
    checkpoint = {"model": deepcopy(model.state_dict()), "optimizer": deepcopy(optimizer.state_dict()),
                  "rng": capture_rng_state(generators, include_cuda=False)}
    checkpoint_path = tmp_path / "tiny-phase-checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    uninterrupted = chemistry_update(model, reference, optimizer, hard, config, protocol, composition, teacher,
                                      generators, update=8, epoch=34, phase_start_epoch=31,
                                      do_step=True, microbatch_size=2)
    after_rng = capture_rng_state(generators, include_cuda=False)
    resumed = deepcopy(model)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=config.learning_rate,
                                          weight_decay=config.weight_decay)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    resumed.load_state_dict(saved["model"], strict=True)
    resumed_optimizer.load_state_dict(saved["optimizer"])
    resumed_generators = {"fresh_prior": torch.Generator()}
    restore_rng_state(saved["rng"], resumed_generators)
    observed = chemistry_update(resumed, reference, resumed_optimizer, hard, config, protocol, composition,
                                _FixedTeacher(model.spec), resumed_generators, update=8, epoch=34,
                                phase_start_epoch=31, do_step=True, microbatch_size=2)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], tensor, rtol=0, atol=0)
    left_state, right_state = optimizer.state_dict()["state"], resumed_optimizer.state_dict()["state"]
    assert left_state.keys() == right_state.keys()
    for index in left_state:
        assert left_state[index].keys() == right_state[index].keys()
        for key in left_state[index]:
            torch.testing.assert_close(right_state[index][key], left_state[index][key], rtol=0, atol=0)
        assert int(left_state[index]["step"]) == 2
    resumed_rng = capture_rng_state(resumed_generators, include_cuda=False)
    assert torch.equal(after_rng["torch"], resumed_rng["torch"])
    assert torch.equal(after_rng["loaders"]["fresh_prior"], resumed_rng["loaders"]["fresh_prior"])
    for key in ("reconstruction", "kl", "repair_real", "repair_prior", "element_js", "charge_js",
                "occupancy_anchor", "total", "pre_clip_gradient_norm"):
        assert observed[key] == uninterrupted[key]


def test_checkpoint_comparison_requires_exact_typed_values_without_mutation():
    left = {"state": {7: {"step": torch.tensor(2.), "exp_avg": torch.tensor([.25, -.125])}},
            "rng": [torch.tensor([1, 2], dtype=torch.uint8)], "group": (False, .01)}
    right = deepcopy(left)
    assert nested_equal(left, right)
    assert left["state"][7]["step"].device.type == "cpu"
    assert nested_equal(left, deepcopy(left))
    right["state"][7]["exp_avg"][0] += .001
    assert not nested_equal(left, right)


@pytest.mark.parametrize("left,right", [
    (torch.tensor(1.), torch.tensor(1., dtype=torch.float64)),
    (torch.tensor(1.), torch.tensor([1.])),
    (torch.tensor(1.), torch.tensor(2.)),
    (torch.tensor(float("nan")), torch.tensor(float("nan"))),
    (torch.tensor(1.), 1.), (1., torch.tensor(1.)),
    ({"x": torch.tensor(1.)}, {}), ([1], (1,)), (True, 1), (1, 1.),
])
def test_checkpoint_comparison_rejects_value_schema_and_dtype_changes(left, right):
    assert nested_equal(left, right) is False


@pytest.mark.parametrize("field,value", [("prior_ratio", 2), ("total_epochs", 101),
                                          ("real_repair_weight", .5), ("prior_repair_weight", .5),
                                          ("parameter_scope", "decoder_only"), ("precision", "fp16")])
def test_read_plan_rejects_changed_scientific_mixture_before_artifact_access(tmp_path, monkeypatch, field, value):
    import latentmol.chemistry_continuation as module
    protocol = {"version": VERSION, "total_epochs": 100, "prior_ratio": 3, "real_batch_size": 256,
                "real_repair_weight": .25, "prior_repair_weight": .75,
                "precision": "fp32", "parameter_scope": "joint_encoder_decoder"}
    protocol[field] = value
    path = tmp_path / "invalid-plan.json"
    path.write_text(json.dumps({"protocol": protocol}))
    calls = []
    monkeypatch.setattr(module, "authenticated", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="declared joint FP32"):
        read_plan(path, "fixture-plan")
    assert len(calls) == 1  # Only the plan itself, no data/checkpoint/source access.


def test_read_plan_rejects_duplicate_scientific_keys_instead_of_last_value_wins(tmp_path, monkeypatch):
    import latentmol.chemistry_continuation as module
    path = tmp_path / "duplicate-plan.json"
    path.write_text('{"protocol":{"prior_ratio":1,"prior_ratio":3}}')
    monkeypatch.setattr(module, "authenticated", lambda *args: None)
    with pytest.raises(ValueError, match="Duplicate scientific key"):
        read_plan(path, "fixture-plan")


def _parent_metadata_fixture():
    """Plain metadata only; actual checkpoint moments/RNG have separate fixtures."""
    config = TrainConfig(mode="vae", epochs=100, batch_size=256,
                         checkpoint_epochs=(25, 50, 75, 100), lr_milestones=((61, .0003), (81, .0001)))
    runtime = {"python": "3.11.9", "torch": "2.5.1+cu121", "cpu_threads": 4,
               "device_name": "NVIDIA A100 80GB PCIe", "device_type": "cuda", "cuda_device_count": 1}
    plan = {"parent_epoch": 35, "parent_updates": 230860, "dataset_sha256": "a" * 64,
            "split_sha256": "b" * 64, "parent_archive_sha256": "c" * 64,
            "parent_manifest_sha256": "d" * 64, "parent_checkpoint_sha256": "e" * 64}
    checkpoint = {"format_version": 1, "epoch": 35, "train_config": asdict(config),
        "model_config": asdict(config.model), "loss_config": asdict(config.loss),
        "dataset_sha256": plan["dataset_sha256"], "run_id": "authenticated-parent",
        "source_archive_sha256": plan["parent_archive_sha256"], "source_sha256": plan["parent_manifest_sha256"],
        "continuation": {"version": 1, "epoch_complete": True, "scheduler_state": None,
                         "updates": 230860, "runtime": deepcopy(runtime), "split_sha256": plan["split_sha256"]}}
    parent_run = {"run_id": checkpoint["run_id"], "config": json.loads(json.dumps(asdict(config))),
                  "runtime_identity": deepcopy(runtime), "status": "running"}
    return checkpoint, config, plan, runtime, parent_run


def test_parent_metadata_authenticates_latest_completed_boundary_with_stopped_run_status():
    checkpoint, config, plan, runtime, parent_run = _parent_metadata_fixture()
    before = deepcopy(checkpoint)
    proof = authenticate_parent_metadata(checkpoint, config, plan, runtime, parent_run)
    assert proof["passed"] and proof["parent_epoch"] == 35 and proof["parent_updates"] == 230860
    assert proof["optimizer_scope"] == "authenticated_parent_AdamW_moments"
    assert proof["rng_scope"] == "all_authenticated_parent_streams_plus_separate_fresh_prior"
    assert checkpoint == before  # No metadata rewrite or invented new checkpoint.
    # SIGTERM can leave a failed/running experiment record while the pinned
    # immutable epoch checkpoint is complete. This is an authorized stop.
    parent_run["status"] = "failed"
    assert authenticate_parent_metadata(checkpoint, config, plan, runtime, parent_run)["passed"]


@pytest.mark.parametrize("path,value", [
    (("format_version",), True), (("epoch",), 36),
    (("continuation", "epoch_complete"), False), (("continuation", "updates"), 230861),
    (("continuation", "version"), 2), (("continuation", "scheduler_state"), {}),
    (("continuation", "runtime", "cpu_threads"), 8),
    (("continuation", "runtime", "device_name"), "NVIDIA A100-SXM4-80GB"),
    (("train_config", "beta"), .002), (("train_config", "epochs"), 100.),
    (("model_config", "latent"), 320), (("loss_config", "bond_weight"), 1.5),
    (("dataset_sha256",), "f" * 64), (("continuation", "split_sha256"), "f" * 64),
    (("source_archive_sha256",), "f" * 64), (("source_sha256",), "f" * 64),
    (("run_id",), "unrelated-parent"),
])
def test_parent_metadata_reports_exact_changed_field_and_never_downgrades_authentication(path, value):
    checkpoint, config, plan, runtime, parent_run = _parent_metadata_fixture()
    target = checkpoint
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    before = deepcopy(checkpoint)
    with pytest.raises(ParentCheckpointMismatch) as caught:
        authenticate_parent_metadata(checkpoint, config, plan, runtime, parent_run)
    assert ".".join(path) in {row["field"] for row in caught.value.differences}
    assert ".".join(path) in str(caught.value)
    assert checkpoint == before


def test_parent_metadata_accumulates_differences_and_missing_scheduler_is_not_default_none():
    checkpoint, config, plan, runtime, parent_run = _parent_metadata_fixture()
    del checkpoint["continuation"]["scheduler_state"]
    checkpoint["continuation"]["runtime"]["cpu_threads"] = 16
    checkpoint["train_config"]["batch_size"] = 128
    checkpoint["epoch"] = 34
    with pytest.raises(ParentCheckpointMismatch) as caught:
        authenticate_parent_metadata(checkpoint, config, plan, runtime, parent_run)
    rows = {row["field"]: row for row in caught.value.differences}
    assert {"continuation.scheduler_state", "continuation.runtime.cpu_threads",
            "train_config.batch_size", "epoch"} <= set(rows)
    assert rows["continuation.scheduler_state"]["actual_present"] is False
    assert rows["continuation.scheduler_state"]["expected_present"] is True
    # The diagnostic remains ordinary strict JSON, suitable for the failed
    # benchmark receipt without dumping any model/optimizer/RNG tensors.
    json.dumps({"differences": caught.value.differences, "actual": caught.value.actual,
                "expected": caught.value.expected}, allow_nan=False)


def test_parent_metadata_checks_independently_bound_originating_runtime_and_config():
    checkpoint, config, plan, runtime, parent_run = _parent_metadata_fixture()
    parent_run["runtime_identity"]["torch"] = "different-version"
    parent_run["config"]["loss"]["bond_weight"] = 3.
    with pytest.raises(ParentCheckpointMismatch) as caught:
        authenticate_parent_metadata(checkpoint, config, plan, runtime, parent_run)
    assert {"parent_run.runtime_identity.torch", "parent_run.config.loss.bond_weight"} <= {
        row["field"] for row in caught.value.differences}


def test_metadata_differences_preserves_json_types_and_reports_nested_schedule_positions():
    assert metadata_differences({"schedule": ((61, .0003), (81, .0001))},
                                {"schedule": [[61, .0003], [81, .0001]]}) == []
    rows = metadata_differences({"schedule": [[61, .0004]], "flag": True},
                                {"schedule": [[61, .0003], [81, .0001]], "flag": 1})
    assert {row["field"] for row in rows} == {"schedule.length", "schedule[0][1]", "flag"}


def test_canonical_parent_json_scientific_notation_preserves_exact_config_and_adamw_types(tmp_path):
    """Reproduce the actual producer, not a hand-written YAML-like fixture.

    The failed native phase read this JSON number as text through PyYAML.
    Loading its genuine JSON representation must pass the existing complete
    parent metadata authentication and supply numeric AdamW hyperparameters.
    """
    checkpoint, config, plan, runtime, parent_run = _parent_metadata_fixture()
    path = tmp_path / "parent-config.json"
    serialized = json.dumps(asdict(config), sort_keys=True, indent=2)
    assert '"weight_decay": 1e-05' in serialized
    path.write_text(serialized, encoding="utf-8")
    restored = load_parent_config(path)
    assert type(restored.weight_decay) is float and restored.weight_decay == 1e-5
    assert asdict(restored) == asdict(config)
    assert restored.checkpoint_epochs == (25, 50, 75, 100)
    assert restored.lr_milestones == ((61, .0003), (81, .0001))
    assert authenticate_parent_metadata(checkpoint, restored, plan, runtime, parent_run)["passed"]
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=restored.learning_rate, weight_decay=restored.weight_decay)
    group = optimizer.param_groups[0]
    assert type(group["weight_decay"]) is float and group["weight_decay"] == 1e-5
    assert type(group["lr"]) is float and group["lr"] == config.learning_rate
    assert optimizer.state == {}  # Parsing/constructing never updates or invents parent moments.


@pytest.mark.parametrize("path,value", [
    (("weight_decay",), "1e-05"), (("weight_decay",), True), (("weight_decay",), -1e-5),
    (("learning_rate",), "0.001"), (("learning_rate",), 0),
    (("beta",), False), (("clip_grad",), float("inf")),
    (("epochs",), 100.), (("batch_size",), "256"), (("seed",), True),
    (("train_probe_limit",), "256"), (("num_workers",), -1), (("device",), 0),
    (("model", "hidden"), 96.), (("model", "latent"), True),
    (("model", "global_hidden"), "320"), (("model", "decoder_variant"), None),
    (("loss", "bond_weight"), "2.0"), (("loss", "size_weight"), -1),
    (("checkpoint_epochs",), [25, True]), (("checkpoint_epochs",), "25,50"),
    (("lr_milestones",), [[61, "0.0003"]]), (("lr_milestones",), [[61., .0003]]),
    (("lr_milestones",), [[61, .0003, 0]]), (("lr_milestones",), [[81, .0001], [61, .0003]]),
])
def test_parent_json_rejects_coercions_nonfinite_values_and_invalid_nested_fields(tmp_path, path, value):
    raw = asdict(_parent_metadata_fixture()[1])
    target = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    location = tmp_path / "invalid-parent-config.json"
    location.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_parent_config(location)


@pytest.mark.parametrize("section", [None, "model", "loss"])
@pytest.mark.parametrize("change", ["missing", "extra"])
def test_parent_json_requires_complete_exact_dataclass_schema(tmp_path, section, change):
    raw = asdict(_parent_metadata_fixture()[1])
    target = raw if section is None else raw[section]
    if change == "missing":
        del target[next(iter(target))]
    else:
        target["unrecognized_default"] = 1
    location = tmp_path / "invalid-schema.json"
    location.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="fields differ"):
        load_parent_config(location)


@pytest.mark.parametrize("replacement", [
    '"weight_decay": 1e-05, "weight_decay": 1e-05',
    '"weight_decay": NaN', '"weight_decay": Infinity', '"weight_decay": 1e999',
])
def test_parent_json_rejects_duplicate_keys_and_nonfinite_number_spellings(tmp_path, replacement):
    raw = json.dumps(asdict(_parent_metadata_fixture()[1]), sort_keys=True)
    location = tmp_path / "malformed-parent-config.json"
    location.write_text(raw.replace('"weight_decay": 1e-05', replacement), encoding="utf-8")
    with pytest.raises(ValueError):
        load_parent_config(location)
