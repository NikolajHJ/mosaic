"""Allocated-HPC exact-semantics fixtures for the opt-in shared classification pool.

Timing properties are observations, never a flaky speed assertion or a native
training throughput claim. The original serial classifier remains the oracle.
"""
from copy import deepcopy
from dataclasses import asdict
import time

import numpy as np
import pytest
import torch

from latentmol import chemistry_classification as classifier
from latentmol import repair_worker_pool as worker_module
from latentmol.atom_preserving_chemistry import AtomPreservingRepairTeacher, repair_loss
from latentmol.chemistry_continuation import nested_equal
from latentmol.failure_guidance_training import FailureAwareRepairTeacher, build_repair_teacher
from latentmol.probability_repair import ProbabilityRepairConfig
from latentmol.repair_failure_guidance import FailureGuidanceConfig, LOSS_V2, failure_aware_repair_loss
from latentmol.repair_progress import ProgressSearchConfig
from latentmol.repair_worker_pool import RepairWorkerPool, validate_classification_execution
from latentmol.representation import GraphSpec
from latentmol.train import capture_rng_state
from test_atom_preserving_chemistry import _decoded, _probabilities


def policy():
    return {"version": worker_module.CLASSIFICATION_EXECUTION_VERSION}


def _config():
    return ProbabilityRepairConfig(max_edits=2, checks_per_depth=8, beam_width=2)


def _output():
    spec = GraphSpec(max_nodes=4)
    return spec, _decoded(spec, ("invalid", "valid", "disconnected", "empty"))


def _failure_policy():
    return dict(version="chemistry-failure-guidance-v1", loss=asdict(FailureGuidanceConfig(version=LOSS_V2)),
                progress_search=asdict(ProgressSearchConfig(max_edits=2, checks_per_depth=8, beam_width=2)))


def _science(receipt):
    return {key: value for key, value in receipt.items() if key != "elapsed_seconds"}


@pytest.mark.parametrize("value", [True, "pool", {}, {"version": "unknown"},
    {"version": worker_module.CLASSIFICATION_EXECUTION_VERSION, "workers": 12}])
def test_explicit_policy_is_strict_and_does_not_default(value):
    with pytest.raises(ValueError, match="classification execution"):
        validate_classification_execution(value)


def test_policy_copy_and_default_do_not_change_the_historical_path():
    assert validate_classification_execution(None) is None
    original = policy(); parsed = validate_classification_execution(original)
    parsed["version"] = "changed"
    assert original == policy()
    spec, _ = _output()
    with AtomPreservingRepairTeacher(spec, _config(), classification_version=classifier.VERSION) as teacher:
        assert teacher.classification_execution is None and teacher.workers == 0


@pytest.mark.parametrize("backend,workers,version", [
    ("reference", 0, classifier.VERSION), ("optimized", 0, classifier.VERSION), ("optimized", 2, None)])
def test_pool_policy_cannot_create_hidden_workers_or_infer_classification(backend, workers, version):
    spec, _ = _output()
    with pytest.raises(ValueError, match="existing optimized worker pool"):
        AtomPreservingRepairTeacher(spec, _config(), backend=backend, workers=workers,
            classification_version=version, classification_execution=policy())


@pytest.mark.parametrize("field", ["tensor", "float_labels", "missing_schema", "missing_field"])
def test_classification_payload_rejects_device_model_or_untyped_inputs_before_submit(field):
    spec, output = _output(); hard = output.harden(spec)
    raw = tuple(value[0].numpy().copy() for value in (hard.atoms, hard.charges, hard.bonds))
    task = (raw, spec)
    if field == "tensor": task = ((hard.atoms[0], *raw[1:]), spec)
    elif field == "float_labels": task = ((raw[0].astype(float), *raw[1:]), spec)
    elif field == "missing_schema": task = (raw, None)
    else: task = (raw,)
    with RepairWorkerPool(2) as pool:
        with pytest.raises(TypeError, match="NumPy integer graph fields"):
            pool.map_classification([task])
        assert pool._executor is None


def test_actual_spawn_classification_preserves_order_schema_inputs_rng_and_pool():
    spec, output = _output(); hard = output.harden(spec)
    graphs = [tuple(value[row].numpy().copy() for value in (hard.atoms, hard.charges, hard.bonds)) for row in range(4)]
    before_graphs = deepcopy(graphs)
    expected = [classifier.classify_repair_graph(*raw, spec) for raw in graphs]
    generators = {"fresh_prior": torch.Generator().manual_seed(7)}
    before_rng = capture_rng_state(generators, False)
    with RepairWorkerPool(2) as pool:
        results = pool.map_classification([(raw, spec) for raw in graphs])
        first_executor = pool._executor
        assert [value["classification"] for value in results] == expected
        assert all(value["schema_sha256"] == spec.fingerprint() and value["worker_seconds"] >= 0 for value in results)
        probabilities = _probabilities(output)
        tasks = [(graphs[row], tuple(value[row].detach().numpy().copy() for value in probabilities), spec, _config()) for row in range(4)]
        assert len(pool.map(tasks)) == 4
        assert pool._executor is first_executor
        assert len(pool.map_classification([(raw, spec) for raw in graphs])) == 4
        assert pool._executor is first_executor
    assert nested_equal(before_rng, capture_rng_state(generators, False))
    assert all(np.array_equal(a, b) for old, new in zip(before_graphs, graphs) for a, b in zip(old, new))


def test_native_saved_representation_failure_is_identical_in_the_worker():
    # Exact raw labels from source128's fifth native-update REAL example,
    # status-news001 SHA299c3d8a...724280. Sanitization succeeds while an
    # unrepresented [nH] distinction fails the declared schema roundtrip.
    spec = GraphSpec(max_nodes=50)
    atoms = np.zeros(50, dtype=np.int64)
    atoms[:14] = [8, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1]
    charges = np.full(50, 2, dtype=np.int64)
    bonds = np.zeros((50, 50), dtype=np.int64)
    for i, j, label in [(0, 1, 1), (1, 2, 4), (1, 13, 4), (2, 3, 4), (3, 4, 4), (4, 5, 4),
            (4, 12, 1), (5, 6, 1), (5, 13, 4), (6, 7, 1), (7, 8, 4), (7, 12, 4),
            (8, 9, 4), (9, 10, 4), (10, 11, 4), (11, 12, 4)]:
        bonds[i, j] = bonds[j, i] = label
    raw = atoms, charges, bonds
    expected = classifier.classify_repair_graph(*raw, spec)
    assert expected["raw_chemical_valid"] and expected["smiles_parse_valid"]
    assert expected["failure_kind"] == "representation" and expected["reason"] == "representation_roundtrip"
    with RepairWorkerPool(1) as pool:
        actual = pool.map_classification([(raw, spec)])[0]
    assert actual["raw_sha256"] == "99783b31e0ff16ec193635d1b99fd21f30772cba1b6232d7f6cc6ba2b171f05f"
    assert actual["classification"] == expected


@pytest.mark.parametrize("cache_capacity", [1, 8])
def test_serial_vs_pool_preserve_all_receipts_targets_losses_and_cache_hits(cache_capacity, record_property):
    spec, output = _output(); observations = []
    for pooled in (False, True):
        with AtomPreservingRepairTeacher(spec, _config(), backend="optimized", workers=2,
                cache_capacity=cache_capacity, classification_version=classifier.VERSION,
                classification_execution=policy() if pooled else None) as teacher:
            started = time.perf_counter()
            initial = teacher.build_targets(output, candidate_ids=[10, 11, 12, 13], update=7)
            replay = teacher.build_targets(output, candidate_ids=[20, 21, 22, 23], update=8)
            wall = time.perf_counter() - started
            phases = dict(teacher.last_snapshot_phases)
            observations.append((initial, replay, repair_loss(output, replay, spec), phases))
            record_property("pooled_wall_seconds" if pooled else "serial_wall_seconds", wall)
            record_property("pooled_phases" if pooled else "serial_phases", str(phases))
    for index in (0, 1):
        left, right = observations[0][index], observations[1][index]
        assert _science(left.receipt) == _science(right.receipt)
        assert [None if target is None else target.record() for target in left.targets] == [
            None if target is None else target.record() for target in right.targets]
    assert torch.equal(observations[0][2], observations[1][2])
    assert observations[0][3]["classification_worker_seconds"] == 0
    assert observations[1][3]["classification_worker_seconds"] > 0
    assert observations[1][3]["classification_wait_seconds"] > 0
    assert all(value >= 0 for result in observations for value in result[3].values())


def test_cache_hits_are_freshly_classified_and_use_the_same_pool(monkeypatch):
    spec, output = _output(); calls = []
    original = RepairWorkerPool.map_classification
    def record(pool, tasks):
        calls.append((pool, len(tasks)))
        return original(pool, tasks)
    monkeypatch.setattr(RepairWorkerPool, "map_classification", record)
    with AtomPreservingRepairTeacher(spec, _config(), backend="optimized", workers=2,
            cache_capacity=8, classification_version=classifier.VERSION,
            classification_execution=policy()) as teacher:
        teacher.build_targets(output, candidate_ids=[0, 1, 2, 3], update=0)
        result = teacher.build_targets(output, candidate_ids=[4, 5, 6, 7], update=1)
        assert result.receipt["cache_hits"] == 4
        assert [count for _, count in calls] == [4, 4]
        assert calls[0][0] is calls[1][0] is teacher._pool


@pytest.mark.parametrize("change", ["raw_sha", "schema_sha", "version", "stage_type", "worker_time", "missing_row"])
def test_misbound_classification_never_falls_back_or_changes_a_point(change):
    from latentmol.atom_preserving_fast import search_atom_preserving_repair_fast
    spec, output = _output()
    class CorruptPool:
        def __init__(self): self.closed = False
        def map(self, tasks): return [search_atom_preserving_repair_fast(*task) for task in tasks]
        def map_classification(self, tasks):
            results = [worker_module._run_classification_task(i, task)["result"] for i, task in enumerate(tasks)]
            if change == "raw_sha": results[0]["raw_sha256"] = "0" * 64
            elif change == "schema_sha": results[0]["schema_sha256"] = "0" * 64
            elif change == "version": results[0]["classification"]["version"] = "unknown"
            elif change == "stage_type": results[0]["classification"]["strict_teacher_valid"] = 0
            elif change == "worker_time": results[0]["worker_seconds"] = float("nan")
            else: results.pop()
            return results
        def close(self): self.closed = True
    pool = CorruptPool()
    with pytest.raises(ValueError):
        with AtomPreservingRepairTeacher(spec, _config(), backend="optimized", workers=2,
                classification_version=classifier.VERSION, classification_execution=policy()) as teacher:
            teacher._pool = pool
            teacher.build_targets(output, candidate_ids=[0, 1, 2, 3], update=0)
    assert pool.closed


def test_failure_teacher_and_live_vjps_are_exact_under_classification_execution():
    spec, output = _output(); observed = []
    for pooled in (False, True):
        with FailureAwareRepairTeacher(spec, _config(), backend="optimized", workers=2,
                failure_guidance=_failure_policy(), classification_execution=policy() if pooled else None) as teacher:
            batch = teacher.build_targets(output, candidate_ids=[0, 1, 2, 3], update=7)
            terms = failure_aware_repair_loss(output, batch.prepared, spec, FailureGuidanceConfig(version=LOSS_V2))
            gradients = torch.autograd.grad(terms["total"], (output.atoms, output.charges, output.bonds), allow_unused=True)
            observed.append((batch, terms, gradients))
            assert set(teacher.last_snapshot_phases).issubset(batch.preparation_phases)
    left, right = observed
    assert _science(left[0].base.receipt) == _science(right[0].base.receipt)
    assert {k: v for k, v in left[0].prepared.receipt.items() if k != "identity"} == {
        k: v for k, v in right[0].prepared.receipt.items() if k != "identity"}
    for name in ("total", "valid_full_nll", "extra_valid_nll", "partial_edited_nll", "local_bond_risk", "component_cut_risk", "nonempty_risk"):
        assert torch.equal(left[1][name], right[1][name])
    for a, b in zip(left[2], right[2]):
        assert (a is None) == (b is None)
        if a is not None: assert torch.equal(a, b)


def test_factory_requires_explicit_policy_and_keeps_reference_oracle_serial():
    spec, _ = _output(); protocol = dict(search=asdict(_config()), failure_guidance=_failure_policy())
    with build_repair_teacher(spec, protocol, {"backend": "optimized", "workers": 2},
                             classification_execution=policy()) as pooled:
        assert pooled.classification_execution == policy()
    with build_repair_teacher(spec, protocol, {"backend": "reference", "workers": 0}) as oracle:
        assert oracle.classification_execution is None
    with pytest.raises(ValueError, match="explicit failure guidance"):
        build_repair_teacher(spec, {"search": asdict(_config())},
                             {"backend": "optimized", "workers": 2}, classification_execution=policy())
