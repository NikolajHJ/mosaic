"""CPU worker ordering and actual search parity; allocated DTU HPC only.

Synthetic executor receipts isolate scheduling/error handling. The real spawn
fixtures compare unchanged per-point searches and all parent RNG states. They
do not establish native training throughput or authorize a training segment.
"""
from concurrent.futures import Future
from copy import deepcopy
import os
import random

import numpy as np
import pytest
import torch

from latentmol import repair_worker_pool as module
from latentmol.probability_repair import ProbabilityRepairConfig
from latentmol.repair_worker_pool import RepairWorkerPool
from latentmol.representation import GraphSpec
from test_atom_preserving_chemistry import _decoded, _probabilities


def _tasks():
    spec = GraphSpec(max_nodes=4)
    decoded = _decoded(spec, ("invalid", "valid", "disconnected", "empty"))
    hard = decoded.harden(spec)
    probabilities = _probabilities(decoded)
    config = ProbabilityRepairConfig(max_edits=2, checks_per_depth=16, beam_width=2)
    return [(tuple(value[row].detach().cpu().numpy().copy() for value in (hard.atoms, hard.charges, hard.bonds)),
             tuple(value[row].detach().cpu().numpy().copy() for value in probabilities), spec, config)
            for row in range(4)]


@pytest.mark.parametrize("workers", [0, -1, True, 1.0, "2", module.MAXIMUM_WORKERS + 1])
def test_worker_count_must_be_explicit_positive_bounded_integer(workers):
    with pytest.raises(ValueError, match="worker count"):
        RepairWorkerPool(workers)


def test_empty_maps_are_lazy_and_close_is_idempotent():
    pool = RepairWorkerPool(2)
    assert pool.map([]) == []
    assert pool._executor is None
    pool.close(); pool.close()
    with pytest.raises(RuntimeError, match="closed"):
        pool.map([])
    with pytest.raises(RuntimeError, match="closed"):
        pool.__enter__()


def test_worker_initializer_caps_its_environment_and_torch_threads(monkeypatch):
    variables = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
    for name in variables:
        monkeypatch.setenv(name, "fixture-parent-setting")
    calls = []
    monkeypatch.setattr(torch, "set_num_threads", lambda value: calls.append(("intra", value)))
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda value: calls.append(("inter", value)))
    module._initialize_worker()
    assert calls == [("intra", 1), ("inter", 1)]
    assert {os.environ[name] for name in variables} == {"1"}


@pytest.mark.parametrize("field,replacement", [
    (0, (torch.zeros(4, dtype=torch.long),) * 3),
    (1, (torch.zeros(4),) * 3), (2, None), (3, None),
])
def test_gpu_or_model_tensor_payloads_and_missing_schema_are_rejected_before_submit(field, replacement):
    task = list(_tasks()[0]); task[field] = replacement
    with RepairWorkerPool(1) as pool:
        with pytest.raises(TypeError, match="NumPy|schema/config"):
            pool.map([tuple(task)])
        assert pool._executor is None


def test_integer_probabilities_and_float_labels_are_not_silently_coerced():
    task = _tasks()[0]
    for replacement in [(tuple(value.astype(np.float64) for value in task[0]), *task[1:]),
                        (task[0], tuple(value.astype(np.int64) for value in task[1]), *task[2:])]:
        with RepairWorkerPool(1) as pool:
            with pytest.raises(TypeError, match="NumPy integer labels and floating probabilities"):
                pool.map([replacement])
            assert pool._executor is None


class _Executor:
    """Hold futures until the test deliberately completes them out of order."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.submitted = []
        self.shutdown_calls = []

    def submit(self, function, index, task):
        future = Future()
        future.task_index = index
        self.submitted.append((function, index, task, future))
        return future

    def shutdown(self, **kwargs):
        self.shutdown_calls.append(kwargs)


def test_out_of_order_completions_still_return_original_indices_and_reuse_pool(monkeypatch):
    executors, completed_order, pending_sizes = [], [], []
    def create(**kwargs):
        executor = _Executor(**kwargs); executors.append(executor)
        return executor
    def finish(pending, *, return_when):
        assert return_when == module.FIRST_COMPLETED
        pending_sizes.append(len(pending))
        future = max(pending, key=lambda value: value.task_index)
        completed_order.append(future.task_index)
        future.set_result({"task_index": future.task_index, "result": {"ordinal": future.task_index}})
        return {future}, set(pending) - {future}
    monkeypatch.setattr(module, "ProcessPoolExecutor", create)
    monkeypatch.setattr(module, "wait", finish)
    tasks = _tasks() * 3
    with RepairWorkerPool(2) as pool:
        assert pool.map(tasks) == [{"ordinal": index} for index in range(len(tasks))]
        assert completed_order != list(range(len(tasks)))
        assert max(pending_sizes) <= 2 * pool.workers
        first = pool._executor
        assert pool.map(tasks[:2]) == [{"ordinal": 0}, {"ordinal": 1}]
        assert pool._executor is first
    assert len(executors) == 1
    assert executors[0].kwargs["max_workers"] == 2
    assert executors[0].kwargs["mp_context"].get_start_method() == "spawn"
    assert executors[0].kwargs["initializer"] is module._initialize_worker
    assert executors[0].shutdown_calls == [{"wait": True, "cancel_futures": True}]


@pytest.mark.parametrize("failure", ["worker_error", "wrong_index", "boolean_index", "missing_result"])
def test_worker_failure_or_misbound_receipt_closes_pool_without_serial_fallback(monkeypatch, failure):
    executors = []
    def create(**kwargs):
        executor = _Executor(**kwargs); executors.append(executor)
        return executor
    def finish(pending, *, return_when):
        future = min(pending, key=lambda value: value.task_index)
        if failure == "worker_error":
            future.set_exception(ValueError("Injected independent search failure"))
        else:
            receipt = {"task_index": future.task_index, "result": {"status": "ok"}}
            if failure == "wrong_index": receipt["task_index"] += 1
            elif failure == "boolean_index": receipt["task_index"] = False
            else: receipt.pop("result")
            future.set_result(receipt)
        return {future}, set(pending) - {future}
    monkeypatch.setattr(module, "ProcessPoolExecutor", create)
    monkeypatch.setattr(module, "wait", finish)
    pool = RepairWorkerPool(2)
    with pytest.raises(ValueError, match="Injected independent search failure|original task index"):
        pool.map(_tasks())
    assert pool._closed and pool._executor is None
    assert executors[0].shutdown_calls == [{"wait": True, "cancel_futures": True}]
    assert all(future.done() or future.cancelled() for *_, future in executors[0].submitted)
    with pytest.raises(RuntimeError, match="closed"):
        pool.map(_tasks())


def test_context_manager_closes_pool_when_parent_body_raises():
    pool = RepairWorkerPool(1)
    with pytest.raises(RuntimeError, match="parent failed"):
        with pool:
            raise RuntimeError("parent failed")
    assert pool._closed


def test_actual_spawn_workers_one_and_two_match_serial_targets_without_parent_rng_changes():
    from latentmol.atom_preserving_fast import search_atom_preserving_repair_fast

    tasks = _tasks()
    originals = deepcopy(tasks)
    expected = [search_atom_preserving_repair_fast(*task) for task in tasks]
    assert {result["status"] for result in expected} >= {"witness_found", "already_valid", "unsupported"}
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    parent_threads = torch.get_num_threads()
    parent_interop_threads = torch.get_num_interop_threads()
    cuda_states = [value.clone() for value in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None
    thread_variables = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
    parent_environment = {name: os.environ.get(name) for name in thread_variables}
    for workers in (1, 2):
        with RepairWorkerPool(workers) as pool:
            assert pool.map(tasks) == expected
            executor = pool._executor
            assert pool.map(tasks) == expected
            assert pool._executor is executor  # Repeated chunks never respawn a new executor.
    assert random.getstate() == python_state
    observed_numpy = np.random.get_state()
    assert numpy_state[0] == observed_numpy[0] and np.array_equal(numpy_state[1], observed_numpy[1])
    assert numpy_state[2:] == observed_numpy[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert torch.get_num_threads() == parent_threads
    assert torch.get_num_interop_threads() == parent_interop_threads
    if cuda_states is not None:
        assert all(torch.equal(before, after) for before, after in zip(cuda_states, torch.cuda.get_rng_state_all()))
    assert {name: os.environ.get(name) for name in thread_variables} == parent_environment
    for original, task in zip(originals, tasks):
        for before, after in zip((*original[0], *original[1]), (*task[0], *task[1])):
            assert np.array_equal(before, after)


def test_actual_spawn_search_exception_is_propagated_and_pool_cannot_continue():
    task = list(_tasks()[0])
    task[1][0][0, 0] = 2.0  # Valid NumPy payload; invalid chemistry probabilities in the worker.
    pool = RepairWorkerPool(1)
    with pytest.raises(ValueError, match="probabilit|categorical"):
        pool.map([tuple(task)])
    assert pool._closed
    with pytest.raises(RuntimeError, match="closed"):
        pool.map([])
