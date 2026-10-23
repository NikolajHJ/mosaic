"""Ordered async repair, snapshots and exact targets; allocated DTU HPC only.

Synthetic event-controlled jobs test lifecycle without timing-based speed claims.
Actual teacher fixtures compare semantic receipts, bounded searches, full NLL,
all logit gradients and RNG state with the unchanged synchronous algorithm.
No fixture establishes end-to-end GPU throughput or authorizes training.
"""
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
import random
from threading import Event

import numpy as np
import pytest
import torch

from latentmol.async_repair import OrderedRepairQueue
from latentmol.atom_preserving_chemistry import AtomPreservingRepairTeacher, repair_loss
from latentmol.probability_repair import ProbabilityRepairConfig
from latentmol.representation import DecoderOutput, GraphSpec
from test_atom_preserving_chemistry import _decoded


TIMEOUT = 30


def _semantic(value):
    if isinstance(value, dict):
        return {key: _semantic(part) for key, part in value.items() if key != "elapsed_seconds"}
    if isinstance(value, (tuple, list)):
        return [_semantic(part) for part in value]
    return value


def _teacher(spec, **kwargs):
    return AtomPreservingRepairTeacher(spec,
        ProbabilityRepairConfig(max_edits=2, checks_per_depth=16, beam_width=3), **kwargs)


@pytest.mark.parametrize("bound", [0, -1, True, 1.0, "8", 65])
def test_queue_bound_is_explicit_positive_integer(bound):
    with pytest.raises(ValueError, match="queue bound"):
        OrderedRepairQueue(bound)
    with pytest.raises(ValueError, match="queue bound"):
        _teacher(GraphSpec(max_nodes=3), async_max_pending=bound)


def test_lazy_fifo_queue_freezes_before_return_and_prevents_cancellation():
    queue, entered, release, calls = OrderedRepairQueue(3), Event(), Event(), []
    source = [1]
    def operation(snapshot):
        calls.append(snapshot[0])
        if snapshot[0] == 1:
            entered.set()
            assert release.wait(TIMEOUT)
        return snapshot[0]
    assert queue._executor is None and queue.pending == 0
    first = queue.submit(lambda: list(source), operation)
    assert isinstance(first, Future) and entered.wait(TIMEOUT)
    source[0] = 2
    second = queue.submit(lambda: list(source), operation)
    source[0] = 999
    assert not first.cancel() and not second.cancel()
    try:
        assert calls == [1] and queue.pending == 2
    finally:
        release.set()
    assert first.result(TIMEOUT) == 1 and second.result(TIMEOUT) == 2
    queue.drain()
    assert calls == [1, 2] and queue.pending == 0
    queue.close(); queue.close()
    with pytest.raises(RuntimeError, match="closed"):
        queue.submit(lambda: 3, operation)


def test_backpressure_applies_before_snapshot_not_after_a_gpu_output_is_queued():
    queue, entered, release, attempted, frozen = OrderedRepairQueue(1), Event(), Event(), Event(), Event()
    def work(value):
        entered.set()
        assert release.wait(TIMEOUT)
        return value
    first = queue.submit(lambda: 1, work)
    assert entered.wait(TIMEOUT)
    def submit_second():
        attempted.set()
        return queue.submit(lambda: (frozen.set(), 2)[1], work)
    with ThreadPoolExecutor(max_workers=1) as producer:
        waiting = producer.submit(submit_second)
        try:
            assert attempted.wait(TIMEOUT)
            assert not frozen.wait(0.05)
            assert queue.pending == 1
        finally:
            release.set()
        second = waiting.result(TIMEOUT)
        assert frozen.is_set()
        assert first.result(TIMEOUT) == 1 and second.result(TIMEOUT) == 2
    queue.close()


def test_invalid_snapshot_consumes_no_chunk_and_does_not_poison_valid_submissions():
    queue = OrderedRepairQueue()
    def invalid():
        raise ValueError("invalid immutable input")
    with pytest.raises(ValueError, match="invalid immutable"):
        queue.submit(invalid, lambda value: value)
    assert queue.pending == 0 and queue._executor is None
    assert queue.submit(lambda: 4, lambda value: value * 2).result(TIMEOUT) == 8
    queue.close()


def test_uncaught_failure_resolves_queued_handles_and_poison_is_not_a_serial_fallback():
    queue, entered, release, called = OrderedRepairQueue(), Event(), Event(), []
    def work(value):
        called.append(value)
        entered.set()
        assert release.wait(TIMEOUT)
        raise ValueError("injected first search failure")
    first = queue.submit(lambda: 1, work)
    assert entered.wait(TIMEOUT)
    second = queue.submit(lambda: 2, work)
    release.set()
    with pytest.raises(ValueError, match="injected first search"):
        first.result(TIMEOUT)
    with pytest.raises(RuntimeError, match="Earlier async repair"):
        second.result(TIMEOUT)
    with pytest.raises(RuntimeError, match="draining"):
        queue.drain()
    assert called == [1] and queue.pending == 0
    with pytest.raises(RuntimeError, match="failed"):
        queue.submit(lambda: 3, work)
    with pytest.raises(RuntimeError, match="failed"):
        queue.synchronous(lambda: called.append(4))
    with pytest.raises(RuntimeError, match="draining"):
        queue.close()
    assert called == [1] and queue._executor is None


def test_close_drains_before_return_and_rejects_new_submissions():
    queue, entered, release, closing = OrderedRepairQueue(), Event(), Event(), Event()
    def work(value):
        entered.set()
        assert release.wait(TIMEOUT)
        return value
    handle = queue.submit(lambda: 5, work)
    assert entered.wait(TIMEOUT)
    def close():
        closing.set()
        queue.close()
    with ThreadPoolExecutor(max_workers=1) as closer:
        closed = closer.submit(close)
        try:
            assert closing.wait(TIMEOUT)
            assert not closed.done()
        finally:
            release.set()
        closed.result(TIMEOUT)
    assert handle.result(TIMEOUT) == 5 and queue.pending == 0
    with pytest.raises(RuntimeError, match="closed"):
        queue.synchronous(lambda: None)


@pytest.mark.parametrize("backend,workers", [("reference", 0), ("optimized", 0), ("optimized", 2)])
def test_async_full_targets_lru_receipts_nll_and_all_head_gradients_match_sync(backend, workers):
    spec = GraphSpec(max_nodes=3)
    chunks = [("invalid", "valid", "invalid", "empty", "invalid"),
              ("invalid", "valid", "disconnected", "invalid"), ("invalid", "empty")]
    with _teacher(spec, cache_capacity=2, backend=backend, workers=workers) as serial, \
         _teacher(spec, cache_capacity=2, backend=backend, workers=workers) as asynchronous:
        expected, outputs, handles = [], [], []
        for update, kinds in enumerate(chunks):
            ids = list(range(100 * update, 100 * update + len(kinds)))
            left, right = _decoded(spec, kinds), _decoded(spec, kinds)
            expected.append(serial.build_targets(left, candidate_ids=ids, update=update))
            outputs.append((left, right))
            handles.append(asynchronous.build_targets_async(right, candidate_ids=ids, update=update))
        for first, handle, (left, right) in zip(expected, handles, outputs):
            second = handle.result(TIMEOUT)
            assert _semantic(first.receipt) == _semantic(second.receipt)
            assert first.candidate_ids == second.candidate_ids
            assert [None if value is None else value.record() for value in first.targets] == [
                None if value is None else value.record() for value in second.targets]
            left_loss, right_loss = repair_loss(left, first, spec), repair_loss(right, second, spec)
            assert torch.equal(left_loss, right_loss)
            gradients = [torch.autograd.grad(loss, (output.atoms, output.charges, output.bonds))
                         for loss, output in ((left_loss, left), (right_loss, right))]
            assert all(torch.equal(a, b) for a, b in zip(*gradients))
        asynchronous.drain()
        assert list(serial._cache.items()) == list(asynchronous._cache.items())
        serial.clear_cache(); asynchronous.clear_cache()
        assert not serial._cache and not asynchronous._cache


def test_async_submission_freezes_storage_ids_and_grad_scope_before_caller_mutates(monkeypatch):
    spec = GraphSpec(max_nodes=3)
    teacher, entered, release, observed = _teacher(spec), Event(), Event(), []
    native = teacher._build_targets_snapshot
    def held(snapshot):
        cpu = snapshot[0]
        assert all(value.device.type == "cpu" and not value.requires_grad and value.grad_fn is None
                   for value in (cpu.atoms, cpu.charges, cpu.bonds))
        observed.append(tuple(value.clone() for value in (cpu.atoms, cpu.charges, cpu.bonds)))
        entered.set()
        assert release.wait(TIMEOUT)
        return native(snapshot)
    monkeypatch.setattr(teacher, "_build_targets_snapshot", held)
    output, ids = _decoded(spec), [17]
    original = DecoderOutput(*(value.detach().clone().requires_grad_() for value in
                               (output.atoms, output.charges, output.bonds)))
    handle = teacher.build_targets_async(output, candidate_ids=ids, update=9)
    assert entered.wait(TIMEOUT)
    try:
        ids[0] = 999
        with torch.no_grad():
            output.atoms.fill_(100)
            output.charges.fill_(-100)
            output.bonds.zero_()
        with pytest.raises(RuntimeError, match="Drain pending"):
            teacher.build_targets(output, candidate_ids=[2], update=10)
        with pytest.raises(RuntimeError, match="Drain pending"):
            teacher.clear_cache()
    finally:
        release.set()
    result = handle.result(TIMEOUT)
    teacher.drain()
    with _teacher(spec) as reference:
        expected = reference.build_targets(original, candidate_ids=[17], update=9)
    assert result.candidate_ids == (17,) and _semantic(result.receipt) == _semantic(expected.receipt)
    assert all(torch.equal(a, b) for a, b in zip(observed[0],
                                               (original.atoms, original.charges, original.bonds)))
    assert all(value.grad is None for value in (output.atoms, output.charges, output.bonds))
    teacher.close()


def test_async_queue_does_not_advance_rng_or_change_parent_torch_threads():
    spec, output = GraphSpec(max_nodes=3), None
    output = _decoded(spec, ("invalid", "valid", "empty"))
    initial = (random.getstate(), deepcopy(np.random.get_state()), torch.get_rng_state().clone())
    threads = (torch.get_num_threads(), torch.get_num_interop_threads())
    copied = [value.detach().clone() for value in (output.atoms, output.charges, output.bonds)]
    with _teacher(spec, backend="optimized", workers=2) as teacher:
        handle = teacher.build_targets_async(output, candidate_ids=[1, 2, 3], update=0)
        assert handle.result(TIMEOUT).receipt["attempted_candidates"] == 3
        teacher.drain()
    assert random.getstate() == initial[0]
    current = np.random.get_state()
    assert current[0] == initial[1][0] and np.array_equal(current[1], initial[1][1]) and current[2:] == initial[1][2:]
    assert torch.equal(torch.get_rng_state(), initial[2])
    assert (torch.get_num_threads(), torch.get_num_interop_threads()) == threads
    assert all(torch.equal(a, b) for a, b in zip(copied, (output.atoms, output.charges, output.bonds)))


def test_teacher_failure_close_still_releases_native_pool_and_failed_cache_can_be_cleared(monkeypatch):
    teacher = _teacher(GraphSpec(max_nodes=3))
    class Pool:
        closed = False
        def close(self):
            self.closed = True
    pool = teacher._pool = Pool()
    def fail(snapshot):
        raise ValueError("worker failure before receipt")
    monkeypatch.setattr(teacher, "_build_targets_snapshot", fail)
    handle = teacher.build_targets_async(_decoded(teacher.spec), candidate_ids=[1], update=0)
    with pytest.raises(ValueError, match="before receipt"):
        handle.result(TIMEOUT)
    with pytest.raises(RuntimeError, match="draining"):
        teacher.drain()
    teacher.clear_cache()
    with pytest.raises(RuntimeError, match="draining"):
        teacher.close()
    assert pool.closed and teacher._pool is None and teacher._async.pending == 0


def test_async_bad_ids_fail_before_submission_and_valid_call_remains_available():
    spec = GraphSpec(max_nodes=3)
    with _teacher(spec) as teacher:
        with pytest.raises(ValueError, match="unique nonnegative"):
            teacher.build_targets_async(_decoded(spec, ("invalid", "valid")), candidate_ids=[1, 1], update=0)
        assert teacher._async.pending == 0 and teacher._async._executor is None
        result = teacher.build_targets_async(_decoded(spec), candidate_ids=[4], update=0).result(TIMEOUT)
        teacher.drain()
        assert result.candidate_ids == (4,)
