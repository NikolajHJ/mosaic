"""Bounded FIFO coordination for detached repair work, without model execution.

One coordinator calls the existing synchronous teacher in submission order. It
does not concurrently map the process pool, reorder LRU mutations, draw random
numbers or change search budgets. ``submit`` freezes its input on the submitting
thread before returning: the background operation never owns live GPU/model
state. All running and queued snapshots count against the explicit bound.

An uncaught operation failure poisons subsequent work. Already submitted handles
receive an exception, rather than a replacement target or serial fallback.
Handles cannot be cancelled: silently dropping a queued scientific chunk would
change the cache/receipt history. ``drain`` and ``close`` propagate failures.
Allocated DTU HPC fixtures are required before scientific use.
"""
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Condition


class OrderedRepairQueue:
    """A lazy single-coordinator queue, bounded in immutable CPU snapshots.

    ``snapshot`` is called synchronously, only after a queue slot is available.
    ``operation`` runs on that private snapshot in FIFO order. Callers must keep
    each submitted handle until its result has been consumed. Queue state stores
    no completed results, so repeated updates do not accumulate old batches.

    This helper changes no process-global Torch/BLAS thread setting. The repair
    coordinator shares the trainer's existing CPU thread policy; independently
    spawned search workers retain their explicit single-thread initialization.
    """

    def __init__(self, maximum_pending=8):
        if type(maximum_pending) is not int or not 1 <= maximum_pending <= 64:
            raise ValueError("Async repair queue bound must be an integer in [1,64]")
        self.maximum_pending = maximum_pending
        self._condition = Condition()
        self._executor = None
        self._pending = 0
        self._synchronous = False
        self._closed = False
        self._failure = None

    @property
    def pending(self):
        with self._condition:
            return self._pending

    def _check(self, *, allow_failed=False):
        if self._closed:
            raise RuntimeError("Async repair queue is closed")
        if self._failure is not None and not allow_failed:
            raise RuntimeError("Async repair queue failed; no further scientific work is allowed") from self._failure

    def synchronous(self, operation, *, allow_failed=False):
        """Reject cache mutation races; the legacy synchronous call is unchanged."""
        with self._condition:
            self._check(allow_failed=allow_failed)
            if self._pending or self._synchronous:
                raise RuntimeError("Drain pending async repair work before a synchronous teacher/cache call")
            self._synchronous = True
        try:
            return operation()
        finally:
            with self._condition:
                self._synchronous = False
                self._condition.notify_all()

    def submit(self, snapshot, operation):
        """Freeze an input now, then enqueue an uncancellable ordered operation.

        Backpressure is applied before copying logits. A malformed snapshot
        raises immediately and consumes no queue slot or cache history. Holding
        the condition through this copy also defines submission order when two
        producers happen to call the API concurrently.
        """
        with self._condition:
            self._check()
            if self._synchronous:
                raise RuntimeError("A synchronous repair teacher/cache call is already active")
            while self._pending >= self.maximum_pending:
                self._condition.wait()
                self._check()
                if self._synchronous:
                    raise RuntimeError("A synchronous repair teacher/cache call is already active")
            frozen = snapshot()
            handle = Future()
            # A handle is logically running from submission. Cancellation must
            # not skip a chunk and silently alter subsequent LRU/receipt state.
            if not handle.set_running_or_notify_cancel():
                raise RuntimeError("Fresh async repair handle was unexpectedly cancelled")
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="repair-coordinator")
            self._pending += 1
            try:
                self._executor.submit(self._execute, operation, frozen, handle)
            except BaseException as error:
                self._pending -= 1
                self._failure = error
                handle.set_exception(error)
                self._condition.notify_all()
                raise
            return handle

    def _execute(self, operation, frozen, handle):
        try:
            with self._condition:
                failure = self._failure
            if failure is not None:
                raise RuntimeError("Earlier async repair work failed; queued chunk was not executed") from failure
            result = operation(frozen)
        except BaseException as error:
            with self._condition:
                if self._failure is None:
                    self._failure = error
            handle.set_exception(error)
        else:
            handle.set_result(result)
        finally:
            with self._condition:
                self._pending -= 1
                self._condition.notify_all()

    def drain(self):
        """Wait for every queued chunk, without retaining its completed result."""
        with self._condition:
            while self._pending or self._synchronous:
                self._condition.wait()
            if self._failure is not None:
                raise RuntimeError("Async repair work failed while draining") from self._failure

    def close(self):
        """Stop submissions, drain every handle and release the coordinator."""
        with self._condition:
            self._closed = True
            executor, self._executor = self._executor, None
            self._condition.notify_all()
        # Never cancel executor tasks: each owns a handle that must be resolved,
        # including chunks rejected because an earlier operation failed.
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
        self.drain()
