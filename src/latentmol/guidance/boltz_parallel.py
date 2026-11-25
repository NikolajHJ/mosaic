"""Run independent batch-one child trajectories in spawned worker processes.

The parent imports no CUDA/model code. Each worker owns its models, graphs,
optimizer state and output directory, and calls ``context.ready(metadata)``
after initialization/warmup. All workers are released together. Actual GPU
kernel overlap and speedup must be measured; a process count is not a speedup.

On DTU, request LSF-managed MPS for several CUDA processes on one exclusive
GPU. Preserve the allocation's MPS environment. This module never changes
visibility, starts an MPS daemon, or shares live CUDA tensors/model objects.
"""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import importlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import tempfile
import time
import traceback


class ParallelWorkerError(RuntimeError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


class WorkerCancelled(RuntimeError):
    """Cooperative group cancellation; the scientific worker must still clean up."""


def _json_value(value):
    # Only plain result/metadata values cross this boundary, never live tensors.
    return json.loads(json.dumps(value, allow_nan=False))


def _write(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


class WorkerContext:
    """Worker-owned paths and the one-time synchronized start handshake."""

    def __init__(self, index, output, connection, event, abort_event, startup_timeout):
        self.index = index
        self.output = Path(output)
        self.pid = os.getpid()
        self.started_monotonic = None
        self._connection, self._event = connection, event
        self._abort_event = abort_event
        self._startup_timeout = startup_timeout
        self._ready_called = False

    @property
    def abort_requested(self):
        return self._abort_event.is_set()

    def check_abort(self):
        """Call between complete optimizer steps; cleanup belongs in ``finally``."""
        if self.abort_requested:
            raise WorkerCancelled("Worker group requested cooperative cancellation")

    def ready(self, metadata=None):
        """Publish initialized identity, then wait for the group start signal.

        CUDA workers must synchronize outstanding warmup work before calling
        this. The callable remains responsible for its native finite/gradient,
        device identity and frozen-state checks.
        """
        if self._ready_called:
            raise RuntimeError("A worker may enter the ready barrier only once")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("Worker ready metadata must be a dictionary")
        self._ready_called = True
        identity = _json_value(dict(metadata or {}, pid=self.pid, worker_index=self.index))
        self._connection.send(dict(type="ready", index=self.index, metadata=identity))
        if not self._event.wait(self._startup_timeout):
            raise TimeoutError("Worker timed out waiting for the group start signal")
        self.check_abort()
        self.started_monotonic = time.perf_counter()
        return self.started_monotonic


def _worker_entry(reference, payload, index, output, connection, event, abort_event, startup_timeout):
    output = Path(output)
    try:
        with (output / "stdout.log").open("x", encoding="utf-8", buffering=1) as stdout:
            with (output / "stderr.log").open("x", encoding="utf-8", buffering=1) as stderr:
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    module, name = reference.split(":")
                    function = getattr(importlib.import_module(module), name)
                    if not callable(function):
                        raise ValueError("Worker reference is not callable")
                    context = WorkerContext(index, output, connection, event, abort_event, startup_timeout)
                    result = function(payload, context)
                    if context.started_monotonic is None:
                        raise RuntimeError("Worker returned without entering the ready barrier")
                    result = _json_value(result)
                    if not isinstance(result, dict):
                        raise ValueError("Worker result must be a JSON-compatible dictionary")
                    message = dict(type="result", index=index, pid=os.getpid(), result=result,
                                   started_monotonic=context.started_monotonic,
                                   returned_monotonic=time.perf_counter())
                    _write(output / "result.json", message)
                    connection.send(message)
    except BaseException as error:
        message = dict(type="error", index=index, pid=os.getpid(),
                       error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        _write(output / "error.json", message)
        try:
            connection.send(message)
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        connection.close()


def _cleanup(processes, connections, *, failed, abort_event, start_event, grace_seconds):
    forced = set()
    messages, ended = [], set()
    if failed:
        abort_event.set()
        # A peer still waiting at ready must get a chance to unwind its CUDA
        # work and leave the MPS server healthy, rather than being killed.
        start_event.set()
    deadline = time.perf_counter() + grace_seconds
    while any(process.is_alive() for process in processes) and time.perf_counter() < deadline:
        # Keep draining peers during cooperative cleanup. A large final result
        # must not block its sender on a full pipe while we wait for its exit.
        for index, connection in enumerate(connections):
            if index in ended:
                continue
            try:
                while connection.poll():
                    messages.append(connection.recv())
            except (EOFError, BrokenPipeError, OSError):
                ended.add(index)
        for process in processes:
            process.join(timeout=min(.05, max(0., deadline-time.perf_counter())))
    for process in processes:
        if process.is_alive():
            forced.add(process.pid)
            process.terminate()
    for process in processes:
        process.join(timeout=5)
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
    return ([dict(pid=p.pid, exitcode=p.exitcode, alive=p.is_alive(), forced_termination=p.pid in forced)
             for p in processes], messages)


def run_spawn_group(worker_reference, jobs, output, *, startup_timeout=1800, run_timeout=600, cleanup_grace=60):
    """Execute a bounded group of independent callable trajectories concurrently.

    ``worker_reference`` is ``module:callable`` and the callable receives
    ``(payload, context)``. It loads its models, warms up, synchronizes CUDA,
    calls ``context.ready(metadata)``, then runs its measured trajectory and
    returns a plain dictionary. Every job is a plain dictionary. Output paths
    are exclusive ``worker-XX`` directories exposed as ``context.output``.

    ``group_wall_seconds`` covers release through the last returned result,
    including any cleanup inside the callable. Worker results may additionally
    report the precise optimizer-only interval. Startup/model loading and final
    process teardown are reported separately. Failures retain partial evidence
    and raise ``ParallelWorkerError`` with the report after stopping children.
    Workers should call ``context.check_abort()`` between optimizer iterations
    and synchronize CUDA in ``finally`` before exiting. Forced termination is a
    last resort and sets ``requires_job_stop``: the caller must not run further
    cases against a potentially unhealthy MPS server in that allocation.
    """
    if (not isinstance(worker_reference, str) or worker_reference.count(":") != 1
            or not all(worker_reference.split(":"))):
        raise ValueError("Worker uses a module:callable reference")
    if not isinstance(jobs, (list, tuple)) or not jobs or any(not isinstance(job, dict) for job in jobs):
        raise ValueError("Require one or more independent dictionary jobs")
    for value in (startup_timeout, run_timeout, cleanup_grace):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Worker timeouts must be finite positive seconds")
    jobs = _json_value(jobs)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for index in range(len(jobs)):
        (output / f"worker-{index:02d}").mkdir()
    started = time.perf_counter()
    report = dict(status="starting", concurrency=len(jobs), start_method="spawn",
                  worker_reference=worker_reference, startup_timeout=startup_timeout, run_timeout=run_timeout,
                  cleanup_grace=cleanup_grace,
                  ready={}, results={},
                  scope="Independent concurrent OS workers; kernel overlap and speedup require measured evidence")
    _write(output / "group.json", report)
    context = multiprocessing.get_context("spawn")
    start_event = context.Event()
    abort_event = context.Event()
    processes, connections = [], []
    released = last_result = None
    failure = None
    try:
        for index, payload in enumerate(jobs):
            receive, send = context.Pipe(duplex=False)
            connections.append(receive)
            process = context.Process(target=_worker_entry,
                args=(worker_reference, payload, index, str(output / f"worker-{index:02d}"), send,
                      start_event, abort_event, startup_timeout), name=f"boltz-child-{index:02d}")
            try:
                process.start()
            finally:
                send.close()
            processes.append(process)
        while len(report["results"]) < len(jobs):
            for index, connection in enumerate(connections):
                if index in report["results"]:
                    continue
                while connection.poll():
                    try:
                        message = connection.recv()
                    except EOFError:
                        if index not in report["results"]:
                            raise RuntimeError(f"Worker {index} exited without a result (exitcode={processes[index].exitcode})")
                        break
                    if message.get("index") != index:
                        raise RuntimeError("Worker result channel identity differs")
                    if message["type"] == "ready":
                        if index in report["ready"] or released is not None:
                            raise RuntimeError("Unexpected duplicate/late worker ready event")
                        report["ready"][index] = message["metadata"]
                    elif message["type"] == "result":
                        if released is None or index not in report["ready"]:
                            raise RuntimeError("Worker returned before the synchronized start")
                        report["results"][index] = message
                        last_result = time.perf_counter()
                        break
                    elif message["type"] == "error":
                        report["worker_error"] = message
                        raise RuntimeError(f"Worker {index} failed: {message['error_type']}: {message['error']}")
                    else:
                        raise RuntimeError("Unknown worker channel message")
            now = time.perf_counter()
            if released is None:
                if len(report["ready"]) == len(jobs):
                    pids = [entry["pid"] for entry in report["ready"].values()]
                    if len(set(pids)) != len(jobs):
                        raise RuntimeError("Worker PIDs must be distinct")
                    released = now
                    report.update(status="running", startup_seconds=released-started)
                    start_event.set()
                    _write(output / "group.json", report)
                elif now-started > startup_timeout:
                    raise TimeoutError("Worker group startup timeout")
            elif now-released > run_timeout:
                raise TimeoutError("Worker group run timeout")
            if len(report["results"]) < len(jobs):
                time.sleep(0.01)
        report.update(status="completed", group_wall_seconds=last_result-released)
    except BaseException as error:
        failure = error
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
    finally:
        report["processes"], cleanup_messages = _cleanup(processes, connections, failed=failure is not None,
            abort_event=abort_event, start_event=start_event, grace_seconds=cleanup_grace)
        if cleanup_messages:
            report["cleanup_messages"] = cleanup_messages
            for message in cleanup_messages:
                if message.get("type") == "result":
                    report["results"].setdefault(message["index"], message)
        for connection in connections:
            connection.close()
        report["elapsed_seconds"] = time.perf_counter()-started
        report["cleanup_complete"] = not any(p["alive"] for p in report["processes"])
        report["requires_job_stop"] = (not report["cleanup_complete"]
            or any(p["forced_termination"] for p in report["processes"]))
        _write(output / "group.json", report)
    if failure is not None:
        raise ParallelWorkerError(str(failure), report) from failure
    if not report["cleanup_complete"] or any(p["exitcode"] != 0 for p in report["processes"]):
        report.update(status="failed", error="Worker process did not exit cleanly")
        _write(output / "group.json", report)
        raise ParallelWorkerError(report["error"], report)
    return report
