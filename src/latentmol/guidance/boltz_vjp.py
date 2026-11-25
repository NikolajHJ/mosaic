"""CPU graph autograd backed by independent resident Boltz2 GPU VJP workers.

Each spawned process owns one batch-one native scorer. The parent keeps its
decoder and probability tensors on CPU and never initializes CUDA. Multiple
workers require an exclusive LSF allocation with managed MPS; this module does
not start a service or change device visibility. Batch dispatch sends one job
to every available worker before collecting independent, unaveraged losses.
"""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import gc
import importlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import tempfile
import threading
import time
import traceback

import numpy as np
import torch

from latentmol.boltz.contracts import PreparedComplex
from latentmol.boltz.provenance import context_digest
from latentmol.experiments import file_sha256
from latentmol.representation import GraphSpec, RelaxedGraph


class BoltzVJPError(RuntimeError):
    def __init__(self, message, report=None):
        super().__init__(message)
        self.report = report or {}


def _plain(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _write(path, data):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def _failure_summary(error):
    detail = getattr(error, "report", {})
    return dict(error_type=type(error).__name__, error=str(error),
        native={key:detail[key] for key in ("worker","request_id","kind","error_type","error","traceback")
                if key in detail})


def _synchronize():
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()


def _memory():
    if not torch.cuda.is_initialized():
        return dict(peak_allocated_bytes=0, peak_reserved_bytes=0)
    return dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())


def _evaluate(scorer, request):
    arrays = request["arrays"]
    tensors = []
    need_grad = request["need_grad"]
    for name in ("atoms", "charges", "bonds"):
        array = arrays[name]
        if not isinstance(array, np.ndarray) or array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError("VJP wire requires finite FP32 NumPy probabilities")
        tensors.append(torch.from_numpy(array).requires_grad_(need_grad))
    active = torch.from_numpy(request["active"])
    started = time.perf_counter()
    with torch.set_grad_enabled(need_grad):
        value, aux = scorer.score_at(RelaxedGraph(*tensors), active)
        if not isinstance(value, torch.Tensor) or value.numel() != 1 or not bool(value.isfinite().all()):
            raise ValueError("Native worker requires a finite scalar DBMol loss")
        gradients = None
        if need_grad:
            if not value.requires_grad:
                raise ValueError("Native worker DBMol path is detached")
            gradient_tensors = torch.autograd.grad(value, tensors, allow_unused=True)
            if any(g is None or not bool(g.isfinite().all()) for g in gradient_tensors):
                raise ValueError("Native worker has a detached/nonfinite required probability gradient")
            gradients = [g.detach().cpu().numpy().astype(np.float32, copy=True) for g in gradient_tensors]
        result = dict(value=float(value.detach()), value_dtype=str(value.dtype),
            gradients=gradients, aux=_plain(aux), need_grad=need_grad,
            started_monotonic=started)
    _synchronize()
    result.update(native_seconds=time.perf_counter()-started, returned_monotonic=time.perf_counter(), **_memory())
    return result


def _worker(config, spec_record, source_sha256, index, output, connection, abort_event,
            cpu_threads, factory_reference):
    """Wire entry point. All native model/CUDA operations happen in this child."""
    output = Path(output)
    scorer = None
    error = None
    shutdown_id = None
    cleanup = dict(cuda_synchronized=False, frozen_unchanged=False, resources_released=False)
    try:
        with (output / "stdout.log").open("x", encoding="utf-8", buffering=1) as stdout:
            with (output / "stderr.log").open("x", encoding="utf-8", buffering=1) as stderr:
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    torch.set_num_threads(cpu_threads)
                    torch.set_num_interop_threads(1)
                    spec = GraphSpec.from_dict(spec_record)
                    if factory_reference is None:
                        if not os.environ.get("LSB_JOBID", "").isdigit():
                            raise RuntimeError("Native Boltz workers require an allocated LSF job")
                        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                            raise RuntimeError("Native Boltz workers require exactly one allocated CUDA device")
                        probe = torch.ones(1, device="cuda:0")
                        torch.cuda.synchronize()
                        properties = torch.cuda.get_device_properties(0)
                        device = dict(platform="cuda", name=properties.name,
                            uuid=str(properties.uuid) if hasattr(properties, "uuid") else None,
                            device_count=torch.cuda.device_count(), total_memory_bytes=int(properties.total_memory),
                            torch=torch.__version__, cuda_runtime=torch.version.cuda,
                            tensor_sum=float(probe.sum()))
                        if device["tensor_sum"] != 1.:
                            raise RuntimeError("Allocated CUDA worker probe returned an invalid value")
                        del probe
                        from latentmol.guidance.boltz_search import BoltzSearchScorer
                        scorer = BoltzSearchScorer.load(config, spec, device="cuda:0", source_sha256=source_sha256)
                    else:
                        module, name = factory_reference.split(":")
                        scorer = getattr(importlib.import_module(module), name)(config, spec, source_sha256)
                        device = dict(platform="cpu-test", name="mock", uuid=None,
                                      device_count=0, total_memory_bytes=None)
                    _synchronize()
                    if torch.cuda.is_initialized():
                        torch.cuda.reset_peak_memory_stats()
                    identity = dict(worker_index=index, pid=os.getpid(), device=device,
                        model=scorer.identity, cpu_threads=cpu_threads,
                        allocation={key: os.environ.get(key) for key in
                            ("LSB_JOBID", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUDA_MPS_PIPE_DIRECTORY",
                             "CUDA_MPS_LOG_DIRECTORY", "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")})
                    _write(output / "identity.json", identity)
                    connection.send(dict(kind="ready", worker=index, identity=identity))
                    while not abort_event.is_set():
                        if not connection.poll(.1):
                            continue
                        request = connection.recv()
                        request_id = request["request_id"]
                        operation = request["operation"]
                        if operation == "shutdown":
                            shutdown_id = request_id
                            break
                        try:
                            if operation == "evaluate":
                                result = _evaluate(scorer, request)
                            elif operation == "verify":
                                result = dict(verification=scorer.close())
                                _synchronize()
                            else:
                                raise ValueError("Unknown Boltz VJP operation")
                            connection.send(dict(kind="response", status="ok", worker=index,
                                                 request_id=request_id, result=result))
                        except BaseException as failure:
                            error = dict(kind="error", worker=index, request_id=request_id,
                                error_type=type(failure).__name__, error=str(failure), traceback=traceback.format_exc())
                            connection.send(error)
                            break
    except BaseException as failure:
        error = dict(kind="error", worker=index, request_id=None,
            error_type=type(failure).__name__, error=str(failure), traceback=traceback.format_exc())
        try:
            connection.send(error)
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        # Cooperative exit is essential for clients sharing an MPS server.
        # Parent force termination is only a last resort and invalidates reuse.
        if scorer is not None:
            try:
                scorer.close()
                cleanup["frozen_unchanged"] = True
            except BaseException as failure:
                cleanup["frozen_error"] = f"{type(failure).__name__}: {failure}"
        try:
            _synchronize()
            cleanup["cuda_synchronized"] = True
        except BaseException as failure:
            cleanup["cuda_error"] = f"{type(failure).__name__}: {failure}"
        scorer = None
        gc.collect()
        try:
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            cleanup["resources_released"] = True
        except BaseException as failure:
            cleanup["release_error"] = f"{type(failure).__name__}: {failure}"
        cleanup["complete"] = all(cleanup[key] for key in
            ("cuda_synchronized", "frozen_unchanged", "resources_released"))
        terminal = dict(kind="shutdown", worker=index, request_id=shutdown_id, cleanup=cleanup, error=error)
        _write(output / "shutdown.json", terminal)
        try:
            connection.send(terminal)
        except (EOFError, BrokenPipeError, OSError):
            pass
        connection.close()
    if error is not None:
        raise SystemExit(1)


class _AttachVJP(torch.autograd.Function):
    @staticmethod
    def forward(context, atoms, charges, bonds, value, ga, gc, gb):
        context.save_for_backward(ga, gc, gb)
        return value

    @staticmethod
    def backward(context, upstream):
        gradients = context.saved_tensors
        return tuple(upstream.to(g)*g for g in gradients) + (None, None, None, None)


class BoltzVJPScorer:
    """Persistent batch-one native workers; the decoder/autograd graph stays CPU."""

    label = "Boltz2 DBMol parallel CPU-probability VJP"
    gradient_scope = ("CPU conditioned graph probabilities -> independent native Boltz2 GPU VJPs "
                      "-> CPU autograd -> latent; full DBMol loss, fixed pose and occupancy")
    claim_limit = "Fixed-pose surrogate gradients; no diffusion or fresh-pose prediction"

    @classmethod
    def load(cls, config, spec, workers=2, *, source_sha256, output=None, managed_mps=False,
             worker_cpu_threads=2, startup_timeout=1800, timeout=300, shutdown_timeout=120,
             _factory_reference=None):
        if _factory_reference is None:
            if not os.environ.get("LSB_JOBID", "").isdigit():
                raise RuntimeError("Native Boltz VJP requires an allocated LSF job")
            if workers > 1 and managed_mps is not True:
                raise ValueError("Multiple native CUDA workers require an explicit LSF-managed MPS profile")
            if workers > 1 and not os.environ.get("CUDA_MPS_PIPE_DIRECTORY"):
                raise ValueError("LSF-managed MPS pipe directory is missing from the allocation")
        if torch.cuda.is_initialized():
            raise ValueError("Create the CPU-parent Boltz VJP pool before initializing parent CUDA")
        if type(workers) is not int or not 1 <= workers <= 8 or type(worker_cpu_threads) is not int or worker_cpu_threads < 1:
            raise ValueError("Require one to eight workers and positive CPU threads")
        for value in (startup_timeout, timeout, shutdown_timeout):
            if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("Worker deadlines must be finite positive seconds")
        if file_sha256(Path(config["prepared"])) != config["prepared_sha256"]:
            raise ValueError("Prepared Boltz context changed")
        prepared = PreparedComplex.load(Path(config["prepared"]), "cpu")
        if prepared.spec != spec:
            raise ValueError("Prepared Boltz and decoder schemas differ")
        pool = cls()
        pool.spec, pool.prepared, pool.active = spec, prepared, prepared.active
        pool._parent_context_digest = context_digest(prepared)
        pool.device = torch.device("cpu")
        pool.output = Path(output) if output is not None else Path(tempfile.mkdtemp(prefix="boltz-vjp-"))
        if output is not None:
            pool.output.mkdir(parents=True, exist_ok=False)
        pool.workers, pool.timeout, pool.shutdown_timeout = workers, timeout, shutdown_timeout
        pool._processes, pool._connections, pool._pending = [], [], {}
        pool._counter, pool._closed, pool._failed = 0, False, False
        pool._lock = threading.RLock()
        context = multiprocessing.get_context("spawn")
        pool._abort = context.Event()
        pool.last_aux, pool.last_worker_metrics = {}, []
        pool.last_native_seconds, pool.last_batch_size, pool.shutdown_status = None, None, None
        pool._worker_identities, pool._cleanup_messages = {}, []
        config = _plain(config)
        try:
            for index in range(workers):
                child_output = pool.output / f"worker-{index:02d}"
                child_output.mkdir()
                parent, child = context.Pipe(duplex=True)
                process = context.Process(target=_worker, args=(config, spec.to_dict(), source_sha256,
                    index, str(child_output), child, pool._abort, worker_cpu_threads, _factory_reference),
                    name=f"boltz-vjp-{index:02d}")
                pool._connections.append(parent)
                try:
                    process.start()
                finally:
                    child.close()
                pool._processes.append(process)
            deadline = time.perf_counter()+startup_timeout
            for index in range(workers):
                response = pool._receive(index, deadline)
                if response.get("kind") != "ready" or response.get("worker") != index:
                    raise BoltzVJPError(f"Worker did not initialize: {response.get('error', response.get('kind'))}", response)
                pool._worker_identities[index] = response["identity"]
            identities = list(pool._worker_identities.values())
            model = identities[0]["model"]
            if any(row["model"] != model for row in identities):
                raise BoltzVJPError("Native worker model/context identities differ")
            uuids = {row["device"]["uuid"] for row in identities if row["device"]["uuid"] is not None}
            if len(uuids) > 1 or len({row["pid"] for row in identities}) != workers:
                raise BoltzVJPError("Workers must be distinct processes on the same allocated GPU")
            if len({(row["device"]["name"], row["device"]["device_count"], row["device"]["total_memory_bytes"])
                    for row in identities}) != 1:
                raise BoltzVJPError("Worker device properties differ")
            first = identities[0]
            pool.device_identity = dict(device_name=first["device"]["name"], device_uuid=first["device"]["uuid"],
                device_count=first["device"]["device_count"], total_memory_bytes=first["device"]["total_memory_bytes"],
                environment=first["allocation"], source="native_worker_pool",
                torch=first["device"].get("torch"), cuda_runtime=first["device"].get("cuda_runtime"))
            pool.identity = dict(protocol="boltz2-cpu-probability-vjp-v1", worker_count=workers,
                workers=identities, model=model, managed_mps_declared=managed_mps,
                source_sha256=source_sha256, parent_cuda_initialized=False,
                objective="dbmol_total_loss", graph_spec=spec.to_dict(), worker_output=str(pool.output),
                device_identity=pool.device_identity)
            _write(pool.output / "identity.json", pool.identity)
            return pool
        except BaseException as error:
            pool._failed = True
            report = pool._stop(abort=True, failure=_failure_summary(error))
            raise BoltzVJPError(f"Boltz VJP startup failed: {error}", report) from error

    def _receive(self, index, deadline):
        connection = self._connections[index]
        while time.perf_counter() < deadline:
            try:
                if connection.poll(min(.05, max(0., deadline-time.perf_counter()))):
                    return connection.recv()
            except (EOFError, BrokenPipeError, OSError) as error:
                raise BoltzVJPError(f"Worker {index} connection ended") from error
            if self._processes[index].exitcode is not None:
                raise BoltzVJPError(f"Worker {index} exited ({self._processes[index].exitcode})")
        raise TimeoutError(f"Boltz VJP worker {index} timed out")

    def _send(self, index, operation, **payload):
        if index in self._pending:
            raise BoltzVJPError("Worker already has an outstanding request")
        self._counter += 1
        self._pending[index] = self._counter
        self._connections[index].send(dict(operation=operation, request_id=self._counter, **payload))

    def _collect(self, index, deadline):
        response = self._receive(index, deadline)
        expected = self._pending.pop(index)
        if (response.get("worker") != index or response.get("request_id") != expected
                or response.get("kind") != "response" or response.get("status") != "ok"):
            raise BoltzVJPError(f"Native Boltz worker {index} failed: {response.get('error', response.get('kind'))}", response)
        return response["result"]

    def _payload(self, graph, active):
        n = self.spec.max_nodes
        shapes = ((1,n,len(self.spec.atomic_numbers)), (1,n,len(self.spec.formal_charges)),
                  (1,n,n,len(self.spec.bond_names)))
        tensors = (graph.atoms, graph.charges, graph.bonds)
        if any(tuple(value.shape) != shape or value.device.type != "cpu" or value.dtype != torch.float32
               or not bool(value.isfinite().all()) for value,shape in zip(tensors, shapes)):
            raise ValueError("VJP parent requires finite CPU FP32 full-schema probability tensors")
        if (active.device.type != "cpu" or active.dtype != torch.bool or tuple(active.shape) != (n,)
                or int(active.sum()) != int(self.active.sum())):
            raise ValueError("VJP parent requires a compatible CPU fixed-occupancy mask")
        return dict(arrays={name:value.detach().numpy().copy() for name,value in
                           zip(("atoms","charges","bonds"),tensors)},
                    active=active.numpy().copy(),
                    need_grad=torch.is_grad_enabled() and any(value.requires_grad for value in tensors))

    def score_batch(self, graphs, active_masks):
        pairs = list(zip(graphs, active_masks, strict=True))
        if not pairs:
            raise ValueError("Require at least one graph")
        payloads = [self._payload(*pair) for pair in pairs]
        with self._lock:
            if self._closed or self._failed:
                raise BoltzVJPError("Boltz VJP worker pool is closed or failed")
            started = time.perf_counter()
            results, metrics = [], []
            try:
                for first in range(0, len(pairs), self.workers):
                    wave = pairs[first:first+self.workers]
                    for index in range(len(wave)):
                        self._send(index, "evaluate", **payloads[first+index])
                    deadline = time.perf_counter()+self.timeout
                    for index, (graph, _) in enumerate(wave):
                        response = self._collect(index, deadline)
                        value = response["value"]
                        if not isinstance(value, (int,float)) or not math.isfinite(value):
                            raise BoltzVJPError("Worker returned a nonfinite loss")
                        dtype_name = response["value_dtype"].removeprefix("torch.")
                        dtype = {"float32":torch.float32,"float64":torch.float64,
                                 "bfloat16":torch.bfloat16,"float16":torch.float16}.get(dtype_name)
                        if dtype is None:
                            raise BoltzVJPError("Worker returned an unsupported scalar dtype")
                        scalar = torch.tensor(value, dtype=dtype)
                        if payloads[first+index]["need_grad"]:
                            arrays = response["gradients"]
                            if not isinstance(arrays, list) or len(arrays) != 3:
                                raise BoltzVJPError("Worker omitted required VJP arrays")
                            gradients = []
                            for array, tensor in zip(arrays, (graph.atoms, graph.charges, graph.bonds)):
                                if not isinstance(array, np.ndarray) or array.dtype != np.float32 or array.shape != tuple(tensor.shape) or not np.isfinite(array).all():
                                    raise BoltzVJPError("Worker returned an incompatible/nonfinite probability gradient")
                                gradients.append(torch.from_numpy(array.copy()))
                            scalar = _AttachVJP.apply(graph.atoms, graph.charges, graph.bonds, scalar, *gradients)
                        results.append((scalar, response["aux"]))
                        metrics.append(dict(worker=index, pid=self._worker_identities[index]["pid"],
                            **{key:response[key] for key in ("need_grad","native_seconds","started_monotonic",
                                  "returned_monotonic","peak_allocated_bytes","peak_reserved_bytes")}))
                self.last_native_seconds, self.last_batch_size = time.perf_counter()-started, len(pairs)
                self.last_aux, self.last_worker_metrics = dict(results[-1][1]), metrics
                return results
            except BaseException as error:
                self._failed = True
                report = self._stop(abort=True, failure=_failure_summary(error))
                raise BoltzVJPError(f"Native Boltz VJP request failed: {error}", report) from error

    def score_at(self, graph, active):
        return self.score_batch([graph], [active])[0]

    def __call__(self, graph):
        return self.score_at(graph, self.active)

    def score_valid_records(self, root, records):
        from latentmol.guidance.boltz_search import BoltzSearchScorer
        return BoltzSearchScorer.score_valid_records(self, root, records)

    def close(self):
        """Verify frozen workers without unloading them; reusable between waves."""
        with self._lock:
            if self._closed:
                return self.shutdown_status
            try:
                if (self.prepared.spec != self.spec or context_digest(self.prepared) != self._parent_context_digest
                        or not torch.equal(self.active, self.prepared.active)):
                    raise BoltzVJPError("CPU prepared context or fixed occupancy changed")
                for index in range(self.workers):
                    self._send(index, "verify")
                deadline = time.perf_counter()+self.timeout
                verification = [self._collect(index, deadline)["verification"] for index in range(self.workers)]
                if any(row.get("frozen_unchanged") is not True for row in verification):
                    raise BoltzVJPError("Worker did not verify its frozen native state")
                return dict(frozen_unchanged=True, workers=verification)
            except BaseException as error:
                self._failed = True
                report = self._stop(abort=True, failure=_failure_summary(error))
                raise BoltzVJPError(f"Boltz VJP verification failed: {error}", report) from error

    def _stop(self, *, abort, failure=None):
        if self._closed:
            return self.shutdown_status
        if abort:
            self._abort.set()
        else:
            for index in range(len(self._processes)):
                self._send(index, "shutdown")
        deadline = time.perf_counter()+self.shutdown_timeout
        ended, cleanup, forced = set(), {}, set()
        while any(p.is_alive() for p in self._processes) and time.perf_counter() < deadline:
            for index, connection in enumerate(self._connections):
                if index in ended:
                    continue
                try:
                    while connection.poll():
                        message = connection.recv()
                        if message.get("kind") == "shutdown":
                            cleanup[index] = message["cleanup"]
                        else:
                            self._cleanup_messages.append(message)
                except (EOFError, BrokenPipeError, OSError):
                    ended.add(index)
            for process in self._processes:
                process.join(timeout=.02)
        for process in self._processes:
            if process.is_alive():
                forced.add(process.pid)
                process.terminate()
        for process in self._processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        # A child can exit immediately after its final send; drain that message.
        for index, connection in enumerate(self._connections):
            try:
                while connection.poll():
                    message = connection.recv()
                    if message.get("kind") == "shutdown":
                        cleanup[index] = message["cleanup"]
                    else:
                        self._cleanup_messages.append(message)
            except (EOFError, BrokenPipeError, OSError):
                pass
            connection.close()
        processes = [dict(pid=p.pid, exitcode=p.exitcode, alive=p.is_alive(), forced_termination=p.pid in forced)
                     for p in self._processes]
        clean = len(cleanup) == len(processes) and all(row.get("complete") for row in cleanup.values())
        self.shutdown_status = dict(status="failed" if abort else "closed", processes=processes,
            cleanup=cleanup, cooperative_cleanup_complete=clean,
            requires_job_stop=bool(forced) or not clean or any(row["alive"] for row in processes),
            discarded_messages=len(self._cleanup_messages))
        if failure is not None:
            self.shutdown_status["failure"] = failure
        self._closed = True
        _write(self.output / "shutdown.json", self.shutdown_status)
        return self.shutdown_status

    def shutdown(self):
        """Synchronize, verify and unload each worker before its process exits."""
        with self._lock:
            return self._stop(abort=self._failed)
