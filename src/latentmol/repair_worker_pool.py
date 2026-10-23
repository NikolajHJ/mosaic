"""Persistent CPU-only workers for independent detached repair searches.

Only NumPy graph/probability arrays and immutable schema/search configuration
cross the process boundary. The caller retains GPU tensors, model weights,
target authentication and the original logical-batch loss. Scheduling cannot
change point order, per-point search budgets or the selected search algorithm.
No task draws randomness or reuses another point's probabilities.

Workers use ``spawn`` rather than inheriting an initialized CUDA process. The
pool is lazy and persistent across calls, with a bounded in-flight window; own
it for the training run and close it at the allocation boundary. Worker and
execution errors propagate and close the pool. No serial fallback is used.
All numerical execution and fixtures belong on allocated DTU HPC.
"""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing
import os
import time


MAXIMUM_WORKERS = 32
CLASSIFICATION_EXECUTION_VERSION = "repair-classification-pool-v1"


def validate_classification_execution(value):
    """Metadata-only opt-in; absence keeps the historical serial classifier."""
    if value is None:
        return None
    if (type(value) is not dict or set(value) != {"version"}
            or value["version"] != CLASSIFICATION_EXECUTION_VERSION):
        raise ValueError("Unsupported explicit classification execution policy")
    return dict(value)


def _initialize_worker():
    """Cap worker CPU libraries without changing the parent trainer's settings."""
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"
    # Import only after setting the subprocess-local library environment. No
    # CUDA query, tensor transfer or model is needed by a search worker.
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _initialize_worker_after(operation, arguments):
    """Keep native thread initialization mandatory before an explicit hook."""
    _initialize_worker()
    operation(*arguments)


def _validate_task(task):
    """Reject model/tensor payloads before anything is submitted to a worker."""
    import numpy as np
    from .probability_repair import ProbabilityRepairConfig
    from .representation import GraphSpec

    if type(task) is not tuple or len(task) != 4:
        raise TypeError("Repair task must be (raw, probabilities, GraphSpec, ProbabilityRepairConfig)")
    raw, probabilities, spec, config = task
    if (type(raw) is not tuple or type(probabilities) is not tuple
            or len(raw) != 3 or len(probabilities) != 3
            or not isinstance(spec, GraphSpec) or not isinstance(config, ProbabilityRepairConfig)):
        raise TypeError("Repair workers require three NumPy label/probability fields and explicit schema/config")
    if (any(type(value) is not np.ndarray or value.dtype.kind not in "iu" for value in raw)
            or any(type(value) is not np.ndarray or value.dtype.kind != "f" for value in probabilities)):
        raise TypeError("Repair workers accept NumPy integer labels and floating probabilities only")


def _run_task(index, task):
    """Return the original task index; native search errors are never swallowed."""
    from .atom_preserving_fast import search_atom_preserving_repair_fast

    result = search_atom_preserving_repair_fast(*task)
    if type(result) is not dict:
        raise TypeError("Repair search worker returned a non-dictionary result")
    return {"task_index": index, "result": result}


def _validate_progress_task(task):
    """A separate opt-in schema; ordinary bond-only tasks are unchanged."""
    from .repair_progress import ProgressSearchConfig
    from .probability_repair import ProbabilityRepairConfig
    if type(task) is not tuple or len(task) != 4 or not isinstance(task[3], ProgressSearchConfig):
        raise TypeError("Progress task requires an explicit ProgressSearchConfig")
    _validate_task((*task[:3], ProbabilityRepairConfig()))


def _run_progress_task(index, task):
    from .repair_progress import search_progress_repair
    result = search_progress_repair(*task)
    if type(result) is not dict:
        raise TypeError("Progress search worker returned a non-dictionary result")
    return {"task_index": index, "result": result}


def _validate_classification_task(task):
    """Only detached integer NumPy graphs and their exact schema cross this boundary."""
    import numpy as np
    from .representation import GraphSpec
    if (type(task) is not tuple or len(task) != 2 or type(task[0]) is not tuple
            or len(task[0]) != 3 or not isinstance(task[1], GraphSpec)
            or any(type(value) is not np.ndarray or value.dtype.kind not in "iu" for value in task[0])):
        raise TypeError("Classification workers require three NumPy integer graph fields and GraphSpec")


def _run_classification_task(index, task):
    """Run the unchanged classifier freshly, including on original-search cache hits."""
    from .chemistry_classification import classify_repair_graph
    from .quality import graph_identity
    raw, spec = task
    started = time.perf_counter()
    classification = classify_repair_graph(*raw, spec)
    result = dict(classification=classification, raw_sha256=graph_identity(*raw, spec),
                  schema_sha256=spec.fingerprint())
    # Execution observations never enter the scientific classifier/search receipt.
    result["worker_seconds"] = time.perf_counter() - started
    return {"task_index": index, "result": result}


class RepairWorkerPool:
    """A single bounded CPU pool, reused across detached teacher chunks.

    ``map(tasks)`` accepts tuples ``(raw, probabilities, spec, config)`` and
    returns search dictionaries in exactly the input order, independent of
    completion order. Arrays must remain immutable until ``map`` returns.
    Explicit worker limits do not reserve CPU resources: the allocated job
    must request enough cores and RAM for its configured pool plus trainer.
    """

    def __init__(self, workers):
        if type(workers) is not int or not 1 <= workers <= MAXIMUM_WORKERS:
            raise ValueError(f"Repair worker count must be an integer in [1,{MAXIMUM_WORKERS}]")
        self.workers = workers
        self.maximum_in_flight = 2 * workers
        self._executor = None
        self._closed = False
        self._mapping = False

    @property
    def started(self):
        """Whether this pool owns an executor; construction alone is still cold."""
        return self._executor is not None

    def _new_executor(self, initializer=_initialize_worker, initargs=()):
        options = dict(max_workers=self.workers,
            mp_context=multiprocessing.get_context("spawn"), initializer=initializer)
        if initargs:
            options["initargs"] = initargs
        return ProcessPoolExecutor(**options)

    def start(self, *, after_initialize=None, initargs=()):
        """Prepare one pristine pool, with optional explicit startup-only work.

        Processes are spawned by the first map. A caller needing every worker
        ready must submit bounded tasks and verify its initializer handshake.
        The native single-thread initializer always runs first. This cannot
        replace an active executor or create a second pool alongside it.
        """
        if self._closed or self._mapping or self.started:
            raise ValueError("Explicit startup requires the pristine cold repair pool")
        if (after_initialize is not None and not callable(after_initialize)) or type(initargs) is not tuple:
            raise TypeError("Worker startup requires a callable hook and tuple arguments")
        if after_initialize is None and initargs:
            raise ValueError("Startup arguments require an explicit initialization hook")
        self._executor = (self._new_executor() if after_initialize is None else
            self._new_executor(_initialize_worker_after, (after_initialize, initargs)))

    def map(self, tasks):
        """Preserve task identity and propagate all failures without substitution."""
        return self._map(tasks, _validate_task, _run_task)

    def map_progress(self, tasks):
        """Reuse the same allocated processes for new detached failed-row work.

        Maps are serial at the coordinator boundary. This never creates a
        second worker pool, oversubscribes cores or silently changes legacy
        repair search; it only selects the explicit progress-task function.
        """
        return self._map(tasks, _validate_progress_task, _run_progress_task)

    def map_classification(self, tasks):
        """Classify all current hard points on the existing bounded CPU pool.

        No classification cache or inferred verdict replaces native inspection.
        The ordinary ordered-index/error/teardown contract applies unchanged.
        """
        return self._map(tasks, _validate_classification_task, _run_classification_task)

    def _map(self, tasks, validate, run):
        if self._closed:
            raise RuntimeError("Repair worker pool is closed")
        if self._mapping:
            raise RuntimeError("Concurrent maps on one repair worker pool are unsupported")
        tasks = list(tasks)
        for task in tasks:
            validate(task)
        if not tasks:
            return []
        self._mapping = True
        pending = {}
        try:
            if self._executor is None:
                self._executor = self._new_executor()
            results = [None] * len(tasks)
            next_index = 0
            while next_index < len(tasks) or pending:
                while next_index < len(tasks) and len(pending) < self.maximum_in_flight:
                    future = self._executor.submit(run, next_index, tasks[next_index])
                    pending[future] = next_index
                    next_index += 1
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    index = pending.pop(future)
                    receipt = future.result()
                    if (type(receipt) is not dict or type(receipt.get("task_index")) is not int
                            or receipt["task_index"] != index or type(receipt.get("result")) is not dict
                            or results[index] is not None):
                        raise ValueError("Repair worker result does not match its original task index")
                    results[index] = receipt["result"]
            if any(result is None for result in results):
                raise RuntimeError("Repair workers did not return every scheduled point")
            return results
        except BaseException:
            for future in pending:
                future.cancel()
            self.close()
            raise
        finally:
            self._mapping = False

    def close(self):
        """Drain running work once and release processes; repeated close is safe."""
        self._closed = True
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Repair worker pool is closed")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False
