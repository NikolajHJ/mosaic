"""Explicit new-objective adapter for the existing streamed VAE trainer.

The legacy teacher stays the default. Each submitted microbatch is first a
complete authenticated teacher batch, then a detached failure-preparation
batch. A result is never created by slicing a receipt from a larger batch.
The existing queue owns CPU snapshots and ordered delivery in both sync and
async modes. Extra search reuses the same allocated worker processes.
All numerical validation must run on allocated DTU HPC before adoption.
"""
from dataclasses import dataclass, field
import time

from .atom_preserving_chemistry import AtomPreservingRepairTeacher, RepairTargetBatch
from .chemistry_runtime import validate_failure_guidance
from .probability_repair import ProbabilityRepairConfig
from .repair_failure_guidance import (FailureGuidanceConfig, PreparedFailureGuidance,
                                      LOSS_V2, prepare_failure_guidance)
from .repair_progress import ProgressSearchConfig


@dataclass(frozen=True)
class FailureTeacherBatch:
    """Keep timing outside immutable scientific reference/identity metadata."""
    base: RepairTargetBatch
    prepared: PreparedFailureGuidance
    preparation_seconds: float
    # Execution observations are deliberately outside scientific identities.
    # These are coordinator wall phases, not additive GPU critical-path costs.
    preparation_phases: dict = field(default_factory=dict)


class FailureAwareRepairTeacher(AtomPreservingRepairTeacher):
    """Optional additional preparation; legacy source121 is never hotpatched."""

    def __init__(self, spec, config, *, failure_guidance, **kwargs):
        self.failure_guidance = validate_failure_guidance(failure_guidance)
        if self.failure_guidance is None:
            raise ValueError("Failure-aware teacher requires explicit versioned objective metadata")
        self.loss_config = FailureGuidanceConfig(**self.failure_guidance["loss"])
        search = self.failure_guidance["progress_search"]
        self.progress_config = None if search is None else ProgressSearchConfig(**search)
        if self.loss_config.version == LOSS_V2:
            from .chemistry_classification import VERSION as classification_version
            kwargs["classification_version"] = classification_version
        super().__init__(spec, config, **kwargs)

    def _progress_map(self, tasks):
        if self.backend == "reference":
            # Keep the historical six-edit algorithm as the serial oracle for
            # exact full-update parity; production workers use invariant reuse.
            from .repair_progress import search_progress_repair_reference
            return [search_progress_repair_reference(*task) for task in tasks]
        if not self.workers:
            from .repair_progress import search_progress_repair
            return [search_progress_repair(*task) for task in tasks]
        if self._pool is None:
            from .repair_worker_pool import RepairWorkerPool
            self._pool = RepairWorkerPool(self.workers)
        return self._pool.map_progress(tasks)

    def _build_targets_snapshot(self, snapshot):
        base = super()._build_targets_snapshot(snapshot)
        base_phases = dict(self.last_snapshot_phases)
        started = time.perf_counter()
        phases = {}
        prepared = prepare_failure_guidance(snapshot[0], base, self.spec,
            progress_search=self.progress_config,
            progress_map=self._progress_map if self.progress_config is not None else None,
            phase_seconds=phases, loss_version=self.loss_config.version)
        phases.update(base_phases)
        return FailureTeacherBatch(base, prepared, time.perf_counter() - started, phases)


def build_repair_teacher(spec, protocol, execution=None, *, classification_execution=None, cache_capacity=None):
    """One factory for training, native resume and backend-parity diagnostics."""
    failure = validate_failure_guidance(protocol.get("failure_guidance"))
    from .chemistry_short_continuation import validate_baseline_continuation
    baseline = validate_baseline_continuation(protocol)
    if classification_execution is not None and failure is None and baseline is None:
        raise ValueError("Pooled diagnostic execution requires explicit failure guidance")
    teacher_type = AtomPreservingRepairTeacher if failure is None else FailureAwareRepairTeacher
    options = ({"backend": execution["backend"], "workers": execution["workers"]}
               if execution is not None else {})
    if cache_capacity is not None:
        options["cache_capacity"] = cache_capacity
    if failure is not None:
        options["failure_guidance"] = failure
    if baseline is not None:
        # Classification observes the same original search, including cache
        # hits. It adds diagnostics, never fallback targets or a changed loss.
        from .chemistry_classification import VERSION as classification_version
        options["classification_version"] = classification_version
    if classification_execution is not None:
        options["classification_execution"] = classification_execution
    return teacher_type(spec, ProbabilityRepairConfig(**protocol["search"]), **options)
