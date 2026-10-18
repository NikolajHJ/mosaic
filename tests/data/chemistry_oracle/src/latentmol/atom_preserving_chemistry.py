"""Detached bond-only repair teachers and differentiable population chemistry guards.

RDKit/search/argmax are outside autograd. A bounded, independently checked valid
witness supplies a constant categorical target; it is neither a globally nearest
valid molecule nor a likelihood from an external chemistry model. Atom identity,
charge, occupancy and slot alignment cannot change during repair. This removes
the old valence proxy's direct incentive to substitute sulfur for carbon.

The population Jensen--Shannon penalty uses every attempted relaxed graph,
including invalid ones. Its element law excludes NO_ATOM, and charge counts are
weighted by expected occupancy. Composition alone cannot protect molecule size
or prevent empty graphs; callers must retain their declared size/occupancy guard.
Numerical validation of this module belongs on allocated DTU HPC.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
import time
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from .async_repair import OrderedRepairQueue
from .chemistry import encode_smiles, inspect_graph
from .probability_repair import (ProbabilityRepairConfig, measure_probability_edits,
                                 probability_identity, validate_probabilities)
from .quality import diagnose_graph, graph_identity
from .repair_search import round_robin_candidates
from .representation import DecoderOutput, GraphSpec, HardGraphBatch
from .sampling import hard_graph_log_prob_terms


VERSION = "atom-preserving-bond-repair-v1"


def _hash_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def repair_policy(config: ProbabilityRepairConfig) -> dict:
    """Declare a restricted search without changing historical repair defaults."""
    return {**config.policy(), "version": VERSION, "allowed_edits": ["bond"],
            "atom_identity": "fixed_exact_slot_labels", "formal_charge": "fixed",
            "occupancy": "fixed", "alignment": "shared_slots",
            "validity": "nonempty_connected_RDKit_sanitization_and_schema_roundtrip",
            "already_valid": "skip_without_self_target", "sampling": "argmax_no_rng"}


def _validated_decoded(decoded: DecoderOutput, spec: GraphSpec) -> int:
    if not isinstance(decoded, DecoderOutput) or decoded.atoms.ndim != 3:
        raise ValueError("Require DecoderOutput with a nonempty batch axis")
    batch, n = decoded.atoms.shape[:2]
    shapes = ((batch, spec.max_nodes, len(spec.atomic_numbers)),
              (batch, spec.max_nodes, len(spec.formal_charges)),
              (batch, spec.max_nodes, spec.max_nodes, len(spec.bond_names)))
    values = (decoded.atoms, decoded.charges, decoded.bonds)
    if batch < 1 or n != spec.max_nodes or any(tuple(v.shape) != s for v, s in zip(values, shapes)):
        raise ValueError("Decoder shapes disagree with the shared graph schema")
    if any(not v.is_floating_point() or v.device != values[0].device
           or v.dtype != values[0].dtype or not bool(torch.isfinite(v).all()) for v in values):
        raise ValueError("Require finite floating logits with a common dtype/device")
    return batch


def _key(graph):
    return tuple(value.tobytes() for value in graph)


def _bond_edits(graph, original, spec, diagnostic, probabilities, config):
    """Generate only unordered bond neighbors; no atom/charge proposal exists."""
    slots = np.flatnonzero(graph[0]).tolist()
    implicated = {int(i) for problem in diagnostic.get("problems", []) for i in problem["slots"]}
    component = {}
    for start in slots:
        if start in component:
            continue
        stack, component[start] = [start], start
        while stack:
            i = stack.pop()
            for j in slots:
                if j not in component and graph[2][i, j]:
                    component[j] = start
                    stack.append(j)
    operations = []
    for offset, i in enumerate(slots):
        for j in slots[offset + 1:]:
            if graph[2][i, j] != original[2][i, j]:
                continue  # Never change the same label twice on one path.
            priority = 0 if i in implicated or j in implicated else 2 if component[i] != component[j] else 3
            old = int(original[2][i, j])
            for new in range(len(spec.bond_names)):
                if new == old:
                    continue
                regret = max(0.0, math.log(max(float(probabilities[2][i, j, old]), config.probability_floor))
                             - math.log(max(float(probabilities[2][i, j, new]), config.probability_floor)))
                operations.append((priority, regret, i, j, new))
    for _, _, i, j, new in sorted(operations):
        candidate = tuple(value.copy() for value in graph)
        candidate[2][i, j] = candidate[2][j, i] = new
        yield candidate


def search_atom_preserving_repair(raw, probabilities, spec: GraphSpec,
                                  config: ProbabilityRepairConfig) -> dict:
    """Observe best valid weighted witness under a bounded bond-only search.

    The shared edit-cost implementation supplies probability regret. Budget
    exhaustion and beam pruning remain explicit; failure is not assigned an
    invented finite distance and never authorizes an atom/charge substitution.
    """
    original = tuple(np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value).copy()
                     for value in raw)
    p = validate_probabilities(probabilities, spec)
    diagnostic = diagnose_graph(*original, spec)
    raw_rdkit_valid = diagnostic["valid"]
    raw_roundtrip_error = None
    if raw_rdkit_valid:
        try:
            if encode_smiles(diagnostic["smiles"], spec)["smiles"] != diagnostic["smiles"]:
                raise ValueError("Canonical schema roundtrip identity differs")
        except ValueError as error:
            # Sanitization alone does not certify schema roundtrip (notably
            # some aromatic label assignments). Continue the bounded search
            # rather than silently treating this candidate as already valid.
            raw_roundtrip_error = f"{type(error).__name__}: {error}"
            diagnostic = {**diagnostic, "valid": False,
                          "problem_burden": max(1, diagnostic.get("problem_burden") or 0)}
    result = {"policy": repair_policy(config), "raw_graph_id": diagnostic.get("graph_id"),
              "probability_sha256": probability_identity(p, spec), "raw_valid": diagnostic["valid"],
              "raw_rdkit_valid": raw_rdkit_valid, "raw_roundtrip_error": raw_roundtrip_error,
              "checks": 0, "checks_by_depth": [], "valid_witnesses_observed": 0,
              "roundtrip_rejected_candidates": 0,
              "weighted_witness": None, "best_observed_weighted_cost": None,
              "distance_is_exact": False}
    if (not diagnostic.get("schema_supported") or not diagnostic.get("heavy_atoms")
            or not diagnostic.get("categorical_support_labels_ok")):
        return {**result, "status": "unsupported", "reason": "Requires nonempty canonical hard-label support"}
    # This validates labels, symmetry, diagonal and padding before any search.
    measure_probability_edits(original, original, p, spec, config)
    if diagnostic["valid"]:
        return {**result, "status": "already_valid", "best_observed_weighted_cost": 0.0}
    seen, beam, best, sequence = {_key(original)}, [(original, diagnostic)], None, 0
    for depth in range(1, config.max_edits + 1):
        frontier, checked, capped = [], 0, False
        streams = [_bond_edits(graph, original, spec, diag, p, config) for graph, diag in beam]
        for candidate in round_robin_candidates(streams):
            identity = _key(candidate)
            if identity in seen:
                continue
            if checked >= config.checks_per_depth:
                capped = True
                break
            seen.add(identity)
            candidate_diag = diagnose_graph(*candidate, spec)
            cost = measure_probability_edits(original, candidate, p, spec, config)
            checked += 1
            result["checks"] += 1
            if cost["edit_count"] != depth or cost["edit_counts"]["atom"] or cost["edit_counts"]["charge"]:
                raise RuntimeError("Bond-only generator violated the direct-cost invariant")
            if candidate_diag["valid"]:
                # Schema representability is stronger than RDKit sanitization.
                try:
                    encoded = encode_smiles(candidate_diag["smiles"], spec)
                    if encoded["smiles"] != candidate_diag["smiles"]:
                        raise ValueError("Canonical schema roundtrip identity differs")
                except ValueError:
                    result["roundtrip_rejected_candidates"] += 1
                    candidate_diag = {**candidate_diag, "valid": False,
                                      "problem_burden": max(1, candidate_diag.get("problem_burden") or 0)}
            if candidate_diag["valid"]:
                result["valid_witnesses_observed"] += 1
                witness = {**cost, "graph_id": candidate_diag["graph_id"], "smiles": candidate_diag["smiles"],
                           "labels": {name: value.tolist() for name, value in
                                      zip(("atoms", "charges", "bonds"), candidate)}}
                if best is None or (cost["weighted_cost"], cost["edit_count"]) < (best["weighted_cost"], best["edit_count"]):
                    best = witness
                continue
            sequence += 1
            burden = candidate_diag.get("problem_burden")
            rank = ((1000000 if burden is None else burden)
                    + max(0, candidate_diag.get("component_count", 1) - 1), cost["weighted_cost"], sequence)
            frontier.append((rank, candidate, candidate_diag))
            frontier.sort(key=lambda value: value[0])
            del frontier[config.beam_width:]
        result["checks_by_depth"].append({"depth": depth, "checks": checked, "capped": capped})
        beam = [(item[1], item[2]) for item in frontier]
        if not beam:
            break
    if best is None:
        return {**result, "status": "censored", "reason": "No valid bond-only witness within the declared budget"}
    return {**result, "status": "witness_found", "weighted_witness": best,
            "best_observed_weighted_cost": best["weighted_cost"]}


@dataclass(frozen=True)
class FrozenAtomPreservingTarget:
    """Private CPU label copies, bound to exact source labels and probabilities."""

    candidate_id: int
    source_update: int
    _graph: HardGraphBatch
    _raw: HardGraphBatch
    target_sha256: str
    raw_sha256: str
    source_probability_sha256: str
    schema_sha256: str
    decisions: int
    weighted_cost: float
    edit_count: int

    @property
    def graph(self) -> HardGraphBatch:
        """Public readers receive copies and cannot mutate the stored reference."""
        return HardGraphBatch(*(value.clone() for value in (self._graph.atoms, self._graph.charges, self._graph.bonds)))

    def verify(self, spec: GraphSpec) -> None:
        if self.schema_sha256 != spec.fingerprint():
            raise ValueError("Repair target schema changed")
        if any(type(value) is not int or value < 0 for value in (self.candidate_id, self.source_update)):
            raise ValueError("Repair point/update identities must be nonnegative integers")
        if type(self.edit_count) is not int or self.edit_count < 1 or isinstance(self.weighted_cost, bool) or not isinstance(self.weighted_cost, (float, int)) or not math.isfinite(self.weighted_cost) or self.weighted_cost < self.edit_count:
            raise ValueError("Repair target edit-cost metadata changed")
        for graph, identity in ((self._graph, self.target_sha256), (self._raw, self.raw_sha256)):
            fields = (graph.atoms, graph.charges, graph.bonds)
            shapes = ((1, spec.max_nodes), (1, spec.max_nodes), (1, spec.max_nodes, spec.max_nodes))
            if any(tuple(value.shape) != shape for value, shape in zip(fields, shapes)):
                raise ValueError("Repair target/source tensor shape changed")
            if any(value.device.type != "cpu" or value.requires_grad or value.dtype != torch.long for value in fields):
                raise ValueError("Repair targets must remain detached int64 CPU labels")
            if graph_identity(*(value[0] for value in fields), spec) != identity:
                raise ValueError("Frozen repair target/source label identity changed")
        if not torch.equal(self._graph.atoms, self._raw.atoms) or not torch.equal(self._graph.charges, self._raw.charges):
            raise ValueError("Repair target changed atom identities or formal charges")
        differences = (self._graph.bonds != self._raw.bonds).triu(1)
        if int(differences.sum()) != self.edit_count:
            raise ValueError("Repair target bond-only edit count changed")
        count = int((self._graph.atoms != 0).sum())
        if self.decisions != spec.max_nodes + count + count * (count - 1) // 2:
            raise ValueError("Repair decision-count normalization changed")

    def record(self) -> dict:
        return {"candidate_id": self.candidate_id, "source_update": self.source_update,
                "target_sha256": self.target_sha256, "raw_sha256": self.raw_sha256,
                "source_probability_sha256": self.source_probability_sha256,
                "schema_sha256": self.schema_sha256, "decisions": self.decisions,
                "weighted_cost": self.weighted_cost, "edit_count": self.edit_count,
                "labels": {name: value[0].tolist() for name, value in
                           zip(("atoms", "charges", "bonds"),
                               (self._graph.atoms, self._graph.charges, self._graph.bonds))}}


@dataclass(frozen=True)
class RepairTargetBatch:
    """One attempt per scheduled point, including missing targets explicitly."""

    targets: tuple[FrozenAtomPreservingTarget | None, ...]
    candidate_ids: tuple[int, ...]
    receipt: dict

    def subset(self, indices: Sequence[int]) -> "RepairTargetBatch":
        indices = tuple(indices)
        if not indices or len(set(indices)) != len(indices) or any(type(i) is not int or i < 0 or i >= len(self.targets) for i in indices):
            raise ValueError("Repair target subset must contain valid batch indices")
        return RepairTargetBatch(tuple(self.targets[i] for i in indices),
                                 tuple(self.candidate_ids[i] for i in indices), self.receipt)


class AtomPreservingRepairTeacher:
    """Fresh-minibatch argmax teacher; no persistent latent training panel.

    Cache hits require identical graph labels, probabilities and restricted
    policy. Targets are always rebuilt with the current point/update IDs; failed
    attempts never reuse an earlier witness from a different decoder state.
    """

    def __init__(self, spec: GraphSpec, config: ProbabilityRepairConfig, *, cache_capacity: int = 128,
                 backend: str = "reference", workers: int = 0, async_max_pending: int = 8):
        if not isinstance(config, ProbabilityRepairConfig) or type(cache_capacity) is not int or cache_capacity < 1:
            raise ValueError("Require explicit repair configuration and positive cache capacity")
        self.spec, self.config, self.cache_capacity = spec, config, cache_capacity
        if backend not in ("reference", "optimized") or type(workers) is not int or not 0 <= workers <= 32:
            raise ValueError("Require an explicit supported repair backend and bounded worker count")
        if backend == "reference" and workers:
            raise ValueError("Reference teacher remains serial; parallel execution uses the optimized backend")
        self.backend, self.workers = backend, workers
        self._cache = OrderedDict()
        self._pool = None
        self._async = OrderedRepairQueue(async_max_pending)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        """Drain queued work before releasing its persistent CPU search pool.

        An async failure still closes both executors and is propagated. No
        background thread may outlive the teacher or mutate its cache after a
        caller restores model/RNG state at a disposable benchmark boundary.
        """
        try:
            self._async.close()
        finally:
            if self._pool is not None:
                self._pool.close()
                self._pool = None

    def drain(self):
        """Wait for all async chunks and propagate any uncaught repair failure."""
        self._async.drain()

    def clear_cache(self):
        """Start a fresh scientific target cache while retaining warmed CPU workers."""
        # Clearing an already drained failed teacher is safe cleanup, but the
        # poisoned queue still rejects subsequent scientific work.
        self._async.synchronous(self._cache.clear, allow_failed=True)

    def _search(self, raw, probabilities):
        if self.backend == "reference":
            return search_atom_preserving_repair(raw, probabilities, self.spec, self.config)
        from .atom_preserving_fast import search_atom_preserving_repair_fast
        return search_atom_preserving_repair_fast(raw, probabilities, self.spec, self.config)

    def _parallel_misses(self, records):
        """Plan exactly the serial LRU misses, then search independent points.

        The simulation preserves hits, evictions, duplicate-point decisions and
        logical check counts. Results return by original row; no search outcome
        can change another point's budget, order, probabilities or target.
        """
        if not self.workers:
            return {}
        from .repair_worker_pool import RepairWorkerPool
        simulated = OrderedDict.fromkeys(self._cache)
        indices, tasks = [], []
        for row, (raw, probabilities, _, _, key) in enumerate(records):
            if key in simulated:
                simulated.move_to_end(key)
                continue
            simulated[key] = None
            if len(simulated) > self.cache_capacity:
                simulated.popitem(last=False)
            indices.append(row)
            tasks.append((tuple(value.numpy().copy() for value in raw),
                          tuple(value.numpy().copy() for value in probabilities), self.spec, self.config))
        if not tasks:
            return {}
        if self._pool is None:
            self._pool = RepairWorkerPool(self.workers)
        results = self._pool.map(tasks)
        if len(results) != len(indices):
            raise RuntimeError("Parallel repair lost an ordered scheduled point")
        return dict(zip(indices, results))

    def _target(self, raw, probabilities, witness, candidate_id, update, probability_sha):
        labels = witness.get("labels")
        if not isinstance(labels, dict) or set(labels) != {"atoms", "charges", "bonds"}:
            raise ValueError("Witness requires exact shared label fields")
        def flatten(value):
            return [part for item in value for part in flatten(item)] if isinstance(value, list) else [value]
        if any(type(value) is not int for field in labels.values() for value in flatten(field)):
            raise ValueError("Witness labels must be exact integer categories")
        graph = HardGraphBatch(*(torch.tensor(labels[name], dtype=torch.long)[None] for name in ("atoms", "charges", "bonds")))
        if any(tuple(value.shape) != shape for value, shape in zip((graph.atoms, graph.charges, graph.bonds),
                                                                 ((1, self.spec.max_nodes), (1, self.spec.max_nodes),
                                                                  (1, self.spec.max_nodes, self.spec.max_nodes)))):
            raise ValueError("Witness is not fully padded and slot-aligned")
        if not torch.equal(graph.atoms[0], raw[0]) or not torch.equal(graph.charges[0], raw[1]):
            raise ValueError("Witness changed atom identity, charge or occupied slots")
        cost = measure_probability_edits(raw, tuple(value[0] for value in (graph.atoms, graph.charges, graph.bonds)),
                                         probabilities, self.spec, self.config)
        if cost["edit_count"] < 1 or cost["edit_counts"]["atom"] or cost["edit_counts"]["charge"]:
            raise ValueError("Witness must be a nonzero bond-only edit")
        if type(witness.get("edit_count")) is not int or witness["edit_count"] != cost["edit_count"]:
            raise ValueError("Witness edit count receipt differs")
        weighted = witness.get("weighted_cost")
        if isinstance(weighted, bool) or not isinstance(weighted, (int, float)) or not math.isfinite(weighted) or not math.isclose(weighted, cost["weighted_cost"], rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("Witness probability-weighted cost receipt differs")
        inspected = inspect_graph(*(value[0] for value in (graph.atoms, graph.charges, graph.bonds)), self.spec)
        if inspected.smiles is None or encode_smiles(inspected.smiles, self.spec)["smiles"] != inspected.smiles:
            raise ValueError("Witness failed independent valid/connected/schema-roundtrip checks")
        identity = graph_identity(*(value[0] for value in (graph.atoms, graph.charges, graph.bonds)), self.spec)
        if witness.get("graph_id") != identity or witness.get("smiles") != inspected.smiles:
            raise ValueError("Witness label or chemistry identity receipt differs")
        raw_copy = HardGraphBatch(*(value.detach().clone()[None].long() for value in raw))
        count = int((graph.atoms != 0).sum())
        target = FrozenAtomPreservingTarget(candidate_id, update, graph, raw_copy, identity,
                                           graph_identity(*raw, self.spec), probability_sha, self.spec.fingerprint(),
                                           self.spec.max_nodes + count + count * (count - 1) // 2,
                                           float(weighted), cost["edit_count"])
        target.verify(self.spec)
        return target

    @torch.no_grad()
    def build_targets(self, decoded: DecoderOutput, *, candidate_ids: Sequence[int], update: int) -> RepairTargetBatch:
        """Audit every fresh point; accepted detached references carry no gradients."""
        return self._async.synchronous(lambda: self._build_targets_snapshot(
            self._snapshot(decoded, candidate_ids, update)))

    def build_targets_async(self, decoded: DecoderOutput, *, candidate_ids: Sequence[int],
                            update: int):
        """Return an ordered future after privately freezing CPU logits now.

        The same canonical labels, probabilities, search, cache transitions and
        authenticated target checks as ``build_targets`` run on one coordinator
        in submission order. Only execution overlaps with the caller. Snapshot
        copies complete before return; mutation or deletion of live decoder
        output cannot change queued targets. At most ``async_max_pending`` CPU
        snapshots are held, including the active chunk; no GPU output is queued.
        Receipt elapsed time excludes queue wait and includes snapshot work.
        Consume results in original chunk order and drain before an optimizer
        update, cache clear, model/RNG restoration or allocation teardown.
        """
        return self._async.submit(lambda: self._snapshot(decoded, candidate_ids, update),
                                  self._build_targets_snapshot)

    @torch.no_grad()
    def _snapshot(self, decoded, candidate_ids, update):
        """Freeze CPU storage on the submitting thread; preserve native precision."""
        batch = _validated_decoded(decoded, self.spec)
        ids = tuple(candidate_ids)
        if len(ids) != batch or len(set(ids)) != batch or any(type(value) is not int or value < 0 for value in ids):
            raise ValueError("Require unique nonnegative integer IDs for each scheduled point")
        if type(update) is not int or update < 0:
            raise ValueError("Update must be a nonnegative integer")
        started = time.perf_counter()
        cpu = DecoderOutput(*(value.detach().cpu().clone() for value in (decoded.atoms, decoded.charges, decoded.bonds)))
        if cpu.atoms.dtype in (torch.float16, torch.bfloat16):
            cpu = DecoderOutput(*(value.float() for value in (cpu.atoms, cpu.charges, cpu.bonds)))
        return cpu, batch, ids, update, time.perf_counter() - started

    @torch.no_grad()
    def _build_targets_snapshot(self, snapshot):
        """The unchanged synchronous CPU target algorithm, on private logits."""
        cpu, batch, ids, update, snapshot_seconds = snapshot
        started = time.perf_counter()
        raw_batch = cpu.harden(self.spec)
        probabilities = (cpu.atoms.softmax(-1), cpu.charges.softmax(-1), cpu.symmetric_bond_logits.softmax(-1))
        targets, attempts, hits, checks = [], [], 0, 0
        statuses = Counter()
        policy = repair_policy(self.config)
        policy_sha = _hash_json(policy)
        records = []
        for row, candidate_id in enumerate(ids):
            raw = tuple(value[row] for value in (raw_batch.atoms, raw_batch.charges, raw_batch.bonds))
            p = tuple(value[row] for value in probabilities)
            probability_sha = probability_identity(p, self.spec)
            raw_sha = graph_identity(*raw, self.spec)
            key = (raw_sha, probability_sha, policy_sha)
            records.append((raw, p, probability_sha, raw_sha, key))
        prefetched = self._parallel_misses(records)
        for row, candidate_id in enumerate(ids):
            raw, p, probability_sha, raw_sha, key = records[row]
            attempt, target = {"candidate_id": candidate_id, "raw_sha256": raw_sha,
                               "probability_sha256": probability_sha, "occupied_slots": int((raw[0] != 0).sum())}, None
            try:
                cache_hit = key in self._cache
                if cache_hit:
                    result = json.loads(json.dumps(self._cache[key]))
                    self._cache.move_to_end(key)
                    hits += 1
                else:
                    if self.workers and row not in prefetched:
                        raise RuntimeError("Parallel repair misses differ from the original serial LRU order")
                    result = prefetched[row] if self.workers else self._search(raw, p)
                    self._cache[key] = json.loads(json.dumps(result))
                    if len(self._cache) > self.cache_capacity:
                        self._cache.popitem(last=False)
                if type(result.get("checks")) is not int or not 0 <= result["checks"] <= self.config.max_edits * self.config.checks_per_depth:
                    raise ValueError("Repair search check count is outside its declared budget")
                # Retain work and the complete result even if witness validation
                # below rejects it. A rejected target still consumed search time.
                checks += 0 if cache_hit else result["checks"]
                attempt.update(cache_hit=cache_hit, search=result)
                if result.get("probability_sha256") != probability_sha or result.get("raw_graph_id") != raw_sha or result.get("policy") != policy:
                    raise ValueError("Repair search provenance or restricted policy differs")
                if result.get("status") not in {"witness_found", "already_valid", "unsupported", "censored"}:
                    raise ValueError("Repair search returned an unknown status")
                if result.get("status") == "witness_found":
                    target = self._target(raw, p, result["weighted_witness"], candidate_id, update, probability_sha)
                    status = "accepted"
                else:
                    status = "skipped_already_valid" if result.get("status") == "already_valid" else result.get("status", "unknown")
                attempt.update(status=status, cache_hit=cache_hit, search=result,
                               target=None if target is None else target.record())
            except (ValueError, RuntimeError, TypeError, KeyError, IndexError) as error:
                target = None
                status = "rejected_or_error"
                attempt.update(status=status, error=f"{type(error).__name__}: {error}")
            statuses[status] += 1
            targets.append(target)
            attempts.append(attempt)
        receipt = {"version": VERSION, "source_update": update, "schema_sha256": self.spec.fingerprint(),
                   "policy": policy, "policy_sha256": policy_sha, "attempted_candidates": batch,
                   "accepted_targets": statuses["accepted"], "status_counts": dict(statuses),
                   "search_checks": checks, "cache_hits": hits,
                   "elapsed_seconds": snapshot_seconds + time.perf_counter() - started,
                   "normalization": "N+K+K*(K-1)/2 per accepted target; average over all scheduled points",
                   "gradient_scope": "decoder logits to fixed target NLL; no argmax/search/chemistry gradient",
                   "attempts": attempts}
        return RepairTargetBatch(tuple(targets), ids, receipt)


def repair_loss(decoded: DecoderOutput, targets: RepairTargetBatch, spec: GraphSpec) -> Tensor:
    """Full categorical NLL, normalized over all points including missing zeros.

    Atoms include every NO_ATOM decision; charges include occupied atoms only;
    each occupied unordered bond pair, including NONE, contributes once. Targets
    stay fixed during this call; there is no probability flooring in the NLL.
    """
    batch = _validated_decoded(decoded, spec)
    if not isinstance(targets, RepairTargetBatch) or len(targets.targets) != batch or len(targets.candidate_ids) != batch:
        raise ValueError("Repair targets and decoder batch disagree")
    if len(set(targets.candidate_ids)) != batch or any(type(value) is not int or value < 0 for value in targets.candidate_ids):
        raise ValueError("Repair loss requires unique nonnegative point IDs")
    # An explicit differentiable zero keeps all three heads connected even when
    # the entire bounded search batch is unsuccessful.
    total = sum(value.reshape(-1)[0] * 0 for value in (decoded.atoms, decoded.charges, decoded.bonds))
    accepted_rows, accepted_targets = [], []
    for row, target in enumerate(targets.targets):
        if target is None:
            continue
        if target.candidate_id != targets.candidate_ids[row]:
            raise ValueError("Repair target and point ID alignment changed")
        target.verify(spec)
        accepted_rows.append(row)
        accepted_targets.append(target)
    if accepted_rows:
        # Batch accepted rows into one categorical computation. Do not move or
        # normalize each graph separately on GPU: high coverage otherwise adds
        # repeated transfers and synchronizations for identical operations.
        one = DecoderOutput(*(value[accepted_rows] for value in (decoded.atoms, decoded.charges, decoded.bonds)))
        if one.atoms.dtype in (torch.float16, torch.bfloat16):
            one = DecoderOutput(*(value.float() for value in (one.atoms, one.charges, one.bonds)))
        graph = HardGraphBatch(*(torch.cat([getattr(target._graph, name) for target in accepted_targets], dim=0)
                                 for name in ("atoms", "charges", "bonds"))).to(one.atoms.device)
        terms = hard_graph_log_prob_terms(one, graph, spec)
        if not bool(terms["supported"].all()):
            raise ValueError("Frozen target lies outside the unconditional graph law")
        decisions = one.atoms.new_tensor([target.decisions for target in accepted_targets])
        nll = -(terms["atoms"] + terms["charges"] + terms["bonds"]) / decisions
        if not bool(torch.isfinite(nll).all()):
            raise ValueError("Repair NLL is nonfinite")
        total = total + nll.sum()
    return total / batch


@dataclass(frozen=True)
class CompositionStatistics:
    """Additive expected counts, suitable for an exact population microbatch VJP."""

    elements: Tensor  # Vocabulary order spec.atomic_numbers[1:], no NO_ATOM.
    charges: Tensor  # spec.formal_charges, weighted by occupancy.
    occupied_mass: Tensor


def composition_statistics(decoded: DecoderOutput, spec: GraphSpec) -> CompositionStatistics:
    """Count all relaxed attempts without invalid filtering or hard argmax."""
    _validated_decoded(decoded, spec)
    atoms, charges = decoded.atoms, decoded.charges
    if atoms.dtype in (torch.float16, torch.bfloat16):
        atoms, charges = atoms.float(), charges.float()
    atom_probability, charge_probability = atoms.softmax(-1), charges.softmax(-1)
    occupancy = atom_probability[..., 1:].sum(-1)
    return CompositionStatistics(atom_probability[..., 1:].sum((0, 1)),
                                 (charge_probability * occupancy[..., None]).sum((0, 1)),
                                 occupancy.sum())


def hard_composition_statistics(graph: HardGraphBatch, spec: GraphSpec) -> CompositionStatistics:
    """Derive empirical reference counts from supported training labels only."""
    if graph.atoms.ndim != 2 or tuple(graph.charges.shape) != tuple(graph.atoms.shape) or graph.atoms.shape[1] != spec.max_nodes:
        raise ValueError("Reference labels disagree with schema")
    if any(value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
           or value.requires_grad for value in (graph.atoms, graph.charges)):
        raise ValueError("Reference requires detached integer labels")
    if bool(((graph.atoms < 0) | (graph.atoms >= len(spec.atomic_numbers))).any()):
        raise ValueError("Reference atom labels outside vocabulary")
    active = graph.atoms != 0
    if bool(((graph.charges[active] < 0) | (graph.charges[active] >= len(spec.formal_charges))).any()):
        raise ValueError("Reference occupied charge labels outside vocabulary")
    elements = torch.bincount(graph.atoms[active].long(), minlength=len(spec.atomic_numbers))[1:].double()
    charges = torch.bincount(graph.charges[active].long(), minlength=len(spec.formal_charges)).double()
    return CompositionStatistics(elements, charges, active.sum().double())


def population_js(counts: Tensor, reference_distribution: Tensor | Sequence[float], *, smoothing: float = 1e-8) -> Tensor:
    """Natural-log Jensen--Shannon divergence with declared probability smoothing.

    Normalize aggregate counts once, not separately per graph or microbatch.
    The reference can contain absent/rare elements. Add the same positive
    probability pseudocount to both normalized laws, then renormalize; the result
    lies in [0, log(2)]. A zero-occupancy population has a uniform fallback law,
    not a claim of perfect chemistry, and must be caught by the caller's guard.
    """
    if not isinstance(counts, Tensor) or counts.ndim != 1 or counts.numel() < 2 or not counts.is_floating_point():
        raise ValueError("Population counts must be a floating category vector")
    if not math.isfinite(smoothing) or not 0 < smoothing < 1:
        raise ValueError("Require probability smoothing in (0,1)")
    if not bool(torch.isfinite(counts).all()) or bool((counts < 0).any()):
        raise ValueError("Population counts must be finite and nonnegative")
    reference = torch.as_tensor(reference_distribution, device=counts.device, dtype=counts.dtype)
    if reference.requires_grad or reference.shape != counts.shape or not bool(torch.isfinite(reference).all()) or bool((reference < 0).any()) or not bool(reference.sum() > 0):
        raise ValueError("Require a constant finite nonnegative supported reference law")
    # Counts can be accumulated in float64 by the caller for population parity.
    count_total = counts.sum()
    # At exact zero occupancy, normalizing by machine tiny would manufacture
    # enormous/undefined cotangents. The declared uniform fallback is constant;
    # this branch receives zero composition gradient and needs the size guard.
    p = counts / count_total if bool(count_total > 0) else counts * 0 + 1.0 / counts.numel()
    q = reference / reference.sum()
    p, q = p + smoothing, q + smoothing
    p, q = p / p.sum(), q / q.sum()
    midpoint = (p + q) * 0.5
    return 0.5 * ((p * (p.log() - midpoint.log())).sum()
                  + (q * (q.log() - midpoint.log())).sum())


def population_js_cotangent(counts: Tensor, reference_distribution: Tensor | Sequence[float], *, smoothing: float = 1e-8) -> tuple[Tensor, Tensor]:
    """Value and detached d(JS)/d(counts) for exact streamed microbatch gradients.

    First sum detached statistics over the complete optimizer-update population.
    Obtain this cotangent once. Re-decode each microbatch with unchanged weights
    and identical stochastic state, and differentiate its dot product with this
    fixed cotangent. This gives the same population VJP as retaining all graphs;
    it must not be replaced by a mean of microbatch JS values. Log the returned
    scalar value, not the linear surrogate's value. The caller owns RNG replay.
    """
    with torch.enable_grad():
        leaf = counts.detach().clone().requires_grad_(True)
        value = population_js(leaf, reference_distribution, smoothing=smoothing)
        gradient, = torch.autograd.grad(value, leaf)
    return value.detach(), gradient.detach()
