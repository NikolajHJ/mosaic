"""Experiment-only parent-owned native-fact cache with ordered CPU workers.

Workers receive hard graphs and schema only, never probabilities or model
tensors. Native facts can be shared by base/progress searches and successive
preparations. Probability-dependent edit costs, candidate order, beam ranks and
witness selection are recomputed by the parent using the original search loops.
No existing scientific run or saved result is changed by importing this module.
"""

from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
import multiprocessing


_ACTIVE = False


def _graph_key(graph, spec):
    import numpy as np

    fields = []
    for value in graph:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        if array.dtype.kind not in "iu":
            raise TypeError("Native graph cache requires integer labels")
        fields.append((array.dtype.str, array.shape, array.tobytes(order="C")))
    return "graph", spec.fingerprint(), tuple(fields)


def _native_bundle(payload):
    """Run original native functions once for an immutable hard-graph miss."""
    from mosaic.small_molecule.chemistry._reference.chemistry import (
        encode_smiles, inspect_graph,
    )
    from mosaic.small_molecule.chemistry._reference.quality import diagnose_graph
    from mosaic.small_molecule.chemistry._reference.repair_progress import diagnostic_vector

    graph, spec = payload
    diagnostic = diagnose_graph(*graph, spec)
    inspected = inspect_graph(*graph, spec)
    encoded, rejected = {}, set()

    def roundtrip(smiles):
        if smiles in rejected:
            return False
        if smiles not in encoded:
            try:
                encoded[smiles] = encode_smiles(smiles, spec)
            except ValueError:
                # This is the reference's explicit native-roundtrip rejection,
                # not a suppressed worker/tool failure or an invented witness.
                rejected.add(smiles)
                return False
        return encoded[smiles]["smiles"] == smiles

    base_diagnostic = diagnostic
    base_rejected = False
    if diagnostic["valid"] and not roundtrip(diagnostic["smiles"]):
        base_rejected = True
        base_diagnostic = {
            **diagnostic, "valid": False,
            "problem_burden": max(1, diagnostic.get("problem_burden") or 0),
        }
    valid = inspected.smiles is not None
    if valid:
        valid = roundtrip(inspected.smiles)
    progress_diagnostic = {**diagnostic, "valid": valid}
    return {
        "raw_diagnostic": diagnostic,
        "inspection": inspected,
        "base_diagnostic": base_diagnostic,
        "base_roundtrip_rejected": base_rejected,
        "progress_diagnostic": progress_diagnostic,
        "progress_vector": diagnostic_vector(progress_diagnostic),
        "encoded": encoded,
        "roundtrip_rejected_smiles": tuple(sorted(rejected)),
    }


class _NativeFacts:
    """One bounded parent LRU; in-flight layer results never mutate its values."""

    def __init__(self, pool, maximum, stats):
        self.pool, self.maximum, self.stats = pool, maximum, stats
        self.values = OrderedDict()

    def lookup(self, key):
        if key not in self.values:
            self.stats["cache_misses"] += 1
            return False, None
        value = self.values.pop(key)
        self.values[key] = value
        self.stats["cache_hits"] += 1
        return True, value

    def put(self, key, value):
        self.values.pop(key, None)
        self.values[key] = deepcopy(value)
        if len(self.values) > self.maximum:
            self.values.popitem(last=False)
            self.stats["cache_evictions"] += 1
        self.stats["cache_entries"] = len(self.values)

    def remember_bundle(self, key, bundle, spec):
        self.put(key, bundle)
        for smiles, encoded in bundle["encoded"].items():
            self.put(("encode", spec.fingerprint(), smiles), encoded)

    def get_bundle(self, graph, spec):
        key = _graph_key(graph, spec)
        found, value = self.lookup(key)
        if found:
            return value
        self.stats["worker_tasks"] += 1
        value = self.pool.submit(_native_bundle, (graph, spec)).result()
        self.remember_bundle(key, value, spec)
        return value

    def layer_bundles(self, payloads):
        """Start unique misses together; deliver successes/errors in input order."""
        keys, ready, pending = [], {}, {}
        for payload in payloads:
            graph, spec = payload[1], payload[3]
            key = _graph_key(graph, spec)
            keys.append(key)
            if key in ready or key in pending:
                self.stats["layer_duplicate_requests"] += 1
                continue
            found, value = self.lookup(key)
            if found:
                ready[key] = value
            else:
                self.stats["worker_tasks"] += 1
                pending[key] = self.pool.submit(_native_bundle, (graph, spec))
        for payload, key in zip(payloads, keys):
            if key not in ready:
                # result() is requested in original candidate order. A failure
                # cannot select a later candidate or become a censored outcome.
                value = pending[key].result()
                ready[key] = value
                self.remember_bundle(key, value, payload[3])
            yield ready[key]


class _OrderedHybridMap:
    """Map protocol consumed by the existing exact parallel search copies."""

    def __init__(self, facts, stats):
        self.facts, self.stats = facts, stats

    def map(self, function, payloads, chunksize=1):
        import parallel_search
        from mosaic.small_molecule.chemistry._reference.probability_repair import (
            measure_probability_edits,
        )

        if chunksize != 1 or function not in (
            parallel_search._base_point, parallel_search._progress_point,
        ):
            raise TypeError("Hybrid map only supports declared ordered repair points")
        payloads = list(payloads)
        for payload, bundle in zip(payloads, self.facts.layer_bundles(payloads)):
            original, candidate, probabilities, spec, config, depth = payload
            self.stats["parent_cost_calls"] += 1
            measured = measure_probability_edits(
                original, candidate, probabilities, spec, config,
            )
            if function is parallel_search._base_point:
                if (
                    measured["edit_count"] != depth
                    or measured["edit_counts"]["atom"]
                    or measured["edit_counts"]["charge"]
                ):
                    raise RuntimeError("Bond-only generator violated the direct-cost invariant")
                yield (
                    deepcopy(bundle["base_diagnostic"]), measured,
                    bundle["base_roundtrip_rejected"],
                )
            else:
                if measured["edit_count"] != depth or measured["edit_counts"]["charge"]:
                    raise RuntimeError("Progress search violated the depth/charge invariant")
                yield (
                    deepcopy(bundle["progress_diagnostic"]),
                    deepcopy(bundle["progress_vector"]), measured,
                )


@contextmanager
def install_hybrid(workers=4, max_entries=8192):
    """Temporarily install a persistent spawn pool and exact native-fact LRU.

    A separate already-tested exact cost adapter retains all candidate validity
    checks and keys cost reuse by complete current probability bytes. Native facts
    never key or cache a probability-dependent repair result. Use only one
    benchmark context at a time, from an executable guarded main entry point.
    """
    global _ACTIVE
    if type(workers) is not int:
        raise TypeError("Hybrid worker count must be an integer")
    if workers not in (2, 4, 8):
        raise ValueError("Declared hybrid worker counts are 2, 4 and 8")
    if type(max_entries) is not int or not 1 <= max_entries <= 65536:
        raise ValueError("Require a bounded positive integer native cache size")
    if _ACTIVE:
        raise RuntimeError("Hybrid contexts cannot be nested")
    import parallel_search
    from exact_reuse import install_reuse
    from mosaic.small_molecule.chemistry import preparation
    from mosaic.small_molecule.chemistry._reference import (
        atom_preserving_chemistry as base,
        chemistry,
        quality,
        repair_progress as progress,
    )

    if parallel_search._ACTIVE:
        raise RuntimeError("Hybrid installation cannot overlap a parallel context")
    stats = {
        name: 0 for name in (
            "cache_hits", "cache_misses", "cache_entries", "cache_evictions",
            "worker_tasks", "layer_duplicate_requests", "parent_cost_calls",
            "native_passthrough", "encode_passthrough",
        )
    }
    stats.update(workers=workers, start_method="spawn", reduction_order="original_candidate_order")
    pool, originals = None, []

    def patch(module, name, value):
        originals.append((module, name, getattr(module, name)))
        setattr(module, name, value)

    original_diagnose, original_inspect = quality.diagnose_graph, chemistry.inspect_graph
    original_encode = chemistry.encode_smiles
    _ACTIVE = True
    try:
        with install_reuse(mode="cost", max_entries=max_entries) as cost_stats:
            stats["cost_reuse"] = cost_stats
            pool = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=parallel_search._worker_initialize,
            )
            facts = _NativeFacts(pool, max_entries, stats)
            mapped = _OrderedHybridMap(facts, stats)

            def native_wrapper(original, field):
                def wrapped(*args, **kwargs):
                    if kwargs or len(args) != 4:
                        stats["native_passthrough"] += 1
                        return original(*args, **kwargs)
                    graph, spec = args[:3], args[3]
                    try:
                        _graph_key(graph, spec)
                    except (ValueError, TypeError, IndexError, AttributeError):
                        stats["native_passthrough"] += 1
                        return original(*args)
                    return deepcopy(facts.get_bundle(graph, spec)[field])
                return wrapped

            def encoded_wrapper(*args, **kwargs):
                if kwargs or len(args) != 2 or not isinstance(args[0], str):
                    stats["encode_passthrough"] += 1
                    return original_encode(*args, **kwargs)
                smiles, spec = args
                key = "encode", spec.fingerprint(), smiles
                found, value = facts.lookup(key)
                if found:
                    return deepcopy(value)
                # Independent final-witness calls and expected rejected SMILES
                # retain the exact original exception class/arguments. All
                # candidate-native cache misses run in the CPU pool above.
                stats["encode_passthrough"] += 1
                value = original_encode(*args)
                facts.put(key, value)
                return deepcopy(value)

            diagnosed = native_wrapper(original_diagnose, "raw_diagnostic")
            inspected = native_wrapper(original_inspect, "inspection")
            for module in (quality, base, progress):
                patch(module, "diagnose_graph", diagnosed)
            for module in (chemistry, quality, progress, preparation):
                patch(module, "inspect_graph", inspected)
            for module in (chemistry, base, progress, preparation):
                patch(module, "encode_smiles", encoded_wrapper)
            patch(
                preparation, "search_atom_preserving_repair",
                lambda raw, probabilities, spec, config: parallel_search._base_search(
                    mapped, raw, probabilities, spec, config,
                ),
            )
            patch(
                preparation, "search_progress_repair",
                lambda raw, probabilities, spec, config=None: parallel_search._progress_search(
                    mapped, raw, probabilities, spec, config,
                ),
            )
            try:
                yield stats
            finally:
                for module, name, value in reversed(originals):
                    setattr(module, name, value)
                originals.clear()
    finally:
        for module, name, value in reversed(originals):
            setattr(module, name, value)
        _ACTIVE = False
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
