"""Allocated-only exactness and lifecycle contracts for the hybrid adapter."""

from dataclasses import replace
from pathlib import Path
import sys

import equinox as eqx
import jax
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from hybrid_search import install_hybrid
import hybrid_search
from test_speed import case, prepare, raw_case, assert_prepared_equal
from mosaic.small_molecule.chemistry import failure_aware_repair_loss
from mosaic.small_molecule.chemistry import preparation
from mosaic.small_molecule.chemistry._reference import atom_preserving_chemistry as atom
from mosaic.small_molecule.chemistry._reference import repair_progress as progress
from mosaic.small_molecule.chemistry._reference import quality


@pytest.mark.parametrize("name", ("chain6", "disconnected6", "invalid6", "random25"))
def test_hybrid_entire_preparation_scientific_identity_matches_original(name):
    logits, spec = case(name)
    original = prepare(logits, spec)
    with install_hybrid(workers=2, max_entries=64) as stats:
        first = prepare(logits, spec)
        warm = prepare(logits, spec)
    assert_prepared_equal(original, first, spec)
    assert_prepared_equal(original, warm, spec)
    assert stats["cache_entries"] <= 64
    assert stats["cache_hits"] > 0
    if name in ("invalid6", "random25"):
        assert stats["worker_tasks"] > 0


@pytest.mark.parametrize("name", ("chain6", "disconnected6", "invalid6"))
def test_hybrid_total_every_loss_component_and_every_gradient_bitwise(name):
    logits, spec = case(name)
    original = prepare(logits, spec)
    with install_hybrid(workers=2):
        hybrid = prepare(logits, spec)
    def evaluate(target):
        def objective(state):
            return failure_aware_repair_loss(state.decoded(), target, spec)
        result = eqx.filter_value_and_grad(objective, has_aux=True)(logits)
        jax.block_until_ready(result)
        return result
    expected, actual = evaluate(original), evaluate(hybrid)
    for left, right in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
        left, right = np.asarray(left), np.asarray(right)
        assert np.isfinite(left).all() and np.isfinite(right).all()
        assert left.dtype == right.dtype and left.shape == right.shape
        assert left.tobytes() == right.tobytes()


def test_hybrid_probability_changes_recompute_cost_target_and_probability_binding():
    logits, spec = case("invalid6")
    changed = eqx.tree_at(lambda x: x.bonds, logits, logits.bonds.at[:, 1].add(0.125))
    np.testing.assert_array_equal(np.argmax(logits.bonds, -1), np.argmax(changed.bonds, -1))
    expected = prepare(changed, spec)
    with install_hybrid(workers=2) as stats:
        previous = prepare(logits, spec)
        hits = stats["cache_hits"]
        actual = prepare(changed, spec)
        assert stats["cache_hits"] > hits
    assert_prepared_equal(expected, actual, spec)
    assert previous.receipt["rows"][0]["probability_sha256"] != actual.receipt["rows"][0]["probability_sha256"]


def test_hybrid_source_update_candidate_ids_not_cached_as_prepared_batch():
    logits, spec = case("chain6")
    with install_hybrid(workers=2):
        first = prepare(logits, spec)
        changed = prepare(logits, spec, update=23, candidate_ids=(49,))
    assert first.identity != changed.identity
    assert changed.receipt["source_update"] == 23
    assert changed.receipt["candidate_ids"] == (49,)


def test_hybrid_hard_graph_schema_invalidate_and_cached_native_facts_are_private():
    raw, _, spec = raw_case()
    changed = tuple(value.copy() for value in raw)
    changed[0][0] = spec.atomic_numbers.index(6)
    schema = replace(spec, atomic_numbers=(0, 7, 6, 8, 16, 9, 17))
    expected = atom.diagnose_graph(*raw, spec)
    expected_changed = atom.diagnose_graph(*changed, spec)
    expected_schema = atom.diagnose_graph(*raw, schema)
    with install_hybrid(workers=2) as stats:
        first = atom.diagnose_graph(*raw, spec)
        first["fixture_mutation"] = [999]
        if first.get("problems"):
            first["problems"][0]["fixture_mutation"] = True
        assert atom.diagnose_graph(*raw, spec) == expected
        misses = stats["cache_misses"]
        assert atom.diagnose_graph(*changed, spec) == expected_changed
        assert stats["cache_misses"] > misses
        misses = stats["cache_misses"]
        assert atom.diagnose_graph(*raw, schema) == expected_schema
        assert stats["cache_misses"] > misses


def test_hybrid_cache_bound_and_eviction_do_not_change_diagnostics():
    raw, _, spec = raw_case()
    graphs = []
    for label in (0, 1, 2, 3):
        graph = tuple(value.copy() for value in raw)
        graph[2][0, 1] = graph[2][1, 0] = label
        graphs.append((graph, atom.diagnose_graph(*graph, spec)))
    with install_hybrid(workers=2, max_entries=2) as stats:
        for graph, expected in graphs:
            assert atom.diagnose_graph(*graph, spec) == expected
        for graph, expected in reversed(graphs):
            assert atom.diagnose_graph(*graph, spec) == expected
        assert stats["cache_entries"] <= 2
        assert stats["cache_evictions"] > 0


def _bindings():
    return (
        preparation.search_atom_preserving_repair,
        preparation.search_progress_repair,
        atom.diagnose_graph,
        progress.diagnose_graph,
        quality.inspect_graph,
        atom.measure_probability_edits,
    )


def test_hybrid_context_restores_original_bindings_after_body_error():
    original = _bindings()
    with pytest.raises(LookupError, match="body"):
        with install_hybrid(workers=2):
            raise LookupError("body")
    assert all(left is right for left, right in zip(original, _bindings()))


def test_hybrid_nested_context_refused_without_losing_outer_context():
    original = _bindings()
    with install_hybrid(workers=2):
        active = _bindings()
        with pytest.raises(RuntimeError):
            with install_hybrid(workers=2):
                pytest.fail("Nested hybrid context was accepted")
        assert all(left is right for left, right in zip(active, _bindings()))
    assert all(left is right for left, right in zip(original, _bindings()))


def test_hybrid_bad_probabilities_still_raise_before_search_and_restore():
    raw, probabilities, spec = raw_case()
    bad = tuple(value.copy() for value in probabilities)
    bad[0][0, 0] = np.nan
    config = atom.ProbabilityRepairConfig(max_edits=1, checks_per_depth=2, beam_width=1)
    with pytest.raises(ValueError) as baseline:
        preparation.search_atom_preserving_repair(raw, bad, spec, config)
    original = _bindings()
    with pytest.raises(type(baseline.value)) as observed:
        with install_hybrid(workers=2):
            preparation.search_atom_preserving_repair(raw, bad, spec, config)
    assert str(observed.value) == str(baseline.value)
    assert all(left is right for left, right in zip(original, _bindings()))


def test_hybrid_real_spawned_worker_error_propagates_and_restores(monkeypatch):
    logits, spec = case("invalid6")
    original = _bindings()
    with pytest.raises(TypeError):
        with install_hybrid(workers=2):
            with monkeypatch.context() as patch:
                # A pickleable builtin raises on the graph/spec tuple in the
                # actual spawned child. No exception can become a cache fact.
                patch.setattr(hybrid_search, "_native_bundle", int)
                prepare(logits, spec)
    assert all(left is right for left, right in zip(original, _bindings()))
    prepare(logits, spec).verify(spec)


@pytest.mark.parametrize("workers", (0, 1, 3, 9, True, 2.0))
def test_hybrid_worker_count_guard(workers):
    with pytest.raises((TypeError, ValueError)):
        with install_hybrid(workers=workers):
            pytest.fail("Unsupported worker count was accepted")


@pytest.mark.parametrize("limit", (0, -1, True, 1.5, 65537))
def test_hybrid_cache_limit_guard(limit):
    with pytest.raises((TypeError, ValueError)):
        with install_hybrid(workers=2, max_entries=limit):
            pytest.fail("Unsupported cache bound was accepted")
