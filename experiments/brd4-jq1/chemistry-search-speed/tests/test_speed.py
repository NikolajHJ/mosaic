"""Allocated HPC equivalence fixtures for separate detached-search experiments.

These fixtures do not import or alter current-run launchers. Tiny search budgets
bound fixture runtime; the saved-input benchmark uses the declared production
budgets. Passing fixtures is not a chemical-efficacy or production-adoption claim.
"""

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from exact_reuse import install_reuse
import parallel_search
from parallel_search import install_parallel
from mosaic.small_molecule.chemistry import (
    FailureGuidanceConfig,
    GraphLogits,
    failure_aware_repair_loss,
    mosaic_spec,
    prepare_failure_guidance,
)
from mosaic.small_molecule.chemistry import preparation
from mosaic.small_molecule.chemistry._reference import atom_preserving_chemistry as atom
from mosaic.small_molecule.chemistry._reference import repair_progress as progress
from mosaic.small_molecule.chemistry._reference.probability_repair import (
    ProbabilityRepairConfig,
    measure_probability_edits,
)
from mosaic.small_molecule.chemistry._reference.repair_progress import ProgressSearchConfig


MODES = ("cache", "cost", "combined", "parallel")


@contextmanager
def accelerated(mode, *, max_entries=64):
    if mode == "parallel":
        with install_parallel(2) as stats:
            yield stats
    else:
        with install_reuse(mode=mode, max_entries=max_entries) as stats:
            yield stats


def case(name):
    """Fixed deterministic occupied, neutral graphs; no trained model or RNG state."""
    n = 25 if name == "random25" else 6
    spec = mosaic_spec(n)
    if name == "random25":
        rng = np.random.default_rng(17)
        logits = GraphLogits(
            jnp.asarray(rng.normal(size=(n, 6)), dtype=jnp.float32),
            jnp.asarray(rng.normal(size=(n * (n - 1) // 2, 5)), dtype=jnp.float32),
        )
    else:
        element = 4 if name == "invalid6" else 0  # F or C in Mosaic ordering.
        atoms = np.full((n, 6), -1.0, dtype=np.float32)
        atoms[:, element] = 2.0
        bonds = np.full((n * (n - 1) // 2, 5), -1.0, dtype=np.float32)
        for k, (i, j) in enumerate(zip(*np.triu_indices(n, 1))):
            present = j == i + 1 and not (name == "disconnected6" and i == 2)
            label = 3 if name == "invalid6" else (1 if present else 0)
            bonds[k, label] = 2.0
        logits = GraphLogits(jnp.asarray(atoms), jnp.asarray(bonds))
    return logits, spec


def prepare(logits, spec, *, update=7, candidate_ids=(31,), extra=True):
    return prepare_failure_guidance(
        logits.decoded(),
        spec,
        update=update,
        candidate_ids=candidate_ids,
        search=ProbabilityRepairConfig(max_edits=2, checks_per_depth=3, beam_width=2),
        progress_search=(
            ProgressSearchConfig(max_edits=2, checks_per_depth=3, beam_width=2)
            if extra else None
        ),
    )


def assert_prepared_equal(left, right, spec):
    left.verify(spec)
    right.verify(spec)
    assert left.identity == right.identity
    assert left.receipt == right.receipt
    assert set(left.arrays) == set(right.arrays)
    for key in left.arrays:
        a, b = np.asarray(left.arrays[key]), np.asarray(right.arrays[key])
        assert a.dtype == b.dtype and a.shape == b.shape
        assert a.tobytes() == b.tobytes(), key
    # Timings are external to identity and must remain honest nonnegative data.
    for item in (left, right):
        assert item.preparation_seconds >= 0
        assert len(item.search_seconds) == item.receipt["attempted_rows"]
        assert all(a >= 0 and b >= 0 for a, b in item.search_seconds)


def raw_case(name="invalid6"):
    logits, spec = case(name)
    decoded = logits.decoded()
    probabilities = tuple(
        np.asarray(jax.nn.softmax(v, -1))[0]
        for v in (decoded.atoms, decoded.charges, decoded.symmetric_bond_logits)
    )
    raw = tuple(np.argmax(v, axis=-1).astype(np.int64) for v in probabilities)
    raw[2][np.diag_indices(spec.max_nodes)] = 0
    return raw, probabilities, spec


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", ("chain6", "disconnected6", "invalid6", "random25"))
def test_complete_preparation_identity_receipt_and_arrays_exact(mode, name):
    logits, spec = case(name)
    expected = prepare(logits, spec)
    with accelerated(mode):
        actual = prepare(logits, spec)
        repeated = prepare(logits, spec)
    assert_prepared_equal(expected, actual, spec)
    assert_prepared_equal(expected, repeated, spec)
    if name == "disconnected6":
        assert actual.receipt["original_valid_targets"] == 1


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name", ("chain6", "disconnected6", "invalid6"))
def test_fixed_target_total_components_and_every_logit_gradient_exact(mode, name):
    logits, spec = case(name)
    expected = prepare(logits, spec)
    with accelerated(mode):
        actual = prepare(logits, spec)
    config = FailureGuidanceConfig()
    def evaluate(prepared):
        def objective(value):
            return failure_aware_repair_loss(value.decoded(), prepared, spec, config)
        (value, terms), gradient = eqx.filter_value_and_grad(objective, has_aux=True)(logits)
        jax.block_until_ready((value, terms, gradient))
        return value, terms, gradient
    baseline = evaluate(expected)
    observed = evaluate(actual)
    for a, b in zip(jax.tree.leaves(baseline), jax.tree.leaves(observed)):
        a, b = np.asarray(a), np.asarray(b)
        assert np.isfinite(a).all() and np.isfinite(b).all()
        assert a.dtype == b.dtype and a.shape == b.shape
        assert a.tobytes() == b.tobytes()
    if name == "invalid6":
        assert float(baseline[0]) > 0
        assert np.any(np.asarray(baseline[2].bonds) != 0)


@pytest.mark.parametrize("mode", MODES)
def test_changed_probabilities_same_hard_graph_recompute_complete_target(mode):
    logits, spec = case("invalid6")
    changed = eqx.tree_at(lambda x: x.bonds, logits, logits.bonds.at[:, 1].add(0.125))
    np.testing.assert_array_equal(np.argmax(logits.bonds, -1), np.argmax(changed.bonds, -1))
    expected = prepare(changed, spec)
    with accelerated(mode):
        first = prepare(logits, spec)
        actual = prepare(changed, spec)
    assert first.receipt["rows"][0]["probability_sha256"] != actual.receipt["rows"][0]["probability_sha256"]
    assert_prepared_equal(expected, actual, spec)


@pytest.mark.parametrize("mode", MODES)
def test_update_and_candidate_identity_not_reused(mode):
    logits, spec = case("chain6")
    with accelerated(mode):
        before = prepare(logits, spec)
        after = prepare(logits, spec, update=8, candidate_ids=(53,))
    assert before.identity != after.identity
    assert after.receipt["source_update"] == 8
    assert after.receipt["candidate_ids"] == (53,)


@pytest.mark.parametrize("mode", MODES)
def test_invalid_source_logits_and_candidate_ids_still_fail(mode):
    logits, spec = case("chain6")
    invalid = eqx.tree_at(lambda x: x.atoms, logits, logits.atoms.at[0, 0].set(jnp.nan))
    with accelerated(mode):
        with pytest.raises(ValueError, match="finite"):
            prepare(invalid, spec)
        with pytest.raises(ValueError, match="candidate"):
            prepare(logits, spec, candidate_ids=(-1,))


@pytest.mark.parametrize("mode", MODES)
def test_context_restores_every_public_search_binding_after_error(mode):
    before = (preparation.search_atom_preserving_repair, preparation.search_progress_repair,
              atom.diagnose_graph, atom.measure_probability_edits, progress.diagnose_graph)
    with pytest.raises(RuntimeError, match="fixture exception"):
        with accelerated(mode):
            raise RuntimeError("fixture exception")
    after = (preparation.search_atom_preserving_repair, preparation.search_progress_repair,
             atom.diagnose_graph, atom.measure_probability_edits, progress.diagnose_graph)
    assert all(a is b for a, b in zip(before, after))


@pytest.mark.parametrize("mode", ("cache", "combined"))
def test_cached_native_diagnostic_returns_deep_copy_and_bounded_entries(mode):
    raw, _, spec = raw_case()
    expected = atom.diagnose_graph(*raw, spec)
    with install_reuse(mode=mode, max_entries=2) as stats:
        first = atom.diagnose_graph(*raw, spec)
        first["fixture_mutation"] = {"nested": [123]}
        if first.get("problems"):
            first["problems"][0]["fixture_mutation"] = True
        again = atom.diagnose_graph(*raw, spec)
        assert again == expected
        assert stats["cache_hits"] > 0
        for value in (0, 1, 2, 3):
            graph = tuple(v.copy() for v in raw)
            graph[2][0, 1] = graph[2][1, 0] = value
            atom.diagnose_graph(*graph, spec)
        assert stats["cache_entries"] <= 2
        assert stats["cache_evictions"] > 0


@pytest.mark.parametrize("mode", ("cache", "combined"))
def test_changed_hard_graph_and_schema_invalidate_native_fact_cache(mode):
    raw, _, spec = raw_case()
    other = tuple(v.copy() for v in raw)
    other[0][0] = spec.atomic_numbers.index(6)
    schema = replace(spec, atomic_numbers=(0, 7, 6, 8, 16, 9, 17))
    expected_other = atom.diagnose_graph(*other, spec)
    expected_schema = atom.diagnose_graph(*raw, schema)
    with install_reuse(mode=mode) as stats:
        atom.diagnose_graph(*raw, spec)
        misses = stats["cache_misses"]
        assert atom.diagnose_graph(*other, spec) == expected_other
        assert stats["cache_misses"] > misses
        misses = stats["cache_misses"]
        assert atom.diagnose_graph(*raw, schema) == expected_schema
        assert stats["cache_misses"] > misses


@pytest.mark.parametrize("mode", ("cost", "combined"))
def test_cost_probabilities_and_all_replacement_labels_revalidated(mode):
    raw, probabilities, spec = raw_case()
    repaired = tuple(v.copy() for v in raw)
    repaired[2][0, 1] = repaired[2][1, 0] = 1
    config = ProbabilityRepairConfig()
    expected = measure_probability_edits(raw, repaired, probabilities, spec, config)
    changed = tuple(v.copy() for v in probabilities)
    changed[2][0, 1, 1] += 0.01
    changed[2][0, 1, 3] -= 0.01
    changed[2][1, 0] = changed[2][0, 1]
    expected_changed = measure_probability_edits(raw, repaired, changed, spec, config)
    with install_reuse(mode=mode):
        assert atom.measure_probability_edits(raw, repaired, probabilities, spec, config) == expected
        assert atom.measure_probability_edits(raw, repaired, changed, spec, config) == expected_changed
    assert expected["weighted_cost"] != expected_changed["weighted_cost"]


@pytest.mark.parametrize("mode", ("cost", "combined"))
def test_cached_cost_terms_deep_copy_config_invalidation_and_bound(mode):
    raw, probabilities, spec = raw_case()
    repaired = tuple(v.copy() for v in raw)
    repaired[2][0, 1] = repaired[2][1, 0] = 1
    config = ProbabilityRepairConfig()
    expected = measure_probability_edits(raw, repaired, probabilities, spec, config)
    with install_reuse(mode=mode, max_entries=2) as stats:
        first = atom.measure_probability_edits(raw, repaired, probabilities, spec, config)
        first["changes"][0]["slots"].append(999)
        assert atom.measure_probability_edits(raw, repaired, probabilities, spec, config) == expected
        for weight in (0.0, 0.5, 2.0, 3.0):
            changed = replace(config, probability_weight=weight)
            actual = atom.measure_probability_edits(raw, repaired, probabilities, spec, changed)
            assert actual == measure_probability_edits(raw, repaired, probabilities, spec, changed)
        assert stats["cost_entries"] <= 2
        assert stats["cost_evictions"] > 0


@pytest.mark.parametrize("mode", MODES)
def test_progress_atom_edit_branch_preserves_order_witness_and_receipt(mode):
    logits, spec = case("invalid6")
    options = {
        "search": ProbabilityRepairConfig(max_edits=1, checks_per_depth=1, beam_width=1),
        "progress_search": ProgressSearchConfig(
            max_edits=2, checks_per_depth=3, beam_width=2,
            allow_atom_edits=True, max_atom_edits=1,
        ),
    }
    expected = prepare_failure_guidance(logits.decoded(), spec, **options)
    with accelerated(mode):
        actual = prepare_failure_guidance(logits.decoded(), spec, **options)
    assert_prepared_equal(expected, actual, spec)


@pytest.mark.parametrize("mode", ("cost", "combined"))
@pytest.mark.parametrize("fault", ("nan", "asymmetric", "negative", "label"))
def test_cost_validation_never_censors_bad_inputs(mode, fault):
    raw, probabilities, spec = raw_case()
    bad = tuple(v.copy() for v in probabilities)
    graph = tuple(v.copy() for v in raw)
    if fault == "nan":
        bad[0][0, 0] = np.nan
    elif fault == "asymmetric":
        bad[2][0, 1] = bad[2][0, 1, ::-1]
    elif fault == "negative":
        bad[0][0, 0] = -0.01
    else:
        graph[0][0] = 999
    config = ProbabilityRepairConfig()
    with pytest.raises(ValueError) as baseline:
        measure_probability_edits(raw, graph, bad, spec, config)
    with install_reuse(mode=mode):
        with pytest.raises(type(baseline.value)) as actual:
            atom.measure_probability_edits(raw, graph, bad, spec, config)
    assert str(actual.value) == str(baseline.value)


@pytest.mark.parametrize("mode", ("cache", "cost", "combined"))
def test_reuse_nesting_refused_without_losing_outer_bindings(mode):
    original = preparation.search_atom_preserving_repair
    with install_reuse(mode=mode):
        outer = preparation.search_atom_preserving_repair
        with pytest.raises(RuntimeError):
            with install_reuse(mode=mode):
                pytest.fail("Nested reuse should fail")
        assert preparation.search_atom_preserving_repair is outer
    assert preparation.search_atom_preserving_repair is original


@pytest.mark.parametrize("kwargs", ({"mode": "unknown"}, {"max_entries": 0}, {"max_entries": True}))
def test_reuse_configuration_fails_closed(kwargs):
    with pytest.raises((ValueError, TypeError)):
        with install_reuse(**kwargs):
            pytest.fail("Invalid cache configuration was accepted")


@pytest.mark.parametrize("workers", (0, 1, 3, 9, True, 2.0))
def test_parallel_configuration_fails_closed(workers):
    with pytest.raises((ValueError, TypeError)):
        with install_parallel(workers):
            pytest.fail("Invalid parallel configuration was accepted")


def test_parallel_nested_context_and_worker_validation_restore_pool():
    original = preparation.search_atom_preserving_repair
    raw, probabilities, spec = raw_case()
    bad = tuple(v.copy() for v in probabilities)
    bad[0][0, 0] = np.nan
    with install_parallel(2) as metadata:
        assert metadata["start_method"] == "spawn"
        assert metadata["reduction_order"] == "original_candidate_order"
        with pytest.raises(RuntimeError):
            with install_parallel(2):
                pytest.fail("Nested pools should fail")
        with pytest.raises(ValueError):
            preparation.search_atom_preserving_repair(raw, bad, spec, ProbabilityRepairConfig())
        # A validation failure cannot poison the subsequent real preparation.
        logits, spec = case("chain6")
        prepared = prepare(logits, spec)
        prepared.verify(spec)
    assert preparation.search_atom_preserving_repair is original


def test_actual_spawned_worker_failure_propagates_and_context_restores(monkeypatch):
    """Use a pickleable builtin to force a real worker TypeError, not a toy pool."""
    original = preparation.search_atom_preserving_repair
    logits, spec = case("invalid6")
    with pytest.raises(TypeError):
        with install_parallel(2):
            # int(tuple_payload) fails inside ProcessPoolExecutor's spawned child.
            with monkeypatch.context() as patch:
                patch.setattr(parallel_search, "_base_point", int)
                prepare(logits, spec)
    assert preparation.search_atom_preserving_repair is original
    # The original sequential path still prepares the same source after cleanup.
    prepare(logits, spec).verify(spec)


def test_prepared_target_mutation_still_rejected_after_acceleration():
    logits, spec = case("invalid6")
    with install_reuse(mode="combined"):
        prepared = prepare(logits, spec)
    prepared.receipt["source_update"] += 1
    with pytest.raises(ValueError, match="identity"):
        failure_aware_repair_loss(logits.decoded(), prepared, spec)
