"""Focused structural-discovery and unchanged-loss controls (allocated CPU)."""
from dataclasses import replace

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from mosaic.small_molecule.chemistry import (
    GraphLogits, failure_aware_repair_loss, mosaic_spec, prepare_failure_guidance,
)
from mosaic.small_molecule.chemistry import structural
from mosaic.small_molecule.chemistry._reference.chemistry import inspect_graph
from mosaic.small_molecule.chemistry._reference.probability_repair import ProbabilityRepairConfig
from mosaic.small_molecule.chemistry._reference.repair_progress import ProgressSearchConfig
from mosaic.small_molecule.chemistry.structural import StructuralRepairConfig, search_structural_repair


def source(elements, edges):
    spec = mosaic_spec(len(elements))
    atoms = np.asarray([spec.atomic_numbers.index(a) for a in elements], np.int64)
    charges = np.zeros(len(elements), np.int64)
    bonds = np.zeros((len(elements), len(elements)), np.int64)
    for i, j, name in edges:
        bonds[i, j] = bonds[j, i] = spec.bond_names.index(name)
    probabilities = []
    for labels, size in zip((atoms, charges, bonds), (len(spec.atomic_numbers), 1, len(spec.bond_names))):
        p = np.full(labels.shape + (size,), .1 / size)
        np.put_along_axis(p, labels[..., None], .9 + .1 / size, axis=-1)
        if size == 1:
            p[...] = 1.
        probabilities.append(p)
    return spec, (atoms, charges, bonds), tuple(probabilities)


def run(elements, edges, **options):
    spec, raw, p = source(elements, edges)
    result = search_structural_repair(raw, p, spec, ProbabilityRepairConfig(), StructuralRepairConfig(**options))
    return spec, raw, p, result


@pytest.mark.parametrize('options', [
    {'version': 'wrong'}, {'max_candidates': 0}, {'max_candidates': 7},
    {'max_candidates': True}, {'max_bond_operations': 0},
    {'max_bond_operations': 4097}, {'max_bond_operations': 1.5},
])
def test_invalid_or_unbounded_options(options):
    with pytest.raises(ValueError):
        StructuralRepairConfig(**options)


@pytest.mark.parametrize('elements,edges', [
    ([6] * 7, [(i, j, 'SINGLE') for i in range(7) for j in range(i + 1, 7)]),
    ([6] * 8, []),
    ([6, 6, 9], [(0, 1, 'SINGLE'), (0, 2, 'SINGLE'), (1, 2, 'SINGLE')]),
    ([9, 9, 6, 6], [(0, 1, 'SINGLE'), (0, 2, 'SINGLE'), (0, 3, 'DOUBLE')]),
])
def test_complete_targets_preserve_labels_inputs_and_connected_roundtrip(elements, edges):
    spec, raw, p = source(elements, edges)
    before = [v.copy() for v in (*raw, *p)]
    result = search_structural_repair(raw, p, spec, ProbabilityRepairConfig())
    assert result['status'] == 'witness_found'
    labels = result['weighted_witness']['labels']
    target = tuple(np.asarray(labels[k]) for k in ('atoms', 'charges', 'bonds'))
    np.testing.assert_array_equal(target[0], raw[0])
    np.testing.assert_array_equal(target[1], raw[1])
    assert inspect_graph(*target, spec).smiles is not None
    assert result['checks'] <= 7
    assert result['distance_is_exact'] is False
    assert result['weighted_witness']['edit_counts']['atom'] == 0
    assert result['weighted_witness']['edit_counts']['charge'] == 0
    for wanted, actual in zip(before, (*raw, *p)):
        np.testing.assert_array_equal(wanted, actual)


def test_many_components_can_require_more_than_the_old_edit_radius():
    _, _, _, result = run([6] * 8, [])
    assert result['status'] == 'witness_found'
    assert result['weighted_witness']['edit_count'] == 7


def test_already_valid_has_no_self_target():
    _, _, _, result = run([6, 6, 8], [(0, 1, 'SINGLE'), (1, 2, 'SINGLE')])
    assert result['status'] == 'already_valid'
    assert result['weighted_witness'] is None and result['raw_valid']


def test_impossible_fixed_halogen_composition_remains_a_miss():
    _, _, _, result = run([9] * 5, [])
    assert result['status'] == 'censored' and result['weighted_witness'] is None
    assert 'degree capacity' in result['reason']


def test_small_budget_does_not_hide_partial_target_as_valid():
    _, _, _, result = run([6] * 8, [], max_bond_operations=1)
    assert result['status'] == 'censored' and result['weighted_witness'] is None


def test_probability_regret_prefers_more_supported_connection():
    spec, raw, p = source([6] * 4, [(0, 1, 'SINGLE'), (2, 3, 'SINGLE')])
    p[2][0, 2] = p[2][2, 0] = np.asarray([.55, .4, .02, .02, .01])
    result = search_structural_repair(raw, p, spec, ProbabilityRepairConfig())
    assert result['status'] == 'witness_found'
    assert result['weighted_witness']['labels']['bonds'][0][2] == spec.bond_names.index('SINGLE')


def test_roundtrip_failure_is_never_a_complete_target(monkeypatch):
    monkeypatch.setattr(structural, 'encode_smiles', lambda *a: {'smiles': 'wrong'})
    _, _, _, result = run([6] * 5, [])
    assert result['status'] == 'censored' and result['weighted_witness'] is None


def test_unexpected_native_error_is_not_swallowed(monkeypatch):
    def broken(*args):
        raise RuntimeError('real tooling error')
    monkeypatch.setattr(structural, 'inspect_graph', broken)
    with pytest.raises(RuntimeError, match='real tooling error'):
        run([6] * 5, [])


def test_default_preparation_and_failed_structural_fallback_keep_same_loss():
    spec = mosaic_spec(3)
    logits = GraphLogits(jnp.zeros((3, 6), jnp.float32).at[:, 4].set(4.),
                        jnp.zeros((3, 5), jnp.float32).at[:, 0].set(4.))
    search = ProbabilityRepairConfig(max_edits=1, checks_per_depth=4, beam_width=1)
    progress = ProgressSearchConfig(max_edits=1, checks_per_depth=4, beam_width=1)
    baseline = prepare_failure_guidance(logits.decoded(), spec, search=search, progress_search=progress)
    candidate = prepare_failure_guidance(logits.decoded(), spec, search=search, progress_search=progress,
                                         structural_repair=StructuralRepairConfig())
    assert 'structural_repair' not in baseline.receipt
    assert candidate.receipt['rows'][0]['structural']['status'] == 'censored'
    assert candidate.receipt['rows'][0]['beam_search_performed']
    for key in baseline.arrays:
        np.testing.assert_array_equal(baseline.arrays[key], candidate.arrays[key])
    old = failure_aware_repair_loss(logits.decoded(), baseline, spec)
    new = failure_aware_repair_loss(logits.decoded(), candidate, spec)
    np.testing.assert_array_equal(old[0], new[0])
    assert old[1].keys() == new[1].keys()
    for key in old[1]:
        np.testing.assert_array_equal(old[1][key], new[1][key])


def test_full_target_uses_existing_loss_and_is_fixed_through_trial_gradients():
    n = 7
    logits = GraphLogits(jnp.zeros((n, 6), jnp.float32).at[:, 0].set(3.),
                        jnp.zeros((n * (n - 1) // 2, 5), jnp.float32).at[:, 1].set(3.))
    spec = mosaic_spec(n)
    prepared = prepare_failure_guidance(logits.decoded(), spec, structural_repair=StructuralRepairConfig())
    assert int(prepared.arrays['route'][0]) == 1
    identity = prepared.identity
    def value(point):
        return failure_aware_repair_loss(point.decoded(), prepared, spec)[0]
    before, gradient = eqx.filter_value_and_grad(value)(logits)
    trial = GraphLogits(logits.atoms - .1 * gradient.atoms, logits.bonds - .1 * gradient.bonds)
    after = value(trial)
    prepared.verify(spec)
    assert prepared.identity == identity and np.isfinite(float(before)) and np.isfinite(float(after))
    assert np.linalg.norm(np.asarray(gradient.bonds)) > 0
    _, terms = failure_aware_repair_loss(logits.decoded(), prepared, spec)
    np.testing.assert_allclose(before, terms['valid_full_nll'], rtol=1e-6)
    assert all(float(v) == 0 for k, v in terms.items() if k != 'valid_full_nll')
