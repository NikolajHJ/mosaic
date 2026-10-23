"""Probability-aware chemistry diagnostics; execute numerical fixtures on HPC."""
import math

import numpy as np
import pytest

from latentmol.chemistry import encode_smiles
from latentmol.probability_repair import (ProbabilityRepairConfig, ProbabilityRepairEvaluator,
    measure_probability_edits, search_probability_repair)
from latentmol.representation import GraphSpec


def fixture():
    spec = GraphSpec(max_nodes=4)
    encoded = encode_smiles('CO', spec)
    valid = tuple(encoded[k].copy() for k in ('atom_types', 'charges', 'bond_types'))
    raw = tuple(a.copy() for a in valid)
    raw[2][0, 1] = raw[2][1, 0] = spec.bond_names.index('TRIPLE')
    p = (np.eye(len(spec.atomic_numbers))[raw[0]], np.eye(len(spec.formal_charges))[raw[1]],
         np.eye(len(spec.bond_names))[raw[2]])
    return spec, raw, valid, p


def with_bond_probabilities(p, single, double, triple):
    p = tuple(a.copy() for a in p)
    p[2][0, 1] = p[2][1, 0] = [0, single, double, triple, 0]
    return p


def test_probability_supported_same_edit_is_closer_and_bond_counts_once():
    spec, raw, valid, p = fixture()
    near = measure_probability_edits(raw, valid, with_bond_probabilities(p, .49, 0, .51), spec)
    far = measure_probability_edits(raw, valid, with_bond_probabilities(p, .001, 0, .999), spec)
    assert near['edit_count'] == far['edit_count'] == 1
    assert near['edit_counts'] == {'atom': 0, 'charge': 0, 'bond': 1}
    assert near['weighted_cost'] == pytest.approx(1 + math.log(.51 / .49))
    assert near['weighted_cost'] < far['weighted_cost']
    assert near['replacement_mass'] == pytest.approx(.49)


def test_search_continues_past_first_witness_and_preserves_raw_graph():
    spec, raw, valid, p = fixture()
    before = tuple(a.copy() for a in raw)
    p = with_bond_probabilities(p, .001, .489, .510)
    report = search_probability_repair(raw, p, spec, ProbabilityRepairConfig(max_edits=1, checks_per_depth=100))
    assert report['status'] == 'witness_found'
    assert report['best_observed_edit_count'] == 1 and not report['distance_is_exact']
    # The strongly supported double bond repairs oxygen valence and wins over
    # the equally short but very unlikely single-bond repair.
    assert report['weighted_witness']['changes'][0]['new_label'] == spec.bond_names.index('DOUBLE')
    assert report['best_observed_weighted_cost'] == pytest.approx(1 + math.log(.510 / .489))
    assert all(np.array_equal(a, b) for a, b in zip(raw, before))


def test_valid_input_has_zero_distance_and_no_candidate_checks():
    spec, raw, valid, p = fixture()
    report = search_probability_repair(valid, p, spec)
    assert report['status'] == 'already_valid' and report['checks'] == 0
    assert report['weighted_proximity'] == 1 and report['distance_is_exact']


def test_probability_aware_cache_does_not_reuse_other_distribution():
    spec, raw, valid, p = fixture()
    evaluator = ProbabilityRepairEvaluator(spec, ProbabilityRepairConfig(max_edits=1, checks_per_depth=100))
    near_p = with_bond_probabilities(p, .49, 0, .51)
    first = evaluator.diagnose(*raw, probabilities=near_p)
    second = evaluator.diagnose(*raw, probabilities=with_bond_probabilities(p, .001, 0, .999))
    assert first['probability_sha256'] != second['probability_sha256']
    assert evaluator.cache_hits == 0
    assert first['best_observed_weighted_cost'] < second['best_observed_weighted_cost']
    first['weighted_witness']['smiles'] = 'mutated external return'
    again = evaluator.diagnose(*raw, probabilities=near_p)
    assert evaluator.cache_hits == 1 and again['weighted_witness']['smiles'] != first['weighted_witness']['smiles']


def test_zero_support_clipping_is_explicit_and_true_mass_stays_zero():
    spec, raw, valid, p = fixture()
    result = measure_probability_edits(raw, valid, p, spec)
    assert result['zero_support_changes'] == result['clipped_changes'] == 1
    assert result['replacement_mass'] == 0 and result['replacement_log_mass'] is None
    assert math.isfinite(result['weighted_cost'])


def test_higher_probability_sample_replacement_still_costs_an_edit():
    spec, raw, valid, p = fixture()
    result = measure_probability_edits(raw, valid, with_bond_probabilities(p, .8, 0, .2), spec)
    assert result['edit_count'] == 1 and result['probability_regret_nats'] == 0
    assert result['weighted_cost'] == 1


def test_fixed_occupancy_rejects_atom_deletion():
    spec, raw, valid, p = fixture()
    deleted = tuple(a.copy() for a in valid)
    deleted[0][1] = 0
    with pytest.raises(ValueError, match='occupancy'):
        measure_probability_edits(raw, deleted, p, spec)


@pytest.mark.parametrize('bad', ['negative', 'unnormalized', 'asymmetric', 'nan'])
def test_probability_contract_is_checked(bad):
    spec, raw, valid, p = fixture()
    p = tuple(a.copy() for a in p)
    if bad == 'negative': p[0][0, 1] = -1
    if bad == 'unnormalized': p[1][0] = 0
    if bad == 'asymmetric': p[2][0, 1] = p[2][0, 0]
    if bad == 'nan': p[0][0, 1] = np.nan
    with pytest.raises(ValueError):
        measure_probability_edits(raw, valid, p, spec)


def test_empty_graph_is_unsupported_and_not_a_perfect_repair():
    spec, raw, valid, p = fixture()
    empty = (np.zeros(4, dtype=int), np.full(4, spec.formal_charges.index(0)), np.zeros((4, 4), dtype=int))
    report = search_probability_repair(empty, p, spec)
    assert report['status'] == 'unsupported' and report['weighted_proximity'] is None


def test_capped_failure_remains_unknown_not_a_large_invented_distance():
    spec = GraphSpec(max_nodes=6)
    encoded = encode_smiles('CCCCCC', spec)
    raw = tuple(encoded[k].copy() for k in ('atom_types', 'charges', 'bond_types'))
    raw[2][:] = 1
    np.fill_diagonal(raw[2], 0)  # Six overvalent carbons: one edit cannot fix all.
    p = (np.eye(len(spec.atomic_numbers))[raw[0]], np.eye(len(spec.formal_charges))[raw[1]],
         np.eye(len(spec.bond_names))[raw[2]])
    report = search_probability_repair(raw, p, spec, ProbabilityRepairConfig(max_edits=1, checks_per_depth=1))
    assert report['status'] == 'censored' and report['checks'] == 1
    assert report['weighted_proximity'] is report['best_observed_edit_count'] is None
    assert report['checks_by_depth'][0]['capped']
