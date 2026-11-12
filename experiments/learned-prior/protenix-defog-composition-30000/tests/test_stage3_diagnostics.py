"""Endpoint repair records are verified witnesses, never repaired run outcomes."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rdkit")
from mosaic.small_molecule.core.state import graph_from_hard

spec = importlib.util.spec_from_file_location("stage3_diagnostics", Path(__file__).parents[1] / "scripts/stage3_diagnostics.py")
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


def graph(elements, pairs):
    n = len(elements)
    i, j = np.triu_indices(n, 1)
    labels = np.zeros(len(i), np.int32)
    lookup = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(i, j))}
    for a, b, category in pairs:
        labels[lookup[tuple(sorted((a, b)))]] = category
    return graph_from_hard(elements, labels)


def assert_verified_witness(result, expected_edits, atoms):
    witness = result["closest_verified_witness"]
    assert witness["edit_count"] == expected_edits
    assert result["distance_upper_bound"] == expected_edits
    assert not result["distance_unknown"]
    assert witness["verification"]["rdkit_sanitized"]
    assert witness["verification"]["component_count"] == 1
    assert witness["verification"]["atoms_retained"] == atoms
    assert len(witness["element_indices"]) == atoms
    verified = diagnostics._verify(np.array(witness["element_indices"]), np.array(witness["bond_upper_indices"]))
    assert verified["rdkit_sanitized"] and verified["fragment_count"] == 1
    json.dumps(result, allow_nan=False)


def test_disconnected_capacity_bridge_has_verified_one_edit_witness():
    endpoint = graph([0, 0, 0, 0], [(0, 1, 1), (2, 3, 1)])
    result = diagnostics.assess_final(endpoint)
    assert result["raw"]["rdkit_sanitized"] and not result["raw"]["valid_connected"]
    assert result["raw"]["component_sizes"] == [2, 2]
    assert result["minimum_bond_additions_for_connectivity"] == 1
    assert_verified_witness(result, 1, 4)
    assert result["closest_verified_witness"]["edits"][0]["operation"] == "bridge"
    assert not result["minimum_is_proven"]


def test_overvalent_bond_deletion_repairs_without_dropping_any_atom():
    endpoint = graph([2, 0, 0, 0], [(0, 1, 1), (0, 2, 1), (0, 3, 1), (1, 2, 1)])
    result = diagnostics.assess_final(endpoint)
    assert not result["raw"]["rdkit_sanitized"]
    assert result["raw"]["neutral_hard_valence"]["violating_atoms"] == [0]
    assert result["raw"]["neutral_hard_valence"]["total_excess"] == 1
    assert_verified_witness(result, 1, 4)
    assert result["closest_verified_witness"]["edits"][0]["operation"] == "delete"


def test_four_bridges_needed_is_censored_beyond_three_edit_budget():
    endpoint = graph([0, 0, 0, 0, 0], [])
    result = diagnostics.assess_final(endpoint, max_edits=3, checks=128)
    assert result["raw"]["component_sizes"] == [1, 1, 1, 1, 1]
    assert result["minimum_bond_additions_for_connectivity"] == 4
    assert result["closest_verified_witness"] is None
    assert result["distance_upper_bound"] is None
    assert result["distance_unknown"]
    assert result["search"]["checks_used"] == 1
    assert result["search"]["termination"] == "connectivity_lower_bound_exceeds_edit_budget"


def test_already_valid_zero_edit_witness_and_no_mutation():
    endpoint = graph([0, 0, 2], [(0, 1, 1), (1, 2, 1)])
    before_x, before_e = np.asarray(endpoint.X).copy(), np.asarray(endpoint.E_upper).copy()
    result = diagnostics.assess_final(endpoint)
    assert result["raw"]["valid_connected"]
    assert_verified_witness(result, 0, 3)
    assert result["minimum_is_proven"]
    assert result["search"]["checks_used"] == 1
    np.testing.assert_array_equal(endpoint.X, before_x)
    np.testing.assert_array_equal(endpoint.E_upper, before_e)


def test_invalid_search_does_not_mutate_and_obeys_total_verification_budget():
    endpoint = graph([0, 0, 0], [])
    before_x, before_e = np.asarray(endpoint.X).copy(), np.asarray(endpoint.E_upper).copy()
    result = diagnostics.assess_final(endpoint, checks=1)
    assert result["distance_unknown"] and result["closest_verified_witness"] is None
    assert result["search"]["checks_used"] == 1
    assert result["search"]["termination"] == "verification_budget_exhausted"
    np.testing.assert_array_equal(endpoint.X, before_x)
    np.testing.assert_array_equal(endpoint.E_upper, before_e)


def test_three_component_search_finds_two_edits_and_count_matches_labels():
    endpoint = graph([0, 0, 0], [])
    result = diagnostics.assess_final(endpoint, max_edits=3, checks=128, beam_width=4)
    assert_verified_witness(result, 2, 3)
    assert sum(label != 0 for label in result["closest_verified_witness"]["bond_upper_indices"]) == 2
    assert result["search"]["checks_used"] <= 128


def test_high_branching_25_atom_endpoint_reserves_checks_for_three_edit_path():
    # Four disconnected carbon chains require three bridges. There are far
    # more than128 one-edit candidate bridges, none of which can finish alone.
    pairs = []
    for start, stop in ((0, 7), (7, 13), (13, 19), (19, 25)):
        pairs.extend((a, a + 1, 1) for a in range(start, stop - 1))
    endpoint = graph([0] * 25, pairs)
    result = diagnostics.assess_final(endpoint, max_edits=3, checks=128, beam_width=4)
    assert_verified_witness(result, 3, 25)
    assert result["search"]["checks_used"] <= 128
    assert result["search"]["depth_reached"] == 3


@pytest.mark.parametrize("settings", [{"checks": 0}, {"beam_width": 0}, {"max_edits": -1}, {"checks": True}])
def test_invalid_budgets_rejected(settings):
    with pytest.raises(ValueError):
        diagnostics.assess_final(graph([0], []), **settings)
