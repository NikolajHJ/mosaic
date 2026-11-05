"""Allocated CPU controls for bounded, scoped atom-exchange repair."""

import math

import numpy as np
import pytest

from mosaic.small_molecule.chemistry._reference.chemistry import inspect_graph
from mosaic.small_molecule.chemistry._reference.probability_repair import (
    ProbabilityRepairConfig,
)
from mosaic.small_molecule.chemistry._reference.representation import GraphSpec
from mosaic.small_molecule.chemistry.atom_exchange import (
    search_atom_exchange,
    within_common_organic_scope,
)
from mosaic.small_molecule.chemistry.structural import search_structural_repair


def fixture(elements, edges):
    spec = GraphSpec(max_nodes=len(elements))
    atoms = np.array([spec.atomic_numbers.index(z) for z in elements])
    charges = np.full(len(elements), spec.formal_charges.index(0))
    bonds = np.zeros((len(elements), len(elements)), dtype=np.int64)
    for i, j, order in edges:
        label = spec.bond_names.index(order)
        bonds[i, j] = bonds[j, i] = label
    probabilities = (
        np.eye(len(spec.atomic_numbers))[atoms],
        np.eye(len(spec.formal_charges))[charges],
        np.eye(len(spec.bond_names))[bonds],
    )
    return spec, (atoms, charges, bonds), probabilities


def set_alternative(raw, probabilities, spec, slot, atomic_number, mass):
    probabilities[0][slot, raw[0][slot]] = 1 - mass
    probabilities[0][slot, spec.atomic_numbers.index(atomic_number)] = mass


def test_supported_atom_exchange_recovers_complete_target_at_declared_cost():
    spec, raw, probabilities = fixture(
        (9, 9, 9, 0), [(0, 1, "SINGLE"), (1, 2, "SINGLE")],
    )
    set_alternative(raw, probabilities, spec, 1, 8, 0.2)
    saved = tuple(x.copy() for x in raw + probabilities)
    search = ProbabilityRepairConfig()
    result = search_atom_exchange(raw, probabilities, spec, search)
    assert result == search_atom_exchange(raw, probabilities, spec, search)
    assert result["baseline"]["weighted_witness"] is None
    assert result["selected"]["status"] == "witness_found"
    assert 1 <= result["substitutions_tested"] <= 4
    witness = result["selected"]["weighted_witness"]
    target = tuple(np.array(witness["labels"][name]) for name in ("atoms", "charges", "bonds"))
    assert [spec.atomic_numbers[i] for i in target[0]] == [9, 8, 9, 0]
    assert np.array_equal(target[1], raw[1])
    assert np.array_equal(target[2], raw[2])
    assert within_common_organic_scope(target, spec)
    assert inspect_graph(*target, spec).smiles == witness["smiles"]
    assert witness["repair_cost"] == pytest.approx(2 + math.log(4))
    assert all(np.array_equal(a, b) for a, b in zip(saved, raw + probabilities))


def test_zero_support_cannot_rescue_an_impossible_bond_only_graph():
    spec, raw, probabilities = fixture(
        (9, 9, 9), [(0, 1, "SINGLE"), (1, 2, "SINGLE")],
    )
    result = search_atom_exchange(raw, probabilities, spec, ProbabilityRepairConfig())
    assert result["selected"]["weighted_witness"] is None
    assert result["substitutions_tested"] == 0


def test_valid_graph_is_left_as_is():
    spec, raw, probabilities = fixture((6, 6), [(0, 1, "SINGLE")])
    set_alternative(raw, probabilities, spec, 0, 8, 0.49)
    result = search_atom_exchange(raw, probabilities, spec, ProbabilityRepairConfig())
    assert result["selected"]["status"] == "already_valid"
    assert result["substitutions_tested"] == 0


def test_atom_surcharge_can_make_bond_only_win_an_exact_tie():
    spec, raw, probabilities = fixture(
        (6, 9, 6), [(0, 1, "SINGLE"), (1, 2, "SINGLE")],
    )
    set_alternative(raw, probabilities, spec, 1, 8, 0.5)
    probabilities[2][:] = (0.5, 0.5, 0, 0, 0)
    search = ProbabilityRepairConfig()
    low = search_atom_exchange(raw, probabilities, spec, search, atom_edit_penalty=0.5)
    tie = search_atom_exchange(raw, probabilities, spec, search, atom_edit_penalty=1.0)
    assert low["selected"]["weighted_witness"]["edit_counts"]["atom"] == 1
    assert low["selected"]["weighted_witness"]["repair_cost"] == pytest.approx(1.5)
    assert tie["selected"]["weighted_witness"]["edit_counts"]["atom"] == 0
    assert tie["selected"]["weighted_witness"]["repair_cost"] == pytest.approx(2.0)


@pytest.mark.parametrize(("neighbors", "orders", "accepted"), [
    ((9,) * 6, ("SINGLE",) * 6, True),
    ((9,) * 5 + (6,), ("SINGLE",) * 6, True),
    ((6, 6, 8, 8), ("SINGLE", "SINGLE", "DOUBLE", "DOUBLE"), True),
    ((6,) * 5 + (9,), ("SINGLE",) * 6, False),
    ((6,) * 6, ("SINGLE",) * 6, False),
    ((9,) * 7, ("SINGLE",) * 7, False),
])
def test_sulfur_scope_distinguishes_coordination_and_valence(neighbors, orders, accepted):
    spec, graph, _ = fixture((16, *neighbors), [(0, i, order) for i, order in enumerate(orders, 1)])
    assert within_common_organic_scope(graph, spec) == accepted
    if accepted:
        assert inspect_graph(*graph, spec).smiles is not None


def test_scope_callback_leaves_legacy_structural_route_unchanged_by_default():
    spec, raw, probabilities = fixture((16, *(6,) * 6), [(0, i, "SINGLE") for i in range(1, 7)])
    search = ProbabilityRepairConfig()
    legacy = search_structural_repair(raw, probabilities, spec, search)
    explicit_default = search_structural_repair(raw, probabilities, spec, search, accept=None)
    assert legacy == explicit_default
    assert legacy["status"] == "already_valid"
    scoped = search_structural_repair(raw, probabilities, spec, search, accept=within_common_organic_scope)
    assert scoped["status"] != "already_valid"
    if scoped["weighted_witness"] is not None:
        labels = scoped["weighted_witness"]["labels"]
        assert within_common_organic_scope(tuple(labels[key] for key in ("atoms", "charges", "bonds")), spec)


def test_new_sulfur_cannot_rescue_overbonded_carbon_by_larger_cap_alone():
    spec, raw, probabilities = fixture((6, 6, 6, 8, 8), [
        (0, 1, "SINGLE"), (0, 2, "SINGLE"), (0, 3, "DOUBLE"), (0, 4, "DOUBLE"),
    ])
    set_alternative(raw, probabilities, spec, 0, 16, 0.49)
    hypothetical_atoms = raw[0].copy()
    hypothetical_atoms[0] = spec.atomic_numbers.index(16)
    hypothetical = (hypothetical_atoms, raw[1], raw[2])
    assert within_common_organic_scope(hypothetical, spec)
    assert inspect_graph(*hypothetical, spec).smiles is not None
    result = search_atom_exchange(raw, probabilities, spec, ProbabilityRepairConfig())
    witness = result["selected"]["weighted_witness"]
    assert witness is not None
    labels = witness["labels"]
    if labels["atoms"][0] == spec.atomic_numbers.index(16):
        bond_order = {"NONE": 0, "SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 1.5}
        assert sum(bond_order[spec.bond_names[i]] for i in labels["bonds"][0]) <= 4
        assert witness["edit_counts"]["bond"] > 0
