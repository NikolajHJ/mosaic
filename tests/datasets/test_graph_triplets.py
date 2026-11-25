"""Verify real molecule identity and disjoint edit maps in a small split."""

import numpy as np

from latentmol.chemistry import encode_smiles
from latentmol.datasets.graph_triplets import select_triplets
from latentmol.representation import GraphSpec


class FixtureSplit:
    spec = GraphSpec(max_nodes=8)

    def __init__(self, smiles):
        self.rows = [encode_smiles(value, self.spec) for value in smiles]
        self.smiles = [value["smiles"] for value in self.rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        value = self.rows[index]
        return dict(atoms=value["atom_types"], charges=value["charges"], bonds=value["bond_types"])


def test_mines_one_edit_on_each_side_and_preserves_canonical_identity():
    split = FixtureSplit(["CC=CCCO", "CCCCCO", "CC=CCCN", "CCOC"])
    config = dict(count=1, center_limit=4, min_atoms=4, max_atoms=8, seed=105)
    rows, report = select_triplets(split, config)
    assert len(rows) == report["selected_triplets"] == 1
    triplet = rows[0]
    assert triplet["b_smiles"] == "CC=CCCO"
    assert {triplet["a_smiles"], triplet["c_smiles"]} == {"CCCCCO", "CC=CCCN"}
    assert {triplet["a_from_b_edit"]["kind"], triplet["c_from_b_edit"]["kind"]} == {"atom", "bond"}
    assert set(triplet["a_from_b_edit"]["slots"]).isdisjoint(triplet["c_from_b_edit"]["slots"])
    center = split[triplet["b_index"]]
    for endpoint in ("a", "c"):
        graph = split[triplet[endpoint + "_index"]]
        order = triplet["mapping"][endpoint + "_original_atom_for_b_slot"]
        size = triplet["atoms"]
        assert sorted(order) == list(range(size))
        assert np.array_equal(center["charges"][:size], graph["charges"][order])
        atom_edits = np.count_nonzero(center["atoms"][:size] != graph["atoms"][order])
        bond_edits = np.count_nonzero(np.triu(center["bonds"][:size, :size] !=
                                             graph["bonds"][np.ix_(order, order)], 1))
        assert atom_edits + bond_edits == 1
    repeat, _ = select_triplets(split, config)
    assert repeat == rows


def test_never_fills_a_chain_with_an_endpoint_outside_the_split():
    split = FixtureSplit(["CC=CCCO", "CCCCCO", "CCOC"])
    rows, report = select_triplets(split, dict(count=1, center_limit=3, min_atoms=4, max_atoms=8, seed=105))
    assert rows == []
    assert report["selected_triplets"] == 0


def test_heldout_derived_endpoints_round_trip_and_exclude_training_identities():
    split = FixtureSplit(["CC=CCCO"])
    excluded = {"CCCCCO", "CC=CCCN", "FC=CCCO"}
    rows, report = select_triplets(split, dict(count=1, center_limit=1, min_atoms=4, max_atoms=8,
        seed=105, allow_derived_endpoints=True), excluded_identities=excluded)
    triplet, = rows
    assert triplet["b_index"] == 0 and triplet["b_smiles"] == "CC=CCCO"
    for side in ("a", "c"):
        smiles = triplet[side + "_smiles"]
        assert smiles not in excluded
        assert triplet[side + "_index"] is None and triplet[side + "_provenance"] == "derived"
        assert encode_smiles(smiles, split.spec)["smiles"] == smiles
    assert report["provenance"] == {"derived_endpoints": 2, "triplets_with_derived_endpoints": 1}


def test_matching_slots_retains_only_chains_with_two_identity_maps():
    split = FixtureSplit(["CCCO", "CCCN", "NCCO", "CC=CCCO", "CCCCCO", "CC=CCCN"])
    config = dict(count=2, center_limit=6, min_atoms=4, max_atoms=8, seed=105)
    baseline, _ = select_triplets(split, config)
    assert len(baseline) == 2
    assert any(order != list(range(row["atoms"])) for row in baseline for order in row["mapping"].values())
    rows, report = select_triplets(split, dict(config, require_matching_slots=True))
    assert len(rows) == report["selected_triplets"] > 0
    assert any(row["b_smiles"] == "CC=CCCO" for row in rows)
    assert all(order == list(range(row["atoms"])) for row in rows for order in row["mapping"].values())
    assert report["require_matching_slots"] is True
    assert report["proposal_counts"]["nonmatching_slot_proposals"] > 0


def test_matching_slots_applies_to_derived_endpoints():
    split = FixtureSplit(["CC=CCCO"])
    config = dict(count=1, center_limit=1, min_atoms=4, max_atoms=8, seed=105,
                  allow_derived_endpoints=True, require_matching_slots=True)
    rows, report = select_triplets(split, config, excluded_identities={"CCCCCO", "CC=CCCN", "FC=CCCO"})
    triplet, = rows
    assert all(order == list(range(triplet["atoms"])) for order in triplet["mapping"].values())
    assert report["provenance"] == {"derived_endpoints": 2, "triplets_with_derived_endpoints": 1}
