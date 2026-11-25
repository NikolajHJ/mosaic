from types import SimpleNamespace

import torch
from torch.nn.functional import one_hot

from latentmol.chemistry import inspect_graph
from latentmol.design.bootstrap_screen import bootstrap, screened_bootstrap
from latentmol.design.latent_protenix import LatentDesignConfig
from latentmol.representation import DecoderOutput, GraphSpec


class SmallDecoder:
    config = SimpleNamespace(latent=1)
    spec = GraphSpec(max_nodes=4)

    def __init__(self, constant=False):
        self.constant = constant

    def decode(self, point):
        choice = 0 if self.constant else int(4 * point.sigmoid().item())
        atoms = torch.tensor([[1, 1, 1, (1, 2, 3, 1)[choice]]])
        charges = torch.full((1, 4), self.spec.formal_charges.index(0))
        bonds = torch.zeros((1, 4, 4), dtype=torch.long)
        for i in range(3):
            if choice == 3 and i == 1:
                continue
            bonds[0, i, i + 1] = bonds[0, i + 1, i] = 1
        return DecoderOutput(one_hot(atoms, len(self.spec.atomic_numbers)).float() * 10,
                             one_hot(charges, len(self.spec.formal_charges)).float() * 10,
                             one_hot(bonds, len(self.spec.bond_names)).float() * 10)


class DisconnectedDecoder(SmallDecoder):
    def decode(self, point):
        decoded = super().decode(point)
        labels = torch.zeros((1, 4, 4), dtype=torch.long)
        return DecoderOutput(decoded.atoms, decoded.charges, one_hot(labels, len(self.spec.bond_names)).float())


def test_screen_selects_replayable_prior_draws_and_keeps_near_valid_start_unrepaired():
    decoder, design = SmallDecoder(), LatentDesignConfig(occupied_atoms=4)
    points, receipt = screened_bootstrap(decoder, design, seed=0, count=4, pool_size=26)
    records = receipt["selected"]
    assert records[0]["role"] == "random" and records[0]["seed"] == 0
    assert "near_valid" in [r["role"] for r in records[1:]]
    assert "valid" in [r["role"] for r in records[1:]]
    assert receipt["draws"] == sum(receipt["pool_categories"].values()) == 26
    for point, record in zip(points, records):
        ordinary = bootstrap(decoder, design, seed=record["seed"])
        assert torch.equal(ordinary.candidate.latent, point.candidate.latent)
        assert torch.equal(ordinary.active, point.active)
        assert torch.equal(ordinary.candidate.hard.bonds, point.candidate.hard.bonds)
        if record["role"] == "near_valid":
            hard = point.candidate.hard
            assert inspect_graph(hard.atoms[0], hard.charges[0], hard.bonds[0], decoder.spec).smiles is None
            assert record["witness_smiles"] and record["witness_edits"] in (1, 2)


def test_screen_reports_prior_fallback_if_pool_has_no_distinct_alternatives():
    decoder = SmallDecoder(constant=True)
    _, receipt = screened_bootstrap(decoder, LatentDesignConfig(occupied_atoms=4), seed=5, count=4, pool_size=6)
    assert [r["role"] for r in receipt["selected"]] == ["random"] + ["fallback_duplicate"] * 3
    assert [r["seed"] for r in receipt["selected"]] == [5, 6, 7, 8]
    assert receipt["native_scoring_calls"] == receipt["repair_checks"] == 0


def test_farther_verified_witness_can_fill_a_slot_without_being_counted_as_near_valid():
    _, receipt = screened_bootstrap(DisconnectedDecoder(), LatentDesignConfig(occupied_atoms=4),
                                    seed=0, count=2, pool_size=12, repair_pool_size=3)
    assert receipt["repair_attempted"] == 3
    chosen = receipt["selected"][1]
    assert chosen["role"] == "farther_witness" and chosen["witness_edits"] > 2
    assert chosen["smiles"] is None and chosen["witness_smiles"]
