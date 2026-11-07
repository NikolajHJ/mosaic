from types import SimpleNamespace

from latentmol.design.protenix_restart_selection import (
    audit_molecule, select_candidates, select_restart_parents,
)


def test_mixed_shortlist_keeps_relaxed_winners_and_spreads_remaining_steps():
    ranked = [SimpleNamespace(step=step, smiles=str(step)) for step in (90, 80, 10, 20, 30, 40)]
    selected = select_candidates(ranked, 4, compare=True)
    assert [row.step for row in selected["mixed"]] == [90, 80, 20, 40]
    assert selected["relaxed"] == ranked[:4]
    assert select_candidates(ranked, 0, compare=False) == {"mixed": []}


def test_diverse_parents_keep_winner_and_respect_margin_and_source_branch():
    archive = {
        smiles: (SimpleNamespace(branch=branch), score)
        for smiles, branch, score in (
            ("c1ccccc1", "first", 2.0),
            ("Cc1ccccc1", "first", 2.01),
            ("CCO", "second", 2.03),
            ("C1CCCCC1", "third", 2.07),
            ("C1CC1", "fourth", 2.2),
        )
    }
    selected = select_restart_parents(archive, 3)
    assert next(iter(selected)) == "c1ccccc1"
    assert set(selected) == {"c1ccccc1", "CCO", "C1CCCCC1"}
    assert select_restart_parents(dict(reversed(archive.items())), 3) == selected
    assert list(select_restart_parents(archive, 3, score_margin=0)) == ["c1ccccc1", "Cc1ccccc1", "CCO"]


def test_molecule_audit_flags_ring_rich_graph_without_removing_validity():
    assert audit_molecule("C1CC1C2CC2C3CC3C4CC4") == {
        "sanitized": True, "connected": True, "three_membered_rings": 4, "small_ring_flag": True,
    }
    assert not audit_molecule("CC.O")["connected"]
