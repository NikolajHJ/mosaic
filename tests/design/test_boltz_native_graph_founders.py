from types import SimpleNamespace

import pytest
import torch

from latentmol.chemistry import inspect_graph
from latentmol.design.boltz_direct_pilot import make_founder
from latentmol.design.boltz_native_graph_founders import native_graph_founders
from latentmol.representation import GraphSpec


def founder_fixture():
    spec = GraphSpec(max_nodes=25)
    active = torch.ones(25, dtype=torch.bool)
    active[[0, 3, 9, 13]] = False
    count = int(active.sum())
    xyz = torch.zeros(count, 3)
    xyz[:, 0] = torch.arange(count) * 1.4
    atoms = torch.full((25,), -1)
    atoms[active] = torch.arange(count)
    hard = make_founder(spec, active, xyz, 617)
    smi = inspect_graph(hard.atoms[0], hard.charges[0], hard.bonds[0], spec).smiles
    prepared = SimpleNamespace(spec=spec, active=active, coordinates=xyz[None, None], slot_to_atom=atoms, seed=hard,
        metadata={"seed_smiles": smi, "target": {"pocket": {"reference_ligand_locator": {"component": "test"}}}})
    scorer = SimpleNamespace(prepared=prepared, execution={"autocast": "none", "tf32": False})
    return scorer, SimpleNamespace(temperature_initial=1., occupied_atoms=count)


def test_native_founder_scores_align_by_pose_in_noncontiguous_slots(tmp_path):
    scorer, design = founder_fixture()
    calls = []

    def score(path, rows):
        calls.extend(rows)
        return [dict(r, loss=float(i)) for i, r in reversed(list(enumerate(rows)))]

    scorer.score_valid_records = score
    state = native_graph_founders(scorer, design, tmp_path, seed=21, count=4)
    assert len(calls) == state["founder_fresh_attempts"] == len(state["archive"]) == 4
    assert state["excluded_smiles"] == set(state["archive"])
    snapshots = torch.load(tmp_path / "founders" / "founders.pt", weights_only=True)
    for i, row in enumerate(snapshots):
        assert "graph_logits" in row and "latent" not in row and row["smiles"] == calls[i]["smiles"]
        assert row["fresh_loss"] == i and torch.equal(row["active"], scorer.prepared.active)
    assert snapshots[0]["smiles"] == scorer.prepared.metadata["seed_smiles"]


def test_rejects_changed_crystal_identity_before_scoring(tmp_path):
    scorer, design = founder_fixture()
    scorer.prepared.metadata["seed_smiles"] = "C"
    with pytest.raises(ValueError, match="valid connected molecule"):
        native_graph_founders(scorer, design, tmp_path, seed=21, count=4)
    assert not (tmp_path / "founders").exists()
