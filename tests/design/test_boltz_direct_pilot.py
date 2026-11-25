import json
from pathlib import Path

import pytest
import torch
from rdkit import Chem

from latentmol.chemistry import inspect_graph
from latentmol.design.boltz_direct_pilot import make_founder, task_budget
from latentmol.guidance.boltz_search import hard_graph_record, hard_pose_key
from latentmol.representation import GraphSpec


def test_spatial_founders_use_only_occupied_slots_and_replay_per_seed():
    spec = GraphSpec(max_nodes=22)
    slots = [0, 2, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 20]
    active = torch.zeros(spec.max_nodes, dtype=torch.bool)
    active[slots] = True
    xyz = torch.zeros(len(slots), 3)
    xyz[:, 0] = torch.arange(len(slots)) * 1.45
    keys = []
    for seed in (517111, 517111, 517133):
        founder = make_founder(spec, active, xyz, seed)
        keys.append(hard_pose_key(hard_graph_record(founder)))
        assert torch.equal(founder.atoms[0] != 0, active)
        mol = inspect_graph(founder.atoms[0], founder.charges[0], founder.bonds[0], spec)
        assert mol.smiles
        molecule = Chem.MolFromSmiles(mol.smiles)
        assert molecule.GetNumBonds() == len(slots) - 1
        assert not molecule.HasSubstructMatch(Chem.MolFromSmarts("[CX3]=[OX1]"))
        assert not molecule.HasSubstructMatch(Chem.MolFromSmarts("[#7,#8]~[#7,#8]"))
    assert keys[0] == keys[1] and keys[1] != keys[2]
    with pytest.raises(ValueError, match="connected"):
        make_founder(spec, active, xyz * 2, 517111)


def test_preset_has_paired_factorial_arms_and_explicit_space():
    root = Path(__file__).resolve().parents[2]
    cfg = json.loads((root / "experiments/latent-design/protenix-vae/restart-boltz-brd4-direct.json").read_text())
    weights = {(v["motif_weight"] > 0, v["triangle_weight"] > 0) for v in cfg["variants"].values()}
    assert weights == {(False, False), (True, False), (False, True), (True, True)}
    assert len({task["seed"] for task in cfg["tasks"].values()}) == len(cfg["tasks"]) == 3
    for name in cfg["tasks"]:
        assert task_budget(cfg, name) == (300, 32)
    with pytest.raises(ValueError, match="graph-logit"):
        task_budget(cfg | {"optimization_space": "latent"}, name)
