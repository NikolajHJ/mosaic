"""Fixed-pose shortlist caching, independent of native Boltz execution."""
from dataclasses import replace
from types import SimpleNamespace

import torch

from latentmol.design.protenix_restarts import Candidate
from latentmol.design.protenix_search import SearchOptions, score_wave
from latentmol.design.protenix_search_frontier import SearchPoint
from latentmol.representation import HardGraphBatch


def test_same_molecule_with_distinct_pose_mapping_is_scored_and_best_is_retained(tmp_path):
    active = torch.tensor([True, True, True, False, False])
    bonds = torch.zeros(1, 5, 5, dtype=torch.long)
    bonds[0, 0, 1] = bonds[0, 1, 0] = bonds[0, 1, 2] = bonds[0, 2, 1] = 1
    hard = HardGraphBatch(torch.tensor([[1, 1, 2, 0, 0]]), torch.full((1,5), 2), bonds)
    source = Candidate("CCN", torch.zeros(1,2), hard, 1., "source", 0)
    parent = SearchPoint(source, active, None, ("source",))

    def result(permutation, index):
        order = torch.tensor(permutation)
        changed = HardGraphBatch(hard.atoms[:,order], hard.charges[:,order], hard.bonds[:,order][:,:,order])
        row = replace(source, hard=changed, branch=f"cycle-01/branch-{index:02d}", step=3)
        return SimpleNamespace(valid=[row], branch=SimpleNamespace(parent=parent))

    results = [result([0,1,2,3,4], 1), result([2,0,1,3,4], 2)]
    archive, attempted, calls = {}, set(), []

    def score(_root, records):
        calls.extend(records)
        return [dict(row, loss=3.-i) for i,row in enumerate(records)]

    options = SearchOptions(hard_scoring="fixed_pose_dbmol", shortlist="relaxed", fresh_per_branch=1)
    losses, successes = score_wave(tmp_path, results, score, archive, attempted, options, None)
    assert losses == [3.,2.] and successes == {"CCN"}
    assert len(attempted) == len(calls) == 2
    assert calls[0]["scoring_key"] != calls[1]["scoring_key"]
    assert archive["CCN"].fresh_loss == 2.
    assert archive["CCN"].candidate is results[1].valid[0]
    assert score_wave(tmp_path, results, score, archive, attempted, options, None) == ([],set())
    third = result([1,2,0,3,4],3)
    score_wave(tmp_path,[third],lambda _root,rows:[dict(row,loss=4.) for row in rows],archive,attempted,options,None)
    assert len(attempted) == 3 and archive["CCN"].fresh_loss == 2.
