"""Small CPU checks for exact parent scoring, rank ties and repair fallback."""
import json

import numpy as np

from repair_frontier import ParentFrontier, _percentiles, hard_graph_key, near_valid_key
from study import PAIR_I, PAIR_J, hard_smiles


class Backend:
    def __init__(self):
        self.calls = []

    def hard_value(self, atoms, bonds):
        assert np.all((atoms == 0) | (atoms == 1)) and np.all((bonds == 0) | (bonds == 1))
        self.calls.append(hard_graph_key((atoms, bonds)))
        return float(atoms[0].argmax() + 3 * atoms[-1].argmax())


class Dock:
    panel_size = 2

    def __init__(self):
        self.calls = []

    def score_many(self, names):
        self.calls.append(list(names))
        return {s: (dict(status="ok", affinity_kcal_mol=-10. if "O" in s else -4.)
                    if "N" not in s else dict(status="failed", reason="embedding", affinity_kcal_mol=None))
                for s in names}


def chain(label=0, position=0, step=1):
    atoms = np.zeros(25, dtype=int)
    atoms[position] = label
    graph = (np.eye(6, dtype=np.float32)[atoms],
             np.eye(5, dtype=np.float32)[(PAIR_J - PAIR_I == 1).astype(int)])
    return dict(graph=graph, smiles=hard_smiles(graph), relaxed_loss=-1., branch="test", step=step)


def test_score_all_ordered_graphs_dock_all_identities_and_retain_distinct_parents():
    old, best, alternate, failed, initial = chain(), chain(2), chain(2, -1), chain(1), chain(3, step=0)
    backend, dock = Backend(), Dock()
    frontier = ParentFrontier(backend, dock)
    assert frontier.hard_contact(best["graph"]) == 2.
    parents, receipt = frontier.select([alternate, best, old, failed, initial, dict(old, step=2)],
                                      [], count=3, vina_weight=.75)
    assert [p["smiles"] for p in parents] == [best["smiles"], old["smiles"], failed["smiles"]]
    assert parents[0] is best and len(receipt["candidates"]) == 4
    assert len(set(backend.calls)) == len(backend.calls) == 4
    assert len(dock.calls) == 2 and len({s for batch in dock.calls for s in batch}) == 3
    assert receipt["selected"][-1]["vina_reason"] == "embedding"
    json.dumps(receipt, allow_nan=False)
    assert _percentiles({"a": 0, "b": 0, "c": 2}) == {"a": .25, "b": .25, "c": 1.}


def test_no_valid_positive_step_ranks_20_invalid_raw_graphs_and_skips_docking():
    backend, dock = Backend(), Dock()
    invalids = []
    for i in range(22):
        p = chain(position=i, label=1)
        p["graph"][1][:] = np.eye(5, dtype=np.float32)[0]
        p.update(smiles=None, repair_smiles=None, repair_edits=None, component_count=25,
                 problem_burden=1, overvalence_excess=0., step=i+1)
        invalids.append(p)
    good = dict(invalids[-1], repair_smiles="C"*25, repair_edits=2)
    assert near_valid_key(good) < near_valid_key(invalids[0])
    invalids[-1] = good
    invalids.append(dict(good))
    frontier = ParentFrontier(backend, dock)
    parents, receipt = frontier.select([chain(step=0), dict(chain(), smiles="bad")],
                                      invalids, count=3, vina_weight=.5)
    assert not backend.calls and not dock.calls
    assert receipt["mode"] == "near_valid" and receipt["near_pool_size"] == 22
    assert len(receipt["near_candidates"]) == 20 and len(parents) == 3 and parents[0] is good
    assert receipt["rejected"][0]["reason"] == "invalid_reparse_or_identity_mismatch"
    assert not parents[0]["smiles"]
    json.dumps(receipt, allow_nan=False)


def test_all_docking_failures_keep_valid_parent_and_do_not_fill_with_invalids():
    point = chain(1)
    frontier = ParentFrontier(Backend(), Dock())
    parents, receipt = frontier.select([point], [], count=2, vina_weight=.5)
    assert len(parents) == 1 and parents[0] is point
    assert receipt["mode"] == "valid" and receipt["effective_vina_weight"] == 0
    assert receipt["selected"][0]["vina_status"] == "failed"
