import json
from types import SimpleNamespace

import numpy as np

from repair_frontier import ParentFrontier, hard_graph_key
from repair_search import perturb, project, run_arm
from repair_signal import evaluate_repair
from study import PAIR_I, PAIR_J


def test_floor_projection_and_seeded_sibling():
    rng = np.random.default_rng(7)
    x = project(rng.normal(size=(25, 6)), .0001)
    y = project(rng.normal(size=(300, 5)), .0001)
    assert np.all(x >= np.float32(.0001)) and np.allclose(x.sum(axis=1), 1)
    a, b = perturb((x, y), .002, rng, .0001)
    assert hard_graph_key((a, b)) == hard_graph_key((x, y))
    assert not np.array_equal(a, x) or not np.array_equal(b, y)


def test_whole_trajectory_selects_only_improved_observed_hard_molecules(tmp_path):
    atom = project(np.eye(6)[np.zeros(25, dtype=int)], .0001)
    bond = project(np.eye(5)[(PAIR_J == PAIR_I + 1).astype(int)], .0001)
    # Make one terminal C/N decision near enough to cross in a single bounded update.
    atom[0] = project(np.array([[.53,.46,.003,.003,.002,.002]]), .0001)[0]
    class Backend:
        def gradient(self, a, b):
            ga = np.zeros_like(a)
            ga[0, 1] = -1.
            return -float(a[0,1]), ga, np.zeros_like(b)
        def hard_value(self, a, b):
            return -float(a[0,1])
    backend = Backend()
    cfg = dict(learning_rate=60., max_step_norm=.5, probability_floor=.0001,
               child_generations=2, parents=2, children_per_parent=2, founder_updates=2,
               child_updates=1, search_versions={"test":{"vina_weight":0.}},
               restart_mass=.002, restart_mass_after_merge=.01)
    ret = run_arm(backend, SimpleNamespace(evaluate=evaluate_repair), ParentFrontier(backend),
                  (atom,bond),cfg,1.,17,"test",tmp_path/"run")
    assert ret["updates"] == 10 and ret["improved_distinct"] > 0
    assert ret["selected"]["score_change"] == -1.
    assert ret["fraction_valid_steps"] == 1 and ret["mean_sulfur_drift"] == 0
    assert ret["selected"]["smiles"] != ret["starting_smiles"]
    assert len(list((tmp_path/"run").rglob("argmax-states.npz"))) == 9
    roles = json.loads((tmp_path/"run"/"hard-readouts.json").read_text())
    assert len(roles) == 27 and any(r["hard_contact_loss"] == -1 for r in roles)
