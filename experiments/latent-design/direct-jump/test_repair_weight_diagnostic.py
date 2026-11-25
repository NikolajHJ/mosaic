import json
from types import SimpleNamespace

import numpy as np

from repair_search import evaluate_segment, projected_repair_comparison


def test_cap_can_hide_weight_change_even_with_a_live_repair_gradient():
    graph = (np.full((1, 3), 1/3, np.float32), np.full((1, 2), .5, np.float32))
    repair = (np.array([[0., -1., 0.]], np.float32), np.zeros_like(graph[1]))
    contact = tuple(np.zeros_like(x) for x in graph)
    cfg = dict(learning_rate=60., max_step_norm=.5, probability_floor=.0001)
    same = projected_repair_comparison(graph, contact, repair, cfg)
    assert same["repair_gradient_norm"] == 1
    assert same["projected_step_norms"][1] > 0
    assert all(c["projected_l2_difference"] == 0 for c in same["comparisons"])
    contact[1][0, 1] = -1.
    distinct = projected_repair_comparison(graph, contact, repair, cfg)
    assert all(0 < c["relative_to_larger_step"] < 1 for c in distinct["comparisons"])
    assert np.allclose(graph[0], 1/3) and np.allclose(graph[1], .5)


def test_only_witnessed_updates_sample_and_no_extra_model_calls(tmp_path):
    graph = (np.full((25, 6), 1/6, np.float32), np.full((300, 5), .2, np.float32))
    calls = dict(model=0, repair=0)
    def gradient(a, b):
        calls["model"] += 1
        return 0., np.zeros_like(a), np.zeros_like(b)
    def evaluate(a, b):
        calls["repair"] += 1
        status = "no_witness" if calls["repair"] < 4 else "verified_witness"
        db = np.zeros_like(b)
        if status == "verified_witness":
            db[0, 1] = -1.
        return 0., np.zeros_like(a), db, dict(status=status, smiles=None, hard_sulfur=0,
                                              component_count=2)
    folder = tmp_path / "branch"
    cfg = dict(learning_rate=60., max_step_norm=.5, probability_floor=.0001)
    _, _, summary, _, _ = evaluate_segment(SimpleNamespace(gradient=gradient), SimpleNamespace(evaluate=evaluate),
                                            None, graph, cfg, 5., 29, "branch", folder, 0)
    rows = [json.loads(s) for s in (folder / "trajectory.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows if "repair_weight_diagnostic" in r] == [3, 28]
    assert summary["repair_weight_diagnostic_samples"] == 2
    assert summary["repair_status_counts"] == dict(verified_witness=26, no_witness=3, already_valid=0)
    assert calls == dict(model=30, repair=30)
