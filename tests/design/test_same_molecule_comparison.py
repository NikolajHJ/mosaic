"""Regression checks for matched latent starts, chemistry identities and fresh-score accounting."""

from io import StringIO
import json
from types import SimpleNamespace

import pytest
import torch

from latentmol.design.same_molecule_comparison import ARM_NAMES, OrderedScorer, matched_fresh_comparison
from latentmol.design.temperature_fresh import score_candidates
from latentmol.representation import RelaxedGraph


def test_resident_worker_checks_frozen_state_without_ending_the_protocol(monkeypatch):
    from latentmol.guidance import mosaic_protenix_worker as worker
    class Server:
        def __init__(self, request):
            self.context = SimpleNamespace(n_atoms=25)
            self.identity, self.devices = {}, []
            self.eval_count = 0
        def close(self):
            return dict(frozen_unchanged=True)
        def evaluate(self, request):
            self.eval_count += 1
            return dict(eval_count=self.eval_count)
    commands = ("init", "evaluate", "verify_frozen", "evaluate", "close")
    lines = "\n".join(json.dumps(dict(op=op, request_id=i)) for i, op in enumerate(commands)) + "\n"
    output = StringIO()
    monkeypatch.setattr(worker, "NativeServer", Server)
    monkeypatch.setattr(worker.sys, "stdin", StringIO(lines))
    monkeypatch.setattr(worker.sys, "stdout", output)
    worker.main()
    results = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [r["status"] for r in results] == ["ok"] * 5
    assert results[2]["frozen_unchanged"] and results[4]["frozen_unchanged"]
    assert results[1]["eval_count"] == 1 and results[3]["eval_count"] == 2


def test_native_mapping_scatters_each_derivative_back_to_its_original_slot():
    class Native:
        identity = {"model": "fixed"}
        checked = False
        def __call__(self, graph):
            return graph.atoms[0, 0, 1] + 2 * graph.charges[0, 1, 0] + 3 * graph.bonds[0, 1, 0, 2]
        def assert_unchanged(self):
            self.checked = True
    x = [torch.randn(*shape, requires_grad=True) for shape in ((1, 3, 2), (1, 3, 2), (1, 3, 3, 3))]
    g = RelaxedGraph(*x)
    native = Native()
    active = torch.tensor([True, False, True])
    prefix = torch.tensor([True, True, False])
    ordered = OrderedScorer(native, active, [2, 0, 1], prefix)
    y = ordered(g)
    expected = x[0][0, 2, 1] + 2 * x[1][0, 0, 0] + 3 * x[2][0, 0, 2, 2]
    assert torch.equal(y, expected)
    assert all(torch.equal(a, b) for a, b in zip(torch.autograd.grad(y, x), torch.autograd.grad(expected, x)))
    ordered.close()
    assert native.checked
    with pytest.raises(ValueError, match="each fixed active slot"):
        OrderedScorer(native, active, [1, 0, 2], prefix)


def test_unreadable_fresh_candidate_is_retained_in_matched_budget():
    batch = [dict(start="s", arm="a", role="initial", smiles="CC"), dict(start="s", arm="a", role="edit", smiles="C.C")]
    scored, count = score_candidates(batch, lambda mol, i: {"loss": .5, "status": "scored"}, record_unrepresentable=True)
    assert count == 2 and scored[1]["status"] == "representation_failed"
    assert scored[1]["loss_delta_from_valid_start"] is None


def test_fresh_report_keeps_missing_baselines_and_failed_edits_visible():
    study = dict(starts={}, checkpoint_sha256="c", optimization_scorer_budget_per_arm=31,
                 fresh_edit_limit_per_arm=4, native_config_sha256="n",
                 same_molecule_panel=dict(parents=12, sha256="p"), parent_slice=[2,4])
    rows = []
    for parent, parent_smiles, base_loss in (("a", "CCC", 1.0), ("b", "CC", None)):
        study["starts"][parent] = {}
        for arm in ARM_NAMES:
            study["starts"][parent][arm] = dict(candidates=dict(initial=dict(smiles=parent_smiles), distinct_valid_edits=0),
                scorer_budget=dict(actual=31), fresh_selection=dict(distinct_valid_edits=2),
                all_scored={"unconditioned": {"distinct_valid_edits": 1}})
            for k, (smi, loss) in enumerate(((parent_smiles, base_loss), ("CCO", .8), ("COC", None))):
                rows.append(dict(start=parent, arm=arm, role="initial" if k==0 else "edit", step=k,
                    smiles=smi, on_committed_trajectory=k == 0, loss=loss,
                    status="scored" if loss is not None else "embedding_failed"))
    fresh = dict(candidates=rows, unique_candidates=4, max_edit_candidates_per_arm=4, geometry_seed=1201)
    report = matched_fresh_comparison(study, fresh, rows)
    for a in ARM_NAMES:
        assert report["per_parent"][1]["arms"][a]["confirmed_improving_edits"] is None
        totals = report["totals"][a]
        assert totals["optimization_calls"] == 62 and totals["scorable_parents"] == 1
        assert totals["confirmed_improving_edits"] == totals["trial_only_improving_edits"] == 1
        assert totals["scored_edits"] == totals["unscored_edits"] == 2
    bad = [dict(r) for r in rows]
    bad[3]["smiles"] = "CNC"
    with pytest.raises(ValueError, match="same fresh parent"):
        matched_fresh_comparison(study, dict(fresh, candidates=bad), bad)
