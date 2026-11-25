import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from latentmol.design import protenix_tacogfn_study as study
from latentmol.design.protenix_synthesis_study import resolve_pilot


def candidate(smiles, step):
    return SimpleNamespace(smiles=smiles, step=step, relaxed_loss=float(step))


def test_taco_pair_sealed_budget_and_order():
    config = json.loads(Path("experiments/latent-design/protenix-vae/restart-taco-guidance.json").read_text())
    orders = []
    for task in config["tasks"]:
        design, budget = resolve_pilot(config, task)
        assert design.updates == 500 and budget["updates"] == 1000
        assert budget["fresh_attempts"] == budget["native_vina_attempts"] == 12
        orders.append(config["tasks"][task]["variants"])
    assert orders[0] == list(reversed(orders[1]))
    config["variants"]["guidance-off"]["ratios"]["synfrag"] = .3
    with pytest.raises(ValueError):
        resolve_pilot(config, "founder-1")


def test_match_counts_deduplicate_and_fill_from_all_valid(monkeypatch):
    off = [candidate("C", 0), candidate("CO", 2), candidate("CO", 1), candidate("CN", 4)]
    on = [candidate("C", 0), candidate("CCO", 2), candidate("CCN", 3), candidate("CCS", 4)]
    monkeypatch.setattr(study, "synthesis_shortlist", lambda *args: [])
    selection, available = study.matched_candidates(
        {"off": SimpleNamespace(valid=off), "on": SimpleNamespace(valid=on)}, {"C"}, 6)
    assert available == {"off": 2, "on": 3}
    assert [(x.smiles, x.step) for x in selection["off"]] == [("CO", 1), ("CN", 4)]
    assert [x.smiles for x in selection["on"]] == ["CCO", "CCN"]


def test_failed_fresh_call_still_gets_native_redocking(tmp_path, monkeypatch):
    class Dock:
        def __init__(self, config):
            self.names = []

        def score_many(self, names):
            self.names = list(names)
            return {n:dict(status="failed", affinity_kcal_mol=None, reason="timeout") for n in names}

        def accounting(self):
            return {"unique_attempts": len(self.names)}

    def fresh(root, records):
        return [row | dict(loss=None, status="embedding_failed") for row in records]

    monkeypatch.setattr(study, "VinaParentPanel", Dock)
    a = study._evaluate_arm(tmp_path / "off", [candidate("CO", 3)], fresh, {}, {"CO"})
    assert a["fresh_attempts"] == a["native_vina"]["unique_attempts"] == 1
    assert a["jointly_scored"] == 0 and not a["candidates"][0]["new_to_history"]
    assert a["candidates"][0]["vina"]["reason"] == "timeout"


def test_boltz_pair_preserves_reference_pose_labels_and_two_arm_budgets(tmp_path):
    import torch
    from latentmol.representation import HardGraphBatch
    from latentmol.guidance.boltz_search import hard_pose_key

    config = json.loads(Path("experiments/latent-design/protenix-vae/restart-boltz-taco-guidance.json").read_text())
    for name in config["tasks"]:
        design, budget = study.resolve_pair(config, name)
        assert design.occupied_atoms == 31
        assert budget["updates"] == 1000 and budget["updates_per_arm"] == 500
        assert budget["fresh_attempts_per_arm"] == budget["native_vina_attempts_per_arm"] == 6
    # Carbon in slot 2, oxygen in slot 0, with a padding site between them.
    hard = HardGraphBatch(torch.tensor([[3, 0, 1]]), torch.tensor([[2, 0, 2]]),
                          torch.tensor([[[0, 0, 1], [0, 0, 0], [1, 0, 0]]]))
    c = SimpleNamespace(smiles="CO", step=8, branch="on", hard=hard)

    def full_fp32(root, records):
        assert len(records) == 1 and records[0]["hard_graph"]["atoms"] == [3, 0, 1]
        assert records[0]["scoring_key"] == hard_pose_key(records[0]["hard_graph"])
        return [records[0] | dict(status="scored", loss=-1.2)]

    assert study._boltz_hard_scores(tmp_path, [c], full_fp32)["CO"]["loss"] == -1.2


def test_half_norm_readout_rejects_a_weaker_nonzero_direction(tmp_path):
    path = tmp_path / "metrics.jsonl"
    meta = dict(actual_ratio=.23, active=True, skip_reason=None, applied_gradient_norm=.23,
                binding_gradient_norm=1., raw_gradient_norm=.23)
    path.write_text(json.dumps(dict(step=0, synthesis=dict(tacogfn=meta))) + "\n")
    with pytest.raises(ValueError, match="did not reach"):
        study._guidance_usage(path, 1, .5, "match")
