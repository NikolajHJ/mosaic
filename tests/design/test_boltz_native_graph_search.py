import json
from pathlib import Path
from types import SimpleNamespace

import torch

from latentmol.design import boltz_allocation_pilot as allocation
from latentmol.design import boltz_search_methods as methods
from latentmol.experiments import file_sha256
from latentmol.representation import GraphSpec, HardGraphBatch


class ToyNativeScorer:
    """Supply a fixed pose and a differentiable three-channel objective on CPU."""

    def __init__(self, spec, pocket):
        self.spec = spec
        self.device = torch.device("cpu")
        self.execution = {"autocast": "none", "tf32": False}
        self.fresh = []
        self.closed = False
        n = spec.max_nodes
        slots = torch.arange(25)
        atoms = torch.zeros(1, n, dtype=torch.long)
        atoms[0, slots] = spec.atomic_numbers.index(6)
        charges = torch.full_like(atoms, spec.formal_charges.index(0))
        bonds = torch.zeros((1, n, n), dtype=torch.long)
        bonds[0, slots[:-1], slots[1:]] = bonds[0, slots[1:], slots[:-1]] = spec.bond_names.index("SINGLE")
        xyz = torch.zeros(1, 1, len(slots), 3)
        xyz[0, 0, :, 0] = slots * 1.45
        mapping = torch.full((n,), -1, dtype=torch.long)
        mapping[slots] = slots
        self.prepared = SimpleNamespace(seed=HardGraphBatch(atoms, charges, bonds), active=atoms[0] != 0,
                                        slot_to_atom=mapping, coordinates=xyz, spec=spec,
                                        metadata={"target_id": "erk2", "target": {"pocket": pocket},
                                                  "seed_smiles": "C" * len(slots)})

    def score_at(self, graph, active):
        assert torch.equal(self.prepared.active, active)
        return (graph.atoms[0, 0, 1] + graph.charges[0, 1, 2] + graph.bonds[0, 1, 2, 1]), {}

    def score_valid_records(self, root, rows):
        self.fresh.extend(row["smiles"] for row in rows)
        return [dict(row, loss=float(i)) for i, row in enumerate(rows)]

    def close(self):
        self.closed = True


def test_allocated_native_search_charges_founders_and_skips_vae(tmp_path, monkeypatch):
    def put(name, data):
        path = tmp_path / name
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def pinned(path):
        return {"path": str(path), "sha256": file_sha256(path)}

    def unexpected_vae(*args, **kwargs):
        raise AssertionError("Native generation search invoked the VAE path")

    monkeypatch.setenv("LSB_JOBID", "987")
    monkeypatch.setenv("LATENTMOL_SOURCE_SHA256", "test")
    monkeypatch.setattr(methods, "load_model", unexpected_vae)
    monkeypatch.setattr(methods, "parent_state", unexpected_vae)
    monkeypatch.setattr(methods, "archive_starts", unexpected_vae)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "CPU fixture")
    monkeypatch.setitem(allocation.POLICY_LENGTHS, "fixed75", 1)

    experiment = Path(__file__).resolve().parents[2] / "experiments/latent-design/protenix-vae"
    cfg = json.loads((experiment / "restart-boltz-erk2-18-native.json").read_text())
    base = json.loads((experiment / cfg["search_config"]).read_text())
    cfg["generation_settings"].update(generations=1, updates_per_generation=8)
    spec = GraphSpec.from_dict(cfg["graph_spec"])
    cfg["history_exclusions"] = pinned(put("history.json", ["CC"]))
    arm = "erk2-18"
    pocket = base["arms"]["erk2"]["target"]["pocket"]
    scorer = ToyNativeScorer(spec, pocket)
    monkeypatch.setattr(methods.BoltzSearchScorer, "load", lambda *args, **kwargs: scorer)
    assets = {"checkpoint": "affinity", "checkpoint_sha256": "test", "confidence_checkpoint": "confidence"}
    prepared = dict(checkpoint=assets["checkpoint"], checkpoint_sha256=assets["checkpoint_sha256"],
                    dbmol={"confidence_checkpoint": assets["confidence_checkpoint"]})
    base["arms"]["erk2"]["prepared_config"] = pinned(put("prepared.json", prepared))
    put(cfg["search_config"], base)
    config = put("config.json", cfg)
    output = tmp_path / "results"
    plan = put("plan.json", dict(config_sha256=file_sha256(config), arms={arm: str(output)},
                                 planned_updates={arm: 8}, planned_fresh_attempts={arm: 8},
                                 optimization_space="relaxed_graph_logits"))
    args = SimpleNamespace(arm=arm, config=config, plan=plan, plan_sha256=file_sha256(plan), output=output,
                           boltz_assets=put("assets.json", assets), checkpoint=None, checkpoint_sha256=None)
    methods.run(args)

    result = json.loads((output / "result.json").read_text())
    assert scorer.closed and result["status"] == "completed" and result["updates"] == 8
    assert result["optimization_space"] == "relaxed_graph_logits" and result["relaxed_observations"] == 16
    assert 4 <= len(scorer.fresh) == result["fresh_attempts"] <= result["fresh_budget"] == 8
    assert result["founder_fresh_attempts"] == 4
    assert result["call_counts"]["hard_fp32_forwards"] == len(scorer.fresh)
    wave = output / "generations/generation-01"
    assert json.loads((wave / "result.json").read_text())["allocation"]["fresh_budget"] == 4
    starts = json.loads((wave / "starts.json").read_text())
    assert "paired_vae_start_hashes" not in starts
    assert len(starts["elites"]) == 4 and len(set(starts["start_hashes"])) == 8
    assert len(set(scorer.fresh[:4])) == 4
    assert set(row["smiles"] for row in result["top20"]).isdisjoint(scorer.fresh[:4] + ["CC"])
    elites = torch.load(wave / "elites.pt", weights_only=True)
    assert all("graph_logits" in row and "latent" not in row for row in elites)
