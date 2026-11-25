from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

from latentmol.design import protenix_near_valid as replay
from latentmol.design.latent_protenix import LatentDesignConfig
from latentmol.design.protenix_restarts import write_json
from latentmol.representation import GraphSpec


def test_near_valid_replay_restores_displaced_exact_slot_and_matches_scoring(tmp_path, monkeypatch):
    decoder = torch.nn.Linear(1, 1)
    decoder.spec = GraphSpec(max_nodes=2)
    decoder.config = SimpleNamespace(latent=1)
    source = tmp_path / "live-source"
    cycle = source / "cycle-02"
    cycle.mkdir(parents=True)
    design = LatentDesignConfig(occupied_atoms=2, learning_rate=60.0, max_step_norm=0.25, noise_initial=0)
    write_json(source / "run.json", dict(runner="restart_search", job_id="123", checkpoint_sha256="checkpoint",
        decoder_sha256=replay._model_identity(decoder), graph_spec=decoder.spec.to_dict(), design=asdict(design),
        search=dict(repair_edit_limit=2, children=3, invalid_children=1, exact_children=1)))
    exact = dict(smiles="CC", fresh_loss=2.2, relaxed_loss=-0.1, branch="cycle-01/branch-05", step=50,
        latent=[[0.2]], labels=dict(atoms=[1, 1], charges=[2, 2], bonds=[[0, 1], [1, 0]]),
        active_slots=[0, 1], ancestry=["founder-02", "cycle-01/branch-05"],
        repair_edits=None, repair_smiles=None, repair_target_sha256=None)
    invalid = exact | dict(smiles=None, fresh_loss=None, branch="cycle-01/branch-03", step=49,
        labels=exact["labels"] | dict(bonds=[[0, 0], [0, 0]]), repair_edits=1,
        repair_smiles="CC", repair_target_sha256="witness")
    plan = dict(wave=2, invalid_selected=True, branches=[
        dict(name="branch-01", policy="raw-invalid", seed=123, source=invalid),
        dict(name="branch-02", policy="latent", seed=124, source=exact),
    ])
    write_json(cycle / "wave-plan.json", plan)
    write_json(cycle / "archive.json", dict(cycle=2, invalid_selected=True, restart_parents=[exact]))
    config = dict(study="near_valid", source_run=str(source), source_job_id=123,
        design={"updates": 500}, fresh_per_arm=3, tasks={"paired": {"waves": [2]}})
    seen = {}

    def execute(_decoder, _scorer, active_design, _root, branches, **kwargs):
        assert active_design.learning_rate == 60.0 and active_design.noise_initial == 0
        assert kwargs["repair_limit"] == 2
        near, control = branches
        assert near.policy == "raw-invalid" and near.parent.repair_target_sha256 == "witness"
        assert control.policy == "none" and control.parent.candidate.branch == "cycle-01/branch-05"
        assert near.seed == control.seed == 123
        return [
            SimpleNamespace(branch=branch, summary=dict(updates=500, first_valid_step=index),
                valid=[replace(branch.parent.candidate, smiles=smiles, step=idx)
                       for idx, smiles in enumerate(pool)])
            for index, (branch, pool) in enumerate(zip(branches, (("CC", "CN", "CO"), ("CC", "CS")), strict=True))
        ]

    def fresh(root, records):
        seen[root.name] = [row["smiles"] for row in records]
        return [row | dict(loss=2.0, status="scored") for row in records]

    monkeypatch.setattr(replay, "execute_branches", execute)
    result = replay.run_paired(decoder, lambda *_: SimpleNamespace(close=lambda: None, identity={"fixture": True}),
                               fresh, config, "paired", tmp_path / "new-run", "checkpoint")
    assert seen == {"baselines": ["CC"], "fresh-near-valid": ["CN"], "fresh-exact-valid": ["CS"]}
    assert result["baseline_attempts"] == 1 and result["descendant_attempts"] == 2
    assert result["pairs"][0]["fresh_baselines"]["witness"]["loss"] == 2.0
    with pytest.raises(ValueError, match="separate from the live source"):
        replay.run_paired(decoder, None, None, config, "paired", source / "bad", "checkpoint")
