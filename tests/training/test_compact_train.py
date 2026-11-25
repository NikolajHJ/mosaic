"""A real compact-model optimizer resumes the same weights across two allocations."""

from dataclasses import replace
import json
from pathlib import Path

import pytest
from rdkit import rdBase
import torch

from latentmol.chemistry import encode_canonical_smiles
from latentmol.evaluation.compact_training_panel import CompactTrainingPanel
from latentmol.experiments import file_sha256
from latentmol.models import GraphAutoencoder
from latentmol.representation import GraphSpec
from latentmol.train import state_digest
from latentmol.training.compact_benchmark import _optimizer
from latentmol.training.compact_train import read_settings, train_segment

CONFIG = Path(__file__).resolve().parents[2] / "experiments/vae/compact435k-real-v100.json"


class Rows(list):
    spec = GraphSpec(max_nodes=5)
    dataset_sha256 = "fixture-dataset"
    manifest = dict(provenance={"canonicalization": "RDKit canonical SMILES; reparsed for stable atom ordering"},
                    rdkit_version=rdBase.rdkitVersion)

    def __init__(self, split, smiles):
        super().__init__()
        self.split = split
        for item in smiles:
            raw = encode_canonical_smiles(item, self.spec)
            self.append({key: torch.as_tensor(raw[source], dtype=torch.long) for key, source in
                         (("atoms", "atom_types"), ("charges", "charges"), ("bonds", "bond_types"))})


def test_optimizer_uses_requested_decay(tmp_path):
    settings = json.loads(CONFIG.read_bytes())
    settings.update(training_config=str((CONFIG.parent / settings["training_config"]).resolve()), weight_decay=.0001)
    path = tmp_path / "train.json"
    path.write_text(json.dumps(settings))
    _, cfg = read_settings(path)
    optimizer = _optimizer(torch.nn.Linear(2, 1), cfg)
    assert optimizer.param_groups[0]["weight_decay"] == .0001


@pytest.mark.parametrize("with_occupancy", [False, True])
def test_matched_mid_epoch_resume_and_checkpoint_rotation(tmp_path, with_occupancy):
    settings, cfg = read_settings(CONFIG)
    data = Rows("train", ["C", "CC", "CO", "CCC", "CCO", "CCN", "CCCl", "CF", "CNC"])
    val = Rows("val", ["N", "CN", "O"])
    cfg = replace(cfg, device="cpu", batch_size=4,
                  model=replace(cfg.model, global_hidden=12, hidden=8, latent=8, layers=1,
                                refinement_rounds=1, encoder_residual_layers=1, pair_hidden=4,
                                triangle_hidden=2, attention_heads=2))
    settings = dict(settings, expected_training_rows=len(data), data_sha256=data.dataset_sha256,
                    expected_parameters=sum(p.numel() for p in GraphAutoencoder(data.spec, cfg.model).parameters()),
                    checkpoint_every_updates=1, keep_checkpoint_updates=[], panel_every_updates=3, panel_early_updates=[1])
    if with_occupancy:
        settings["prior_occupancy"] = dict(real_per_prior=3, seed=2026111927,
                                           count_kl_weight=1., argmax_count_weight=1.)
        data.manifest = {**data.manifest, "active_atom_counts": {"train": {"1": 1, "2": 3, "3": 5}}}
    panel = CompactTrainingPanel(val, cfg, validation_count=3, prior_count=3, seed=17119, batch_size=2)

    def run(name, **kwargs):
        output = tmp_path / name
        output.mkdir()
        return train_segment(settings, cfg, data, output, "cpu", "fixture-config", panel, **kwargs)

    single = run("full")
    first = run("first", max_updates=1)
    assert first["status"] == "paused" and first["real_points"] == 4
    restore = {"resume": Path(first["latest"]["checkpoint"]), "resume_sha": first["latest"]["sha256"]}
    # A checkpoint written before atom-update dropout was added describes p=0.
    if not with_occupancy:
        old = torch.load(restore["resume"], weights_only=True)
        old["train_config"]["model"].pop("atom_refinement_dropout")
        legacy = tmp_path / "legacy.pt"
        torch.save(old, legacy)
        restore = {"resume": legacy, "resume_sha": file_sha256(legacy)}
        old["train_config"]["model"]["atom_refinement_dropout"] = .05
        changed = tmp_path / "changed.pt"
        torch.save(old, changed)
        with pytest.raises(ValueError, match="different compact training recipe"):
            run("reject-changed", resume=changed, resume_sha=file_sha256(changed))
    resumed = run("second", **restore)
    assert resumed["status"] == single["status"] == "completed"
    assert resumed["updates"] == 3 and resumed["real_points"] == len(data)
    assert resumed["row"] == 0 and resumed["epoch"] == 1 and resumed["segment_updates"] == 2
    checkpoints = [torch.load(row["latest"]["checkpoint"], weights_only=True) for row in (single, resumed)]
    assert state_digest(checkpoints[0]["model_state"]) == state_digest(checkpoints[1]["model_state"])
    torch.testing.assert_close(checkpoints[0]["optimizer_state"], checkpoints[1]["optimizer_state"], rtol=0, atol=0)
    assert torch.equal(checkpoints[0]["continuation"]["rng"]["loaders"]["epsilon"],
                       checkpoints[1]["continuation"]["rng"]["loaders"]["epsilon"])
    if with_occupancy:
        assert first["total_prior_points"] == 1 and resumed["total_prior_points"] == single["total_prior_points"] == 3
        assert torch.equal(checkpoints[0]["continuation"]["rng"]["loaders"]["prior_occupancy"],
                           checkpoints[1]["continuation"]["rng"]["loaders"]["prior_occupancy"])
        traces = [json.loads(s) for s in (tmp_path / "full/training.jsonl").read_text().splitlines()]
        assert [row["total_prior_points"] for row in traces] == [1, 3]
        for row in traces:
            parts = row["prior_guidance"]
            assert row["weighted_prior_loss"] == pytest.approx(parts["count_kl_per_slot"] + parts["argmax_count_loss"])
            assert row["prior_count"] == 1 and "bond_loss" not in parts and "count_cdf_loss" not in parts
    for name in ("full", "second"):
        assert len(list((tmp_path / name).glob("checkpoint-*.pt"))) == 1
    assert [json.loads(line)["updates"] for line in (tmp_path / "full/quality-panel.jsonl").read_text().splitlines()] == [0, 1, 3]
    with pytest.raises(ValueError, match="digest mismatch"):
        run("bad-sha", resume=restore["resume"], resume_sha="wrong")
    with pytest.raises(FileExistsError):
        train_segment(settings, cfg, data, tmp_path / "full", "cpu", "fixture-config", panel)
