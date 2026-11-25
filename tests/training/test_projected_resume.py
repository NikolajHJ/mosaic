from copy import deepcopy
from contextlib import nullcontext
from dataclasses import asdict
from functools import partial
import json
from pathlib import Path

import pytest
import torch

from latentmol.models import GraphAutoencoder, ModelConfig
from latentmol.representation import GraphSpec
from latentmol.train import TrainConfig, capture_rng_state
from latentmol.training.combined_continuation import _train
from latentmol.training.projected_continuation import _arm_identity, _read_resume, _source_identity, _student, projected_update


def test_one_time_objective_handoff_changes_only_motif_policy(tmp_path):
    old, new = (tmp_path / name for name in ("old.json", "new.json"))
    common = {"batch_size": 240, "shuffle_seed": 101}
    old.write_text(json.dumps(dict(common, functional_groups={"weight": .3})))
    new.write_text(json.dumps(dict(common, functional_groups={"weight": .4})))
    old_identity = _source_identity(new, old, "parent-digest", 3)
    assert old_identity == _arm_identity(old, "parent-digest", 3)
    assert old_identity != _arm_identity(new, "parent-digest", 3)
    new.write_text(json.dumps(dict(common, shuffle_seed=102, functional_groups={"weight": .4})))
    with pytest.raises(ValueError, match="differing only in functional_groups"):
        _source_identity(new, old, "parent-digest", 3)


def test_roundtrip_recipe_transition_only_adds_the_new_loss(tmp_path):
    ring = Path(__file__).resolve().parents[2] / "experiments/vae/epoch48-seven-round-ring-sizes.json"
    old = json.loads(ring.read_bytes())
    new = tmp_path / "roundtrip.json"
    changed = dict(old, generated_roundtrip=dict(draws=16, max_parents=8, every_updates=4, weight=.1))
    new.write_text(json.dumps(changed))
    assert _source_identity(new, ring, "parent", 3) == _arm_identity(ring, "parent", 3)
    with pytest.raises(ValueError, match="may only add"):
        _source_identity(new, ring, "parent", 0)
    new.write_text(json.dumps(dict(changed, functional_groups=old["functional_groups"]["activation_gate"]["control"])))
    with pytest.raises(ValueError, match="may only add"):
        _source_identity(new, ring, "parent", 3)


def test_projected_resume_matches_unbroken_training(tmp_path):
    """Interrupt after one real projected update and compare the rest through the epoch boundary."""
    class Rows(list):
        manifest = {"dataset_sha256": "test"}

    spec = GraphSpec(max_nodes=5)
    arch = ModelConfig(hidden=12, latent=8, layers=1, global_hidden=12, pair_hidden=6, triangle_hidden=3,
                       decoder_variant="triangle_refine", refinement_rounds=4, encoder_pooling="canonical",
                       atom_refinement_dropout=.05)
    model = GraphAutoencoder(spec, arch)
    train = TrainConfig(model=arch, mode="vae", device="cpu", batch_size=2)
    rng = capture_rng_state({"fresh_prior": torch.Generator().manual_seed(15)}, False)
    parent = dict(train_config=asdict(train), loss_config=asdict(train.loss),
                  optimizer_state=torch.optim.AdamW(model.parameters()).state_dict(),
                  continuation=dict(updates=700, rng=rng))
    config = dict(data_sha256="test", epochs=1, batch_size=2, microbatch=2, parent_epoch=47,
                  parent_updates=700, shuffle_seed=13, checkpoint_every_updates=1, atom_order="graph",
                  atom_refinement_dropout=.05, projection=dict(reference_priors=1, max_edits=2, weight=.375),
                  prior_only=dict(guidance=dict(occupancy_weight=1., conditional_weight=1.)), step_control=None,
                  functional_groups=dict(draws=2, every_updates=2, seed=9, weight=.3,
                                         aromatic_floor=.08, amide_floor=.12))
    rows = Rows()
    for i in range(6):
        atoms = torch.tensor([1, 1 + i % 3, 0, 0, 0])
        bonds = torch.zeros((5, 5), dtype=torch.long)
        bonds[0, 1] = bonds[1, 0] = 1
        rows.append(dict(atoms=atoms, charges=torch.full_like(atoms, spec.formal_charges.index(0)), bonds=bonds))
    assets = (model, parent, deepcopy(model).eval().requires_grad_(False), None, None)

    def run(name, resume=None, limit=None):
        output = tmp_path / name
        output.mkdir()
        return _train(config, "lineage", rows, assets, output, "cpu", 2, resume, limit, None, 2,
                      student_factory=partial(_student, count=3, seed=21, atom_order="graph"),
                      teacher_factory=lambda *_: nullcontext(), update_fn=projected_update)

    full = run("full")
    paused = run("first-leg", limit=1)
    assert paused["status"] == "paused" and paused["row"] == 2
    path = tmp_path / "first-leg" / "latest.json"
    read = partial(_read_resume, config=config, model=model, payload=parent, count=3, train_size=len(rows))
    _, resume = read(path, "lineage")
    with pytest.raises(ValueError, match="training arm"):
        read(path, "different-lineage")
    tampered = dict(json.loads(path.read_bytes()), row=4)
    bad_path = path.with_name("wrong-row.json")
    bad_path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="cursor"):
        read(bad_path, "lineage")
    resumed = run("next-allocation", resume)
    assert resumed["status"] == full["status"] == "completed"
    assert resumed["epoch"] == 48 and resumed["updates"] == 703 and resumed["row"] == 0
    states = [torch.load(Path(report["latest"]["checkpoint"]), map_location="cpu", weights_only=True)
              for report in (full, resumed)]
    for key in ("model_state", "optimizer_state"):
        torch.testing.assert_close(states[0][key], states[1][key], rtol=0., atol=0.)
    saved_rng = [state["continuation"]["rng"] for state in states]
    assert saved_rng[0]["python"] == saved_rng[1]["python"]
    for key in ("torch", "cuda", "loaders"):
        torch.testing.assert_close(saved_rng[0][key], saved_rng[1][key], rtol=0., atol=0.)
    # The mid-epoch update remains scheduled globally and sees the same latent and data rows.
    def events(name):
        return [{k: v for k, v in json.loads(line).items() if k != "seconds"}
                for line in (tmp_path / name / "training.jsonl").read_text().splitlines()]

    assert events("next-allocation") == events("full")[1:]
