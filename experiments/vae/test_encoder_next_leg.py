"""The next objective must restore the trained attention blocks, AdamW and data position."""

import json
from argparse import Namespace
from dataclasses import asdict, replace

import pytest
import torch

from experiments.vae import encoder_next_leg as leg
from latentmol.experiments import file_sha256, write_json
from latentmol.models import GraphAutoencoder, ModelConfig
from latentmol.representation import GraphSpec
from latentmol.train import TrainConfig, capture_rng_state
from latentmol.training.combined_continuation import VERSION
from latentmol.training.motif_activation import policies


@pytest.mark.parametrize("operator_stop", [False, True])
def test_terminal_candidate_restores_trained_blocks_rng_and_optimizer(tmp_path, monkeypatch, operator_stop):
    recipe = json.loads(leg.PILOT_RECIPE.read_bytes())
    base = leg.PILOT_RECIPE.with_name(recipe["source_recipe"])
    cfg = json.loads(base.read_bytes())
    policy = {"parent_source": "eval", "false_positive_bonds": {"weight": .1, "max_pairs": 4}}
    write_json(tmp_path / "policy.json", policy)
    original = tmp_path / "predecessor" / "candidate"
    directory = tmp_path / "cutover" / "source" if operator_stop else original
    directory.mkdir(parents=True)
    parent = {"sha256": "a"*64, "checkpoint": "parent.pt"}
    fork_start = {"sha256": "b"*64, "updates": 611116}
    proto = leg._arm_identity(base, parent["sha256"], 3)
    candidate = leg._digest(dict(experiment="epoch48-three-block-encoder", pilot=recipe,
                                 predecessor_sha256=fork_start["sha256"], source_identity=proto))
    old = GraphAutoencoder(GraphSpec(max_nodes=5), ModelConfig(hidden=12, latent=8, layers=1,
        global_hidden=12, pair_hidden=6, triangle_hidden=3, refinement_rounds=4,
        decoder_variant="triangle_refine", encoder_pooling="canonical", atom_refinement_dropout=.05))
    full = GraphAutoencoder(old.spec, replace(old.config, encoder_graph_attention_layers=3,
                                              encoder_degree_features=True, extra_bond_rounds=3))
    train = TrainConfig(mode="vae", model=old.config, device="cpu")
    payload = dict(config_sha256=cfg["parent_config_sha256"], dataset_sha256=cfg["data_sha256"],
                   continuation={"updates": cfg["parent_updates"]}, train_config=asdict(train),
                   loss_config=asdict(train.loss))
    optimizer = torch.optim.AdamW(full.parameters())
    for i,p in enumerate(full.parameters(), 1):
        p.grad = torch.full_like(p, i / 1000)
    optimizer.step()
    rng = capture_rng_state({"fresh_prior": torch.Generator().manual_seed(97)}, False)
    update = fork_start["updates"] + 20
    row = (update - cfg["parent_updates"]) * cfg["batch_size"]
    cursor = dict(updates=update, row=row, real_points=row, epoch=cfg["parent_epoch"])
    metadata = dict(encoder_upgrade={"parent_sha256": fork_start["sha256"], "policy": recipe},
        motif_activation={"protocol": "rings", "winner": "control", "adopted_at_update": 598116,
                         "policy_sha256": leg._digest(policies(cfg)[1]["control"])},
        roundtrip_handoff={"target_protocol": proto, "source_protocol": "rings"})
    state = dict(format_version=1, combined_version=VERSION, epoch=cursor["epoch"],
        dataset_sha256=cfg["data_sha256"], config_sha256=candidate,
        continuation=dict(cursor, rng=rng, epoch_complete=False), model_config=asdict(full.config),
        graph_spec=old.spec.to_dict(), experiment_state=metadata, model_state=full.state_dict(),
        loss_config=asdict(replace(train.loss, atom_order="graph")), optimizer_state=optimizer.state_dict(),
        batch_size=cfg["batch_size"], microbatch_size=cfg["microbatch"])
    path = directory / f"checkpoint-{update:09d}.pt"
    torch.save(state, path)
    receipt = dict(cursor, completed_epoch=cursor["epoch"], sha256=file_sha256(path), checkpoint=str(path))
    fork = dict(parent=parent, resume=fork_start, source_config_sha256=proto, config_sha256=candidate,
                production_control=str(original.parent / "control"),
                encoder_seed=recipe["encoder_seed"], motif_winner="control",
                generated_roundtrip=dict(cfg["generated_roundtrip"], excess_confidence=recipe["excess_confidence"]))
    write_json(directory / "latest.json", receipt)
    write_json(directory / "fork.json", fork)
    summary = dict(latest=receipt, status="paused", stop_reason="allocation_budget_reached",
                   segment_updates=20, updates=update)
    stop_path = None
    if operator_stop:
        original.mkdir(parents=True)
        (original / "fork.json").write_bytes((directory / "fork.json").read_bytes())
        stop_path = directory.parent / "operator-stop.json"
        stop = dict(status="operator_stopped", job_id=30330582, terminal_status="EXIT",
                    source_directory=str(original), source_latest=dict(receipt, checkpoint=str(original / path.name)),
                    latest=receipt, last_logged_update=update + 17, fork_sha256=file_sha256(directory / "fork.json"))
        write_json(stop_path, stop)
        for changes in (dict(terminal_status="RUN"), dict(last_logged_update=update-1),
                        dict(source_latest=dict(stop["source_latest"], sha256="c"*64))):
            stop_path.write_text(json.dumps(dict(stop, **changes)))
            with pytest.raises(ValueError, match="Operator-stop handoff"):
                leg.check_operator_stop(stop_path, directory, receipt, fork)
        stop_path.write_text(json.dumps(stop))
    else:
        write_json(directory / "summary.json", summary)
    class TestCorpus:
        def split(self, name):
            return range(60_000_000) if name == "train" else range(300)
    args = Namespace(source_dir=directory, config=base, policy=tmp_path / "policy.json", output=tmp_path/"result",
                     operator_stop_receipt=stop_path)
    monkeypatch.setenv("LSB_JOBID","123")
    monkeypatch.setenv("DBMOL_LAUNCH_READY", str(tmp_path / "ready.json"))
    monkeypatch.setenv("DBMOL_LAUNCH_NONCE","hello")
    monkeypatch.setattr(leg.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(leg, "_check_cuda_requirement", lambda *_: None)
    monkeypatch.setattr(leg, "verify_canonical_training_cache", lambda *_: None)
    monkeypatch.setattr(leg, "BalancedSparseCorpus", lambda *a, **kw: TestCorpus())
    monkeypatch.setattr(leg, "_load", lambda *a: (old, payload, None, None, None))
    def training(config, identity, data, assets, dest, device, size, resumed, *args, **kwargs):
        assert config["quality_panel"]["generated_cycle_bins"]
        assert config["generated_roundtrip"] == dict(fork["generated_roundtrip"], **policy)
        assert resumed["continuation"]["row"] == row and resumed["continuation"]["updates"] == update
        assert kwargs["experiment_state"]["encoder_upgrade"] == metadata["encoder_upgrade"]
        assert identity == leg._digest(kwargs["experiment_state"]["generated_bond_handoff"])
        if operator_stop:
            assert kwargs["experiment_state"]["generated_bond_handoff"]["operator_stop"]["saved_update"] == update
        student, opt, training_cfg, generators = kwargs["student_factory"](
            assets[0], payload, None, size, "cpu", resumed, atom_refinement_dropout=.05)
        assert student.config == full.config and training_cfg.loss.atom_order == "graph"
        torch.testing.assert_close(student.state_dict(), full.state_dict(), rtol=0, atol=0)
        torch.testing.assert_close(opt.state_dict(), optimizer.state_dict(), rtol=0, atol=0)
        assert torch.equal(generators["fresh_prior"].get_state(), rng["loaders"]["fresh_prior"])
        assert torch.equal(torch.get_rng_state(), rng["torch"])
        return "restored"
    monkeypatch.setattr(leg, "_train", training)
    assert leg.run(args) == "restored"
    assert json.loads((tmp_path/"ready.json").read_bytes())["nonce"] == "hello"
    with pytest.raises(ValueError, match="new output"):
        leg.run(args)
    with pytest.raises(ValueError, match="terminal"):
        leg.check_terminal(dict(summary, updates=update-1), receipt, fork)
    with pytest.raises(ValueError, match="three-block"):
        leg.successor_config(cfg, dict(fork, config_sha256=proto), state, proto, recipe, policy)
