from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from latentmol.guidance.valence_projection import projection_loss, projection_targets
from latentmol.models import GraphAutoencoder, ModelConfig
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch
from latentmol.train import TrainConfig
from latentmol.training import projected_continuation
from latentmol.training.losses import LossConfig
from latentmol.training.projected_continuation import append_rounds, projected_update


def test_projected_update_pairs_randomness_and_only_applies_motif_once(monkeypatch):
    spec = GraphSpec(max_nodes=5)
    architecture = ModelConfig(hidden=12, latent=8, layers=1, global_hidden=12, decoder_variant="triangle_refine",
                              pair_hidden=6, triangle_hidden=3, refinement_rounds=1, atom_refinement_dropout=.05,
                              encoder_pooling="canonical")
    parent = GraphAutoencoder(spec, architecture)
    ref = deepcopy(parent).eval().requires_grad_(False)
    training = TrainConfig(mode="vae", device="cpu", model=architecture, loss=LossConfig(atom_order="graph"))
    atoms = torch.tensor([[1, 1, 0, 0, 0]] * 4)
    bonds = torch.zeros((4, 5, 5), dtype=torch.long)
    bonds[:, 0, 1] = bonds[:, 1, 0] = 1
    hard = HardGraphBatch(atoms, torch.full_like(atoms, spec.formal_charges.index(0)), bonds)
    baseline = dict(parent_updates=700, projection=dict(reference_priors=1, weight=.375, max_edits=2),
                    prior_only=dict(guidance=dict(occupancy_weight=1., conditional_weight=1.)),
                    step_control=None)
    policy = dict(draws=2, every_updates=4, seed=103, weight=.01, aromatic_floor=.05, amide_floor=.08)
    called = []

    def controlled_loss(decoded, graph_spec, generator, **floors):
        assert graph_spec == spec and len(decoded.atoms) == 2
        assert floors == dict(aromatic_floor=.05, amide_floor=.08)
        called.append(torch.rand((), generator=generator).item())
        return decoded.atoms[..., 1].mean(), {"draws": 2}

    monkeypatch.setattr(projected_continuation, "functional_group_loss", controlled_loss)

    def run(update, config):
        model = deepcopy(parent).train()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        fresh = torch.Generator().manual_seed(93)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(21)
            metrics = projected_update(config, model, ref, optimizer, hard, training, None, None,
                                       {"fresh_prior": fresh}, update, 48, 2)
            rng = torch.get_rng_state()
        return metrics, model.decoder.atom_head.weight.detach(), rng, fresh.get_state()

    control, w0, rng0, prior0 = run(703, baseline)
    skip, w1, _, _ = run(702, dict(baseline, functional_groups=policy))
    assert control == skip and torch.equal(w0, w1) and not called
    enabled, w2, rng2, prior2 = run(703, dict(baseline, functional_groups=policy))
    assert len(called) == 1 and enabled["functional_groups"]["draws"] == 2
    assert enabled["total"] == pytest.approx(control["total"] + enabled["functional_groups"]["weighted_loss"])
    assert torch.equal(rng2, rng0) and torch.equal(prior2, prior0)
    assert not torch.equal(w2, w0)
    with pytest.raises(ValueError, match="microbatch"):
        run(703, dict(baseline, functional_groups=dict(policy, draws=3)))

    # The pooled frequency arm still spends only the first two local draws
    # on site supervision, and replays dropout without advancing base RNGs.
    monkeypatch.undo()
    pooled, _, pooled_rng, pooled_prior = run(703, dict(baseline, functional_groups=dict(policy, score_draws=4)))
    assert (pooled["functional_groups"]["draws"], pooled["functional_groups"]["local_draws"]) == (4, 2)
    assert torch.equal(pooled_rng, rng0) and torch.equal(pooled_prior, prior0)
    assert pooled["total"] == pytest.approx(control["total"] + pooled["functional_groups"]["weighted_loss"])
    with pytest.raises(ValueError, match="logical prior batch"):
        run(703, dict(baseline, functional_groups=dict(policy, score_draws=5)))


def test_roundtrip_reuses_first_prior_and_changes_only_encoder_gradients():
    spec = GraphSpec(max_nodes=3)
    arch = ModelConfig(hidden=12, latent=8, layers=1, global_hidden=12, decoder_variant="triangle_refine",
                       pair_hidden=6, triangle_hidden=3, refinement_rounds=1, atom_refinement_dropout=.05,
                       encoder_pooling="canonical")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(51)
        source = GraphAutoencoder(spec, arch)
    with torch.no_grad():
        # Produce valid cyclopropane centers while retaining live input Jacobians.
        for head, label in ((source.decoder.atom_head, 1), (source.decoder.charge_head, spec.formal_charges.index(0)),
                            (source.decoder.bond_head[-1], 1)):
            head.bias.fill_(-3)
            head.bias[label] = 3
    ref = deepcopy(source).eval().requires_grad_(False)
    training = TrainConfig(mode="vae", device="cpu", model=arch, loss=LossConfig(atom_order="graph"), clip_grad=1e5)
    atoms = torch.tensor([[1, 1, 0]] * 4)
    bonds = torch.zeros(4, 3, 3, dtype=torch.long)
    bonds[:, 0, 1] = bonds[:, 1, 0] = 1
    hard = HardGraphBatch(atoms, torch.full_like(atoms, spec.formal_charges.index(0)), bonds)
    cfg = dict(parent_updates=700, projection=dict(reference_priors=1, weight=.375, max_edits=2),
               prior_only=dict(guidance=dict(occupancy_weight=1., conditional_weight=1.)), step_control=None,
               functional_groups=dict(draws=2, score_draws=4, every_updates=4, seed=19, weight=.3,
                                      neighbor_weight=3., aromatic_floor=.87056, amide_floor=.47988))
    policy = dict(draws=2, max_parents=1, every_updates=4, weight=.1)

    def run(config):
        model = deepcopy(source).train()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        fresh = torch.Generator().manual_seed(37)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(23)
            metrics = projected_update(config, model, ref, optimizer, hard, training, None, None,
                                       {"fresh_prior": fresh}, 703, 48, 2)
            return metrics, model, torch.get_rng_state(), fresh.get_state()

    baseline, before, rng, priors = run(cfg)
    for disabled in (dict(policy, every_updates=5), dict(policy, weight=0.)):
        skipped, model, skipped_rng, skipped_priors = run(dict(cfg, generated_roundtrip=disabled))
        assert skipped == baseline
        torch.testing.assert_close(model.state_dict(), before.state_dict(), rtol=0, atol=0)
        assert torch.equal(skipped_rng, rng) and torch.equal(skipped_priors, priors)
    active, after, active_rng, active_priors = run(dict(cfg, generated_roundtrip=policy))
    cycle = active["generated_roundtrip"]
    assert (cycle["draws"], cycle["examined"], cycle["selected"]) == (2, 1, 1)
    assert cycle["weighted_loss"] == pytest.approx(cycle["reconstruction"] * .1)
    assert active["total"] == pytest.approx(baseline["total"] + cycle["weighted_loss"])
    assert active["functional_groups"] == baseline["functional_groups"]
    assert active["fresh_prior_points"] == baseline["fresh_prior_points"] == 4
    assert torch.equal(active_rng, rng) and torch.equal(active_priors, priors)
    assert not torch.equal(before.encoder.mu.weight.grad, after.encoder.mu.weight.grad)
    torch.testing.assert_close(before.decoder.state_dict(), after.decoder.state_dict(), rtol=0, atol=0)
    for a, b in zip(before.decoder.parameters(), after.decoder.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in ref.parameters())
    for invalid in (dict(policy, draws=3), dict(policy, max_parents=3), dict(policy, every_updates=0),
                    dict(policy, weight=float("nan"))):
        with pytest.raises(ValueError, match="Round-trip"):
            run(dict(cfg, generated_roundtrip=invalid))


def test_projection_changes_one_bond_and_protects_bridges():
    spec = GraphSpec(max_nodes=5)
    atoms = torch.ones(2, 5, dtype=torch.long)  # two carbon stars
    charges = torch.full_like(atoms, 2)
    bonds = torch.zeros(2, 5, 5, dtype=torch.long)
    bonds[:, 0, 1:] = bonds[:, 1:, 0] = 1
    bonds[0, 0, 1] = bonds[0, 1, 0] = 2  # one extra bond-order unit, a bridge
    logits = (4. * F.one_hot(bonds, 5).float()).requires_grad_()
    decoded = DecoderOutput(F.one_hot(atoms, 10).float(), F.one_hot(charges, 5).float(), logits)
    loss, counts = projection_loss(decoded, spec)
    assert counts == dict(over_valent=1, still_over_valent=0, changed_graphs=1, changed_edges=1)
    expected = F.cross_entropy(logits[0, 0, 1:2], torch.tensor([1])) / 2
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad[0, 0, 1, 1] < 0 and logits.grad[0, 1, 0, 2] > 0
    assert logits.grad.any(-1).sum() == 2
    # Five single bonds in a tree cannot be dropped while preserving its components.
    spec = GraphSpec(max_nodes=6)
    i, j = np.triu_indices(6, 1)
    bonds = (i == 0).astype(int)
    original, target, stats = projection_targets(np.ones((1, 6), int), np.full((1, 6), 2),
                                                np.eye(5)[bonds][None], spec, 8)
    assert np.array_equal(original, target) and stats["still_over_valent"] == 1


def test_projection_breaks_cycles_without_fragmenting_or_editing_aromatic_bonds():
    spec = GraphSpec(max_nodes=6)
    i, j = np.triu_indices(6, 1)
    labels = np.eye(5)[np.ones(len(i), int)][None].repeat(2, axis=0)
    labels[1] = np.eye(5)[np.full(len(i), 4)]
    original, target, stats = projection_targets(np.ones((2, 6), int), np.full((2, 6), 2), labels, spec, 8)
    adj = np.zeros((6, 6))
    adj[i, j] = target[0]
    adjacency = adj + adj.T + np.eye(6)
    assert (np.linalg.matrix_power(adjacency, 5) > 0).all()
    assert np.array_equal(original[1], target[1]) and (target[0] != original[0]).any()
    # Equal-confidence ties produce a protected star; its centre retains one excess unit.
    assert stats["still_over_valent"] == 1 and adjacency.sum(1)[0] == 6


def test_extension_is_identity_and_preserves_adamw_and_rng():
    config = ModelConfig(hidden=12, latent=8, layers=1, global_hidden=12, decoder_variant="triangle_refine",
                         pair_hidden=6, triangle_hidden=3, refinement_rounds=4)
    model = GraphAutoencoder(GraphSpec(max_nodes=5), config)
    optimizer = torch.optim.AdamW(model.parameters())
    z = torch.randn(3, 8)
    model.decode(z).bonds.square().mean().backward()
    optimizer.step()
    initial = model.decode(z)
    old_moments = {p: s["exp_avg"].clone() for p, s in optimizer.state.items()}
    rng = torch.get_rng_state()
    append_rounds(model, optimizer, 3, 321)
    assert torch.equal(torch.get_rng_state(), rng)
    extended = model.decode(z)
    for head in ("atoms", "charges", "bonds"):
        assert torch.equal(getattr(initial, head), getattr(extended, head))
    assert all(torch.equal(v, optimizer.state[p]["exp_avg"]) for p, v in old_moments.items())
    optimizer.zero_grad(set_to_none=True)
    extended.bonds.square().mean().backward()
    for block in model.decoder.extra_rounds:
        assert block.atom_gate.grad.abs() > 0 and block.pair_gate.grad.abs() > 0
    optimizer.step()
    fresh = GraphAutoencoder(model.spec, model.config)
    fresh.load_state_dict(model.state_dict())
    restored = torch.optim.AdamW(fresh.parameters())
    restored.load_state_dict(deepcopy(optimizer.state_dict()))
    assert torch.equal(fresh.decode(z).bonds, model.decode(z).bonds)


def test_parent_gate_follows_paused_parent_and_releases_only_completed_pair(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts/hpc"))
    import vae_paired_first_leg as gate
    recipe = {"dataset_sha256": "dataset", "training_molecules": 10, "parent_epoch": 46, "max_segments": 3}
    predecessor = dict(run_root=str(tmp_path), sparse=recipe)
    plan = dict(parent_updates=77, arms=[dict(name="four"), dict(name="seven")])
    seen = []
    monkeypatch.setattr(gate, "watch", lambda *args: seen.append(args[-1]) or 888)
    def publish(segment, finished):
        directory = tmp_path / f"sparse-{segment:02d}"
        directory.mkdir()
        path = directory / "checkpoint-077.pt"
        path.write_bytes(b"test")
        latest = dict(updates=77 if finished else 72, row=0 if finished else 5, epoch=47 if finished else 46,
                      real_points=10 if finished else 5, completed_epoch=47 if finished else 46,
                      checkpoint=str(path), sha256=hashlib.sha256(b"test").hexdigest())
        record = dict(latest, latest=latest, dataset_sha256="dataset", training_molecules=10, segment_updates=5,
                      status="completed" if finished else "paused",
                      stop_reason="epoch_target_reached" if finished else "allocation_budget_reached")
        (directory / "summary.json").write_text(json.dumps(record))
        (directory / "latest.json").write_text(json.dumps(latest))
    publish(1, False)
    assert gate.advance(plan, tmp_path, predecessor, 1)["next_parent_gate"] == 888 and seen == [2]
    assert not (tmp_path / "parent.json").exists()
    shared = tmp_path / "original-pair"
    shared.mkdir()
    plan["parent_receipt"] = str(shared / "parent.json")
    publish(2, True)
    for name, job in (("four", 123), ("seven", 456)):
        (tmp_path / (name + ".receipt.json")).write_text(json.dumps({"id": job}))
    released = []
    monkeypatch.setattr(gate.subprocess, "check_output", lambda *args, **kwargs: "PSUSP\n")
    monkeypatch.setattr(gate.subprocess, "run", lambda cmd, **kwargs: released.append(cmd))
    result = gate.advance(plan, tmp_path, predecessor, 2)
    assert released == [["bresume", "123"], ["bresume", "456"]]
    assert result["parent"] == json.loads((shared / "parent.json").read_bytes())
    assert not (tmp_path / "parent.json").exists()
