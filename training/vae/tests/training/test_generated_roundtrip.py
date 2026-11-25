import pytest
import torch

from latentmol.chemistry import encode_smiles
from latentmol.evaluation.atom_order import permute_batch
from latentmol.models import GraphAutoencoder, ModelConfig
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch
from latentmol.training.generated_roundtrip import canonical_generated_parents, generated_roundtrip_loss
from latentmol.training.confidence_margins import matched_confidence_loss
from latentmol.training.losses import LossConfig, reconstruction_losses


def _graphs(spec, *smiles):
    records = [encode_smiles(s, spec) for s in smiles]
    return HardGraphBatch(*(torch.stack([torch.tensor(r[key], dtype=torch.long) for r in records])
                            for key in ("atom_types", "charges", "bond_types")))


def _logits(graph, spec):
    labels = graph.to_relaxed(spec)
    return DecoderOutput(*(head.requires_grad_() for head in (labels.atoms, labels.charges, labels.bonds)))


def test_screening_preserves_all_labels_and_accounts_for_skips_and_cap():
    spec = GraphSpec(max_nodes=8)
    hard = _graphs(spec, "C", "c1ccccc1", "[NH3+]CC(=O)[O-]", "CC#N", "CO")
    hard.atoms[0] = 0
    # Valid benzene only after changing its input bond categories to aromatic.
    hard.bonds[1] = 0
    for i in range(6):
        j = (i + 1) % 6
        hard.bonds[1, i, j] = hard.bonds[1, j, i] = 1 + i % 2
    order = torch.tensor([6, 2, 0, 7, 4, 3, 5, 1]).expand(5, -1)
    selected, stats = canonical_generated_parents(permute_batch(hard, order), spec, max_parents=2,
                                                  include_smiles=True)
    identities = [encode_smiles(s, spec)["smiles"] for s in ("[NH3+]CC(=O)[O-]", "CC#N")]
    assert stats == dict(draws=5, examined=4, selected=2, rows=[2, 3], smiles=identities, rejections={
        "canonical_input_empty": 1, "canonical_input_not_label_preserving": 1})
    for key in ("atoms", "charges", "bonds"):
        assert torch.equal(getattr(selected, key), getattr(hard, key)[2:4])


def test_frozen_decoder_pass_preserves_joint_gradients_rng_and_modes():
    torch.manual_seed(271)
    spec = GraphSpec(max_nodes=8)
    model = GraphAutoencoder(spec, ModelConfig(hidden=8, latent=4, layers=1, global_hidden=8,
                             pair_hidden=4, triangle_hidden=2, refinement_rounds=4, extra_bond_rounds=3,
                             decoder_variant="triangle_refine", atom_refinement_dropout=.1,
                             encoder_pooling="canonical")).train()
    model.decoder.rounds[0].eval()
    modes = [m.training for m in model.modules()]
    values = [p.detach().clone() for p in model.decoder.parameters()]
    hard = _graphs(spec, "CCO", "[NH3+]CC(=O)[O-]")
    parents = _logits(hard, spec)
    policy = LossConfig(atom_order="graph")
    rng = torch.get_rng_state()
    loss, stats = generated_roundtrip_loss(model, parents, policy, max_parents=2)
    assert stats["examined"] == stats["selected"] == 2
    assert stats["reconstruction"] == pytest.approx(loss.item())
    assert torch.equal(rng, torch.get_rng_state()) and [m.training for m in model.modules()] == modes
    loss.backward()
    for p in (model.encoder.mu.weight, model.encoder.atom_projection.weight):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0
    assert model.encoder.logvar.weight.grad is None
    assert all(p.grad is None for p in model.decoder.parameters())
    assert all(t.grad is None for t in (parents.atoms, parents.charges, parents.bonds))
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    optimizer.step()
    assert all(torch.equal(p, v) for p, v in zip(model.decoder.parameters(), values))

    # Joint real-data gradients still accumulate on the original parameters.
    optimizer.zero_grad(set_to_none=True)
    output = model(hard, sample_latent=True, from_canonical_cache=True)
    reconstruction_losses(output.decoded, hard, policy)["reconstruction"].backward()
    decoder_grad = model.decoder.atom_head.weight.grad.clone()
    assert decoder_grad.norm() > 0 and model.encoder.mu.weight.grad.norm() > 0
    generated_roundtrip_loss(model, parents, policy, max_parents=2)[0].backward()
    assert torch.equal(decoder_grad, model.decoder.atom_head.weight.grad)


def test_generated_roundtrip_is_equal_parent_mean():
    spec = GraphSpec(max_nodes=8)
    model = GraphAutoencoder(spec, ModelConfig(hidden=8, latent=4, layers=1, encoder_pooling="canonical"))
    policy = LossConfig(atom_order="graph")
    parents = _logits(_graphs(spec, "C", "CC(=O)N"), spec)
    together, _ = generated_roundtrip_loss(model, parents, policy, max_parents=2)
    separate = [generated_roundtrip_loss(model, _logits(_graphs(spec, s), spec), policy, max_parents=1)[0]
                for s in ("C", "CC(=O)N")]
    torch.testing.assert_close(together, torch.stack(separate).mean())


def test_no_eligible_parents_contribute_zero_and_nonfinite_logits_raise():
    spec = GraphSpec(max_nodes=4)
    model = GraphAutoencoder(spec, ModelConfig(hidden=8, latent=4, layers=1, encoder_pooling="canonical"))
    hard = _graphs(spec, "C", "CC")
    hard.atoms.zero_()
    parents = _logits(hard, spec)
    loss, stats = generated_roundtrip_loss(model, parents, LossConfig(), max_parents=1)
    assert loss.item() == 0 and not loss.requires_grad
    assert stats == dict(draws=2, examined=2, selected=0, rejections={"canonical_input_empty": 2})
    with torch.no_grad():
        parents.atoms[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        generated_roundtrip_loss(model, parents, LossConfig(), max_parents=1)


def test_excess_margin_matches_symmetry_and_only_trains_active_overconfident_sites():
    spec = GraphSpec(max_nodes=6)
    targets = _graphs(spec, "CC", "CO")
    order = torch.tensor([[4, 3, 5, 2, 1, 0], [1, 3, 2, 0, 5, 4]])
    teacher = _logits(permute_batch(targets, order), spec)
    student = _logits(targets, spec)
    with torch.no_grad():
        # Both carbons in ethane are equivalent; sorting must tolerate switched margins.
        teacher.atoms[0, 4] *= 4
        student.atoms[0, 0] *= 4
        # CO has two occupied atoms, one present bond: two extra gap units on
        # its carbon above a tolerance of one gives (2/2 atoms)/2 heads/2 parents.
        student.atoms[1, 0] *= 4
        student.atoms[:, 4:, 0] = 1000
        student.charges.mul_(10)
        student.bonds[:, 2:, :, 0] = 1000
    loss, stats = matched_confidence_loss(student, teacher, targets, spec, 1.)
    assert stats["exact_pairs"] == 2
    assert loss.item() == pytest.approx(.25)
    loss.backward()
    assert student.atoms.grad[1, 0].abs().sum() > 0
    assert torch.count_nonzero(student.atoms.grad[:, 2:]) == 0
    assert torch.count_nonzero(student.bonds.grad) == 0
    assert teacher.atoms.grad is None and teacher.bonds.grad is None and student.charges.grad is None

    with torch.no_grad():
        student.charges[1, 0, 0] = 50
    skipped, stats = matched_confidence_loss(student, teacher, targets, spec, 1.)
    assert stats["reference_matches"] == 2 and stats["exact_pairs"] == 1 and skipped.item() == 0


def test_margin_uses_eval_reference_without_changing_baseline_targets_or_training_rng():
    from types import SimpleNamespace
    spec = GraphSpec(max_nodes=6)
    hard = _graphs(spec, "CO")
    base = _logits(hard, spec)
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mean = torch.nn.Parameter(torch.tensor(1.))
        def forward(self, targets, *, from_canonical_cache=False):
            mu = self.mean.expand(len(targets.atoms), 1)
            return mu, mu * 0
    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.))
            self.dropout = torch.nn.Dropout(.5)
        def forward(self, z):
            scale = self.scale * (1 + self.dropout(z.flatten()))
            return DecoderOutput(*(getattr(base, name).detach() * scale.reshape(-1, *([1] * ndims))
                                   for name, ndims in (("atoms", 2), ("charges", 2), ("bonds", 3))))
    model = SimpleNamespace(encoder=Encoder(), decoder=Decoder(), spec=spec, config=SimpleNamespace(latent=1))
    # Training-mode parent logits deliberately differ from its eval confidence.
    input_logits = DecoderOutput(*(getattr(base, name) * 17 for name in ("atoms", "charges", "bonds")))
    prior = torch.zeros((1, 1), requires_grad=True)
    rng = torch.get_rng_state()
    kwargs = dict(max_parents=1, prior_latents=prior, excess_confidence={"weight": .1, "tolerance": .5})
    loss, stats = generated_roundtrip_loss(model, input_logits, LossConfig(), **kwargs)
    assert stats["excess_confidence"]["exact_pairs"] == 1
    assert stats["excess_confidence"]["loss"] == pytest.approx(.5)
    assert loss.item() == pytest.approx(stats["reconstruction"] + .05)
    assert model.decoder.dropout.training and torch.equal(rng, torch.get_rng_state())
    loss.backward()
    assert model.encoder.mean.grad is not None and model.encoder.mean.grad.isfinite()
    assert prior.grad is None and model.decoder.scale.grad is None

    # Dropout can even make the training parent invalid. Its reconstruction
    # eligibility must remain unchanged while the independent eval parent trains.
    empty = _graphs(spec, "C")
    empty.atoms.zero_()
    loss, stats = generated_roundtrip_loss(model, _logits(empty, spec), LossConfig(), **kwargs)
    assert not stats["selected"] and stats["excess_confidence"]["exact_pairs"] == 1
    assert loss.item() == pytest.approx(.05)


@pytest.mark.parametrize("invent_edge", [True, False])
def test_shared_eval_target_and_single_posterior_decode(monkeypatch, invent_edge):
    from types import SimpleNamespace
    from latentmol.training import matching
    from latentmol.training.projected_continuation import _scheduled_roundtrip_policy

    spec = GraphSpec(max_nodes=5)
    hard = _graphs(spec, "CCO")
    labels = hard.to_relaxed(spec)
    i, j = torch.nonzero(((hard.atoms[0] != 0)[:, None] & (hard.atoms[0] != 0)[None, :]
                           & (hard.bonds[0] == 0)).triu(1))[0].tolist()
    off = torch.zeros_like(labels.bonds)
    off[0, i, j, 1] = off[0, j, i, 1] = 5 * invent_edge
    calls = []
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mean = torch.nn.Parameter(torch.tensor(1.))
        def forward(self, targets, *, from_canonical_cache):
            assert torch.equal(targets.bonds, hard.bonds)
            calls.append("encode")
            return self.mean.reshape(1, 1), self.mean.reshape(1, 1)*0
    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.))
        def forward(self, z):
            assert not self.training
            calls.append("decode")
            value = z.squeeze() * self.scale
            return DecoderOutput(labels.atoms*(1 + value), labels.charges, labels.bonds + value*off)
    model = SimpleNamespace(encoder=Encoder(), decoder=Decoder(), config=SimpleNamespace(latent=1), spec=spec)
    original_align = matching.align_reconstruction_targets
    def counted(*args, **kwargs):
        calls.append("align")
        return original_align(*args, **kwargs)
    monkeypatch.setattr(matching, "align_reconstruction_targets", counted)
    prior = torch.zeros((1, 1), requires_grad=True)
    empty = _graphs(spec, "C")
    empty.atoms.zero_()
    train_draw = _logits(empty, spec)
    options = dict(parent_source="eval", false_positive_bonds={"max_pairs": 4, "weight": .1},
                   excess_confidence={"tolerance": 0., "weight": .1})
    config = dict(parent_updates=0, generated_roundtrip=dict(**options, weight=.1, max_parents=1,
                                                             draws=1, every_updates=4))
    assert _scheduled_roundtrip_policy(config, 2, 4, 2) is None
    assert _scheduled_roundtrip_policy(config, 3, 4, 2) == config["generated_roundtrip"]
    loss, metrics = generated_roundtrip_loss(model, train_draw, LossConfig(atom_order="graph"),
                                             prior_latents=prior, max_parents=1, **options)
    assert calls == ["decode", "encode", "decode", "align"]
    assert metrics["parent_source"] == "eval" and metrics["selected"] == 1
    assert metrics["excess_confidence"]["exact_pairs"] == int(not invent_edge)
    focus = metrics["false_positive_bonds"]
    assert focus["selected_pairs"] == int(invent_edge)
    assert loss.item() == pytest.approx(metrics["reconstruction"] + focus["weighted_loss"]
                                       + metrics["excess_confidence"]["weighted_loss"])
    loss.backward()
    assert model.encoder.mean.grad is not None and torch.isfinite(model.encoder.mean.grad)
    assert model.decoder.scale.grad is None and model.decoder.training
    assert prior.grad is None and train_draw.bonds.grad is None
