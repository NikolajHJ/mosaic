"""Lossless hard-input invariance must retain the pretrained ordered information."""
from dataclasses import replace

import pytest
import torch

from latentmol.chemistry import MoleculeRejected, canonical_slot_order, encode_smiles
from latentmol.datasets.ordering import canonical_encoder_input
from latentmol.evaluation.atom_order import permute_batch
from latentmol.models import GraphAutoencoder, ModelConfig
from latentmol.representation import GraphSpec, HardGraphBatch, RelaxedGraph
from latentmol.training.invariant_pilot import initialize_invariant


def hard_graph(smiles, spec):
    row = encode_smiles(smiles, spec)
    return HardGraphBatch(*(torch.tensor(row[k], dtype=torch.long)[None]
                           for k in ("atom_types", "charges", "bond_types")))


@pytest.mark.parametrize("smiles", ["C", "CCO", "CC(C)C", "c1ccccc1", "O=S(=O)(C)C",
    "[NH3+]CC(=O)[O-]", "CC(=O)N", "Clc1ccc(Br)cc1", "CCOP(=O)(O)O",
    "C12C3C1C1C2C31", "C12C3C4C1C3C24"])
def test_full_slot_permutations_restore_every_canonical_label(smiles):
    spec = GraphSpec(max_nodes=16)
    target = hard_graph(smiles, spec)
    rng = torch.Generator().manual_seed(911)
    for _ in range(4):
        shuffled = permute_batch(target, torch.randperm(16, generator=rng)[None])
        expected = target.to_relaxed(spec)
        actual = canonical_encoder_input(shuffled.to_relaxed(spec), spec)
        for key in ("atoms", "charges", "bonds"):
            assert torch.equal(getattr(actual, key), getattr(expected, key))


def test_wrapper_preserves_parent_codes_and_parameter_gradients():
    torch.manual_seed(17)
    spec = GraphSpec(max_nodes=8)
    parent = GraphAutoencoder(spec, ModelConfig(hidden=16, latent=8, layers=4))
    model, receipt = initialize_invariant(parent, pooling="canonical")
    assert receipt["encoder_initially_identical"] and not receipt["initialized"] and not receipt["removed"]
    target = hard_graph("CC(=O)N", spec)
    shuffled = permute_batch(target, torch.tensor([[6, 2, 0, 7, 4, 3, 5, 1]]))
    mu, logvar = model.encode(shuffled.to_relaxed(spec))
    expected_mu, expected_logvar = parent.encode(target.to_relaxed(spec))
    assert torch.equal(mu, expected_mu) and torch.equal(logvar, expected_logvar)
    cached = model(target, from_canonical_cache=True)
    assert torch.equal(cached.mu, mu) and torch.equal(cached.logvar, logvar)
    (mu.square().mean()+logvar.square().mean()).backward()
    for layer in [model.encoder.atom_projection, model.encoder.mu, model.encoder.logvar]:
        assert layer.weight.grad is not None and bool(torch.isfinite(layer.weight.grad).all())
        assert float(layer.weight.grad.norm()) > 0
    # New config persists the discrete boundary; no tensor name or shape changed.
    loaded = GraphAutoencoder(spec, replace(parent.config, encoder_pooling="canonical"))
    loaded.load_state_dict(parent.state_dict(), strict=True)
    torch.testing.assert_close(loaded.encode(shuffled.to_relaxed(spec))[0], expected_mu)


def test_neural_set_encoder_still_collides_on_regular_cages_but_wrapper_does_not():
    spec = GraphSpec(max_nodes=8)
    targets = [hard_graph(s, spec) for s in ("C12C3C1C1C2C31", "C12C3C4C1C3C24")]
    torch.manual_seed(31)
    old = GraphAutoencoder(spec, ModelConfig(hidden=16, latent=8, layers=4,
        encoder_pooling="invariant", encoder_aggregation="sum", encoder_degree_features=True))
    codes = [old.encode(g.to_relaxed(spec))[0] for g in targets]
    assert float((codes[0]-codes[1]).norm()) < 1e-6
    parent = GraphAutoencoder(spec, ModelConfig(hidden=16, latent=8, layers=4))
    wrapper, _ = initialize_invariant(parent, pooling="canonical")
    preserved = [wrapper.encode(g.to_relaxed(spec))[0] for g in targets]
    assert float((preserved[0]-preserved[1]).norm()) > 1e-4


def test_reject_soft_nonfinite_required_input_gradient_and_chemistry_changes():
    spec = GraphSpec(max_nodes=8)
    hard = hard_graph("CCO", spec)
    value = hard.to_relaxed(spec)
    soft_atoms = .99*value.atoms+.01/len(spec.atomic_numbers)
    with pytest.raises(ValueError, match="discrete one-hot"):
        canonical_encoder_input(RelaxedGraph(soft_atoms, value.charges, value.bonds), spec)
    with pytest.raises(ValueError, match="input gradients"):
        canonical_encoder_input(RelaxedGraph(value.atoms.clone().requires_grad_(), value.charges, value.bonds), spec)
    nan_atoms = value.atoms.clone(); nan_atoms[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        canonical_encoder_input(RelaxedGraph(nan_atoms, value.charges, value.bonds), spec)
    benzene = hard_graph("c1ccccc1", spec)
    benzene.bonds.zero_()
    for i in range(6):
        j = (i+1) % 6
        benzene.bonds[0, i, j] = benzene.bonds[0, j, i] = 1+i % 2
    with pytest.raises(MoleculeRejected, match="not_label_preserving"):
        canonical_slot_order(benzene.atoms[0], benzene.charges[0], benzene.bonds[0], spec)


def test_training_reuses_canonical_cache_and_checkpoint_handles_shuffled_input(tmp_path, monkeypatch):
    from latentmol.datasets import ordering
    from latentmol.datasets.data import prepare_dataset
    from latentmol.evaluation.evaluate import load_model
    from latentmol.train import TrainConfig, train

    spec = GraphSpec(max_nodes=8)
    source = tmp_path / "molecules.smi"
    source.write_text("C\nCC\nCCO\nCCN\nCCF\nCCC\nCO\nCN\n")
    cache = tmp_path / "cache"
    prepare_dataset(source, cache, spec=spec, train_count=4, val_count=2, seed=17)
    calls = []

    def track_ordering(graph, spec):
        calls.append(len(graph.atoms))
        return canonical_encoder_input(graph, spec)

    monkeypatch.setattr(ordering, "canonical_encoder_input", track_ordering)
    config = TrainConfig(mode="vae", epochs=1, batch_size=2, validation_limit=2,
                         chemistry_limit=2, device="cpu", checkpoint_every=1,
                         model=ModelConfig(hidden=8, latent=4, layers=1, encoder_pooling="canonical"))
    run = train(config, cache, tmp_path / "runs")
    assert calls == [4, 2]  # One startup verification per split, none per batch.

    model, payload = load_model(run / "last.pt", "cpu", spec)
    assert payload["model_config"]["encoder_pooling"] == "canonical"
    target = hard_graph("CCO", spec)
    shuffled = permute_batch(target, torch.tensor([[6, 2, 0, 7, 4, 3, 5, 1]]))
    with torch.inference_mode():
        reference = model(target, from_canonical_cache=True)
        actual = model(shuffled.to_relaxed(spec))
    assert torch.equal(actual.mu, reference.mu)
    assert torch.equal(actual.logvar, reference.logvar)
    for key in ("atoms", "charges", "bonds"):
        assert torch.equal(getattr(actual.decoded, key), getattr(reference.decoded, key))
