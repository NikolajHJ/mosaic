from dataclasses import replace

import pytest
import torch

from latentmol.chemistry import encode_smiles
from latentmol.evaluation.atom_order import permute_batch
from latentmol.models import GraphAutoencoder, ModelConfig
from latentmol.models.encoder import _graph_attention_inputs
from latentmol.representation import GraphSpec, HardGraphBatch


def graphs(spec, *smiles):
    records = [encode_smiles(s, spec) for s in smiles]
    return HardGraphBatch(*(torch.as_tensor([record[key].tolist() for record in records], dtype=torch.long)
                            for key in ("atom_types", "charges", "bond_types")))


def test_path_buckets_use_bonds_and_ignore_padding():
    spec = GraphSpec(max_nodes=8)
    atoms = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0]])
    bonds = torch.zeros(1, 8, 8, dtype=torch.long)
    # Six connected atoms, one separate occupied atom, one padding slot.
    for i in range(5):
        bonds[0, i, i+1] = bonds[0, i+1, i] = 2 if i == 1 else 1
    hard = HardGraphBatch(atoms, torch.full_like(atoms, 2), bonds)
    occupied, bond_type, distance = _graph_attention_inputs(hard.to_relaxed(spec))
    assert occupied.tolist() == [[True] * 7 + [False]]
    assert bond_type[0, 1, 2] == 2
    assert distance[0, 0].tolist() == [0, 1, 2, 3, 4, 5, 5, 5]
    assert distance[0, 7].tolist() == [5] * 8


def test_graft_is_exact_then_all_three_blocks_learn_and_keep_order_robustness():
    torch.manual_seed(2311)
    spec = GraphSpec(max_nodes=8)
    config = ModelConfig(hidden=12, latent=4, layers=2, encoder_pooling="canonical")
    parent = GraphAutoencoder(spec, config)
    child = GraphAutoencoder(spec, replace(config, encoder_degree_features=True, encoder_graph_attention_layers=3))
    missing, unexpected = child.load_state_dict(parent.state_dict(), strict=False)
    fresh = set(child.state_dict()) - set(parent.state_dict())
    assert not unexpected and set(missing) == fresh
    assert "encoder.degree_projection.weight" in fresh
    assert all(key.startswith("encoder.graph_attention.") for key in fresh - {"encoder.degree_projection.weight"})
    hard = graphs(spec, "CC(=O)O", "c1ccncc1")
    original = parent.encoder(hard, from_canonical_cache=True)
    output = child.encoder(hard, from_canonical_cache=True)
    assert all(torch.equal(a, b) for a, b in zip(original, output))
    opt = torch.optim.SGD(child.encoder.parameters(), lr=.05)
    for step in range(2):
        mu, logvar = child.encoder(hard, from_canonical_cache=True)
        (mu.square().mean() + logvar.square().mean()).backward()
        assert child.encoder.degree_projection.weight.grad.abs().sum() > 0
        for block in child.encoder.graph_attention:
            targets = [block.attention_output.weight, block.feedforward[-1].weight]
            if step:
                targets += [block.qkv.weight, block.distance_bias.weight, block.bond_bias.weight]
            assert all(torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in targets)
        opt.step()
        opt.zero_grad(set_to_none=True)
    shuffled = permute_batch(hard, torch.arange(7, -1, -1).expand(2, -1))
    expected = child.encoder(hard, from_canonical_cache=True)
    actual = child.encoder(shuffled.to_relaxed(spec))
    assert all(torch.equal(a, b) for a, b in zip(expected, actual))


def test_post_update_mask_excludes_padding_and_keeps_empty_graph_finite():
    spec = GraphSpec(max_nodes=5)
    config = ModelConfig(hidden=12, latent=4, encoder_pooling="canonical", encoder_graph_attention_layers=3)
    block = GraphAutoencoder(spec, config).encoder.graph_attention[0]
    torch.nn.init.normal_(block.attention_output.weight)
    torch.nn.init.normal_(block.feedforward[-1].weight)
    hard = graphs(spec, "CC", "CC")
    hard.atoms[1] = 0
    structural = _graph_attention_inputs(hard.to_relaxed(spec))
    h = torch.randn(2, 5, 12)
    altered = h.clone()
    altered[~structural[0]] = 17.5
    output = block(h, *structural)
    assert torch.equal(output, block(altered, *structural))
    assert torch.isfinite(output).all() and torch.count_nonzero(output[~structural[0]]) == 0


def test_attention_requires_hard_canonical_path_and_compatible_width():
    with pytest.raises(ValueError, match="canonical"):
        ModelConfig(encoder_graph_attention_layers=3)
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(encoder_graph_attention_layers=3, encoder_pooling="canonical", hidden=10, attention_heads=4)
