"""The encoder fork must retain AdamW moments even for modules after the new branch."""

from copy import deepcopy

import torch

from latentmol.models import GraphAutoencoder, ModelConfig
from latentmol.representation import GraphSpec
from latentmol.training.encoder_expansion import expand_encoder


def test_graft_retains_parameter_identity_rng_and_reloadable_moments():
    model = GraphAutoencoder(GraphSpec(), ModelConfig(hidden=12, latent=8, layers=1,
                                                       encoder_pooling="canonical"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    inherited = dict(model.named_parameters())
    for index, parameter in enumerate(inherited.values(), start=1):
        parameter.grad = torch.full_like(parameter, index / 100.)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    before = torch.random.get_rng_state()
    moments = {key: optimizer.state[p]["exp_avg"].clone() for key, p in inherited.items()}
    assert expand_encoder(model, optimizer, seed=13) > 0
    assert torch.equal(torch.random.get_rng_state(), before)
    expanded = dict(model.named_parameters())
    assert all(expanded[name] is p for name, p in inherited.items())
    assert all(expanded[name] not in optimizer.state for name in expanded.keys() - inherited.keys())
    clone = GraphAutoencoder(model.spec, model.config)
    clone.load_state_dict(model.state_dict())
    restored = torch.optim.AdamW(clone.parameters())
    restored.load_state_dict(deepcopy(optimizer.state_dict()))
    for name, p in clone.named_parameters():
        if name in inherited:
            assert torch.equal(restored.state[p]["exp_avg"], moments[name])
            assert restored.state[p]["step"].item() == 1
