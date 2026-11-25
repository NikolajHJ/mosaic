"""Numerical contract tests. Execute on DTU HPC, including CPU allocations."""

import torch

from latentmol.losses import LossConfig, latent_kl, reconstruction_losses
from latentmol.models import GraphAutoencoder, ModelConfig
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch


def small_graph():
    spec = GraphSpec(max_nodes=4)
    atoms = torch.tensor([[1, 3, 0, 0]])  # carbon--oxygen and two empty slots
    charges = torch.full_like(atoms, spec.formal_charges.index(0))
    bonds = torch.zeros(1, 4, 4, dtype=torch.long)
    bonds[0, 0, 1] = bonds[0, 1, 0] = 1
    return spec, HardGraphBatch(atoms, charges, bonds)


def test_decoder_invariants_and_latent_gradient():
    torch.manual_seed(3)
    spec, _ = small_graph()
    model = GraphAutoencoder(spec, ModelConfig(hidden=16, latent=8, layers=1))
    z = torch.randn(2, 8, requires_grad=True)
    relaxed = model.decode(z).probabilities()
    assert torch.allclose(relaxed.bonds, relaxed.bonds.transpose(1, 2))
    assert torch.allclose(relaxed.bonds.sum(-1), torch.ones(2, 4, 4))
    assert torch.allclose(relaxed.bonds[:, range(4), range(4), 0], torch.ones(2, 4))
    assert torch.all(relaxed.bonds[:, range(4), range(4), 1:] == 0)
    (relaxed.expected_size.sum() + relaxed.bonds[..., 1].sum()).backward()
    assert z.grad is not None and torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0


def test_absent_atom_cannot_retain_bonds():
    spec, _ = small_graph()
    atoms = torch.full((1, 4, len(spec.atomic_numbers)), -1000.0)
    atoms[..., 0] = 0.0
    bonds = torch.zeros(1, 4, 4, len(spec.bond_names))
    bonds[..., 1] = 10.0
    output = DecoderOutput(atoms, torch.zeros(1, 4, len(spec.formal_charges)), bonds)
    graph = output.probabilities()
    assert torch.all(graph.bonds[..., 0] == 1)
    assert torch.all(output.harden(spec).bonds == 0)


def test_reconstruction_uses_same_undirected_bonds_as_inference():
    spec, hard = small_graph()
    torch.manual_seed(5)
    output = DecoderOutput(torch.randn(1, 4, len(spec.atomic_numbers)),
                           torch.randn(1, 4, len(spec.formal_charges)),
                           torch.randn(1, 4, 4, len(spec.bond_names)))
    reversed_output = DecoderOutput(output.atoms, output.charges, output.bonds.transpose(1, 2))
    first = reconstruction_losses(output, hard, LossConfig())
    second = reconstruction_losses(reversed_output, hard, LossConfig())
    assert torch.allclose(first["reconstruction"], second["reconstruction"])


def test_padding_charge_does_not_change_reconstruction():
    spec, target = small_graph()
    model = GraphAutoencoder(spec, ModelConfig(hidden=16, latent=8, layers=1))
    output = model(target.to_relaxed(spec)).decoded
    first = reconstruction_losses(output, target, LossConfig())["reconstruction"]
    target.charges[0, 2:] = 0
    second = reconstruction_losses(output, target, LossConfig())["reconstruction"]
    assert torch.allclose(first, second)


def test_ae_and_vae_share_weights_and_allow_backward():
    spec, target = small_graph()
    model = GraphAutoencoder(spec, ModelConfig(hidden=16, latent=8, layers=1))
    graph = target.to_relaxed(spec)
    ae = model(graph, sample_latent=False)
    vae = model(graph, sample_latent=True)
    assert torch.equal(ae.z, ae.mu)
    assert torch.equal(ae.mu, vae.mu)
    assert not torch.equal(vae.z, vae.mu)
    loss = reconstruction_losses(vae.decoded, target, LossConfig())["reconstruction"] + 0.01 * latent_kl(vae.mu, vae.logvar)
    loss.backward()
    assert model.encoder.logvar.weight.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_gaussian_kl_and_schema_roundtrip():
    assert latent_kl(torch.zeros(2, 3), torch.zeros(2, 3)) == 0
    assert torch.allclose(latent_kl(torch.ones(2, 3), torch.zeros(2, 3)), torch.tensor(1.5))
    spec = GraphSpec()
    assert GraphSpec.from_dict(spec.to_dict()) == spec
    assert GraphSpec(max_nodes=49).fingerprint() != spec.fingerprint()


def test_small_autoencoder_can_reduce_reconstruction_loss():
    torch.manual_seed(11)
    spec, target = small_graph()
    model = GraphAutoencoder(spec, ModelConfig(hidden=16, latent=8, layers=1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    graph = target.to_relaxed(spec)
    before = float(reconstruction_losses(model(graph).decoded, target, LossConfig())["reconstruction"].detach())
    for _ in range(40):
        optimizer.zero_grad()
        loss = reconstruction_losses(model(graph).decoded, target, LossConfig())["reconstruction"]
        loss.backward()
        optimizer.step()
    assert float(loss.detach()) < before * 0.2
