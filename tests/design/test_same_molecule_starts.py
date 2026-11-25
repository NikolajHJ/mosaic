"""A different slot permutation cannot give one latent a different native pose assignment."""

from types import SimpleNamespace

import pytest
import torch

from latentmol.chemistry import MoleculeRejected, encode_canonical_smiles
from latentmol.design.same_molecule_starts import _draw_triplet, _identity
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch


def test_native_order_verifies_labels_with_padding_and_permuted_slots():
    spec = GraphSpec(max_nodes=6)
    data = encode_canonical_smiles("C=C[NH3+]", spec)
    labels = HardGraphBatch(*(torch.as_tensor(data[k]).long()[None] for k in ("atom_types", "charges", "bond_types")))
    r = labels.to_relaxed(spec)
    perm = [3, 2, 4, 0, 1, 5]
    permuted = DecoderOutput(15*r.atoms[:,perm], 15*r.charges[:,perm], 15*r.bonds[:,perm][:,:,perm])
    model = SimpleNamespace(spec=spec, config=SimpleNamespace(latent=2), decode=lambda z: permuted)
    smiles, active, order = _identity(model, torch.zeros(1,2), 3)
    assert smiles == "C=C[NH3+]"
    assert order[:3] == [3, 4, 1]
    assert active.nonzero().flatten().tolist() == [1, 3, 4]
    actual = permuted.harden(spec)
    assert torch.equal(actual.atoms[:,order], labels.atoms)
    assert torch.equal(actual.charges[:,order], labels.charges)
    assert torch.equal(actual.bonds[:,order][:,:,order], labels.bonds)
    with pytest.raises(MoleculeRejected, match="atom_count"):
        _identity(model, torch.zeros(1,2), 2)


def test_mismatching_single_posterior_draw_rejects_whole_triplet(monkeypatch):
    from latentmol.design import same_molecule_starts as starts
    observations = iter(["CCO", "CCO", "COC"])
    calls = []
    def identity(model, z, count):
        calls.append(z.clone())
        return next(observations), torch.tensor([True,True,True]), [0,1,2]
    monkeypatch.setattr(starts, "_identity", identity)
    mu = torch.full((1,2), 0.5)
    sigma = torch.tensor([[1.0, 0.01]])
    monkeypatch.setattr(starts, "_encode", lambda m, s: (mu,sigma))
    model = SimpleNamespace(config=SimpleNamespace(latent=2))
    attempt = {}
    assert _draw_triplet(model, torch.zeros(1,2), 17, 3, set(), attempt) is None
    assert attempt == dict(stage="posterior_sample", smiles="CCO", reason="hard_identity_changed", decoded_smiles="COC")
    assert len(calls) == 3
    assert torch.equal(calls[-1], mu + sigma * starts._normal(model, 17))


def test_extending_exhausted_screen_keeps_seed_sequence_and_denominator():
    from latentmol.design.same_molecule_starts import generate_panel
    model = torch.nn.Linear(2, 2).eval()
    model.spec = GraphSpec(max_nodes=3)
    model.config = SimpleNamespace(latent=2)
    # This screen rejects all zero-atom proposals before attempting chemistry.
    def decode(z):
        return DecoderOutput(torch.zeros(len(z),3,10), torch.zeros(len(z),3,5), torch.zeros(len(z),3,3,5))
    model.decode = decode
    prefix = generate_panel(model, "sha", 1, 1, 7, 112)
    extra = generate_panel(model, "sha", 1, 1, 23, 112, prefix=prefix)
    assert extra == generate_panel(model, "sha", 1, 1, 23, 112)
    assert extra["proposals"] == 23 and len(prefix["attempts"]) == 7
