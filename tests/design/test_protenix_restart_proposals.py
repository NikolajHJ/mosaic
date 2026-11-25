import pytest
import torch
from torch.nn import functional as F

from latentmol.chemistry import inspect_graph
from latentmol.design.latent_protenix import hard_from_graph
from latentmol.design.protenix_restart_noise import propose_noise_start
from latentmol.design.protenix_restart_proposals import propose_start
from latentmol.design.protenix_restarts import Candidate
from latentmol.graph_policy import graph_from_decoder
from latentmol.representation import DecoderOutput, GraphSpec


class Decoder:
    spec = GraphSpec(max_nodes=5)

    def decode(self, latent):
        atoms = F.one_hot(torch.tensor([[1, 1, 1, 1, 0]]), len(self.spec.atomic_numbers)).float() * 3
        atoms[0, 0, 3] = 2.98 + latent[0, 0]
        charges = F.one_hot(torch.full((1, 5), 2), len(self.spec.formal_charges)).float() * 6
        bond_labels = torch.zeros((1, 5, 5), dtype=torch.long)
        for i in range(3):
            bond_labels[0, i, i + 1] = bond_labels[0, i + 1, i] = 1
        bonds = F.one_hot(bond_labels, len(self.spec.bond_names)).float() * 6
        return DecoderOutput(atoms, charges, bonds)


@pytest.fixture
def inputs():
    decoder = Decoder()
    latent = torch.zeros(1, 4)
    active = torch.tensor([True, True, True, True, False])
    graph = graph_from_decoder(decoder.decode(latent), decoder.spec, fixed_active=active)
    hard = hard_from_graph(graph)
    smiles = inspect_graph(hard.atoms[0], hard.charges[0], hard.bonds[0], decoder.spec).smiles
    return decoder, Candidate(smiles, latent, hard, 0.0, "fixture", 0), active


def test_sparse_leaves_none_rows_unchanged_and_fades(inputs):
    decoder, parent, active = inputs
    point, offset, receipt = propose_start(*inputs, policy="sparse", seed=2)
    base = decoder.decode(point)
    graph = graph_from_decoder(offset(base, 0), decoder.spec, fixed_active=active)
    original = graph_from_decoder(base, decoder.spec, fixed_active=active)
    torch.testing.assert_close(graph.bonds[parent.hard.bonds == 0], original.bonds[parent.hard.bonds == 0],
                               rtol=0, atol=0)
    assert torch.equal(graph.bonds, graph.bonds.transpose(1, 2))
    assert receipt["proposal"]["edit_count"] == 0
    assert len(receipt["perturbed_atom_slots"]) == 3
    assert len(receipt["perturbed_bonds"]) == 2
    assert offset(base, 25) is base


def test_latent_policy_is_seeded_and_prefers_a_valid_edit(inputs):
    point, offset, receipt = propose_start(*inputs, policy="latent", seed=3)
    other, _, repeat = propose_start(*inputs, policy="latent", seed=3)
    assert offset is None
    assert receipt["attempted"] == 12
    assert receipt["proposal"]["edit_count"] == 1
    assert receipt["proposal"]["valid"] and receipt["proposal"]["different"]
    assert receipt == repeat
    torch.testing.assert_close(point, other, rtol=0, atol=0)
    assert receipt["latent_displacement"] == pytest.approx(receipt["proposal"]["radius"])


def test_latent_policy_keeps_parent_when_all_kicks_are_invalid(inputs, monkeypatch):
    decoder, parent, active = inputs
    decode = decoder.decode

    def decode_invalid_kick(point):
        decoded = decode(point)
        if not torch.equal(point, parent.latent):
            # A fluorine with two bonds fails sanitization.
            decoded.atoms[0, 1, decoder.spec.atomic_numbers.index(9)] = 10
        return decoded

    monkeypatch.setattr(decoder, "decode", decode_invalid_kick)
    point, offset, receipt = propose_start(decoder, parent, active, policy="latent", seed=3)
    assert offset is None
    assert receipt["attempted"] == 12 and receipt["valid_trials"] == 0
    assert receipt["fallback_to_parent"] and receipt["proposal"]["valid"]
    assert torch.equal(point, parent.latent)


def test_all_row_preserves_parent_but_inflates_soft_nonbonds(inputs):
    _, offset, receipt = propose_start(*inputs, policy="all-row", seed=2)
    assert offset is not None
    assert receipt["proposal"]["edit_count"] == 0
    assert receipt["proposal"]["soft_bonds"] > 3
    _, no_offset, control = propose_start(*inputs, policy="none", seed=2)
    assert no_offset is None
    assert control["proposal"] == control["parent"]


def test_pre_fade_parent_cannot_silently_change_identity(inputs):
    decoder, parent, active = inputs
    parent.latent[0, 0] = 1
    with pytest.raises(ValueError, match="does not reproduce saved graph"):
        propose_start(decoder, parent, active, policy="latent", seed=2)


def test_raw_latent_doses_share_direction_without_screening(inputs, monkeypatch):
    decoder, parent, active = inputs
    decode = decoder.decode

    def decode_invalid_move(point):
        decoded = decode(point)
        if not torch.equal(point, parent.latent):
            decoded.atoms[0, 1, decoder.spec.atomic_numbers.index(9)] = 10
        return decoded

    monkeypatch.setattr(decoder, "decode", decode_invalid_move)
    low, offset, receipt = propose_noise_start(*inputs, policy="raw-latent", seed=12, strength=0.1)
    high, _, _ = propose_noise_start(*inputs, policy="raw-latent", seed=12, strength=0.5)
    torch.testing.assert_close(high - parent.latent, 5 * (low - parent.latent))
    assert receipt["latent_displacement"] == pytest.approx(0.1)
    assert receipt["proposal"]["valid"] is False
    assert receipt["proposal"]["parent_similarity"] is None
    assert offset is None
    prior, _, diagnostics = propose_noise_start(*inputs, policy="prior", seed=12)
    torch.testing.assert_close(prior, torch.randn(parent.latent.shape, generator=torch.Generator().manual_seed(12)))
    assert not diagnostics["proposal"]["valid"]


def test_graph_doses_preserve_argmax_and_protect_nonbonds(inputs):
    decoder, parent, active = inputs
    base = decoder.decode(parent.latent)
    original = graph_from_decoder(base, decoder.spec, fixed_active=active)
    for policy in ("sparse-mix", "dense-protected"):
        _, low, low_receipt = propose_noise_start(*inputs, policy=policy, seed=12, strength=0.25)
        _, high, high_receipt = propose_noise_start(*inputs, policy=policy, seed=12, strength=1)
        graph = graph_from_decoder(high(base, 0), decoder.spec, fixed_active=active)
        assert torch.equal(graph.bonds, graph.bonds.transpose(1, 2))
        assert torch.equal(graph.atoms[..., 0], original.atoms[..., 0])
        assert high_receipt["proposal"]["edit_count"] == 0
        assert high_receipt["proposal"]["parent_similarity"] == 1
        assert high_receipt["mixing"]["atoms"]["mean_fraction"] == pytest.approx(
            4 * low_receipt["mixing"]["atoms"]["mean_fraction"])
        assert high(base, 25) is base and low(base, 25) is base
        before = high_receipt["mixing"]["no_bond_soft_mass_before"]
        after = high_receipt["mixing"]["no_bond_soft_mass_after"]
        if policy == "sparse-mix":
            torch.testing.assert_close(graph.bonds[parent.hard.bonds == 0],
                                       original.bonds[parent.hard.bonds == 0], rtol=0, atol=0)
            assert high_receipt["mixing"]["no_bonds"]["rows"] == 0
        else:
            assert len(high_receipt["mixing"]["atom_slots"]) == int(active.sum())
            assert before < after <= 1.2 * before + 1e-6


@pytest.mark.parametrize("policy", ("sparse-mix", "dense-protected"))
def test_no_bond_weight_matches_support_and_reduces_added_bond_mass(inputs, policy):
    decoder, parent, active = inputs
    base = decoder.decode(parent.latent)
    kwargs = dict(policy=policy, seed=12, strength=1)
    _, one, one_receipt = propose_noise_start(*inputs, **kwargs, no_bond_weight=1.)
    _, eight, eight_receipt = propose_noise_start(*inputs, **kwargs, no_bond_weight=8.)
    _, legacy, legacy_receipt = propose_noise_start(*inputs, **kwargs)
    _, explicit_none, none_receipt = propose_noise_start(*inputs, **kwargs, no_bond_weight=None)

    assert one_receipt["no_bond_weight"] == 1. and eight_receipt["no_bond_weight"] == 8.
    assert "no_bond_weight" not in legacy_receipt and legacy_receipt == none_receipt
    assert torch.equal(legacy(base, 0).bonds, explicit_none(base, 0).bonds)
    for field in ("atom_slots", "atoms", "charges", "existing_bonds", "no_bonds"):
        assert one_receipt["mixing"][field] == eight_receipt["mixing"][field]
    assert torch.equal(one(base, 0).atoms, eight(base, 0).atoms)
    assert torch.equal(one(base, 0).charges, eight(base, 0).charges)

    graphs = [graph_from_decoder(offset(base, 0), decoder.spec, fixed_active=active)
              for offset in (one, eight)]
    for graph in graphs:
        hard = hard_from_graph(graph)
        assert torch.equal(hard.atoms, parent.hard.atoms)
        assert torch.equal(hard.charges, parent.hard.charges)
        assert torch.equal(hard.bonds, parent.hard.bonds)
    one_mass, eight_mass = (graph.bonds[..., 1:].sum(dim=-1) for graph in graphs)
    active_pairs = torch.triu(active[:, None] & active[None, :], diagonal=1)[None, ...]
    existing = active_pairs & (parent.hard.bonds != 0)
    absent = active_pairs & ~existing
    assert torch.all(eight_mass[active_pairs] <= one_mass[active_pairs])
    assert torch.any(eight_mass[existing] < one_mass[existing])
    if policy == "dense-protected":
        assert torch.all(eight_mass[absent] < one_mass[absent])
    else:
        assert torch.equal(eight_mass[absent], one_mass[absent])


@pytest.mark.parametrize("policy,strength,weight", (
    ("none", None, 1.), ("raw-latent", .25, 8.), ("dense-protected", 1., 0.),
    ("sparse-mix", 1., float("inf")), ("sparse-mix", 1., True),
))
def test_no_bond_weight_requires_a_positive_finite_graph_mix_weight(inputs, policy, strength, weight):
    with pytest.raises(ValueError, match="no-bond weight"):
        propose_noise_start(*inputs, policy=policy, seed=12, strength=strength, no_bond_weight=weight)
