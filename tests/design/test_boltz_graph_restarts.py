from types import SimpleNamespace

import pytest
import torch

from latentmol.chemistry import inspect_graph
from latentmol.design import boltz_graph_restarts as restarts
from latentmol.design.boltz_speed import _point_digest
from latentmol.design.latent_protenix import hard_from_graph
from latentmol.design.protenix_restarts import Candidate
from latentmol.design.protenix_search_frontier import SearchPoint
from latentmol.graph_policy import graph_from_decoder
from latentmol.representation import DecoderOutput, GraphSpec


def test_graph_coordinates_replay_asymmetric_input_and_expose_all_three_gradients():
    spec = GraphSpec(max_nodes=6)
    active = torch.tensor([False, True, False, True, True, False])
    gen = torch.Generator().manual_seed(10)
    decoded = DecoderOutput(*(torch.randn(shape, generator=gen) for shape in
                              [(1, 6, len(spec.atomic_numbers)), (1, 6, len(spec.formal_charges)),
                               (1, 6, 6, len(spec.bond_names))]))
    adapter = restarts.DirectGraphCoordinates(spec, active)
    packed = adapter.pack(decoded).requires_grad_()
    expected = graph_from_decoder(decoded, spec, fixed_active=active)
    actual = graph_from_decoder(adapter.decode(packed), spec, fixed_active=active)
    for k in ("atoms", "charges", "bonds"):
        assert torch.equal(getattr(expected, k), getattr(actual, k))
    total = actual.atoms[..., 1].sum() + actual.charges[..., 1].sum() + actual.bonds[..., 1].sum()
    grad, = torch.autograd.grad(total, packed)
    assert all(rows.norm() > 0 for rows in adapter.rows(grad))
    with pytest.raises(ValueError, match="packed graph"):
        adapter.decode(packed[:, :-1])
    with pytest.raises(ValueError, match="packed graph"):
        adapter.decode(torch.full_like(packed, float("nan")))


def test_graph_collision_refill_preserves_pose_and_labels_checkpoint(monkeypatch):
    spec = GraphSpec(max_nodes=4)
    active = torch.tensor([True, False, True, True])
    adapter = restarts.DirectGraphCoordinates(spec, active)
    packed = torch.zeros(1, sum(adapter.sizes))
    atoms, charges, bonds = adapter.rows(packed)
    atoms[:, spec.atomic_numbers.index(6) - 1] = 1
    charges[:, spec.formal_charges.index(0)] = 1
    bonds[[0, 2], spec.bond_names.index("SINGLE")] = 1
    graph = graph_from_decoder(adapter.decode(packed), spec, fixed_active=active)
    hard = hard_from_graph(graph)
    smiles = inspect_graph(hard.atoms[0], hard.charges[0], hard.bonds[0], spec).smiles
    assert smiles == "CCC"
    parent = SearchPoint(Candidate(smiles, packed, hard, -.8, "founder", 4), active, -1., ("founder",))
    state = dict(archive={smiles: parent}, ineligible=set())
    seen = {_point_digest(packed, active)}
    # Exercise the bound explicitly when all sampled proposals return an identical parent.
    monkeypatch.setattr(restarts, "propose_start", lambda *a, **k: (
        packed, None, dict(parent=dict(smiles=smiles, edit_count=0))))
    cfg = SimpleNamespace(occupied_atoms=3, noise_initial=0, temperature_initial=1.)
    points, receipts = restarts.direct_archive_starts(spec, cfg, state, 2, 84, parent_count=1, seen=seen)
    assert len(seen) == 3 and all(p.candidate.smiles == smiles for p in points)
    assert all(r["rejected_exact_starts"] == 9 for r in receipts)
    for p in points:
        checkpoint = restarts.pack_graph_point(p)
        assert "latent" not in checkpoint
        assert torch.equal(checkpoint["graph_logits"], p.candidate.latent)
        assert torch.equal(checkpoint["bonds"], parent.candidate.hard.bonds)
