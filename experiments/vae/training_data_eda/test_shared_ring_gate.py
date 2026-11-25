import numpy as np
import torch

from experiments.vae.training_data_eda.shared_ring_gate import (
    SharedRingGate, apply_ring_gate, find_ring_sites, gate_training_loss, relaxed_ring_bonds,
)
from latentmol.chemistry import encode_smiles, inspect_graph
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch

SPEC = GraphSpec(max_nodes=16)


def encode(*smiles):
    rows = [encode_smiles(s, SPEC) for s in smiles]
    return HardGraphBatch(*(torch.from_numpy(np.stack([r[key] for r in rows])).long()
                            for key in ("atom_types", "charges", "bond_types")))


def test_masked_features_and_atom_order():
    g = encode("Cc1ccccc1", "CC1CCCCC1", "C1CCC2CCCCC2C1")
    gate = SharedRingGate(SPEC)
    sites = find_ring_sites(g, SPEC)
    assert [(s.row, s.aromatic) for s in sites] == [(0, True), (1, False)]
    scores = gate(g.to_relaxed(SPEC), sites)
    # The two rings differ only in perimeter labels. The target is unavailable.
    torch.testing.assert_close(scores[0], scores[1])
    order = torch.arange(g.atoms.size(1)).flip(0)
    other = HardGraphBatch(g.atoms[:,order], g.charges[:,order], g.bonds[:,order][:,:,order])
    torch.testing.assert_close(scores, gate(other.to_relaxed(SPEC), find_ring_sites(other, SPEC)))
    loss, counts = gate_training_loss(gate, g, SPEC)
    assert counts == dict(rows=3, sites=2, positive=1, negative=1)
    loss.backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in gate.parameters())


def test_joint_decision_and_preserved_invalids():
    g = encode("CC1CCCCC1", "CC1CCCCC1", "c1ccccc1")
    g.bonds[1].zero_()  # disconnected row remains in the evaluation population
    gate = SharedRingGate(SPEC)
    with torch.no_grad():
        for t in gate.parameters():
            t.zero_()
        gate.readout[-1].bias.fill_(50)
    out, metrics = apply_ring_gate(gate, g, SPEC)
    assert metrics["eligible_by_row"] == [1, 0, 1]
    assert metrics["changed_by_row"] == [1, 0, 0]
    assert inspect_graph(out.atoms[0],out.charges[0],out.bonds[0],SPEC).smiles == "Cc1ccccc1"
    assert (out.bonds != g.bonds).sum() == 12
    assert torch.equal(out.bonds,out.bonds.transpose(1,2))
    with torch.no_grad():
        gate.readout[-1].bias.zero_()
    for seed in range(5):
        draw, _ = apply_ring_gate(gate, g, SPEC, generator=torch.Generator().manual_seed(seed))
        assert (draw.bonds != g.bonds).sum().item() in (0,12)


def test_relaxed_gradient_reaches_latent_and_preserves_marginal_presence():
    hard = encode("CC1CCCCC1")
    gate = SharedRingGate(SPEC)
    graph = hard.to_relaxed(SPEC)
    z = torch.tensor(1.,requires_grad=True)
    decoded = DecoderOutput(graph.atoms*z*3,graph.charges*z*3,graph.bonds*z*3)
    base = decoded.probabilities().bonds
    mixed = relaxed_ring_bonds(gate,hard,decoded,SPEC)
    torch.testing.assert_close(mixed.sum(-1),torch.ones_like(mixed[...,0]))
    torch.testing.assert_close(mixed[...,0],base[...,0])
    torch.testing.assert_close(mixed,mixed.transpose(1,2))
    site, = find_ring_sites(hard,SPEC)
    aromatic = SPEC.bond_names.index("AROMATIC")
    objective = mixed[0,site.slots[0],site.slots[1],aromatic]
    z_grad, gate_grad = torch.autograd.grad(objective,(z,gate.readout[-1].bias))
    assert torch.isfinite(z_grad) and z_grad.abs() > 0 and gate_grad.abs() > 0
