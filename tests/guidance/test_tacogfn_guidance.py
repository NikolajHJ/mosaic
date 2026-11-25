"""Per-graph docking gradients reanchor to valid hard graphs and skip invalid ones."""

import torch
from torch import nn
from torch.nn import functional as F

from latentmol.graph_policy import graph_from_decoder
from latentmol.guidance import tacogfn_head
from latentmol.guidance.tacogfn import TacoGFNGuidance
from latentmol.representation import DecoderOutput, GraphSpec


class _Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("bias", torch.zeros(()))

    def forward(self, atoms, bonds, edges):
        return atoms[:, 0].sum() - bonds[:, 1].sum() + self.bias


class _Base:
    def evaluate(self, graph, active, point, binding_gradient, step):
        return torch.full_like(point, 0.125), {"base_called": True}


def test_reanchor_and_skip_share_base_guidance(monkeypatch):
    spec = GraphSpec(max_nodes=3)
    atoms = torch.tensor([1, 1, 3])
    charges = torch.full((3,), spec.formal_charges.index(0))
    chain = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    labels = {"atoms": atoms.tolist(), "bonds": chain.tolist(), "charges": charges.tolist()}
    founder = {"smiles": "CCO", "active_slots": [0, 1, 2], "labels": labels}
    topology = {"x": [0, 0, 2], "edge_index": [[0, 1], [1, 2]], "edge_attr": [0, 0], "hard_score": 2.0}
    head = _Head().eval()
    guide = TacoGFNGuidance(head, founder, spec, topology, base_guidance=_Base())
    calls = []

    def ring_topology(smiles):
        calls.append(smiles)
        return {"x": [0, 0, 2], "edge_index": [[0, 1, 2], [1, 2, 0]], "edge_attr": [0, 0, 0]}

    monkeypatch.setattr(tacogfn_head, "topology_from_smiles", ring_topology)
    active = torch.ones(3, dtype=torch.bool)
    point = torch.zeros(1, requires_grad=True)
    reference = torch.full_like(point, 0.2)

    def evaluate(bonds, step):
        atom_logits = 4 * F.one_hot(atoms, 10).float() + point * torch.arange(10)
        decoded = DecoderOutput(atom_logits[None], F.one_hot(charges, len(spec.formal_charges)).float()[None],
                                (4 * F.one_hot(bonds, 5).float())[None])
        graph = graph_from_decoder(decoded, spec, temperature=1., fixed_active=active)
        grad, metadata = guide.evaluate(graph, active, point, reference, step)
        assert metadata["base_called"]
        return grad, metadata["tacogfn"]

    grad, initial = evaluate(chain, 0)
    assert initial["used_founder_graph"] and initial["edges"] == 2 and initial["active"]
    assert torch.isclose((grad - 0.125).norm(), 0.05 * reference.norm())
    ring = chain.clone()
    ring[0, 2] = ring[2, 0] = 1
    _, changed = evaluate(ring, 1)
    _, repeat = evaluate(ring, 2)
    assert calls == ["C1CO1"]
    assert changed["edges"] == 3 and changed["active"] and not changed["used_founder_graph"]
    assert repeat["topology_cache_hit"] and repeat["smiles"] == "C1CO1"
    grad, invalid = evaluate(torch.zeros_like(chain), 3)
    assert not invalid["active"] and invalid["skip_reason"] == "invalid:disconnected"
    assert invalid["applied_gradient_norm"] == 0. and torch.equal(grad, torch.full_like(point, 0.125))


def test_matched_ratio_scales_in_both_directions_and_skips_undefined_norms():
    spec = GraphSpec(max_nodes=2)
    active = torch.ones(2, dtype=torch.bool)
    atoms = torch.tensor([1, 1])
    charges = torch.full((2,), spec.formal_charges.index(0))
    bonds = torch.tensor([[0, 1], [1, 0]])
    founder = dict(smiles="CC", active_slots=[0, 1], labels=dict(atoms=atoms, charges=charges, bonds=bonds))
    topology = dict(x=[0, 0], edge_index=[[0], [1]], edge_attr=[0], hard_score=2.0)
    guide = TacoGFNGuidance(_Head().eval(), founder, spec, topology, cap_ratio=.5,
                            norm_mode="match", base_guidance=_Base())

    for reference_size, multiplier in [(1e-4, 1.), (10., 1.), (0., 1.), (1., 0.)]:
        point = torch.zeros(2, requires_grad=True)
        logits = 4 * F.one_hot(atoms, 10).float() + multiplier * point[0] * torch.arange(10)
        decoded = DecoderOutput(logits[None], F.one_hot(charges, len(spec.formal_charges)).float()[None],
                                (4 * F.one_hot(bonds, 5).float())[None])
        graph = graph_from_decoder(decoded, spec, temperature=1., fixed_active=active)
        binding_gradient, = torch.autograd.grad(reference_size * point[1], point)
        extra, report = guide.evaluate(graph, active, point, binding_gradient, step=0)
        meta = report["tacogfn"]
        if reference_size == 0 or multiplier == 0:
            assert not meta["active"] and meta["skip_reason"].startswith("zero_")
            assert meta["applied_gradient_norm"] == 0 and torch.equal(extra, torch.full_like(point, .125))
        else:
            assert meta["norm_mode"] == "match" and meta["target_ratio"] == .5
            assert abs(meta["actual_ratio"] - .5) < 1e-7
            assert (meta["scale"] > 1) == (reference_size == 10.)
            assert torch.isclose((extra - .125).norm(), .5 * binding_gradient.norm())
            assert extra[0] < .125 and extra[1] == .125
