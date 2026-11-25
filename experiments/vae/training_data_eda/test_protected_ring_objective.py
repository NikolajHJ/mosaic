"""Small fixed-graph tests for conditional target probability and anchor gradient isolation."""

import math

import pytest
import torch
from torch.nn import functional as F

from experiments.vae.training_data_eda.aromatic_encodings_probe import patterns, select_sites
from experiments.vae.training_data_eda.protected_ring_objective import protected_ring_loss
from experiments.vae.training_data_eda.teacher_motif_training_pilot import encode
from latentmol.representation import DecoderOutput, GraphSpec


def batch(smiles="CCC1CCCCC1"):
    spec = GraphSpec()
    hard = encode([smiles], spec)
    out = DecoderOutput(*(4 * F.one_hot(value, classes).double() for value, classes in (
        (hard.atoms, len(spec.atomic_numbers)), (hard.charges, len(spec.formal_charges)),
        (hard.bonds, len(spec.bond_names)))))
    source = DecoderOutput(out.atoms, out.charges, out.bonds.detach().clone().requires_grad_())
    out.bonds.requires_grad_()
    return spec, hard, out, source


def test_equivalent_encodings_probability_and_zero_initial_retention_gradient():
    spec, hard, output, anchor = batch()
    a, info = protected_ring_loss(output, anchor, spec)
    b, _ = protected_ring_loss(output, anchor, spec, retention_weight=0)
    assert info["sites"] == 1
    assert info["offsite_type_kl"] == info["presence_kl"] == 0
    assert info["protected_near_edges"] == 1 and info["protected_external_edges"] == 1
    assert info["presence_edges"] == 8
    grad_a, source_grad = torch.autograd.grad(a, (output.bonds, anchor.bonds), allow_unused=True)
    grad_b, = torch.autograd.grad(b, output.bonds)
    assert source_grad is None
    torch.testing.assert_close(grad_a, grad_b, atol=1e-15, rtol=1e-14)
    assert grad_b[..., 0].count_nonzero() == 0 and grad_b.abs().sum() > 0
    rings,_,_ = select_sites(hard,spec)
    _,i,j = torch.tensor(rings[0]).T
    p = output.bonds[0,i,j,1:].softmax(-1)
    event = sum(p[torch.arange(6),assignment-1].prod() for assignment in patterns(spec))
    assert info["ring_nll_per_ring"] == pytest.approx(-math.log(float(event.detach())))
    assert info["ring_nll_per_edge"] == pytest.approx(info["ring_nll_per_ring"]/6)


@pytest.mark.parametrize("bond_type,with_presence", [("AROMATIC", False), ("NONE", True)])
def test_restoring_gradient_after_unrelated_edge_perturbation(bond_type, with_presence):
    spec,hard,out,source = batch()
    ring,_,_ = select_sites(hard,spec)
    ring_slots = {i for _,i,_ in ring[0]}
    # The ethyl chain has one real bond joining two atoms outside the ring.
    i,j = next((i,j) for i,j in (hard.bonds[0].triu(1) != 0).nonzero().tolist()
               if i not in ring_slots and j not in ring_slots)
    idx = spec.bond_names.index(bond_type)
    offset = torch.zeros_like(out.bonds)
    offset[0,i,j,idx] = offset[0,j,i,idx] = 1
    drift = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    changed = DecoderOutput(out.atoms, out.charges, out.bonds + drift*offset)
    retained, info = protected_ring_loss(changed, source, spec, preserve_presence=with_presence)
    raw, _ = protected_ring_loss(changed, source, spec, retention_weight=0, preserve_presence=with_presence)
    plain_grad, = torch.autograd.grad(raw, drift, retain_graph=True)
    restoring_grad, source_grad = torch.autograd.grad(retained, (drift,source.bonds),allow_unused=True)
    assert abs(float(plain_grad)) < 1e-12
    assert restoring_grad > 0 and source_grad is None
    assert info["offsite_type_kl" if bond_type=="AROMATIC" else "presence_kl"] > 0


def test_zero_site_frame_contributes_no_retention():
    spec,_,out,source = batch("CCCC")
    out = DecoderOutput(out.atoms, out.charges, out.bonds + torch.randn_like(out.bonds)*.1)
    value, info = protected_ring_loss(out, source, spec)
    assert info["eligible_rows"] == info["protected_external_edges"] == info["presence_edges"] == 0
    assert info["ring_nll_per_edge"] == 0 and abs(float(value.detach())) < 1e-15


def test_other_molecules_in_a_ring_batch_never_receive_guard_gradients():
    spec,_,ring,source_ring = batch()
    _,_,acyclic,source_acyclic = batch("CCCC")
    def join(a,b):
        return DecoderOutput(*(torch.cat([getattr(a,k),getattr(b,k)]) for k in ("atoms","charges","bonds")))
    out = join(ring,acyclic)
    anchor = join(source_ring,source_acyclic)
    perturb = torch.zeros_like(out.bonds)
    perturb[1,...,spec.bond_names.index("DOUBLE")] = .5
    shifted = DecoderOutput(out.atoms,out.charges,out.bonds+perturb)
    loss,info = protected_ring_loss(shifted,anchor,spec)
    grad, = torch.autograd.grad(loss, out.bonds)
    assert info["eligible_rows"] == 1 and info["presence_edges"] == 8
    assert grad[1].count_nonzero() == 0 and info["offsite_type_kl"] == info["presence_kl"] == 0
