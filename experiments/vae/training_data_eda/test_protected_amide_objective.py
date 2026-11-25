"""A verified single-bond amide edit and an incompatible ether supply opposing bond gradients."""

import torch
from torch.nn import functional as F

from experiments.vae.training_data_eda.protected_amide_objective import amide_sites, protected_amide_loss
from experiments.vae.training_data_eda.teacher_motif_training_pilot import encode
from latentmol.representation import DecoderOutput, GraphSpec


def test_positive_and_incompatible_matched_carbon_oxygen_directions():
    spec = GraphSpec()
    hard = encode(["CC(O)N(C)C", "COC"], spec)
    output = DecoderOutput(
        *(4 * F.one_hot(values, size).double() for values, size in
          [(hard.atoms, len(spec.atomic_numbers)), (hard.charges, len(spec.formal_charges)),
           (hard.bonds, len(spec.bond_names))])
    )
    output.bonds.requires_grad_()
    anchor = DecoderOutput(output.atoms, output.charges, output.bonds.detach().clone().requires_grad_())
    sites = amide_sites(output, spec)
    loss, info = protected_amide_loss(output, anchor, spec, sites=sites)
    assert info["contrast_amide_pairs"] == 1 and info["blocked_edges"] == 2
    r,i,j,n,k,l = sites[0][0]
    assert (r,n) == (0,1)
    gradient, frozen = torch.autograd.grad(loss, (output.bonds, anchor.bonds), allow_unused=True)
    double = spec.bond_names.index("DOUBLE")
    assert frozen is None and gradient[...,0].count_nonzero() == 0
    assert gradient[r,i,j,double] < 0 and gradient[n,k,l,double] > 0
    assert info["blocked_upward_drift"] == 0
    offset = torch.zeros_like(output.bonds)
    offset[n,k,l,double] = offset[n,l,k,double] = 1.
    increment = torch.tensor(.5, dtype=torch.float64, requires_grad=True)
    drifted = DecoderOutput(output.atoms, output.charges, output.bonds + increment*offset)
    changed, metrics = protected_amide_loss(drifted, anchor, spec, sites=sites)
    assert metrics["blocked_upward_drift"] > 0 and torch.autograd.grad(changed, increment)[0] > 0
