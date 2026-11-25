import numpy as np
import torch
from torch.nn import functional as F

from latentmol.chemistry import encode_smiles
from latentmol.guidance.motif_contrast import contrast_completion_loss
from latentmol.representation import DecoderOutput, GraphSpec


def encode(smiles):
    spec = GraphSpec(max_nodes=15)
    graphs = [encode_smiles(s, spec) for s in smiles]
    logits = [torch.tensor(np.stack([g[key] for g in graphs])).long()
              for key in ("atom_types", "charges", "bond_types")]
    output = DecoderOutput(*(F.one_hot(t, len(v)).float().mul(8).requires_grad_() for t, v in
                             zip(logits, (spec.atomic_numbers, spec.formal_charges, spec.bond_names))))
    return output, spec


def test_matched_sites_and_global_bias_invariance():
    decoded, spec = encode(["CCC1CCCCC1", "NC(O)COC"])
    loss, sites = contrast_completion_loss(decoded, spec, need_ring=True, need_amide=True)
    assert sites["contrast_aromatic_sites"] == sites["contrast_amide_sites"] == 1
    assert sites["contrast_aromatic_local_pairs"] == 6
    assert sites["contrast_amide_local_pairs"] == 1
    bias = torch.tensor([0., 0.2, -0.5, 0.1, 0.4], requires_grad=True)
    shifted = DecoderOutput(decoded.atoms, decoded.charges, decoded.bonds + bias)
    shifted_loss, _ = contrast_completion_loss(shifted, spec, need_ring=True, need_amide=True)
    torch.testing.assert_close(loss, shifted_loss)
    grad, grad_bias = torch.autograd.grad(shifted_loss, (decoded.bonds, bias))
    torch.testing.assert_close(grad_bias, torch.zeros_like(grad_bias), atol=1e-7, rtol=0.)
    assert grad[..., 0].count_nonzero() == grad[..., spec.bond_names.index("TRIPLE")].count_nonzero() == 0
    assert (grad[..., spec.bond_names.index("DOUBLE")] > 0).any()
    assert (grad[..., spec.bond_names.index("DOUBLE")] < 0).any()
    torch.testing.assert_close(grad, grad.transpose(1, 2))


def test_missing_element_matched_negative_skips_whole_site():
    decoded, spec = encode(["C1CCCCC1", "NC(O)C"])
    # Isolate each to exclude cross-molecule negatives.
    for k, group in enumerate(("aromatic", "amide")):
        one = DecoderOutput(decoded.atoms[k:k+1], decoded.charges[k:k+1], decoded.bonds[k:k+1])
        loss, sites = contrast_completion_loss(one, spec, need_ring=True, need_amide=True)
        assert sites[f"contrast_{group}_eligible_sites"] == 1
        assert sites[f"contrast_{group}_sites"] == 0
        assert loss.requires_grad and loss == 0.
