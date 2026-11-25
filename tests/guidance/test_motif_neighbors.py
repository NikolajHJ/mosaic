import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from latentmol.chemistry import encode_smiles
from latentmol.guidance.motif_neighbors import prior_motif_neighbor_loss
from latentmol.representation import DecoderOutput, GraphSpec


SPEC = GraphSpec(max_nodes=14)


def center_logits(smiles):
    records = [encode_smiles(s, SPEC) for s in smiles]
    return DecoderOutput(*(F.one_hot(torch.tensor(np.stack([r[key] for r in records])).long(), len(v)).float()
                         .mul(12).requires_grad_() for key, v in (("atom_types", SPEC.atomic_numbers),
                         ("charges", SPEC.formal_charges), ("bond_types", SPEC.bond_names))))


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.atom = nn.Parameter(torch.zeros(14,10))
        self.charge = nn.Parameter(torch.zeros(14,5))
        self.bond = nn.Parameter(torch.zeros(14,14,5))
        self.child = nn.Dropout(.2)
        self.evaluations = []

    def decode(self, z):
        self.evaluations.append((z.detach().clone(), self.training, self.child.training))
        # Even if a decoder draws noise in eval mode, its auxiliary pass must preserve global RNG.
        torch.rand(3)
        return DecoderOutput(*(p.unsqueeze(0).expand(len(z),*p.shape) for p in (self.atom, self.charge, self.bond)))


def test_strict_motifs_share_one_neighbor_and_restore_rng_and_modes():
    centers = center_logits(["CC", "CC(=O)N", "c1ccc(C(=O)N)cc1", "c1ccccc1", "c1ccncc1"])
    model = Decoder().train()
    model.child.eval()
    z = torch.arange(20,dtype=torch.float32).reshape(5,4).requires_grad_()
    g = torch.Generator().manual_seed(17)
    global_rng, private_rng = torch.get_rng_state(), g.get_state()
    loss, stats = prior_motif_neighbor_loss(model,z,centers,SPEC,g)
    assert stats["neighbor_aromatic_sites"] == stats["neighbor_amide_sites"] == 2
    assert stats["neighbor_center_count"] == 3 and len(model.evaluations) == 1
    noise = torch.randn((3,4),generator=torch.Generator().manual_seed(17))
    torch.testing.assert_close(model.evaluations[0][0],.995*z[[1,2,3]] + (1-.995**2)**.5 * noise)
    assert model.evaluations[0][1:] == (False,False) and model.training and not model.child.training
    assert torch.equal(torch.get_rng_state(), global_rng) and not torch.equal(g.get_state(), private_rng)
    loss.backward()
    assert torch.isfinite(loss) and loss > 0 and all(p.grad.isfinite().all() and p.grad.any() for p in model.parameters())
    assert z.grad is None and centers.bonds.grad is None and centers.atoms.grad is None and centers.charges.grad is None
    grad = model.bond.grad
    assert not grad.diagonal(dim1=0,dim2=1).count_nonzero()
    torch.testing.assert_close(grad, grad.transpose(0,1))


def test_no_natural_motif_skips_and_whole_batch_denominator_is_preserved():
    model = Decoder()
    g = torch.Generator().manual_seed(55)
    no_site, stats = prior_motif_neighbor_loss(model,torch.zeros(2,4),center_logits(["C1CCCCC1","NC(O)C"]),SPEC,g)
    assert no_site.requires_grad and float(no_site.detach()) == 0 and stats["neighbor_center_count"] == 0
    assert model.evaluations == []
    # The center's chemically valid near miss is not accepted as a true motif.
    a, _ = prior_motif_neighbor_loss(model,torch.zeros(1,4),center_logits(["CC(=O)N"]),SPEC,g,rho=1.)
    b, _ = prior_motif_neighbor_loss(model,torch.zeros(2,4),center_logits(["CC(=O)N","CC"]),SPEC,g,rho=1.)
    torch.testing.assert_close(a, 2*b)
