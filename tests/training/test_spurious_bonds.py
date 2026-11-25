import pytest
import torch
from torch.nn import functional as F

from latentmol.representation import DecoderOutput, HardGraphBatch
from latentmol.training.losses import LossConfig
from latentmol.training.matching import align_reconstruction_targets
from latentmol.training.spurious_bonds import spurious_bond_loss


def test_only_top_invented_edges_on_shared_occupied_slots_get_a_gradient():
    atoms = torch.ones(2, 7, dtype=torch.long)
    atoms[0, 5] = 0
    predicted_atoms = atoms.clone()
    predicted_atoms[0, 5:7] = torch.tensor([1, 0])
    target_bonds = torch.zeros(2, 7, 7, dtype=torch.long)
    target_bonds[0, 0, 1] = target_bonds[0, 1, 0] = 1
    target_bonds[0, 1, 2] = target_bonds[0, 2, 1] = 1
    logits = torch.zeros(2, 7, 7, 3)
    for i, j, gap in [(0, 2, 1), (0, 3, 2), (0, 4, 3), (1, 3, 4), (1, 4, 5),
                      (0, 1, 9), (0, 5, 9), (0, 6, 9), (0, 0, 9)]:
        logits[0, i, j, 2] = logits[0, j, i, 2] = gap
    logits.requires_grad_()
    out = DecoderOutput(F.one_hot(predicted_atoms, 2).float(), torch.zeros(2, 7, 1), logits)
    target = HardGraphBatch(atoms, torch.zeros_like(atoms), target_bonds)
    loss, counts = spurious_bond_loss(out, target)
    assert counts == dict(parents=2, affected_parents=1, false_positive_pairs=5, selected_pairs=4)
    torch.testing.assert_close(loss.detach(), F.softplus(torch.tensor([2., 3., 4., 5.])).mean()/2)
    loss.backward()
    for i, j in [(0, 3), (0, 4), (1, 3), (1, 4)]:
        assert logits.grad[0, i, j, 0] < 0
        assert logits.grad[0, i, j, 2] > 0
        assert logits.grad[0, i, j, 1] == 0
        torch.testing.assert_close(logits.grad[0, i, j], logits.grad[0, j, i])
    assert logits.grad.count_nonzero() == 4 * 2 * 2


def test_exact_graph_in_different_slots_has_connected_zero_after_matching():
    atoms = torch.tensor([[1, 2, 3]])
    charges = torch.zeros_like(atoms)
    bonds = torch.tensor([[[0, 1, 0], [1, 0, 2], [0, 2, 0]]])
    order = [2, 0, 1]
    target = HardGraphBatch(atoms, charges, bonds)
    logits = (F.one_hot(bonds[:, order][:, :, order], 3).float()*8).requires_grad_()
    decoded = DecoderOutput(F.one_hot(atoms[:, order], 4).float()*8, torch.zeros(1, 3, 1), logits)
    _, raw_counts = spurious_bond_loss(decoded, target)
    assert raw_counts["false_positive_pairs"] > 0
    aligned = align_reconstruction_targets(decoded, target, LossConfig(atom_order="graph"))
    loss, counts = spurious_bond_loss(decoded, aligned)
    assert counts["false_positive_pairs"] == 0 and loss.item() == 0
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


def test_one_atom_has_no_pairs():
    a = torch.tensor([[1]])
    b = torch.zeros(1, 1, 1, 2, requires_grad=True)
    out = DecoderOutput(F.one_hot(a, 2).float(), a[..., None].float(), b)
    target = HardGraphBatch(a, a, torch.zeros(1, 1, 1, dtype=torch.long))
    loss, counts = spurious_bond_loss(out, target)
    assert counts["selected_pairs"] == 0 and loss.item() == 0
    loss.backward()
    torch.testing.assert_close(b.grad, torch.zeros_like(b))
    with pytest.raises(ValueError, match="positive integer"):
        spurious_bond_loss(out, target, max_pairs=0)
