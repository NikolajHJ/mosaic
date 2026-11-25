from math import log
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from latentmol.evaluation.decoder_bond_ranking import _rank, decoder_bond_ranking
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch
from latentmol.training.losses import LossConfig


def test_ties_never_split_and_zero_positive_panels_have_no_threshold():
    scores = np.array([0.9, 0.8, 0.8, 0.8, 0.1])
    truth = np.array([True, False, True, False, True])
    result = _rank(scores, truth)
    assert result["selected_pairs"] == 4
    assert result["true_positives"] == 2
    assert result["precision"] == .5 and result["actual_recall"] == 2 / 3
    assert result["average_precision"] == pytest.approx((1 + .5 + .6) / 3)
    assert _rank(scores, truth & False)["average_precision"] is None
    assert _rank(scores[:0], truth[:0])["prevalence"] is None
    with pytest.raises(FloatingPointError):
        _rank(np.full_like(scores, np.nan), truth)


def test_target_alignment_and_conditional_ranking_exclude_occupancy():
    spec = GraphSpec(max_nodes=4)
    atoms = torch.full((1, 4, len(spec.atomic_numbers)), -50.)
    atoms[0, 0, 1] = atoms[0, 2, 3] = atoms[0, 3, 0] = 0.
    atoms[0, 1, 0], atoms[0, 1, 2] = log(.49), log(.51)
    charges = torch.zeros(1, 4, len(spec.formal_charges))
    charges[..., spec.formal_charges.index(0)] = 10
    bonds = torch.full((1, 4, 4, len(spec.bond_names)), -50.)
    bonds[..., 0] = 0
    for i, j, probability in ((0, 1, .52), (1, 2, .52), (0, 2, .49),
                              (0, 3, .95), (1, 3, .95), (2, 3, .95)):
        bonds[0, i, j, 0] = bonds[0, j, i, 0] = log(1 - probability)
        bonds[0, i, j, 1] = bonds[0, j, i, 1] = log(probability)
    prediction = DecoderOutput(atoms, charges, bonds)
    hard = prediction.harden(spec)
    order = [3, 2, 0, 1]
    shuffled = HardGraphBatch(hard.atoms[:, order], hard.charges[:, order], hard.bonds[:, order][:, :, order])

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.spec = spec
            self.anchor = nn.Parameter(torch.tensor(0.))

        def forward(self, target, sample_latent, from_canonical_cache):
            assert from_canonical_cache and not sample_latent
            assert not self.training and not torch.is_grad_enabled()
            return SimpleNamespace(decoded=prediction)

    model = Model()
    loss = LossConfig(atom_order="graph")
    original = decoder_bond_ranking(model, [hard], loss)
    result = decoder_bond_ranking(model, [shuffled], loss)
    assert result == original
    assert model.training and model.anchor.grad is None
    marginal = result["all_pairs_marginal"]
    conditional = result["target_occupied_pairs_conditional"]
    # Two true bonds to the .51 occupancy atom have marginal score .265.
    # The false bond between the fully occupied atoms has marginal score .49.
    assert marginal["pairs"] == 6 and marginal["target_bonds"] == 2
    assert marginal["precision"] == pytest.approx(2 / 3)
    # Among the three occupied target pairs, the two true bonds have greater
    # conditional probability than the false one. High-scoring padding is excluded.
    assert conditional["pairs"] == 3 and conditional["target_bonds"] == 2
    assert conditional["precision"] == conditional["average_precision"] == 1.
    assert conditional["threshold"] == pytest.approx(.52)
    assert conditional["actual_recall"] == 1.
