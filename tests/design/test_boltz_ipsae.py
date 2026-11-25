"""Small mathematical checks; no native-model or calibration claim."""
from types import SimpleNamespace

import pytest
import torch

from latentmol.boltz.dbmol_model import ligand_ipsae_from_logits
from latentmol.boltz.dbmol_objective import (
    CONFIDENCE_WEIGHTS, CONFIDENCE_SIGNS, reduce_objective,
)


def test_ligand_ipsae_direction_and_gradient():
    # The second protein frame has lower predicted PAE. The reverse direction
    # is deliberately different and must not influence the ligand score.
    logits = torch.zeros(1, 4, 4, 2, dtype=torch.float64)
    logits[0, 0, 2:] = torch.tensor([0., 2.])
    logits[0, 1, 2:] = torch.tensor([2., 0.])
    logits[0, 2:, :2] = torch.tensor([5., 0.])
    logits.requires_grad_()
    protein = torch.tensor([True, True, False, False])
    ligand = ~protein
    score = ligand_ipsae_from_logits(logits, protein, ligand)
    probabilities = torch.softmax(torch.tensor([2., 0.], dtype=torch.float64), -1)
    expected_pae = probabilities @ torch.tensor([8., 24.], dtype=torch.float64)
    torch.testing.assert_close(score, 1 / (1 + (expected_pae / 2).square()))
    score.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 1, 2:].abs().sum() > 0
    assert logits.grad[0, 0].abs().sum() == 0
    assert logits.grad[0, 2:].abs().sum() == 0


def test_ligand_ipsae_rejects_missing_and_overlapping_masks():
    logits = torch.zeros(1, 2, 2, 4)
    with pytest.raises(ValueError, match="disjoint nonempty"):
        ligand_ipsae_from_logits(logits, torch.tensor([False, False]), torch.tensor([False, True]))
    with pytest.raises(ValueError, match="disjoint nonempty"):
        ligand_ipsae_from_logits(logits, torch.tensor([True, False]), torch.tensor([True, False]))


def test_confidence_objective_opt_in_and_gradient():
    base = dict(affinity_regression_weight=1., affinity_logit_weight=1.,
                contact_weight=2.5, confidence_weight=2.5,
                confidence_weights=dict(CONFIDENCE_WEIGHTS))
    values = {k: torch.tensor(.5, requires_grad=True) for k in CONFIDENCE_WEIGHTS}
    values.update(affinity_pred_value=torch.tensor(1.), affinity_logits_binary=torch.tensor(2.),
                  dbmol_contact_loss=torch.tensor(3.), ligand_ipsae=torch.tensor(.7, requires_grad=True))
    baseline = reduce_objective(values, SimpleNamespace(**base, ipsae_weight=0.))
    modified = reduce_objective(values, SimpleNamespace(**base, ipsae_weight=.1))
    torch.testing.assert_close(modified['dbmol_confidence_loss'],
                               baseline['dbmol_confidence_loss'] - .1 * values['ligand_ipsae'])
    modified['dbmol_total_loss'].backward()
    assert values['ligand_ipsae'].grad == pytest.approx(-.25)
    for key in CONFIDENCE_WEIGHTS:
        assert values[key].grad == pytest.approx(2.5 * CONFIDENCE_SIGNS[key] * CONFIDENCE_WEIGHTS[key])
