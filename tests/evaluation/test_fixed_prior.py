"""CPU fixture for the matched in-training validity probe."""

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from latentmol.evaluation.fixed_prior import FixedPriorProbe, compare_fixed_prior
from latentmol.representation import DecoderOutput, GraphSpec


class TwoCarbonDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.spec = GraphSpec(max_nodes=2)
        self.config = SimpleNamespace(latent=2)
        self.bond_shift = torch.nn.Parameter(torch.tensor(0.))
        self.dropout = torch.nn.Dropout()

    def decode(self, codes):
        assert not self.training and not self.dropout.training
        # A diagnostic can consume RNG internally without perturbing training.
        random.random()
        np.random.random()
        torch.rand(1)
        batch, nodes = len(codes), self.spec.max_nodes
        atoms = codes.new_full((batch, nodes, len(self.spec.atomic_numbers)), -30.)
        atoms[..., self.spec.atomic_numbers.index(6)] = 10.
        charges = codes.new_full((batch, nodes, len(self.spec.formal_charges)), -30.)
        charges[..., self.spec.formal_charges.index(0)] = 10.
        bonds = codes.new_full((batch, nodes, nodes, len(self.spec.bond_names)), -30.)
        bonds[..., 0] = 0.
        bonds[..., self.spec.bond_names.index("SINGLE")] = (codes[:, 0] + self.bond_shift)[:, None, None]
        return DecoderOutput(atoms, charges, bonds)


def test_fixed_prior_pairs_validity_and_preserves_training_state():
    model = TwoCarbonDecoder().train()
    model.dropout.eval()
    model.bond_shift.grad = torch.tensor(0.25)
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    panel = FixedPriorProbe(2, count=19, latent_seed=314, categorical_seed=211, batch_size=8)
    before = panel.observe(model)
    assert before == panel.observe(model)
    assert model.training and not model.dropout.training
    assert random.getstate() == python_state
    np.testing.assert_equal(np.random.get_state(), numpy_state)
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert model.bond_shift.grad.item() == 0.25
    assert model.bond_shift.item() == 0.
    for method in ("argmax", "categorical"):
        assert before[method]["attempts"] == 19
        assert before[method]["failure_reasons"] == {"chemical:disconnected": 19 - before[method]["valid_count"]}

    with torch.no_grad():
        model.bond_shift.fill_(12.)
    after = panel.observe(model)
    paired = compare_fixed_prior(before, after)
    for method in paired:
        assert after[method]["valid_count"] == 19
        assert paired[method]["gained"] == 19 - before[method]["valid_count"]
        assert paired[method]["lost"] == 0
    mismatched = dict(after, panel=dict(after["panel"], categorical_seed=210))
    with pytest.raises(ValueError, match="identical panel"):
        compare_fixed_prior(before, mismatched)
