"""CPU checks for decoder-faithful one-slot proposals and latent gradients."""

import pytest
import torch

from latentmol.protenix_occupancy import raw_occupancy
from latentmol.protenix_occupancy.raw_occupancy import occupancy_objective, toggle_slot
from latentmol.representation import DecoderOutput, GraphSpec


def _decoder_with_thirty_atoms():
    spec = GraphSpec()
    atoms = torch.zeros(1, 50, len(spec.atomic_numbers))
    charges = torch.zeros(1, 50, len(spec.formal_charges))
    bonds = torch.zeros(1, 50, 50, len(spec.bond_names))
    carbon, nitrogen = (spec.atomic_numbers.index(element) for element in (6, 7))
    neutral, positive = (spec.formal_charges.index(charge) for charge in (0, 1))
    single, double = (spec.bond_names.index(bond) for bond in ("SINGLE", "DOUBLE"))
    occupied = [0, *range(2, 30), 47]
    atoms[..., 0] = 2
    atoms[0, occupied, carbon] = 4
    atoms[0, 1, nitrogen] = 1
    charges[..., neutral] = 2
    charges[0, [1, 47], positive] = 4
    bonds[..., 0] = 2
    for left, right, bond in ((0, 2, double), (2, 47, single),
                               (0, 1, single), (1, 49, double)):
        bonds[0, left, right, bond] = bonds[0, right, left, bond] = 4
    return DecoderOutput(atoms, charges, bonds), spec


@pytest.mark.parametrize(("slot", "present", "expected_count"), [(1, True, 31), (47, False, 29)])
def test_toggle_uses_raw_variable_count_and_preserves_other_labels(slot, present, expected_count):
    decoded, spec = _decoder_with_thirty_atoms()
    raw = decoded.harden(spec)
    saved_logits = tuple(x.clone() for x in (decoded.atoms, decoded.charges, decoded.bonds))
    alternative = toggle_slot(decoded, spec, slot, present)

    assert int(raw.atoms.count_nonzero()) == 30
    assert decoded.atoms[0, 1].softmax(-1)[1:].sum().item() > .5
    assert int(raw.atoms[0, 1]) == 0
    assert int(alternative.atoms.count_nonzero()) == expected_count
    core = torch.arange(spec.max_nodes) != slot
    torch.testing.assert_close(alternative.atoms[:, core], raw.atoms[:, core])
    torch.testing.assert_close(alternative.charges[:, core], raw.charges[:, core])
    torch.testing.assert_close(alternative.bonds[:, core][:, :, core], raw.bonds[:, core][:, :, core])
    torch.testing.assert_close(alternative.bonds, alternative.bonds.transpose(-1, -2))
    assert int(alternative.bonds[0, slot, slot]) == 0

    if present:
        assert spec.atomic_numbers[int(alternative.atoms[0, slot])] == 7
        assert spec.formal_charges[int(alternative.charges[0, slot])] == 1
        assert spec.bond_names[int(alternative.bonds[0, slot, 0])] == "SINGLE"
        assert int(alternative.bonds[0, slot, 49]) == 0
    else:
        assert int(alternative.atoms[0, slot]) == 0
        assert spec.formal_charges[int(alternative.charges[0, slot])] == 0
        assert not alternative.bonds[0, slot].any()
    for actual, expected in zip((decoded.atoms, decoded.charges, decoded.bonds), saved_logits):
        torch.testing.assert_close(actual, expected)


def test_occupancy_objective_has_probability_weighted_add_and_drop_gradients():
    q = torch.tensor([.1, .8, .5], dtype=torch.float64)
    z = q.logit().requires_grad_()
    split = torch.tensor([.7, .3], dtype=z.dtype).log()
    atoms = torch.cat((torch.zeros_like(z[:, None]), z[:, None] + split), dim=-1)[None, ...]
    charges = torch.zeros(1, 3, 1, dtype=z.dtype, requires_grad=True)
    bonds = torch.zeros(1, 3, 3, 1, dtype=z.dtype, requires_grad=True)
    decoded = DecoderOutput(atoms, charges, bonds)

    # Both changes improve their score relative to the raw 2.0 baseline.
    loss = occupancy_objective(decoded, 2.0, [(0, True, 1.5), (1, False, 1.0)])
    gradient, charge_gradient, bond_gradient = torch.autograd.grad(
        loss, (z, charges, bonds), allow_unused=True)

    assert loss.item() == pytest.approx(2.0 - .5 * .1 - 1.0 * .2)
    torch.testing.assert_close(gradient, torch.tensor([-.045, .16, 0.], dtype=z.dtype))
    assert charge_gradient is None and bond_gradient is None


@pytest.mark.parametrize(("budget", "final_count", "status"),
                         [(2, 2, "native_budget"), (3, 4, "accepted")])
def test_random_direction_shares_backtracking_and_respects_native_budget(
        monkeypatch, tmp_path, budget, final_count, status):
    class Decoder:
        spec = GraphSpec(max_nodes=4)

        def decode(self, point):
            distance = point.abs().item()
            count = 3 if distance > .2 else 4 if distance > .1 else 2
            atoms = torch.zeros(1, 4, len(self.spec.atomic_numbers))
            charges = torch.zeros(1, 4, len(self.spec.formal_charges))
            bonds = torch.zeros(1, 4, 4, len(self.spec.bond_names))
            atoms[0, :count, self.spec.atomic_numbers.index(6)] = 1
            charges[..., self.spec.formal_charges.index(0)] = 1
            for i in range(count - 1):
                bonds[0, i, i + 1, 1] = bonds[0, i + 1, i, 1] = 1
            return DecoderOutput(atoms, charges, bonds)

    def score(_native, mol, *_args, **_kwargs):
        count = mol.GetNumAtoms()
        return {2: 1., 3: 2., 4: .5}[count], {"heavy_atoms": count}

    monkeypatch.setattr(raw_occupancy, "score_topology", score)
    oracle = raw_occupancy.ProtenixOracle(
        None, {"protein_sequence": "A", "residues": []}, tmp_path, budget)
    config = {"geometry_seed": 294, "trial_steps": [.25, .125], "latent_radius": 1.0}
    generator = torch.Generator().manual_seed(1910)
    result = raw_occupancy.random_trajectory(Decoder(), torch.zeros(1, 1), oracle,
                                             config, generator, max_directions=2)
    assert result["search_calls"] == budget
    assert result["final"]["atom_count"] == final_count
    assert result["updates"][0]["status"] == status
    assert result["updates"][0]["trials"][0]["loss"] == 2.
