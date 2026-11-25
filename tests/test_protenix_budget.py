"""CPU checks for the supported molecules and differentiable topology mixture."""

from itertools import product
import json
from pathlib import Path

import pytest
import torch

from latentmol.design.latent import count_band_penalty
from latentmol.protenix_occupancy.budget import topology_family, topology_probabilities
from latentmol.representation import GraphSpec


def test_budget_config_covers_a_shared_25_atom_start_and_28_atom_endpoint():
    path = Path(__file__).resolve().parents[1] / "experiments/latent-design/protenix-vae/atom-budget-dtu.json"
    config = json.loads(path.read_text())
    family = topology_family(config["start_smiles"], config["fluorine_sites"], GraphSpec())
    assert len(family) == 16
    assert family[(True, False, False, False)][0].GetNumAtoms() == 25
    assert {mol.GetNumAtoms() for mol, _, _ in family.values()} == set(range(24, 29))
    assert all(len(retained) == 24 for _, retained, _ in family.values())


def test_topology_mixture_count_and_budget_gradient():
    masks = torch.tensor(list(product((False, True), repeat=4)))
    q = torch.tensor([0.8, 0.9, 0.95, 0.9], dtype=torch.float64, requires_grad=True)
    probabilities = topology_probabilities(q, masks)
    counts = 24 + masks.sum(-1)
    expected_count = (probabilities * counts).sum()
    assert probabilities.sum().item() == pytest.approx(1)
    assert expected_count.item() == pytest.approx(24 + q.sum().item())
    count_loss = .01 * count_band_penalty(expected_count, 25, 2)
    gradient, = torch.autograd.grad(count_loss, q)
    assert gradient.tolist() == pytest.approx([.011] * 4)
    start = torch.tensor([.8, .2, .2, .2])
    assert count_band_penalty(24 + start.sum(), 25, 2).item() == 0


@pytest.mark.parametrize("message, expected", [
    ("Invariant Violation\nupper bound not greater than lower bound", "geometry"),
    ("unexpected native runtime failure", "runtime"),
])
def test_geometry_bounds_failure_has_a_scoped_unscorable_result(monkeypatch, tmp_path, message, expected):
    from rdkit import Chem
    from latentmol.protenix_occupancy import budget_native

    def fail(*args):
        raise RuntimeError(message)

    monkeypatch.setattr(budget_native.AllChem, "EmbedMolecule", fail)
    error = budget_native.ReferenceGeometryError if expected == "geometry" else RuntimeError
    with pytest.raises(error, match="ETKDG failed" if expected == "geometry" else message):
        budget_native.score_topology(None, Chem.MolFromSmiles("CC"), (0, 1), "ACD", (1,),
                                     geometry_seed=1201, mol_path=tmp_path / "candidate.mol")
