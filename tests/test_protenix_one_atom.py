"""CPU checks for the occupancy expectation and latent-slot bookkeeping."""

import pytest
import torch

from latentmol.protenix_occupancy.one_atom import (
    assess_case, expected_loss, labels, latent_probe, optional_slot,
)
from latentmol.representation import DecoderOutput, GraphSpec


@pytest.mark.parametrize(("full", "deleted", "element"), [
    ("O=C1CC(I)=CC=N1", "O=C1CC=CC=N1", 53),
    ("NC(=O)c1ccc(I)cc1", "NC(=O)c1ccccc1", 53),
    ("Brc1ccnc2ncccc12", "c1cnc2ncccc2c1", 35),
])
def test_native_smiles_have_a_single_aligned_optional_slot(full, deleted, element):
    spec = GraphSpec()
    graph = labels(full, spec, "cpu")
    slot = optional_slot(graph, spec, element, deleted)
    assert spec.atomic_numbers[graph.atoms[0, slot]] == element


def test_expected_loss_endpoints_and_low_occupancy_gradient():
    deleted, full = torch.tensor([0.8, 0.3], dtype=torch.float64)
    assert expected_loss(torch.zeros_like(full), deleted, full) == deleted
    assert expected_loss(torch.ones_like(full), deleted, full) == full
    u = torch.tensor(-2.1972245773362196, dtype=torch.float64, requires_grad=True)
    score = expected_loss(u.sigmoid(), deleted, full)
    assert torch.autograd.grad(score, u)[0].item() == pytest.approx(-0.045)


def test_latent_optimization_records_raw_and_controlled_choices():
    spec = GraphSpec(max_nodes=5)
    full, deleted = "CCI", "CC"
    graph = labels(full, spec, "cpu")
    slot = optional_slot(graph, spec, 53, deleted)
    atoms, charges, bonds = graph.to_relaxed(spec).__dict__.values()
    occupancy_direction = torch.zeros_like(atoms)
    occupancy_direction[0, slot, spec.atomic_numbers.index(53)] = 8

    class FixtureDecoder:
        def encode(self, graph):
            return torch.zeros(1, 1), None

        def decode(self, point):
            return DecoderOutput(atoms * 8 + point.reshape(1, 1, 1) * occupancy_direction,
                                 charges * 8, bonds * 8)

    model = FixtureDecoder()
    model.spec = spec
    metadata = dict(full_smiles=full, deleted_smiles=deleted)
    losses = {key: torch.tensor(value, dtype=torch.float64) for key, value in
              dict(full=0.3, deleted=0.8, fresh_full=0.4, fresh_deleted=0.9).items()}
    probe = latent_probe(model, {"metadata": metadata}, losses, 53)
    repeats = {key: {"loss": loss.clone()} for key, loss in losses.items()}
    equality = {key: {"passed": True} for key in ("full", "deleted")}
    result = assess_case(losses, repeats, equality, probe, metadata)
    assert result["useful_gradient"]
    assert not result["controlled_preferred_at_start"]
    assert result["controlled_preferred_at_end"]
    assert result["raw_preferred_at_start"] and result["raw_preferred_at_end"]
    assert probe["trace"][0]["q"] == pytest.approx(0.1)
    assert probe["trace"][-1]["q"] > 0.5
    assert probe["trace"][-1]["latent_radius"] <= 1.00001
    # The internal ranking alone must not count as success when the separately
    # prepared geometry favors the opposite topology.
    losses["fresh_deleted"] = losses["fresh_full"] - 0.1
    assert not assess_case(losses, repeats, equality, probe, metadata)["useful_gradient"]
