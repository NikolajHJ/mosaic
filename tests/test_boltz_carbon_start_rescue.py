"""Deterministic local-start policy; these tests run only on DTU HPC."""

from types import SimpleNamespace

import pytest
import torch

from latentmol.boltz import carbon_start_rescue as rescue
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch, RelaxedGraph


def _fixture():
    spec = GraphSpec(max_nodes=4, atomic_numbers=(0, 6, 7),
                     formal_charges=(-1, 0, 1),
                     bond_names=("NONE", "SINGLE"))
    atoms = torch.tensor([[1, 1, 0, 0]])
    charges = torch.ones((1, 4), dtype=torch.long)
    bonds = torch.zeros((1, 4, 4), dtype=torch.long)
    bonds[0, 0, 1] = bonds[0, 1, 0] = 1
    seed = HardGraphBatch(atoms, charges, bonds)
    context = SimpleNamespace(spec=spec, seed=seed,
                              active=torch.tensor([True, True, False, False]))
    atom_logits = torch.tensor([[[0., 10., 0.], [0., 10., 0.],
                                 [10., 0., 0.], [10., 0., 0.]]])
    charge_logits = torch.tensor([[[0., 10., 0.]]] * 4).transpose(0, 1)
    bond_logits = torch.zeros((1, 4, 4, 2))
    bond_logits[..., 0] = 10.
    bond_logits[0, 0, 1] = bond_logits[0, 1, 0] = torch.tensor([0., 10.])
    return context, DecoderOutput(atom_logits, charge_logits, bond_logits)


def test_requires_ordinary_and_fixed_exact_native_slot_labels():
    context, exact = _fixture()
    assert rescue.exact_native_graphs(exact, context)["exact_native_c13"]

    # Fixed occupancy restores an atom, but ordinary decoding still says absent.
    missing = DecoderOutput(exact.atoms.clone(), exact.charges, exact.bonds)
    missing.atoms[0, 0, 0] = 11.
    audit = rescue.exact_native_graphs(missing, context)
    assert not audit["ordinary_fields_equal"]["atoms"]
    assert audit["fixed_fields_equal"]["atoms"]
    assert not audit["exact_native_c13"]

    charged = DecoderOutput(exact.atoms, exact.charges.clone(), exact.bonds)
    charged.charges[0, 0, 2] = 11.
    audit = rescue.exact_native_graphs(charged, context)
    assert not audit["ordinary_fields_equal"]["charges"]
    assert not audit["fixed_fields_equal"]["charges"]

    unbonded = DecoderOutput(exact.atoms, exact.charges, exact.bonds.clone())
    unbonded.bonds[0, 0, 1, 0] = unbonded.bonds[0, 1, 0, 0] = 11.
    audit = rescue.exact_native_graphs(unbonded, context)
    assert not audit["ordinary_fields_equal"]["bonds"]
    assert not audit["fixed_fields_equal"]["bonds"]

    # A matching SMILES is insufficient if a padding-slot label differs.
    different_padding = SimpleNamespace(
        spec=context.spec, active=context.active,
        seed=HardGraphBatch(context.seed.atoms.clone(),
                            context.seed.charges.clone(), context.seed.bonds.clone()))
    different_padding.seed.charges[0, 3] = 2
    assert not rescue.exact_native_graphs(exact, different_padding)["exact_native_c13"]


def test_active_secant_counts_each_undirected_bond_once():
    context, _ = _fixture()
    atom = torch.zeros((1, 4, 3))
    charge = torch.zeros((1, 4, 3))
    bond0 = torch.zeros((1, 4, 4, 2))
    bond1 = bond0.clone()
    bond1[0, 0, 1] = bond1[0, 1, 0] = torch.tensor([-0.2, 0.2])
    anchor = RelaxedGraph(atom, charge, bond0)
    candidate = RelaxedGraph(atom, charge, bond1)
    response = rescue.active_secant_response(candidate, anchor, context.active, 0.5)
    expected = float(torch.tensor([-0.2, 0.2]).norm())
    assert response["element_l2"] == 0
    assert response["charge_l2"] == 0
    assert response["bond_upper_triangle_l2"] == pytest.approx(expected)
    assert response["per_latent_distance"] == pytest.approx(expected / 0.5)


def test_neighborhood_is_33_decoder_only_evaluations_and_reproducible():
    context, base = _fixture()

    class IdentityReference:
        def denormalize(self, point):
            return point

    class SoftModel:
        calls = 0

        def decode(self, point):
            self.calls += 1
            atom = base.atoms.clone()
            atom[:, :2, 2] = atom[:, :2, 2] + point[:, 0:1].expand(-1, 2)
            charge = base.charges.clone()
            charge[:, :2, 2] = charge[:, :2, 2] + point[:, 1:2].expand(-1, 2)
            bond = base.bonds.clone()
            bond[:, 0, 1, 0] = bond[:, 0, 1, 0] + point[:, 2]
            bond[:, 1, 0, 0] = bond[:, 1, 0, 0] + point[:, 2]
            return DecoderOutput(atom, charge, bond)

    model = SoftModel()
    anchor = torch.zeros((1, 3))
    journal = []
    rows, latents = rescue.decoder_neighborhood(model, IdentityReference(),
                                                context, anchor,
                                                record_candidate=journal.append)
    assert model.calls == 33
    assert len(rows) == len(latents) == 33
    assert journal == rows
    assert rows[0]["candidate_id"] == "anchor"
    assert all(row["exact_native_c13"] for row in rows)
    finalists = rescue.select_decoder_finalists(rows)
    assert len(finalists) == 4
    assert [row["radius"] for row in finalists] == list(rescue.RADII)
    again, again_latents = rescue.decoder_neighborhood(
        SoftModel(), IdentityReference(), context, anchor)
    assert rows == again
    for key in latents:
        torch.testing.assert_close(latents[key], again_latents[key])


def test_finalist_rank_excludes_nonexact_candidate_and_has_stable_ties():
    rows = [{"candidate_id": "anchor", "radius": 0., "exact_native_c13": True}]
    rows += [{"candidate_id": "r0.25-d2", "radius": 0.25,
              "direction_index": 2, "exact_native_c13": True,
              "active_secant": {"per_latent_distance": 0.4}},
             {"candidate_id": "r0.25-d1", "radius": 0.25,
              "direction_index": 1, "exact_native_c13": True,
              "active_secant": {"per_latent_distance": 0.4}},
             {"candidate_id": "r0.25-d0", "radius": 0.25,
              "direction_index": 0, "exact_native_c13": False,
              "active_secant": {"per_latent_distance": 900.0}},
             {"candidate_id": "r1-d0", "radius": 1.0,
              "direction_index": 0, "exact_native_c13": True,
              "active_secant": {"per_latent_distance": 0.1}}]
    finalists = rescue.select_decoder_finalists(rows)
    assert [row["candidate_id"] for row in finalists] == ["r0.25-d1", "r1-d0"]


def test_fd_gate_uses_binary_objective_and_prescribed_five_scales(monkeypatch):
    def fake_score_state(point, *args, **kwargs):
        return {"binary_probability": -0.2 * float(point[0, 0])}, None, None

    class Run:
        def __init__(self):
            self.events = []

        def log(self, value):
            self.events.append(value)

    monkeypatch.setattr(rescue, "_score_state", fake_score_state)
    run = Run()
    result = rescue._finite_difference_at(
        torch.tensor([[1., 0.]]), torch.tensor([[0.2, 0.]]),
        None, None, None, None, run, candidate_id="r0.25-d0")
    assert result["gate"]["passed"]
    assert [row["epsilon"] for row in result["rows"]] == list(rescue.FD_EPSILONS)
    assert len(run.events) == 5


def test_finite_gradient_entries_with_overflowed_norm_cannot_pass(monkeypatch):
    large = torch.tensor([[1e20, 1e20]], dtype=torch.float32)

    def fake_score_state(*args, **kwargs):
        return {"binary_probability": 0.5, "affinity_pred_value": 1.0}, None, large

    monkeypatch.setattr(rescue, "_score_state", fake_score_state)
    screen, gradient = rescue._screen_gradient(
        torch.zeros((1, 2)), None, None, None, None, candidate_id="r0.25-d0")
    assert screen["gradient_finite"]
    assert screen["normalized_gradient_norm"] is None
    assert not screen["viable"]
    with pytest.raises(ValueError, match="finite nonzero analytic gradient norm"):
        rescue._finite_difference_at(torch.zeros((1, 2)), gradient,
                                     None, None, None, None, None,
                                     candidate_id="r0.25-d0")
