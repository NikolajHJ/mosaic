"""Paired edit readouts, hard scoring eligibility, and local latent gradient smoke."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from rdkit import Chem

from latentmol.chemistry import encode_smiles
from latentmol.design.decoder_comparison import (
    ARCHITECTURES, POLICIES, RAYS, candidate_rows, permute_for_native, probe, summarize,
)
from latentmol.design.temperature_fresh import check_edit_limits, score_candidates
from latentmol.models import GraphAutoencoder
from latentmol.representation import GraphSpec, HardGraphBatch, RelaxedGraph
from latentmol.train import TrainConfig


def observation(smiles):
    return dict(smiles=smiles, signature=smiles or "invalid-id",
                failure=None if smiles else "disconnected")


def fake_record(start, base, successive):
    trials = {ray: dict(samples=[
        dict(radius=radius, graphs={policy: observation(smiles) for policy in POLICIES})
        for radius, smiles in enumerate(successive, 1)
    ]) for ray in RAYS}
    return dict(start=start, architecture="triangle_refine", source_smiles="CCC",
                baseline={policy: observation(base) for policy in POLICIES}, trials=trials)


def test_edits_need_comparable_scored_centers_and_are_deduplicated_and_capped():
    records = [
        fake_record("mixed", "CCO", ["CCO", None, "CCC", "COC", "COC", "CCN", "CCF"]),
        fake_record("rescue", None, [None, "COC"]),
        fake_record("same", "CCC", ["CCC", "CCC"]),
        fake_record("failed", "CCF", ["COC"]),
        fake_record("worse", "COC", ["CCO"]),
    ]
    requested = candidate_rows(records, 2)
    check_edit_limits(requested, 2)
    losses = {"CCC": 1.5, "CCO": 1.3, "COC": .9, "CCN": 1.4, "CCF": None}

    def fake_score(molecule, index):
        score = losses[Chem.MolToSmiles(molecule)]
        return dict(status="scored" if score is not None else "embedding_failed", loss=score)

    scored, unique = score_candidates(requested, fake_score)
    assert unique == 5
    report = summarize(records, scored, 2)
    key = "triangle_refine/descent/fixed"
    per_start = {r["start"]: r for r in report["starts"] if r["arm"] == key}
    mixed = per_start["mixed"]
    assert (mixed["fresh_selected"], mixed["distinct_valid_edits"], mixed["invalid_observations"],
            mixed["unchanged_observations"], mixed["source_return_observations"]) == (2, 3, 1, 1, 1)
    assert mixed["best_delta_from_center"] == pytest.approx(-.4)
    assert mixed["best_delta_from_input"] == pytest.approx(-.6)
    assert per_start["rescue"]["rescued_invalid_center"] and not per_start["rescue"]["improved"]
    assert per_start["failed"]["outcome"] == "unscored_center"
    assert per_start["same"]["outcome"] == "unchanged"
    assert per_start["worse"]["outcome"] == "scored_without_improvement"
    assert report["by_arm"][key]["improved_fraction_all_starts"] == .2
    assert report["by_arm"][key]["improved_fraction_comparable"] == .5
    with pytest.raises(ValueError, match="identities"):
        summarize(records, scored[1:], 2)


def test_native_relabeling_preserves_both_bond_axes_and_input_gradient():
    bonds = torch.randn(1, 4, 4, 2, requires_grad=True)
    graph = RelaxedGraph(torch.randn(1, 4, 3), torch.randn(1, 4, 5), bonds)
    order = torch.tensor([1, 3, 0, 2])
    native = permute_for_native(graph, order)
    assert torch.equal(native.atoms[:, :2], graph.atoms[:, [1, 3]])
    native.bonds[0, 0, 1, 1].backward()
    assert bonds.grad.nonzero().tolist() == [[0, 1, 3, 1]]


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_real_compact_models_decode_and_receive_latent_derivative_from_analytic_scorer(architecture):
    # Only a smoke test. The actual jobs use the frozen Mosaic and native FILE contact scorer.
    torch.manual_seed(21)
    spec = GraphSpec(max_nodes=6)
    base = TrainConfig.from_yaml(Path(__file__).resolve().parents[2] / "configs/vae_compact_triangle.yaml")
    model = GraphAutoencoder(spec, replace(base.model, decoder_variant=architecture)).eval().requires_grad_(False)
    graph = encode_smiles("CCO", spec)
    hard = HardGraphBatch(*(torch.as_tensor(graph[f]).long().unsqueeze(0)
                            for f in ("atom_types", "charges", "bond_types")))

    def toy_contact(graph):
        return (graph.atoms[0, :3, 1] * torch.arange(3)).sum() + graph.bonds[0, 0, 1, 0], {}

    random = torch.randn(1, model.config.latent)
    cfg = dict(active_count=3, temperature=1.0, radii=[.01])
    result = probe("heldout", hard, "CCO", "CCO", model, toy_contact, cfg, random / random.norm())
    assert result["gradient_norm"] > 0
    assert len(result["native_slots"]) == 3
    assert result["trials"]["descent"]["samples"][0]["loss_delta"] < 0
    assert all(len(result["trials"][ray]["samples"]) == 1 for ray in RAYS)
    assert all(p.grad is None for p in model.parameters())
