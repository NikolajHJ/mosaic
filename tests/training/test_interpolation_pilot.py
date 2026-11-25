"""Small checks for the mapped objective and the matched training entry point."""

from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from latentmol.atom_preserving_chemistry import repair_loss
from latentmol.chemistry import encode_smiles
from latentmol.datasets.data import prepare_dataset
from latentmol.evaluation.molecular_pairs import permute_graph, relabel_categories
from latentmol.experiments import file_sha256
from latentmol.guidance.repair_proposals import ProposalRepairTeacher, RepairProposalConfig
from latentmol.models import GraphAutoencoder, ModelConfig
from latentmol.probability_repair import ProbabilityRepairConfig
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch
from latentmol.training.interpolation_pilot import (
    FRACTIONS, _backward_with_interpolation_cap, mapped_interpolation_loss, run_pilot,
)
from latentmol.training.losses import LossConfig


def _graph(smiles, spec):
    row = encode_smiles(smiles, spec)
    return HardGraphBatch(*(torch.tensor(row[key], dtype=torch.long)[None]
                           for key in ("atom_types", "charges", "bond_types")))


def test_mapped_soft_targets_have_the_same_optimum_in_both_frames():
    spec = GraphSpec(max_nodes=6)
    a, b = _graph("CCO", spec), _graph("CCN", spec)
    mapped_b = permute_graph(b, [2, 0, 1])
    order = torch.tensor([[2, 0, 1, 3, 4, 5]])
    left, right = a.to_relaxed(spec), mapped_b.to_relaxed(spec)
    logits = []
    for name in ("atoms", "charges", "bonds"):
        x, y = getattr(left, name), getattr(right, name)
        mixed = torch.cat([(1 - fraction) * x + fraction * y for fraction in FRACTIONS])
        logits.append(mixed.clamp_min(1e-8).log())
    losses = []
    for values in (logits, relabel_categories(*logits, order.argsort(-1).expand(9, -1))):
        live = tuple(value.clone().requires_grad_() for value in values)
        loss, _ = mapped_interpolation_loss(DecoderOutput(*live), a, mapped_b, order, spec, LossConfig())
        gradients = torch.autograd.grad(loss, live)
        assert all(float(gradient.abs().max()) < 1e-6 for gradient in gradients)
        losses.append(float(loss.detach()))
    assert losses[0] == pytest.approx(losses[1], abs=1e-7)


@pytest.mark.parametrize("auxiliary_size, expected_weight", [(100., .0125), (2., .1), (0., .1)])
def test_interpolation_gradient_cap_preserves_base_gradient(auxiliary_size, expected_weight):
    parameter = torch.tensor([1., 2.], requires_grad=True)
    base, auxiliary = 5 * parameter[0], auxiliary_size * parameter[1]
    cap = _backward_with_interpolation_cap(base, auxiliary, (parameter,), .1, .25)
    assert cap["base_gradient_norm"] == 5.
    assert cap["interpolation_weight_used"] == pytest.approx(expected_weight)
    assert cap["capped_interpolation_gradient_norm"] <= 1.25
    torch.testing.assert_close(parameter.grad, torch.tensor([5., auxiliary_size * expected_weight]))


@pytest.mark.parametrize("chemistry", [False, True])
def test_short_pilot_runs_matched_arms_with_all_nine_points(tmp_path, chemistry):
    spec = GraphSpec(max_nodes=8)
    source = tmp_path / "molecules.smi"
    source.write_text("\n".join(["CCO", "CCN", "CCS", "CCC", "CCF", "CCCl", "CCBr", "COC",
                                 "CNC", "CC=O", "CC#N", "CC=C", "CC#C", "C=CO", "C=CN"]))
    data = tmp_path / "cache"
    prepare_dataset(source, data, spec=spec, train_count=7, val_count=7, seed=31)
    manifest = json.loads((data / "manifest.json").read_text())
    torch.manual_seed(19)
    model = GraphAutoencoder(spec, ModelConfig(hidden=8, latent=4, layers=1))
    checkpoint = tmp_path / "epoch45.pt"
    torch.save(dict(format_version=1, epoch=45, graph_spec=spec.to_dict(), model_config=asdict(model.config),
        model_state=model.state_dict(), dataset_sha256=manifest["dataset_sha256"],
        train_config=dict(mode="vae"), loss_config=asdict(LossConfig())),
        checkpoint)
    config = json.loads((Path(__file__).resolve().parents[2] /
                        "experiments/vae/epoch45-interpolation-pilot.json").read_text())
    config.update(checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint), data=str(data),
                  data_manifest_sha256=file_sha256(data / "manifest.json"), dataset_sha256=manifest["dataset_sha256"],
                  updates=1, batch_size=2, pairs_per_update=1, prior_anchor_count=2)
    for key in ("train_pairs", "validation_pairs"):
        config[key].update(pool_count=7, pair_count=2, min_atoms=2, max_atoms=8, min_similarity=.01,
                           min_core_fraction=.5, max_candidate_pairs=30)
    config["path_evaluation"].update(batch_size=4, categorical_draws=1, probability_snapshot_pairs=1)
    config["comparison"].update(reconstruction_count=4, prior_count=4, prior_seeds=[22],
                                anchor_count=2, directions=1, exclude_reconstruction_panels=[],
                                atom_order=dict(count=2, permutations=2, seed=23))
    if chemistry:
        config["interpolation_chemistry"] = json.loads((Path(__file__).resolve().parents[2] /
            "experiments/vae/epoch45-interpolation-chemistry.json").read_text())
        config["interpolation_chemistry"]["workers"] = 0
    output = tmp_path / "run"
    result = run_pilot(config, output, device="cpu")
    assert result["status"] == "completed"
    assert file_sha256(checkpoint) == config["checkpoint_sha256"]
    assert list(result["paired_argmax_validity"]) == [str(fraction) for fraction in FRACTIONS]
    for summary in result["path_evaluation"].values():
        assert [row["fraction"] for row in summary["points"]] == [0., *FRACTIONS, 1.]
    control = json.loads((output / "control/training.jsonl").read_text())
    treatment = json.loads((output / "interpolation/training.jsonl").read_text())
    assert control["reconstruction"] == treatment["reconstruction"]
    assert treatment["total"] == pytest.approx(control["total"] + .1 * treatment["interpolation"])
    assert all(value > 0 for value in treatment["interpolation_path_gradient_norms"].values())
    if chemistry:
        extra = json.loads((output / "interpolation_chemistry/training.jsonl").read_text())
        assert extra["interpolation"] == treatment["interpolation"]
        assert sum(extra["chemistry_status_counts"].values()) == len(FRACTIONS)
        assert extra["total"] == pytest.approx(treatment["total"] + .25 * extra["chemistry"])
        assert list(result["paired_chemistry_argmax_validity"]) == [str(fraction) for fraction in FRACTIONS]
    else:
        assert "interpolation_chemistry" not in result["checkpoints"]


def test_chemistry_teacher_repairs_disconnected_interpolation_output():
    spec = GraphSpec(max_nodes=5)
    hard = _graph("CC", spec)
    graph = hard.to_relaxed(spec)
    bonds = graph.bonds.clamp_min(1e-8).log()
    bonds[0, 0, 1, :2] = bonds[0, 1, 0, :2] = torch.tensor([0.0, -0.2])
    decoded = DecoderOutput(graph.atoms.clamp_min(1e-8).log(), graph.charges.clamp_min(1e-8).log(),
                            bonds.requires_grad_())
    with ProposalRepairTeacher(spec, ProbabilityRepairConfig(checks_per_depth=32),
                              proposals=RepairProposalConfig(candidate_checks=96, max_bond_changes=6)) as teacher:
        targets = teacher.build_targets(decoded, candidate_ids=[0], update=1)
    assert targets.receipt["accepted_targets"] == 1
    loss = repair_loss(decoded, targets, spec)
    gradient, = torch.autograd.grad(loss, bonds)
    assert gradient.isfinite().all()
    assert gradient[0, 0, 1, 1] < 0 < gradient[0, 0, 1, 0]


@pytest.mark.parametrize("capped", [False, True])
def test_verified_graph_pilot_uses_matched_arms_and_hard_middle_targets(tmp_path, capped):
    spec = GraphSpec(max_nodes=8)
    source = tmp_path / "graphs.smi"
    source.write_text("\n".join(["CCCO", "CCCN", "NCCO", "CCCOC", "COCCN", "CCCON", "CCCCCC"]))
    data = tmp_path / "cache"
    manifest = prepare_dataset(source, data, spec=spec, train_count=3, val_count=3, seed=430)
    model = GraphAutoencoder(spec, ModelConfig(hidden=8, latent=4, layers=1))
    checkpoint = tmp_path / "epoch45.pt"
    torch.save(dict(format_version=1, epoch=45, graph_spec=spec.to_dict(), model_config=asdict(model.config),
        model_state=model.state_dict(), dataset_sha256=manifest["dataset_sha256"],
        train_config=dict(mode="vae"), loss_config=asdict(LossConfig())), checkpoint)
    configs = Path(__file__).resolve().parents[2] / "experiments/vae"
    config = {**json.loads((configs / "epoch45-interpolation-pilot.json").read_text()),
              **json.loads((configs / "epoch45-graph-interpolation-pilot.json").read_text())}
    config.update(checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint), data=str(data),
                  data_manifest_sha256=file_sha256(data / "manifest.json"), dataset_sha256=manifest["dataset_sha256"],
                  updates=1, batch_size=2, pairs_per_update=1, prior_anchor_count=2)
    for key in ("train_triplets", "validation_triplets"):
        config[key].update(count=1, center_limit=3, min_atoms=3, max_atoms=8)
    config["path_evaluation"]["batch_size"] = 4
    config["comparison"].update(reconstruction_count=3, prior_count=4, prior_seeds=[22], anchor_count=2,
                                directions=1, exclude_reconstruction_panels=[],
                                atom_order=dict(count=2, permutations=2, seed=23))
    if capped:
        config.update(interpolation_max_gradient_ratio=.25, early_prior=dict(
            steps=[0, 1], count=4, batch_size=2, latent_seed=24, categorical_seed=25))

    output = tmp_path / "verified"
    result = run_pilot(config, output, device="cpu")
    assert result["status"] == "completed"
    assert result["frozen_parent_verified"]
    assert result["paired_exact_midpoint"]["attempts"] == 1
    assert result["path_evaluation"]["interpolation"]["triplets"] == 1
    control = json.loads((output / "control/training.jsonl").read_text())
    treatment = json.loads((output / "interpolation/training.jsonl").read_text())
    assert control["reconstruction"] == treatment["reconstruction"]
    weight = treatment.get("interpolation_weight_used", .1)
    assert treatment["total"] == pytest.approx(control["total"] + weight * treatment["interpolation"])
    assert all(value > 0 for value in treatment["interpolation_path_gradient_norms"].values())
    if capped:
        assert 0 < weight <= .1
        assert treatment["capped_interpolation_gradient_norm"] <= .25 * treatment["base_gradient_norm"] + 1e-8
        assert set(result["early_prior_paired_validity"]) == {0, 1}
        control_prior = json.loads((output / "control/prior-step-0000.json").read_text())
        treatment_prior = json.loads((output / "interpolation/prior-step-0000.json").read_text())
        assert control_prior["observation"] == treatment_prior["observation"]
