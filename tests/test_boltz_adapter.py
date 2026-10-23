"""Boundary regressions runnable without optional Boltz dependencies, on HPC."""

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from latentmol.boltz import PreparedComplex, adapt_graph, fixed_occupancy_graph
from latentmol.boltz.contracts import BOLTZ_COMMIT
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch


def prepared_fixture():
    """Deliberately noncontiguous, permuted ligand map among protein/padded rows."""
    spec = GraphSpec(max_nodes=4)
    atoms = torch.tensor([[1, 0, 3, 0]])
    charges = torch.tensor([[2, 2, 1, 2]])
    bonds = torch.zeros(1, 4, 4, dtype=torch.long)
    bonds[0, 0, 2] = bonds[0, 2, 0] = 1
    seed = HardGraphBatch(atoms, charges, bonds)
    types = torch.zeros(1, 4, 4, dtype=torch.long)
    types[0, 1, 3] = types[0, 3, 1] = 2
    types[0, 0, 2] = types[0, 2, 0] = 1  # native OTHER, fixed protein context
    pool = torch.zeros(1, 5, 4)
    pool[0, :4] = torch.eye(4)
    features = {
        "ref_element": F.one_hot(torch.tensor([[7, 8, 6, 6, 0]]), 128),
        "ref_charge": torch.tensor([[0., -1., 0., 0., 0.]]),
        "token_pad_mask": torch.ones(1, 4), "atom_pad_mask": torch.tensor([[1., 1., 1., 1., 0.]]),
        "affinity_token_mask": torch.tensor([[0., 1., 0., 1.]]),
        "mol_type": torch.tensor([[0, 3, 0, 3]]), "type_bonds": types,
        "token_bonds": (types > 0).float()[..., None], "atom_to_token": pool,
        "token_to_rep_atom": pool.transpose(1, 2), "ref_pos": torch.randn(1, 5, 3),
    }
    return PreparedComplex(features, torch.randn(1, 1, 5, 3),
                           torch.tensor([3, -1, 1, -1]), torch.tensor([3, -1, 1, -1]),
                           seed, spec, {"boltz_commit": BOLTZ_COMMIT})


def random_logits(spec):
    return DecoderOutput(torch.randn(1, spec.max_nodes, len(spec.atomic_numbers), requires_grad=True),
                         torch.randn(1, spec.max_nodes, len(spec.formal_charges), requires_grad=True),
                         torch.randn(1, spec.max_nodes, spec.max_nodes, len(spec.bond_names), requires_grad=True))


def test_hard_mapping_preserves_native_features_and_fixed_context():
    prepared = prepared_fixture()
    prepared.validate()
    mapped, bonds = adapt_graph(prepared.seed.to_relaxed(prepared.spec), prepared)
    for key in ("ref_element", "ref_charge", "token_bonds"):
        torch.testing.assert_close(mapped[key], prepared.features[key].to(mapped[key]), rtol=0, atol=0)
    torch.testing.assert_close(bonds, F.one_hot(prepared.features["type_bonds"], 7).float(), rtol=0, atol=0)
    assert mapped["ref_pos"] is prepared.features["ref_pos"]
    assert mapped["atom_to_token"] is prepared.features["atom_to_token"]


def test_fixed_occupancy_conditions_before_bond_gating():
    prepared = prepared_fixture()
    decoded = random_logits(prepared.spec)
    baseline = fixed_occupancy_graph(decoded, prepared.active, prepared.spec)
    changed = decoded.atoms.detach().clone()
    changed[..., 0] += 1000  # would nearly erase every atom in ordinary probabilities()
    alternate = fixed_occupancy_graph(DecoderOutput(changed, decoded.charges, decoded.bonds), prepared.active, prepared.spec)
    for left, right in ((baseline.atoms, alternate.atoms), (baseline.charges, alternate.charges), (baseline.bonds, alternate.bonds)):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        torch.testing.assert_close(left.sum(-1), torch.ones_like(left[..., 0]))
    torch.testing.assert_close(baseline.occupancy, prepared.active.float()[None], rtol=0, atol=0)
    assert baseline.bonds[0, 0, 2, 1:].sum() > 0
    assert torch.equal(baseline.bonds[0, :, :, 0].diagonal(), torch.ones(4))
    assert baseline.bonds[0, 1, :, 1:].count_nonzero() == 0


def test_soft_connectivity_is_probability_not_expected_order_and_none_is_embedded():
    prepared = prepared_fixture()
    graph = prepared.seed.to_relaxed(prepared.spec)
    mixture = torch.tensor([0.25, 0., 0., 0.75, 0.])  # NONE or TRIPLE
    graph.bonds[0, 0, 2] = graph.bonds[0, 2, 0] = mixture
    mapped, native = adapt_graph(graph, prepared)
    assert mapped["token_bonds"][0, 3, 1, 0] == 0.75
    assert native[0, 3, 1, 0] == 0.25
    assert native[0, 3, 1, 4] == 0.75
    assert native[0, 3, 1, 3] == 0


def test_all_supported_logits_receive_gradients_and_absent_slots_do_not():
    prepared = prepared_fixture()
    decoded = random_logits(prepared.spec)
    graph = fixed_occupancy_graph(decoded, prepared.active, prepared.spec)
    features, types = adapt_graph(graph, prepared)
    # This scalar tests adapter connectivity, not a fake affinity predictor.
    scalar = (features["ref_element"] * torch.arange(128.)).sum()
    scalar = scalar + features["ref_charge"].sum() + features["token_bonds"].sum()
    scalar = scalar + (types * torch.arange(7.)).sum()
    gradients = torch.autograd.grad(scalar, (decoded.atoms, decoded.charges, decoded.bonds))
    assert all(torch.isfinite(g).all() and g.norm() > 0 for g in gradients)
    assert gradients[0][..., 0].count_nonzero() == 0
    assert gradients[0][:, ~prepared.active].count_nonzero() == 0
    assert gradients[1][:, ~prepared.active].count_nonzero() == 0
    assert gradients[2][:, ~prepared.active].count_nonzero() == 0


def test_reject_soft_occupancy_instead_of_silently_reconditioning_marginals():
    prepared = prepared_fixture()
    with pytest.raises(ValueError, match="exact prepared occupancy"):
        adapt_graph(random_logits(prepared.spec).probabilities(), prepared)


def test_reject_incomplete_or_duplicate_mapping():
    prepared = prepared_fixture()
    with pytest.raises(ValueError):
        replace(prepared, slot_to_token=torch.tensor([3, -1, 3, -1])).validate()
    bad = dict(prepared.features)
    bad["affinity_token_mask"] = torch.tensor([[1., 1., 0., 1.]])
    with pytest.raises(ValueError, match="cover exactly"):
        replace(prepared, features=bad).validate()


def test_reject_swapped_map_even_when_shapes_and_coverage_match():
    prepared = prepared_fixture()
    with pytest.raises(ValueError, match="Seed and mapped native"):
        replace(prepared, slot_to_atom=torch.tensor([1, -1, 3, -1]),
                slot_to_token=torch.tensor([1, -1, 3, -1])).validate()
    with pytest.raises(ValueError, match="sentinel"):
        replace(prepared, slot_to_atom=torch.tensor([3, -2, 1, -1])).validate()


def test_prepared_snapshot_preserves_plain_tensor_contract(tmp_path):
    original = prepared_fixture()
    path = tmp_path / "prepared.pt"
    original.save(path)
    recovered = PreparedComplex.load(path)
    assert recovered.spec == original.spec
    assert torch.equal(recovered.slot_to_atom, original.slot_to_atom)
    assert torch.equal(recovered.coordinates, original.coordinates)
    with pytest.raises(FileExistsError):
        original.save(path)


def test_loader_replaces_obsolete_checkpoint_runtime_groups(monkeypatch, tmp_path):
    """Catch the real HPC constructor failure without loading multi-GB weights.

    This optional integration regression uses the installed pinned Boltz classes
    and their current constructor signatures. Other AE/VAE environments skip it.
    """
    pytest.importorskip("boltz")
    import inspect
    from types import SimpleNamespace
    from boltz.model.models.boltz2 import Boltz2
    from boltz.model.modules.diffusionv2 import AtomDiffusion
    from latentmol.boltz import model as adapter_model
    from latentmol.boltz import prepare

    captured = {}

    class FakeLoadedModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.msa_module = SimpleNamespace(subsample_msa=True)
            self.affinity_prediction = True
            self.run_trunk_and_structure = True

    def fake_load(path, **kwargs):
        # Lightning overrides this whole dictionary, not individual saved keys.
        saved = {"diffusion_process_args": {"mse_rotational_alignment": True}}
        saved.update(kwargs)
        diffusion = saved["diffusion_process_args"]
        assert "mse_rotational_alignment" not in diffusion
        assert set(diffusion) <= set(inspect.signature(AtomDiffusion.__init__).parameters)
        assert diffusion["step_scale"] == 1.5  # Boltz-2, not Boltz-1 defaults
        captured.update(kwargs)
        return FakeLoadedModel()

    monkeypatch.setattr(adapter_model, "file_sha256", lambda path: adapter_model.AFFINITY_SHA256)
    monkeypatch.setattr(prepare, "installed_boltz_provenance", lambda: {})
    monkeypatch.setattr(Boltz2, "load_from_checkpoint", fake_load)
    result = adapter_model.load_affinity_model(tmp_path / "official.ckpt")
    assert captured["strict"] is True
    assert captured["pairformer_args"]["num_blocks"] == 64
    assert captured["msa_args"]["subsample_msa"] is False
    assert captured["msa_args"]["use_paired_feature"] is True
    assert not any(captured["steering_args"][name] for name in
                   ("fk_steering", "physical_guidance_update", "contact_guidance_update"))
    assert captured["predict_args"]["max_parallel_samples"] == 1
    assert captured["use_kernels"] is False
    assert captured["affinity_mw_correction"] is False
    assert result.training is False
    assert result.msa_module.subsample_msa is False
    assert not result.weight.requires_grad


def test_eval_checkpoints_preserve_two_distinct_blocks_outputs_gradients_and_names():
    """Verify recomputation and autograd.grad without train mode or state renaming."""
    from latentmol.boltz.model import checkpoint_eval_block

    class FrozenBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)
            self.dropout = torch.nn.Dropout(0.75)
            self.calls = 0

        def forward(self, value):
            assert not self.training
            self.calls += 1
            return value + self.dropout(self.linear(value).tanh())

    network = torch.nn.Sequential(FrozenBlock(), FrozenBlock()).double().eval().requires_grad_(False)
    original_state = {name: value.clone() for name, value in network.state_dict().items()}
    names = tuple(name for name, _ in network.named_parameters())
    point = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    reference = network(point)
    reference_gradient, = torch.autograd.grad(reference.square().sum(), point)
    for block in network:
        checkpoint_eval_block(block)
        first_forward = block.forward
        checkpoint_eval_block(block)
        assert block.forward is first_forward  # Idempotent; no nested wrappers.
    checkpoint_point = point.detach().clone().requires_grad_(True)
    result = network(checkpoint_point)
    gradient, = torch.autograd.grad(result.square().sum(), checkpoint_point)
    torch.testing.assert_close(result, reference, rtol=0, atol=0)
    torch.testing.assert_close(gradient, reference_gradient, rtol=1e-12, atol=1e-12)
    assert all(block.calls >= 3 for block in network)  # baseline, forward, recompute
    assert tuple(name for name, _ in network.named_parameters()) == names
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in network.parameters())
    for name, value in network.state_dict().items():
        torch.testing.assert_close(value, original_state[name], rtol=0, atol=0)
    with torch.no_grad():
        torch.testing.assert_close(network(point.detach()), reference, rtol=0, atol=0)
    assert all(not block.training for block in network)
    network.train()
    with pytest.raises(ValueError, match="remain in eval mode"):
        network(point)


def test_strict_finite_difference_gate_requires_two_distinct_agreeing_scales():
    from latentmol.boltz.smoke import finite_difference_gate
    one_good = [{"epsilon": 0.0001, "relative_error": 0.01},
                {"epsilon": 0.001, "relative_error": 0.03}]
    assert not finite_difference_gate(one_good)["passed"]
    assert not finite_difference_gate([one_good[0], one_good[0]])["passed"]
    two_good = one_good + [{"epsilon": 0.0003, "relative_error": 0.015}]
    result = finite_difference_gate(two_good)
    assert result["passed"]
    assert result["agreeing_epsilons"] == [0.0001, 0.0003]
    assert result["relative_tolerance"] == 0.02
    assert not finite_difference_gate([{"epsilon": 0.0001, "relative_error": float("nan")},
                                       {"epsilon": 0.0003, "relative_error": 0.001}])["passed"]


def test_discrete_projection_scores_only_valid_one_hot_graphs_without_autograd():
    from latentmol.boltz.smoke import score_discrete_projection
    prepared = prepared_fixture()
    graph = prepared.seed.to_relaxed(prepared.spec)
    graph.atoms[0, 0, 1] = 0.6
    graph.atoms[0, 0, 2] = 0.4
    graph.atoms.requires_grad_(True)
    calls = []

    def scorer(features, types, coordinates):
        assert not torch.is_grad_enabled()
        assert torch.all((features["ref_element"] == 0) | (features["ref_element"] == 1))
        assert torch.all((types == 0) | (types == 1))
        assert coordinates is prepared.coordinates
        assert features["ref_pos"] is prepared.features["ref_pos"]
        calls.append(True)
        return {"affinity_pred_value": torch.tensor([2.]),
                "affinity_probability_binary": torch.tensor([0.25])}

    row, hard = score_discrete_projection(graph, prepared, scorer)
    assert row["discrete_valid"]
    assert row["discrete_smiles"]
    assert row["fixed_context_discrete_affinity_pred_value"] == 2.
    assert row["fixed_context_discrete_affinity_probability_binary"] == 0.25
    assert "no pose or native feature rebuild" in row["discrete_score_policy"]
    assert torch.equal(hard.atoms, prepared.seed.atoms)
    assert graph.atoms.grad is None
    invalid_graph = prepared.seed.to_relaxed(prepared.spec)
    invalid_graph.bonds.zero_()
    invalid_graph.bonds[..., 0] = 1.  # Separate carbon and oxygen fragments.
    invalid, _ = score_discrete_projection(invalid_graph, prepared, scorer)
    assert not invalid["discrete_valid"]
    assert invalid["fixed_context_discrete_affinity_pred_value"] is None
    assert invalid["fixed_context_discrete_affinity_probability_binary"] is None
    assert len(calls) == 1
