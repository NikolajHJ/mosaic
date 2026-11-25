"""Restart scorer plumbing with a small frozen mock, not native model evidence."""
from copy import deepcopy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from latentmol.boltz.contracts import BOLTZ_COMMIT, PreparedComplex
from latentmol.guidance.boltz_search import BoltzSearchScorer, hard_graph_record, hard_pose_key, resolve_execution
from latentmol.graph_policy import graph_from_decoder
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch


class MockDBMol(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.), requires_grad=False)
        self.recycling_steps = 0
        self.calls = 0
        self.manifest_config = dict(profile="dbmol_fixed_pose_v1",
            confidence_checkpoint=dict(path="unused", sha256="c" * 64))

    def forward(self, features, types, coordinates):
        self.calls += 1
        elements = (features["ref_element"] * torch.arange(128.)).sum(-1)
        # Position-dependent weights mimic a fixed-pose score, not a predictor.
        value = (elements * torch.tensor([[1., 3.]])).sum()
        value = value + features["ref_charge"].sum() + (types * torch.arange(7.)).sum()
        value = self.weight * value
        return dict(dbmol_total_loss=value + 7., dbmol_affinity_loss=value - 2.,
                    dbmol_contact_loss=value / 3., dbmol_confidence_loss=value / 5.,
                    affinity_pred_value=value + 1000.)


@pytest.fixture(autouse=True)
def restore_cuda_math_flags():
    previous = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


def fixture(execution=None, model_factory=MockDBMol):
    spec = GraphSpec(max_nodes=4)
    bonds = torch.zeros(1, 4, 4, dtype=torch.long)
    bonds[0, 0, 1] = bonds[0, 1, 0] = 1
    seed = HardGraphBatch(torch.tensor([[1, 2, 0, 0]]), torch.full((1, 4), 2), bonds)
    native_bonds = bonds[:, :2, :2]
    pool = torch.eye(2)[None]
    features = dict(ref_element=F.one_hot(torch.tensor([[6, 7]]), 128), ref_charge=torch.zeros(1, 2),
        token_pad_mask=torch.ones(1, 2), atom_pad_mask=torch.ones(1, 2), affinity_token_mask=torch.ones(1, 2),
        mol_type=torch.full((1, 2), 3), type_bonds=native_bonds * 2,
        token_bonds=native_bonds.float()[..., None], atom_to_token=pool, token_to_rep_atom=pool,
        ref_pos=torch.zeros(1, 2, 3))
    prepared = PreparedComplex(features, torch.zeros(1, 1, 2, 3), torch.tensor([0, 1, -1, -1]),
        torch.tensor([0, 1, -1, -1]), seed, spec, dict(boltz_commit=BOLTZ_COMMIT))
    model = model_factory()
    scorer = BoltzSearchScorer(prepared, model, source_sha256="a" * 64, execution=execution)
    return prepared, model, scorer


def test_live_graph_repacking_preserves_all_gradients_and_full_dbmol_total():
    prepared, model, scorer = fixture()
    active = torch.tensor([False, True, False, True])
    spec = prepared.spec
    logits = DecoderOutput(torch.randn(1, 4, len(spec.atomic_numbers), requires_grad=True),
        torch.randn(1, 4, len(spec.formal_charges), requires_grad=True),
        torch.randn(1, 4, 4, len(spec.bond_names), requires_grad=True))
    graph = graph_from_decoder(logits, spec, fixed_active=active)
    value, aux = scorer.score_at(graph, active)
    assert float(value.detach()) == aux["dbmol_total_loss"]
    assert aux["dbmol_total_loss"] != aux["affinity_pred_value"]
    gradients = torch.autograd.grad(value, (logits.atoms, logits.charges, logits.bonds))
    assert all(torch.isfinite(g).all() and g.norm() > 0 for g in gradients)
    assert gradients[0][:, ~active].count_nonzero() == 0
    assert gradients[1][:, ~active].count_nonzero() == 0
    assert model.weight.grad is None
    assert scorer.identity["allow_lossy_charge"] is True
    assert scorer.close()["frozen_unchanged"]
    # close is a verification boundary; later waves/hard scoring can reuse it.
    scorer.score_at(graph, active)


def test_hard_scoring_distinguishes_same_molecule_in_different_pose_slots(tmp_path):
    prepared, model, scorer = fixture()
    left = hard_graph_record(prepared.seed)
    right = deepcopy(left)
    right["atoms"][:2] = [2, 1]
    left_key, right_key = hard_pose_key(left), hard_pose_key(right)
    assert left_key != right_key
    records = [dict(smiles="CN", hard_graph=graph, scoring_key=hard_pose_key(graph))
               for graph in (left, right, left)]
    results = scorer.score_valid_records(tmp_path, records)
    assert len(results) == 3 and model.calls == 2
    assert results[0]["loss"] != results[1]["loss"]
    assert results[0]["loss"] == results[2]["loss"]
    assert results[0]["details"]["dbmol_total_loss"] == results[0]["loss"]
    assert results[0]["scoring_key"] == left_key and results[1]["scoring_key"] == right_key
    import json
    summary = json.loads((tmp_path / "fresh/summary.json").read_text())
    assert summary["unique_pose_graphs"] == 2 and summary["unique_molecules"] == 1


def test_hard_key_ignores_padding_but_keeps_occupied_order():
    _, _, scorer = fixture()
    record = hard_graph_record(scorer.prepared.seed)
    moved = dict(atoms=[0, 1, 0, 2], charges=[2, 2, 2, 2], bonds=[[0]*4 for _ in range(4)])
    moved["bonds"][1][3] = moved["bonds"][3][1] = 1
    assert hard_pose_key(moved) == hard_pose_key(record)
    moved["atoms"][1] = 2
    assert hard_pose_key(moved) != hard_pose_key(record)


def test_scorer_rejects_smiles_only_wrong_graph_identity_and_mutation(tmp_path):
    prepared, _, scorer = fixture()
    with pytest.raises(ValueError, match="hard_graph"):
        scorer.score_valid_records(tmp_path, [dict(smiles="CN")])
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="declared valid"):
        scorer.score_valid_records(other, [dict(smiles="CC", hard_graph=hard_graph_record(prepared.seed))])
    with torch.no_grad():
        scorer.scorer.weight.add_(1)
    with pytest.raises(ValueError, match="model state changed"):
        scorer.close()


def test_execution_defaults_and_serial_full_input_gradient_probe():
    prepared, _, scorer = fixture()
    assert scorer.execution == dict(tf32=False, autocast="none", checkpoint_every=1, checkpoint_preserve_rng_state=True)
    assert scorer.identity["execution"]["numerical_variant"] is False
    result = scorer.probe_value_and_grad(prepared.seed.to_relaxed(prepared.spec), prepared.active)
    assert set(result["gradients"]) == {"atoms", "charges", "bonds"}
    assert all(gradient.isfinite().all() for gradient in result["gradients"].values())
    assert all(norm > 0 for norm in result["gradient_norms"].values())
    assert result["peak_allocated_bytes"] is None and result["peak_reserved_bytes"] is None
    assert result["total_seconds"] >= result["forward_seconds"] >= 0


def test_search_can_reject_unstable_graph_identity_and_score_next_candidate(tmp_path, monkeypatch):
    import latentmol.guidance.boltz_search as module
    from latentmol.chemistry import DecodeResult

    prepared, model, scorer = fixture()
    # Graph inspection produced this literal identity at ERK2 wave 7, child 7.
    # RDKit assigns different aromatic bonds when reading its own SMILES back.
    unstable = "COc1cnc(N2c3ccc(F)c2c3C2c3c(N)c(N)n(C)c32)nn1"
    inspect = module.inspect_graph

    def inspection(atoms, charges, bonds, spec):
        if atoms[0].item() == 2:
            return DecodeResult(smiles=unstable)
        return inspect(atoms, charges, bonds, spec)

    monkeypatch.setattr(module, "inspect_graph", inspection)
    graph = hard_graph_record(prepared.seed)
    second_pose = dict(graph, atoms=[2, 1, 0, 0])
    rows = scorer.score_valid_records(tmp_path, [dict(smiles=unstable, hard_graph=second_pose),
                                                  dict(smiles="CN", hard_graph=graph)],
                                      reject_unstable_identity=True)
    assert rows[0]["status"] == "rejected_unstable_identity" and rows[0]["loss"] is None
    assert rows[0]["smiles"] == unstable and rows[0]["details"]["reparsed_smiles"] != unstable
    assert rows[1]["status"] == "scored" and model.calls == 1
    other = tmp_path / "mismatch"
    other.mkdir()
    with pytest.raises(ValueError, match="declared valid"):
        scorer.score_valid_records(other, [dict(smiles="CC", hard_graph=second_pose)],
                                   reject_unstable_identity=True)


@pytest.mark.parametrize("options", [dict(tf32=1), dict(autocast="float16"),
    dict(checkpoint_every=-1), dict(checkpoint_every=True), dict(compile=True),
    dict(checkpoint_preserve_rng_state="false")])
def test_rejects_ambiguous_or_unsupported_execution_options(options):
    with pytest.raises(ValueError):
        resolve_execution(options)


def test_tf32_is_explicit_and_rejects_conflicting_process_global_flags():
    prepared, _, scorer = fixture(dict(tf32=True))
    assert scorer.execution_record["numerical_variant"] is True
    assert torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32
    scorer.score_at(prepared.seed.to_relaxed(prepared.spec), prepared.active)
    torch.backends.cuda.matmul.allow_tf32 = False
    with pytest.raises(ValueError, match="Process-global TF32"):
        scorer.score_at(prepared.seed.to_relaxed(prepared.spec), prepared.active)


class TracedLinear(nn.Linear):
    def forward(self, value):
        result = super().forward(value)
        self.inner_output_dtype = result.dtype
        return result


class CheckpointMock(MockDBMol):
    def __init__(self):
        super().__init__()
        from latentmol.boltz.model import checkpoint_eval_block
        self.layers = nn.ModuleList([TracedLinear(1, 1, bias=False) for _ in range(4)])
        for layer in self.layers:
            layer.eval().requires_grad_(False)
            checkpoint_eval_block(layer)

    def forward(self, features, types, coordinates):
        geometry = torch.bmm(torch.eye(2)[None], coordinates.reshape(1, 2, 3))
        self.geometry_dtype = geometry.dtype
        outputs = super().forward(features, types, coordinates)
        value = outputs["dbmol_total_loss"].reshape(1, 1)
        for layer in self.layers:
            value = torch.tanh(layer(value))
        return outputs | dict(dbmol_total_loss=value.reshape(()))


def test_checkpoint_cadence_changes_recomputation_only_and_preserves_state():
    torch.manual_seed(71)
    prepared, baseline_model, baseline = fixture(model_factory=CheckpointMock)
    graph = prepared.seed.to_relaxed(prepared.spec)
    expected = baseline.probe_value_and_grad(graph, prepared.active)
    state = deepcopy(baseline_model.state_dict())
    for every, kept in ((0, 0), (2, 2)):
        torch.manual_seed(71)
        _, model, scorer = fixture(dict(checkpoint_every=every), CheckpointMock)
        model.load_state_dict(state)
        measured = scorer.probe_value_and_grad(graph, prepared.active)
        assert measured["loss"] == expected["loss"]
        assert scorer.execution_record["checkpoint_policy"]["checkpointed_blocks"] == kept
        assert scorer.execution_record["numerical_variant"] is False
        for key in measured["gradients"]:
            torch.testing.assert_close(measured["gradients"][key], expected["gradients"][key], rtol=0, atol=0)
        assert scorer.close()["frozen_unchanged"]


def test_rng_preservation_opt_out_uses_eval_fixed_msa_and_retains_mock_gradients():
    torch.manual_seed(71)
    prepared, _, baseline = fixture(model_factory=CheckpointMock)
    expected = baseline.probe_value_and_grad(prepared.seed.to_relaxed(prepared.spec), prepared.active)
    torch.manual_seed(71)
    _, model, scorer = fixture(dict(checkpoint_preserve_rng_state=False), CheckpointMock)
    actual = scorer.probe_value_and_grad(prepared.seed.to_relaxed(prepared.spec), prepared.active)
    assert actual["loss"] == expected["loss"]
    for name in actual["gradients"]:
        torch.testing.assert_close(actual["gradients"][name], expected["gradients"][name], rtol=0, atol=0)
    assert scorer.execution_record["checkpoint_policy"]["preserve_rng_state"] is False
    assert scorer.close()["frozen_unchanged"]
    model.subsample_msa = True
    with pytest.raises(ValueError, match="fixed MSA"):
        BoltzSearchScorer(prepared, model, source_sha256="a"*64,
                          execution=dict(checkpoint_preserve_rng_state=False))


class BF16ReductionMock(MockDBMol):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 1, bias=False).eval().requires_grad_(False)
        self.reduction_dtypes = None

    def forward(self, features, types, coordinates):
        head = self.linear(features["ref_element"].float()).sum()
        self.head_dtype = head.dtype
        return self._reduce(dict(head=head), dict(head=head), head[None], features, pair=head[None])

    def _reduce(self, affinity, structure, distogram, features, *, pair=None):
        self.reduction_dtypes = affinity["head"].dtype, structure["head"].dtype, distogram.dtype, pair.dtype
        return dict(dbmol_total_loss=affinity["head"] + structure["head"] + distogram.sum() + pair.sum())


def test_bf16_autocast_keeps_input_weights_and_reduction_boundary_fp32():
    prepared, model, scorer = fixture(dict(autocast="bfloat16"), BF16ReductionMock)
    graph = prepared.seed.to_relaxed(prepared.spec)
    loss, _ = scorer.score_at(graph, prepared.active)
    assert model.head_dtype == torch.bfloat16
    assert model.reduction_dtypes == (torch.float32,)*4
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert graph.atoms.dtype == torch.float32 and loss.dtype == torch.float32
    assert scorer.execution_record["numerical_variant"] is True
    assert scorer.close()["frozen_unchanged"]


def test_block_only_bf16_keeps_parent_geometry_and_model_outputs_fp32():
    prepared, model, scorer = fixture(dict(autocast="bfloat16_blocks"), CheckpointMock)
    loss, _ = scorer.score_at(prepared.seed.to_relaxed(prepared.spec), prepared.active)
    assert model.geometry_dtype == torch.float32
    assert all(layer.inner_output_dtype == torch.bfloat16 for layer in model.layers)
    assert loss.dtype == torch.float32
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert scorer.execution_record["checkpoint_policy"]["neural_block_autocast"] is True
    assert scorer.execution_record["numerical_variant"] is True
    assert scorer.close()["frozen_unchanged"]


def test_local_gradient_preserves_exact_graph_derivative_and_uses_native_hard_scores(tmp_path):
    from latentmol.guidance.boltz_gradient_reuse import LocalBoltzGradient
    from latentmol.representation import RelaxedGraph

    prepared, model, scorer = fixture()
    wrapper = LocalBoltzGradient(scorer)
    hard = prepared.seed.to_relaxed(prepared.spec)
    graph = RelaxedGraph(*(v.clone().requires_grad_() for v in (hard.atoms, hard.charges, hard.bonds)))
    inputs = (graph.atoms, graph.charges, graph.bonds)
    expected, _ = scorer.score_at(graph, prepared.active)
    expected_grads = torch.autograd.grad(expected, inputs)
    for step, kind in enumerate(("exact", "linear", "exact")):
        value, aux = wrapper.score_branch_step(graph, prepared.active, branch="b1", step=step, terminal=step == 2)
        assert aux["gradient_reuse"]["kind"] == kind
        torch.testing.assert_close(value, expected)
        for got, wanted in zip(torch.autograd.grad(value, inputs), expected_grads, strict=True):
            torch.testing.assert_close(got, wanted)
    assert model.calls == 3  # the comparison forward and the two anchors
    rows = wrapper.score_valid_records(tmp_path, [dict(smiles="CN", hard_graph=hard_graph_record(prepared.seed))])
    assert model.calls == 4 and rows[0]["loss"] == float(expected.detach())
    counts = wrapper.diagnostics()
    assert (counts["full_gradient_calls"], counts["proxy_calls"], counts["hard_fp32_forwards"]) == (2, 1, 1)
    with pytest.raises(ValueError, match="consecutive states"):
        wrapper.score_branch_step(graph, prepared.active, branch="b2", step=1, terminal=False)
    _, first = wrapper.score_branch_step(graph, prepared.active, branch="b2", step=0, terminal=True)
    assert first["gradient_reuse"]["reason"] == "initial"
    assert wrapper.close()["frozen_unchanged"]


def test_local_gradient_checks_curvature_and_refreshes_outside_shrunk_radius(monkeypatch):
    from latentmol.guidance.boltz_gradient_reuse import LocalBoltzGradient
    from latentmol.representation import RelaxedGraph

    prepared, _, scorer = fixture()
    seed = prepared.seed.to_relaxed(prepared.spec)
    def curved_score(graph, active):
        return (100 * graph.atoms[0, 0, 2].square() +
                graph.charges.sum() * 0 + graph.bonds.sum() * 0), {}

    monkeypatch.setattr(scorer, "score_at", curved_score)
    wrapper = LocalBoltzGradient(scorer, max_graph_distance=0.3, absolute_tolerance=1e-6)
    def observe(step, amount):
        atoms = seed.atoms.clone()
        atoms[0, 0, 1:3] = torch.tensor([1 - amount, amount])
        graph = RelaxedGraph(*(v.clone().requires_grad_() for v in (atoms, seed.charges, seed.bonds)))
        return wrapper.score_branch_step(graph, prepared.active, branch="test", step=step, terminal=False)[1]["gradient_reuse"]

    observe(0, 0.0)
    assert observe(1, .05)["kind"] == "linear"
    refresh = observe(2, .10)
    assert refresh["predicted_reduction"] == 0
    assert refresh["actual_reduction"] == pytest.approx(-1.0)
    assert not refresh["agreement"] and refresh["radius_after"] == pytest.approx(.15)
    assert observe(3, .30)["reason"] == "radius"
