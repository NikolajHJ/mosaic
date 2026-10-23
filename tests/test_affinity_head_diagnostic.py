"""Allocated CPU engineering fixtures for the cached affinity-head observer.

Run on DTU HPC. These toy networks test algebra, parity and bounded planning;
they do not establish native Boltz success or approve historical failed gates.
"""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest
import torch
from torch import nn


SOURCE = Path(__file__).parents[1] / "src/latentmol/boltz/affinity_head_diagnostic.py"
SPEC = importlib.util.spec_from_file_location("_affinity_head_fixture", SOURCE)
observer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(observer)


def source093_baseline_schema():
    """Actual archived schema: graph hash on forward, gradient hash on analytic.

    This reduced metadata fixture retains the asymmetric fields that caused
    both native source099 failures. Numerical toy-head tests alone did not
    exercise the historical report schema.
    """
    values = {"affinity_logits_binary": -.3, "affinity_pred_value": 1.2,
              "dbmol_affinity_loss": 1.5, "combined": 4.}
    forward = {"repeat": 0, "grad_enabled": False, "graph_sha256": "a" * 64,
               "values": deepcopy(values), "dtypes": {name: "torch.float64" for name in values},
               "target_sha256": None, "weight": 0.}
    analytic = {"repeat": 0, "grad_enabled": True, "values": deepcopy(values),
                "loss_dtype": "torch.float64", "sum_gradient_identity": {"combined_gradient_sha256": "b" * 64},
                "scalar_assembly_identity": {"exact_combined_scalar_match": True}}
    return {"baselines": [forward, analytic]}


def test_source093_asymmetric_baseline_pair_keeps_graph_and_backward_roles_separate():
    reference = source093_baseline_schema()
    forward, analytic = observer.reference_baselines(reference)
    assert "graph_sha256" not in analytic  # Precisely the original native failure.
    assert "sum_gradient_identity" not in forward
    assert forward is reference["baselines"][0]
    assert analytic is reference["baselines"][1]
    assert forward["graph_sha256"] == "a" * 64
    assert analytic["sum_gradient_identity"]["combined_gradient_sha256"] == "b" * 64
    assert forward["values"] == analytic["values"]


@pytest.mark.parametrize("corruption", ["missing_forward", "missing_graph", "duplicate_forward",
    "different_repeat", "changed_scalar", "missing_backward_digest", "contradictory_analytic_graph"])
def test_source093_baseline_pair_rejects_missing_or_conflicting_identity(corruption):
    reference = source093_baseline_schema()
    forward, analytic = reference["baselines"]
    if corruption == "missing_forward":
        reference["baselines"].remove(forward)
    elif corruption == "missing_graph":
        del forward["graph_sha256"]
    elif corruption == "duplicate_forward":
        reference["baselines"].append(deepcopy(forward))
    elif corruption == "different_repeat":
        forward["repeat"] = 1
    elif corruption == "changed_scalar":
        forward["values"]["dbmol_affinity_loss"] += .01
    elif corruption == "missing_backward_digest":
        del analytic["sum_gradient_identity"]["combined_gradient_sha256"]
    elif corruption == "contradictory_analytic_graph":
        analytic["graph_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="source093"):
        observer.reference_baselines(reference)


class Heads(nn.Module):
    """Native topology with tiny widths and deterministic nonzero weights."""
    def __init__(self, second_bias=1.):
        super().__init__()
        self.affinity_out_mlp = nn.Sequential(nn.Linear(1, 1), nn.ReLU(), nn.Linear(1, 1), nn.ReLU())
        self.to_affinity_pred_value = nn.Sequential(nn.Linear(1, 1), nn.ReLU(), nn.Linear(1, 1), nn.ReLU(), nn.Linear(1, 1))
        self.to_affinity_pred_score = nn.Sequential(nn.Linear(1, 1), nn.ReLU(), nn.Linear(1, 1), nn.ReLU(), nn.Linear(1, 1))
        self.to_affinity_logits_binary = nn.Linear(1, 1)
        with torch.no_grad():
            for layer in self.modules():
                if isinstance(layer, nn.Linear):
                    layer.weight.fill_(1)
                    layer.bias.fill_(1)
            self.affinity_out_mlp[0].bias.fill_(-.0001)
            self.affinity_out_mlp[2].weight.fill_(2)
            self.affinity_out_mlp[2].bias.fill_(second_bias)
            self.to_affinity_pred_score[-1].weight.fill_(.2)
        self.eval().requires_grad_(False)

    def forward(self, x):
        g = self.affinity_out_mlp(x)
        value = self.to_affinity_pred_value(g).reshape(-1, 1)
        logit = self.to_affinity_logits_binary(self.to_affinity_pred_score(g)).reshape(-1, 1)
        return value, logit


def native(heads, points):
    value, logits = zip(*(head(point) for head, point in zip(heads, points)))
    return {"affinity_pred_value": torch.stack(value).mean(0),
            "affinity_logits_binary": torch.stack(logits).mean(0),
            "affinity_probability_binary": torch.stack([x.sigmoid() for x in logits]).mean(0)}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cached_native_ensemble_outputs_and_vjp_are_identical(dtype):
    heads = tuple(Heads().to(dtype=dtype) for _ in range(2))
    points = tuple(torch.tensor([[.3 + index]], dtype=dtype, requires_grad=True) for index in range(2))
    expected = native(heads, points)
    outputs, _, _ = observer.heads_forward(heads, points)
    assert set(outputs) == set(expected)
    assert all(torch.equal(outputs[key], expected[key]) for key in expected)
    direct = torch.autograd.grad(observer.head_scalar(expected, 1.7, .9), points)
    replay = torch.autograd.grad(observer.head_scalar(outputs, 1.7, .9), points)
    assert all(torch.equal(a, b) for a, b in zip(direct, replay))


@pytest.mark.parametrize("scope", ["shared", "all"])
def test_baseline_sign_held_output_and_derivative_match_but_endpoint_changes(scope):
    heads = tuple(Heads().double() for _ in range(2))
    points = tuple(torch.zeros((1, 1), dtype=torch.float64, requires_grad=True) for _ in heads)
    outputs, _, pre = observer.heads_forward(heads, points)
    masks = {name: value > 0 for name, value in pre.items()}
    held, _, _ = observer.heads_forward(heads, points, held_scope=scope, masks=masks)
    assert all(torch.equal(outputs[key], held[key]) for key in outputs)
    gradient = torch.autograd.grad(observer.head_scalar(outputs, 1., 1.), points)
    held_gradient = torch.autograd.grad(observer.head_scalar(held, 1., 1.), points)
    assert all(torch.equal(a, b) for a, b in zip(gradient, held_gradient))
    endpoint = tuple(torch.full_like(point, .002) for point in points)
    ordinary = observer.heads_forward(heads, endpoint)[0]
    changed = observer.heads_forward(heads, endpoint, held_scope=scope, masks=masks)[0]
    assert not torch.equal(ordinary["affinity_pred_value"], changed["affinity_pred_value"])


@pytest.mark.parametrize("second_bias", [1., -.001])
def test_algebraic_two_relu_contribution_matches_full_shared_counterfactual(second_bias):
    heads = tuple(Heads(second_bias).double() for _ in range(2))
    base = tuple(torch.zeros((1, 1), dtype=torch.float64) for _ in heads)
    pre = observer.heads_forward(heads, base)[2]
    masks = {name: value > 0 for name, value in pre.items()}
    endpoint = tuple(torch.full_like(x, .002) for x in base)
    cotangents = (torch.tensor([[.7]], dtype=torch.float64), torch.tensor([[-.2]], dtype=torch.float64))
    ordinary = observer.heads_forward(heads, endpoint)[1]
    held = observer.heads_forward(heads, endpoint, held_scope="shared", masks=masks)[1]
    expected = float(observer.projection(tuple(ordinary[str(i)] - held[str(i)] for i in range(2)), cotangents))
    actual = observer.shared_switch_contributions(heads, endpoint, masks, cotangents)
    assert actual["total_predicted_shared_counterfactual_difference"] == pytest.approx(expected, abs=1e-14)
    assert sum(row["contracted_switch_effect"] for row in actual["layers"].values()) == pytest.approx(expected, abs=1e-14)
    assert any(row["units"] for row in actual["layers"].values())


def test_crossings_retain_unit_indices_and_actual_values():
    baseline = {"head.0.affinity_out_mlp.1": torch.tensor([[-.1, .2, 0.]])}
    endpoint = {"head.0.affinity_out_mlp.1": torch.tensor([[.05, .1, .3]])}
    result = observer.crossings(endpoint, baseline)["head.0.affinity_out_mlp.1"]
    assert result["positive_branch_changes"] == 2
    assert [unit["index"] for unit in result["changed_units"]] == [[0, 0], [0, 2]]
    assert result["changed_units"][0]["baseline"] == pytest.approx(-.1)
    assert result["changed_units"][0]["endpoint"] == pytest.approx(.05)
    assert result["baseline_exact_zeros"] == 1


def test_native_readout_widens_unequal_magnitude_fp32_operands_before_subtraction():
    endpoint = (torch.tensor([[100_000_000.]], dtype=torch.float32),)
    baseline = (torch.tensor([[1.]], dtype=torch.float32),)
    cotangent = (torch.ones((1, 1), dtype=torch.float32),)
    observed = float(observer.centered_projection(endpoint, baseline, cotangent))
    incorrectly_rounded = float(observer.projection(tuple(a - b for a, b in zip(endpoint, baseline)), cotangent))
    assert observed == 99_999_999.
    assert incorrectly_rounded == 100_000_000.
    assert observed != incorrectly_rounded


def test_centered_secant_preserves_source097_two_scalar_reduction_order():
    plus = (torch.tensor([[1e16]], dtype=torch.float64),)
    minus = (torch.tensor([[1e16 - 2]], dtype=torch.float64),)
    baseline = (torch.tensor([[-1e16]], dtype=torch.float64),)
    cotangent = (torch.ones((1, 1), dtype=torch.float64),)
    observed = observer.centered_secant(plus, minus, baseline, cotangent, 1.)
    changed_order = float(observer.projection(tuple(a - b for a, b in zip(plus, minus)), cotangent)) / 2
    assert observed == 0.
    assert changed_order == 1.
    assert observed != changed_order


def test_open_cell_radius_is_symmetric_and_handles_zero_and_constant_units():
    result = observer.branch_radius({"a": torch.tensor([[-.1, .2]], dtype=torch.float64)},
                                    {"a": torch.tensor([[2., -.5]], dtype=torch.float64)})
    assert result["symmetric_open_cell_radius"] == pytest.approx(.05)
    assert result["limiting_unit"]["flat_index"] == 0
    assert not result["baseline_zero_with_nonzero_tangent"]
    zero = observer.branch_radius({"a": torch.tensor([[0.]])}, {"a": torch.tensor([[1.]])})
    assert zero["baseline_zero_with_nonzero_tangent"]
    constant = observer.branch_radius({"a": torch.tensor([[0.]])}, {"a": torch.tensor([[0.]])})
    assert constant["symmetric_open_cell_radius"] is None
    assert not constant["baseline_zero_with_nonzero_tangent"]


def test_tiny_head_only_tangent_agrees_in_fp64_inside_same_relu_cell():
    originals = tuple(Heads() for _ in range(2))
    heads = {name: tuple(deepcopy(head).to(dtype=getattr(torch, name)) for head in originals)
             for name in ("float32", "float64")}
    pooled = tuple(torch.tensor([[.3 + index]]) for index in range(2))
    tangent = (torch.tensor([[.7]], dtype=torch.float64), torch.tensor([[-.2]], dtype=torch.float64))
    masks = {name: value > 0 for name, value in observer.heads_forward(heads["float32"], pooled)[2].items()}
    before = [deepcopy(head.state_dict()) for head in originals]
    report = observer.tangent_probes(heads, pooled, tangent, masks, regression_weight=1.3, logit_weight=.9,
                                    rtol=.01, atol=1e-7)
    fp64 = [row for row in report["probes"] if row["dtype"] == "float64"]
    assert len(fp64) == 3
    assert all(row["same_relu_branches"] and row["head_only_same_branch_agreement"] for row in fp64)
    assert all(row["epsilon"] < report["cell"]["symmetric_open_cell_radius"] for row in report["probes"])
    assert all(torch.equal(expected[key], head.state_dict()[key]) for expected, head in zip(before, originals) for key in expected)


def test_head_only_zero_at_active_tangent_cannot_claim_symmetric_cell():
    originals = tuple(Heads().double() for _ in range(2))
    pooled = tuple(torch.zeros((1, 1), dtype=torch.float64) for _ in originals)
    with torch.no_grad():
        for head in originals:
            head.affinity_out_mlp[0].bias.zero_()
    heads = {name: tuple(deepcopy(head).to(dtype=getattr(torch, name)) for head in originals)
             for name in ("float32", "float64")}
    masks = {name: value > 0 for name, value in observer.heads_forward(heads["float64"], pooled)[2].items()}
    report = observer.tangent_probes(heads, pooled, tuple(torch.ones_like(x) for x in pooled), masks,
                                    regression_weight=1., logit_weight=1., rtol=.01, atol=1e-7)
    assert report["status"] == "no_symmetric_same_branch_neighborhood"
    assert not report["probes"]


def test_invalid_head_operations_and_missing_masks_are_rejected():
    heads = (Heads(), Heads())
    pooled = (torch.zeros((1, 1)), torch.zeros((1, 1)))
    with pytest.raises(ValueError, match="baseline masks"):
        observer.heads_forward(heads, pooled, held_scope="all")
    heads[0].affinity_out_mlp[1] = nn.Sigmoid()
    with pytest.raises(ValueError, match="Unsupported"):
        observer.heads_forward(heads, pooled)


def test_forward_budget_has_all_original_endpoints_and_controller_headroom():
    budget = observer.forward_budget([2, 4])
    assert budget["explicit_native_affinity_forwards"] == 110
    assert budget["total_native_affinity_traversals_upper_bound"] == 116
    assert budget["maximum_child_seconds"] + budget["controller_headroom_seconds"] == budget["gpu_wall_seconds"]
    assert budget["controller_headroom_seconds"] == 300
    assert budget["new_defog_calls"] == budget["optimizer_updates"] == 0
    with pytest.raises(ValueError):
        observer.forward_budget([2, 5])


def test_cached_head_budgets_are_enforced_before_any_further_forward(monkeypatch):
    monkeypatch.setattr(observer, "_ENFORCE_HEAD_BUDGET", True)
    monkeypatch.setattr(observer, "_HEAD_CALLS", observer.MAXIMUM_CACHED_HEAD_CALLS)
    monkeypatch.setattr(observer, "_SWITCH_CALLS", observer.MAXIMUM_SHARED_SWITCH_CALLS)
    with pytest.raises(ValueError, match="Cached ensemble-head call budget"):
        observer.heads_forward((), ())
    with pytest.raises(ValueError, match="Shared-switch attribution call budget"):
        observer.shared_switch_contributions((), (), {}, ())


def test_unsubmitted_boundary_checkout_is_rejected_before_source_imports():
    case = {"parent_artifacts_sha256": {
        "/work3/s234818/latentmol/releases/source-new/src/latentmol/boltz/operation_boundary_diagnostic.py": "a" * 64}}
    with pytest.raises(ValueError, match="exact sealed source097"):
        observer.boundary(case)


def test_head_tolerance_is_original_and_never_approves_failed_gate():
    row = observer.scalar_comparison(4.075, 4.54, rtol=.02, atol=.0001)
    assert not row["within_original_tolerance"]
    assert row["original_tolerance"] == pytest.approx(.0001 + .02 * 4.54)
    assert row["historical_gate_approved"] is False


def test_submitter_gates_new_head_observer_without_changing_native_gpu_policy():
    scripts = Path(__file__).parents[1] / "scripts/hpc"
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location("_affinity_head_submit_fixture", scripts / "submit_affinity_head_diagnostic.py")
        submitter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(submitter)
        release = Path("/work3/s234818/latentmol/releases/source-test")
        work = Path("/work3/s234818/latentmol")
        jobs = submitter.workflow(release, work, work / "plan.json", "a" * 64, "b" * 64, "c" * 64)
        assert [row["stage"] for row in jobs] == ["cpu", "native"]
        assert "tests/test_affinity_head_diagnostic.py" in jobs[0]["job_command"]
        assert "latentmol.boltz.affinity_head_diagnostic" in jobs[1]["job_command"]
        assert jobs[1]["dependency"] == "done:cpu"
        assert jobs[1]["arguments"][jobs[1]["arguments"].index("-W") + 1] == "2:00"
        assert "select[gpu80gb]" in jobs[1]["arguments"]
    finally:
        sys.path.remove(str(scripts))
