"""HPC-only synthetic affinity-cut fixtures; no native checkpoint/network."""
from copy import deepcopy
import weakref

import pytest
import torch
from torch import nn

from latentmol.boltz.affinity_causal_diagnostic import (EXACT_MODES, LEVELS, MODES,
    activation_observations, assert_detached_tree, baseline_ready, capture_affinity, detached_phase, diagnostic_helpers,
    evaluate_members, exact_parity, fixed_linear, head_helpers, key, selected_case,
    require_tensor_parity, sequential, validate_architecture)


class SmallHead(nn.Module):
    """Native topology with deterministic, interpretable fixture weights."""
    def __init__(self, scale=1., dtype=torch.float64):
        super().__init__()
        self.affinity_out_mlp = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2), nn.ReLU())
        self.to_affinity_pred_value = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1))
        self.to_affinity_pred_score = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1))
        self.to_affinity_logits_binary = nn.Linear(1, 1)
        self.to(dtype=dtype)
        with torch.no_grad():
            for layer in self.modules():
                if isinstance(layer, nn.Linear):
                    layer.weight.zero_()
                    layer.bias.zero_()
                    if layer.weight.shape == (2, 2):
                        layer.weight.copy_(torch.eye(2, dtype=dtype))
                    else:
                        layer.weight.fill_(scale)
        self.eval().requires_grad_(False)

    def forward(self, pooled):
        shared = self.affinity_out_mlp(pooled)
        return self.to_affinity_pred_value(shared)


def held_masks(captured):
    return [{block: tuple(captured[key(member, f"{block}_relu_{index}")].detach() > 0 for index in (0, 1))
             for block in ("shared", "regression")} for member in range(2)]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cached_and_held_sign_baselines_have_exact_native_values_and_latent_gradients(dtype):
    heads = (SmallHead(1., dtype), SmallHead(-.25, dtype))
    validate_architecture(heads)
    point = torch.tensor([[0., 1.]], dtype=dtype, requires_grad=True)
    with capture_affinity(heads) as captured:
        values = [module(point) for module in heads]
        native = torch.stack(values).mean()
        gradient, = torch.autograd.grad(native, point, retain_graph=True)
        masks = held_masks(captured)
        for mode in MODES:
            cached = torch.stack(evaluate_members(heads, captured, masks, mode)).mean()
            observed, = torch.autograd.grad(cached, point, retain_graph=True)
            parity = exact_parity(native, cached, gradient, observed)
            assert parity["scalar_exact"] and parity["gradient_exact"]
        # Held zero gate has exactly native ReLU's zero derivative.
        assert gradient[0, 0] == 0


def test_each_two_member_boundary_is_complete_but_ancestor_cuts_are_not_additive():
    heads = (SmallHead(1.), SmallHead(-.25))
    point = torch.tensor([[.5, 1.]], dtype=torch.float64, requires_grad=True)
    with capture_affinity(heads) as captured:
        # Actual members have distinct upstream pooled tensors. Reusing the
        # identical point object twice would not represent a two-branch cut.
        native = torch.stack([module(point + 0.) for module in heads]).mean()
        direct, = torch.autograd.grad(native, point, retain_graph=True)
        observed = []
        for level in LEVELS:
            tensors = tuple(captured[key(member, level)] for member in range(2))
            cotangents = torch.autograd.grad(native, tensors, retain_graph=True)
            parts = [torch.autograd.grad(value, point, grad_outputs=cotangent, retain_graph=True)[0]
                     for value, cotangent in zip(tensors, cotangents)]
            assert head_helpers().chain_completeness(direct, sum(parts))["engineering_chain_complete"]
            assert not head_helpers().chain_completeness(direct, parts[0])["engineering_chain_complete"]
            observed.append(sum(parts))
            for value, cotangent in zip(tensors, cotangents):
                readout = fixed_linear(value, value.detach(), cotangent.detach())
                analytic, = torch.autograd.grad(readout, point, retain_graph=True)
                reference, = torch.autograd.grad(value, point, grad_outputs=cotangent, retain_graph=True)
                assert torch.equal(analytic, reference)
        assert not head_helpers().chain_completeness(direct, sum(observed))["engineering_chain_complete"]


def test_held_branch_counterfactual_explains_kink_but_does_not_certify_native_endpoints():
    heads = (SmallHead(), SmallHead())
    point = torch.tensor([[1e-4, 1.]], dtype=torch.float64, requires_grad=True)
    with capture_affinity(heads) as captured:
        native = torch.stack([module(point) for module in heads]).mean()
        direct, = torch.autograd.grad(native, point, retain_graph=True)
        base = {name: value.detach().clone() for name, value in captured.items()}
        masks = held_masks(captured)
        pre = {name: value for name, value in captured.items() if "_relu_" in name}
        cotangents = dict(zip(pre, torch.autograd.grad(native, tuple(pre.values()), retain_graph=True)))
    values, crossings = {}, []
    for sign in (1, -1):
        endpoint = point.detach() + torch.tensor([[sign * .001, 0.]], dtype=point.dtype)
        with torch.no_grad(), capture_affinity(heads) as captured:
            originals = [module(endpoint) for module in heads]
            values[sign] = {"original": torch.stack(originals).mean()}
            for mode in ("float64_pooled", "frozen_both", "frozen_shared", "frozen_regression"):
                values[sign][mode] = torch.stack(evaluate_members(heads, captured, masks, mode)).mean()
            crossings.append(activation_observations(captured, base, cotangents))
    slopes = {name: float((values[1][name] - values[-1][name]) / .002) for name in values[1]}
    assert slopes["original"] == pytest.approx(.55)
    assert slopes["float64_pooled"] == pytest.approx(.55)  # Precision cannot remove this branch crossing.
    assert slopes["frozen_both"] == pytest.approx(float(direct[0, 0]))
    assert slopes["frozen_shared"] == pytest.approx(.55)
    assert slopes["frozen_regression"] == pytest.approx(.55)
    assert sum(row["sign_crossings"] for row in crossings[1].values()) == 8


def test_widening_small_head_arithmetic_cannot_recover_already_rounded_upstream_inputs():
    layer = nn.Sequential(nn.Linear(1, 1)).eval().requires_grad_(False)
    with torch.no_grad():
        layer[0].weight.fill_(1.)
        layer[0].bias.fill_(2**24)
    endpoints = [torch.tensor([[sign * .25]], dtype=torch.float32) for sign in (1, -1)]
    native = [sequential(layer, value) for value in endpoints]
    wide = [sequential(layer, value, double=True) for value in endpoints]
    assert torch.equal(native[0], native[1])
    assert float((wide[0] - wide[1]) / .5) == 1.
    # Move the rounding upstream of the observed cut. Widening the cache now
    # sees identical inputs and cannot recreate the lost endpoint difference.
    with torch.no_grad():
        layer[0].bias.zero_()
    upstream = [value + 2**24 for value in endpoints]
    wide_upstream = [sequential(layer, value, double=True) for value in upstream]
    assert torch.equal(wide_upstream[0], wide_upstream[1])


def test_architecture_and_capture_fail_closed_and_hooks_restore_after_error():
    heads = (SmallHead(), SmallHead())
    with pytest.raises(ValueError, match="both native"):
        validate_architecture(heads[:1])
    heads[0].affinity_out_mlp[1] = nn.GELU()
    with pytest.raises(ValueError, match="Unexpected affinity"):
        validate_architecture(heads)
    heads = (SmallHead(), SmallHead())
    point = torch.ones(1, 2, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="fixture"):
        with capture_affinity(heads):
            for module in heads:
                module(point)
            raise RuntimeError("fixture")
    assert all(not module._forward_hooks and not module._forward_pre_hooks
               for head in heads for module in head.modules())
    with pytest.raises(ValueError, match="unexpectedly replayed"):
        with capture_affinity(heads):
            heads[0](point)
            heads[0](point)


def test_observer_overrides_restore_even_on_error():
    helper = head_helpers()
    before = helper.baseline_probe, helper.endpoint_probe
    with pytest.raises(RuntimeError, match="fixture"):
        with diagnostic_helpers():
            assert (helper.baseline_probe, helper.endpoint_probe) != before
            raise RuntimeError("fixture")
    assert (helper.baseline_probe, helper.endpoint_probe) == before


def valid_baseline():
    return {"affinity_cut_completeness": {name: {"engineering_chain_complete": True} for name in LEVELS},
        "cached_member_sum_completeness": {name: {"engineering_chain_complete": True} for name in MODES},
        "cached_exact_baseline_parity": {name: {"scalar_exact": True, "gradient_exact": True,
            "member_values_exact": True, "member_gradients_exact": True} for name in EXACT_MODES}}


def test_all_baseline_controls_required_and_failure_retained_before_endpoints(monkeypatch):
    from latentmol.boltz.affinity_causal_diagnostic import probe_case
    baseline = valid_baseline()
    assert baseline_ready(baseline)
    del baseline["cached_exact_baseline_parity"]["frozen_both"]
    assert not baseline_ready(baseline)
    baseline = valid_baseline()
    baseline["cached_exact_baseline_parity"]["frozen_both"]["gradient_exact"] = False
    helper, report, retained = head_helpers(), {}, []
    def incomplete(*args, report, persist, **kwargs):
        report["baseline"] = baseline
        persist()
        pytest.fail("Invalid baseline must abort before endpoints")
    monkeypatch.setattr(helper, "probe_case", incomplete)
    with pytest.raises(ValueError, match="without endpoint attribution"):
        probe_case(report=report, persist=lambda: retained.append(deepcopy(report)))
    assert retained[-1]["affinity_baseline_validated"] is False
    assert retained[-1]["affinity_status"] == "baseline_parity_or_cut_incomplete"


def test_only_original_saved_prior5_is_selected():
    plan = {"cases": [{"id": "prior-0", "step": 20}, {"id": "prior-5", "step": 20}]}
    assert selected_case(plan) == plan["cases"][1]
    plan["cases"][1]["step"] = 21
    with pytest.raises(ValueError, match="prior-5"):
        selected_case(plan)


def test_fresh_phases_match_derivatives_and_release_graph_before_next_phase():
    heads = (SmallHead(1.), SmallHead(-.25))
    point = torch.tensor([[.5, 1.]], dtype=torch.float64)
    weak_outputs, progress = [], []
    def phase():
        latent = point.detach().clone().requires_grad_()
        value = torch.stack([module(latent) for module in heads]).mean()
        weak_outputs.append(weakref.ref(value))
        gradient, = torch.autograd.grad(value, latent, retain_graph=False)
        return {"value": value.detach(), "gradient": gradient.detach()}, {"measured": True}
    first = detached_phase("first", phase, point.device, lambda label, row: progress.append((label, row)))
    assert weak_outputs[0]() is None
    second = detached_phase("second", phase, point.device, lambda label, row: progress.append((label, row)))
    assert weak_outputs[1]() is None
    assert torch.equal(first["value"], second["value"])
    assert torch.equal(first["gradient"], second["gradient"])
    assert [row["status"] for _, row in progress] == ["running", "completed", "running", "completed"]
    assert all(row["no_live_autograd_tensor_returned"] for _, row in progress if row["status"] == "completed")


def test_live_phase_returns_rejected_and_later_failure_preserves_completed_receipt():
    progress = {}
    save = lambda label, row: progress.update({label: row})
    detached_phase("first", lambda: ({"tensor": torch.ones(1)}, {"complete": True}), "cpu", save)
    def broken():
        return {"tensor": torch.ones(1, requires_grad=True).square()}, {}
    with pytest.raises(ValueError, match="retained a live"):
        detached_phase("broken", broken, "cpu", save)
    assert progress["first"]["status"] == "completed"
    assert progress["broken"]["status"] == "failed"
    assert progress["broken"]["error_type"] == "ValueError"
    with pytest.raises(ValueError, match="retained a live"):
        assert_detached_tree({"nested": [{"tensor": torch.ones(1, requires_grad=True)}]})


def test_fresh_phase_native_boundaries_reject_value_dtype_and_signed_zero_drift():
    original = {"pooled": torch.tensor([[0., 1.]])}
    assert require_tensor_parity({"pooled": original["pooled"].clone()}, original, "fixture")
    with pytest.raises(ValueError, match="tensor changed"):
        require_tensor_parity({"pooled": torch.tensor([[0., 2.]])}, original, "fixture")
    with pytest.raises(ValueError, match="tensor changed"):
        require_tensor_parity({"pooled": original["pooled"].double()}, original, "fixture")
    with pytest.raises(ValueError, match="tensor bytes changed"):
        require_tensor_parity({"pooled": torch.tensor([[-0., 1.]])}, original, "fixture")
