"""Arithmetic/provenance fixtures only; native diagnosis remains an HPC job."""

import torch

from latentmol.boltz.active_set_diagnostic import (finite_difference_row, tensor_description,
                                                  realized_step, repeat_comparison, gate_paths, EPSILONS,
                                                  probe_schedule, runtime_description, rng_digest,
                                                  prepared_mask_report)
import pytest


def test_float32_spacing_and_small_signal_are_retained():
    x = torch.tensor(-0.079055354, dtype=torch.float32)
    description = tensor_description(x, scalar=True)
    assert description["dtype"] == "torch.float32"
    assert description["float32_ulp"] == description["output_ulp"]
    assert description["float32_ulp"] > 0
    row = finite_difference_row({"loss": description}, {"loss": description},
                                8.6e-6, .001, rtol=.05, atol=1e-6)
    assert row["central_difference"] == 0
    assert row["difference_in_float32_ulps"] == 0
    assert row["expected_signal_in_float32_ulps"] < 3
    assert not row["passed_original_tolerance"]


def test_fd_uses_original_tolerance_without_automatic_precision_excuse():
    def scalar(value):
        return {"loss": tensor_description(torch.tensor(value, dtype=torch.float64), scalar=True)}
    row = finite_difference_row(scalar(1.002), scalar(.998), .2, .01, rtol=.05, atol=1e-6)
    assert row["passed_original_tolerance"]
    assert abs(row["central_difference"] - .2) < 1e-12
    failed = finite_difference_row(scalar(1.002), scalar(.998), .3, .01, rtol=.05, atol=1e-6)
    assert not failed["passed_original_tolerance"]
    assert failed["relative_error"] > .3


def test_realized_step_does_not_hide_rounded_away_float32_perturbation():
    point = torch.tensor([1e8], dtype=torch.float32)
    plus, minus, row = realized_step(point, torch.ones_like(point), torch.ones_like(point), .001)
    assert torch.equal(plus, point) and torch.equal(minus, point)
    assert row["realized_plus_projection"] == row["realized_minus_projection"] == 0
    assert row["actual_secant_linear_prediction"] == 0
    assert row["analytic_dot_float64"] == 1


def test_repeat_comparison_exposes_mode_bias_and_within_mode_noise():
    values = [(False, 1.0), (False, 1.0), (False, 1.125),
              (True, 2.0), (True, 2.0), (True, 2.0)]
    repeats = [{"grad_enabled": mode, "loss": {"value": value}} for mode, value in values]
    row = repeat_comparison(repeats, 1.5)
    assert row["no_grad_range"] == .125
    assert row["grad_enabled_range"] == 0
    assert row["first_grad_minus_no_grad"] == 1
    assert row["first_grad_minus_original"] == .5


def test_toy_derivative_detects_wrong_backward_even_with_perfect_repeatability():
    class WrongBackward(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            ctx.save_for_backward(value)
            return value.square()

        @staticmethod
        def backward(ctx, output_gradient):
            value, = ctx.saved_tensors
            return 3 * value * output_gradient

    x = torch.tensor(2., dtype=torch.float64, requires_grad=True)
    correct, = torch.autograd.grad(x.square(), x)
    wrong, = torch.autograd.grad(WrongBackward.apply(x), x)
    epsilon = .001
    plus = {"loss": tensor_description((x.detach() + epsilon).square(), scalar=True)}
    minus = {"loss": tensor_description((x.detach() - epsilon).square(), scalar=True)}
    assert finite_difference_row(plus, minus, float(correct), epsilon,
                                 rtol=.05, atol=1e-6)["passed_original_tolerance"]
    assert not finite_difference_row(plus, minus, float(wrong), epsilon,
                                     rtol=.05, atol=1e-6)["passed_original_tolerance"]


def test_one_sided_slopes_expose_curvature_without_reclassifying_a_gate():
    def score(value):
        return {"loss": tensor_description(torch.tensor(value, dtype=torch.float64), scalar=True)}
    row = finite_difference_row(score(1.04), score(.98), .3, .1,
                                rtol=.05, atol=1e-6, baseline=1.)
    assert row["forward_slope"] == pytest.approx(.4)
    assert row["backward_slope"] == pytest.approx(.2)
    assert row["forward_backward_slope_difference"] == pytest.approx(.2)
    assert row["central_second_difference"] == pytest.approx(2.)


def test_saved_gate_paths_support_fixed_control_and_transition_steps(tmp_path):
    gate, state = gate_paths(tmp_path, "fixed_initial_mask", 4)
    assert gate == tmp_path / "fixed_initial_mask/gate-0004.json"
    assert state == tmp_path / "fixed_initial_mask/state-0004.pt"
    for arm, step in (("../active_set", 0), ("active_set", True), ("active_set", -1),
                      ("active_set", 10000)):
        with pytest.raises(ValueError):
            gate_paths(tmp_path, arm, step)


def test_replay_grid_contains_small_and_large_steps_but_no_production_override():
    assert min(EPSILONS) == 1e-5 and max(EPSILONS) == .1
    assert all(value in EPSILONS for value in (3e-5, 1e-4, 3e-4, .001, .003, .01, .03))


def test_focused_cpu_profile_has_fixed_small_budget_and_both_forward_modes():
    schedule = probe_schedule("focused_cpu")
    assert schedule["epsilons"] == (1e-5, 1e-4, .001)
    assert schedule["random_directions"] == 0
    assert schedule["base_repeats_per_mode"] == 2 and schedule["backward_repeats"] == 1
    assert schedule["both_modes_at_all_epsilons"] is True
    # Two native parity calls + four base calls + six pairs of FD endpoints.
    assert 2 + 2 * schedule["base_repeats_per_mode"] + 4 * len(schedule["epsilons"]) == 18
    with pytest.raises(ValueError):
        probe_schedule("guess")


def test_cpu_runtime_and_rng_description_never_query_a_cuda_device(monkeypatch):
    def reject(*args, **kwargs):
        raise AssertionError("CPU diagnostic must not query a CUDA device")
    for name in ("get_device_name", "get_device_capability", "get_rng_state"):
        monkeypatch.setattr(torch.cuda, name, reject)
    report = runtime_description("cpu")
    assert report["backend"] == "cpu" and report["torch_num_threads"] > 0
    assert "cuda_device_name" not in report
    assert isinstance(rng_digest("cpu"), str)


def test_fully_masked_padding_is_distinct_from_receptor_cross_pair_masking():
    features = {"token_pad_mask": torch.tensor([[1, 1, 1, 0]]),
                "atom_pad_mask": torch.tensor([[1, 1, 1, 1, 0, 0]]),
                "mol_type": torch.tensor([[0, 0, 1, 0]]),
                "affinity_token_mask": torch.tensor([[0, 0, 1, 0]])}
    report = prepared_mask_report(features)
    assert report["token_pad_zero"] == 1
    assert report["atom_pad_zero"] == 2 and report["atom_pad_nonzero"] == 4
    for name in ("trunk_pair_attention_key_mask", "affinity_pair_attention_key_mask"):
        assert report[name]["fully_masked_rows"] == 1
        assert report[name]["fully_masked_nonpadding_rows"] == 0
        assert report[name]["fully_masked_receptor_rows"] == 0
        assert report[name]["fully_masked_token_indices"] == [[3]]
