"""HPC-only synthetic observer fixtures; no native weights or local numerics."""
from copy import deepcopy

import pytest
import torch

from latentmol.boltz.native_head_gradient_diagnostic import (annotate_rows, channel_name,
    diagnostic_helpers, head_helpers, isolated_head_vjps, native_rounding_budget,
    require_complete_head_sum, selected_case)
from test_head_causal_diagnostic import Toy, _point


def test_isolated_channels_sum_to_binding_including_distogram_pde_dependency():
    head = head_helpers()
    model, point = Toy(), _point()
    with head.capture_heads(model) as heads:
        _, binding = model(point)
        direct, = torch.autograd.grad(binding, point, retain_graph=True)
        _, cotangents, _ = head.cut_vjp(binding, heads, point)
        isolated = isolated_head_vjps(heads, cotangents, point)
        assert require_complete_head_sum(direct, isolated)["engineering_chain_complete"]
        assert all(float(value.abs().sum()) > 0 for value in isolated.values())
        broken = dict(isolated)
        broken["distogram"] = torch.zeros_like(point)
        assert not require_complete_head_sum(direct, broken)["engineering_chain_complete"]
        del broken["distogram"]
        with pytest.raises(ValueError, match="membership"):
            require_complete_head_sum(direct, broken)


def test_channel_readouts_match_their_separate_vjps():
    head = head_helpers()
    model, point = Toy(), _point().double().detach().requires_grad_()
    with head.capture_heads(model) as heads:
        _, binding = model(point)
        _, cotangents, _ = head.cut_vjp(binding, heads, point)
        isolated = isolated_head_vjps(heads, cotangents, point)
        base = {key: value.detach().clone() for key, value in heads.items()}
        for key in head.HEADS:
            readout = ((heads[key] - base[key]) * cotangents[key]).sum()
            observed, = torch.autograd.grad(readout, point, retain_graph=True)
            torch.testing.assert_close(observed, isolated[key], rtol=0, atol=0)
    epsilon = 1e-5
    direction = torch.arange(1, point.numel() + 1, dtype=point.dtype).reshape_as(point)
    direction /= direction.norm()
    values = []
    for sign in (1, -1):
        with torch.no_grad(), head.capture_heads(model) as heads:
            model(point + sign * epsilon * direction)
            _, contributions = head.fixed_readout(heads, base, cotangents)
            values.append(contributions)
    for key in head.HEADS:
        numerical = (values[0][key] - values[1][key]) / (2 * epsilon)
        analytic = float((isolated[key] * direction).sum())
        assert numerical == pytest.approx(analytic, rel=1e-7, abs=1e-8)


def test_native_budget_uses_native_spacing_and_absolute_cotangents():
    value = torch.tensor([1., -1., 0.], dtype=torch.float32)
    cotangent = torch.tensor([2., -3., 0.])
    row = native_rounding_budget(value, cotangent)
    assert row["native_dtype"] == "torch.float32"
    assert row["weighted_half_ulp_sum"] == pytest.approx(2.5 * 2**-23)
    assert row["maximum_native_spacing"] == 2**-23
    assert native_rounding_budget(value, torch.zeros_like(value))["weighted_half_ulp_sum"] == 0
    wide = native_rounding_budget(value.double(), cotangent.double())
    assert wide["weighted_half_ulp_sum"] < row["weighted_half_ulp_sum"]
    with pytest.raises(ValueError, match="Nonfinite"):
        native_rounding_budget(torch.tensor([float("nan")]), torch.ones(1))


def test_endpoint_budget_and_channel_cancellation_are_observational_only():
    head = head_helpers()
    terms = {channel_name(key): {"absolute_error": .4, "analytic_dot_float64": (-1.)**index,
                               "central_difference": 2 * (-1.)**index, "passed": False}
             for index, key in enumerate(head.HEADS)}
    endpoint = {"native_head_rounding": {key: {"weighted_half_ulp_sum": .01} for key in head.HEADS}}
    row = {"epsilon": .1, "terms": deepcopy(terms), "plus": endpoint, "minus": endpoint}
    report = {"finite_differences": [row]}
    annotate_rows(report)
    accounting = row["isolated_head_accounting"]
    assert accounting["sum_channel_analytic_dot_float64"] == 0
    assert accounting["sum_absolute_channel_analytic"] == 6
    assert accounting["rounding_only"]["distogram"]["central_slope_final_rounding_budget"] == pytest.approx(.1)
    assert accounting["rounding_only"]["distogram"]["observed_error_exceeds_final_rounding_budget"]
    assert row["terms"] == terms  # A rounding classification never overrides FD.


def test_helper_override_is_scoped_to_observer_and_restored_on_failure():
    head = head_helpers()
    before = head.baseline_probe, head.endpoint_probe
    with pytest.raises(RuntimeError, match="fixture"):
        with diagnostic_helpers():
            assert (head.baseline_probe, head.endpoint_probe) != before
            raise RuntimeError("fixture")
    assert (head.baseline_probe, head.endpoint_probe) == before


def test_incomplete_head_sum_is_retained_then_aborts_before_endpoint_probes(monkeypatch):
    from latentmol.boltz.native_head_gradient_diagnostic import probe_case
    head = head_helpers()
    report, retained = {}, []
    def incomplete(*args, report, persist, **kwargs):
        report["baseline"] = {"isolated_head_sum": {"engineering_chain_complete": False}}
        persist()
        pytest.fail("An incomplete cut must stop before any endpoint")
    monkeypatch.setattr(head, "probe_case", incomplete)
    with pytest.raises(ValueError, match="no endpoint localization"):
        probe_case(report=report, persist=lambda: retained.append(deepcopy(report)))
    assert retained[-1]["isolated_head_status"] == "incomplete"
    assert retained[-1]["engineering_isolated_head_sum_complete"] is False


def test_only_original_prior5_step20_is_selected():
    plan = {"cases": [{"id": "prior-0", "step": 20}, {"id": "prior-3", "step": 50}, {"id": "prior-5", "step": 20}]}
    assert selected_case(plan) == plan["cases"][2]
    plan["cases"][2]["step"] = 21
    with pytest.raises(ValueError, match="prior-5"):
        selected_case(plan)
