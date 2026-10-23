"""HPC-only component-specific graph-cut fixtures; no native model weights."""
from types import SimpleNamespace

import pytest
import torch

from latentmol.boltz.affinity_decoder_causal_diagnostic import (observers,
    dual_fd_attribution, regression_graph_cut, require_complete_chain, require_decoder_baseline,
    selected_case, source087_reference)
from latentmol.boltz.local_fd import endpoint_row
from latentmol.graph_policy import graph_from_decoder
from latentmol.multimodel.study import tensor_digest
from test_decoder_causal_diagnostic import _model


def test_complete_canonical_cut_matches_direct_component_and_missing_path_fails():
    affinity, _ = observers()
    point = torch.tensor([[1., 2.]], requires_grad=True)
    fields = point * 2, point * 3, point * 4
    output = fields[0].sum() + 2 * fields[1].sum() + 3 * fields[2].sum()
    direct, chain, cotangents, unused = regression_graph_cut(output, torch.tensor(.5), fields, point)
    require_complete_chain(affinity.head_helpers().chain_completeness(direct, chain))
    assert torch.equal(direct, torch.full_like(point, 10.))
    assert all(not value.requires_grad for value in (direct, chain, *cotangents))
    assert unused == []
    point = point.detach().requires_grad_()
    fields = point * 2, point * 3, point * 4
    output = fields[0].sum() + 2 * fields[1].sum() + 3 * fields[2].sum()
    incomplete = fields[:2] + (point * 0.,)
    direct, chain, _, unused = regression_graph_cut(output, torch.tensor(.5), incomplete, point)
    assert unused == ["bonds"]
    with pytest.raises(ValueError, match="no decoder attribution"):
        require_complete_chain(affinity.head_helpers().chain_completeness(direct, chain))


def test_chain_roundoff_is_reported_separately_from_exactness_and_fd_tolerances():
    affinity, _ = observers()
    direct = torch.tensor([1., 2.])
    rounded = torch.nextafter(direct, torch.full_like(direct, float("inf")))
    record = affinity.head_helpers().chain_completeness(direct, rounded)
    assert not record["exact_equal"]
    assert record["historical_fd_tolerance_changed"] is False
    require_complete_chain(record)
    with pytest.raises(ValueError, match="incomplete"):
        require_complete_chain(affinity.head_helpers().chain_completeness(direct, direct + .01))


def test_unused_canonical_channel_is_explicit_zero_without_losing_complete_cut():
    point = torch.tensor([[1., 2.]], requires_grad=True)
    fields = point * 2, point * 3, point * 4
    direct, chain, cotangents, unused = regression_graph_cut(
        fields[0].sum() + fields[2].sum(), torch.tensor(1.), fields, point)
    assert torch.equal(direct, chain)
    assert unused == ["charges"]
    assert torch.count_nonzero(cotangents[1]) == 0
    with pytest.raises(ValueError, match="three live canonical"):
        regression_graph_cut(torch.tensor(1.), torch.tensor(1.), fields[:2], point)


def test_full_binding_decoder_agreement_can_hide_regression_specific_abs_kink():
    _, decoder = observers()
    point = torch.tensor([1e-4], dtype=torch.float64, requires_grad=True)
    fields = point.abs(), point * 1., point * 2.
    regression = fields[0].sum()
    # A second binding component cancels the decoder abs contribution. A
    # full-binding-only cut therefore does not diagnose the regression channel.
    binding = regression - fields[0].sum() + fields[1].sum()
    raw = torch.autograd.grad(binding, fields, retain_graph=True, allow_unused=True)
    binding_cot = tuple(torch.zeros_like(field) if value is None else value.detach()
                        for field, value in zip(fields, raw))
    direct, chain, regression_cot, _ = regression_graph_cut(regression, torch.tensor(1., dtype=point.dtype), fields, point)
    assert torch.equal(direct, chain)
    base = tuple(value.detach() for value in fields)
    outputs = []
    for sign in (1, -1):
        z = point.detach() + sign * .001
        graph = SimpleNamespace(atoms=z.abs(), charges=z, bonds=2 * z)
        outputs.append((decoder.fixed_readout(graph, base, binding_cot),
                        decoder.fixed_readout(graph, base, regression_cot)))
    assert float((outputs[0][0] - outputs[1][0]) / .002) == pytest.approx(1.)
    assert float((outputs[0][1] - outputs[1][1]) / .002) == pytest.approx(.1)
    assert float(direct) == 1.


def test_reused_decoder_controls_keep_same_graph_and_pointwise_component_vjp():
    _, helpers = observers()
    model = _model()
    point = torch.tensor([[.3, -.2, .4, .1]])
    active = torch.tensor([True, True, False])
    probe = point.clone().requires_grad_()
    graph = graph_from_decoder(model.decode(probe), model.spec, fixed_active=active)
    fields = tuple(getattr(graph, name) for name in ("atoms", "charges", "bonds"))
    base = tuple(value.detach() for value in fields)
    cotangents = tuple(torch.arange(value.numel(), dtype=value.dtype).reshape_as(value) / value.numel() for value in base)
    expected, = torch.autograd.grad(fields, probe, grad_outputs=cotangents)
    identity = tensor_digest(dict(zip(("atoms", "charges", "bonds"), base)))
    observer = helpers.DecoderObserver()
    with helpers.decoder_control(model, observer):
        for mode in ("original_fp32", "frozen_abs_fp32"):
            row, gradient = helpers.evaluate_decoder(model, point, active, 1., base, cotangents,
                observer, mode, gradients=True, capture=mode == "original_fp32")
            assert require_decoder_baseline(row, gradient, identity, expected)
            assert row["trace"]["crossings_with_nonzero_baseline_output_cotangent"] == 0
    assert not observer.active


def test_decoder_baseline_requires_graph_scalar_and_exact_gradient_together():
    expected = torch.tensor([1., 2.])
    row = {"graph_sha256": "a" * 64, "value": 0.}
    assert require_decoder_baseline(row, expected.clone(), row["graph_sha256"], expected)
    for wrong_row, gradient in (({**row, "value": 1e-12}, expected),
                                ({**row, "graph_sha256": "b" * 64}, expected),
                                (row, expected + .01)):
        with pytest.raises(ValueError, match="baseline graph or gradient"):
            require_decoder_baseline(wrong_row, gradient, "a" * 64, expected)


def _row(analytic, numerical=.7):
    return endpoint_row(torch.tensor([0.]), torch.tensor([1.]), torch.tensor([analytic]), .001,
        {"loss": .001 * numerical, "loss_dtype": "torch.float64"},
        {"loss": -.001 * numerical, "loss_dtype": "torch.float64"}, 0., rtol=.05, atol=1e-6)


def test_dual_fd_preserves_both_rows_and_separates_small_chain_roundoff_from_failure():
    native, chain = _row(1.), _row(1.000001)
    original = dict(native), dict(chain)
    result = dual_fd_attribution(native, chain)
    assert result["status"] == "consistent_failure_separated_from_chain_roundoff"
    assert result["failure_localization_eligible"]
    assert (native, chain) == original
    assert native["passed"] is chain["passed"] is False
    assert dual_fd_attribution(_row(1., 1.), _row(.999999, 1.))["status"] == "consistent_pass"


def test_dual_fd_ambiguous_when_classification_signal_or_chain_separation_disagrees():
    flipped = dual_fd_attribution(_row(1., 1.), _row(1.06, 1.))
    assert not flipped["classifications_agree"] and not flipped["attribution_eligible"]
    # Both remain failures, but chain differences now explain too much of the
    # residual for this prespecified localization criterion.
    not_separated = dual_fd_attribution(_row(1.), _row(1.01))
    assert not_separated["classifications_agree"] and not not_separated["failure_localization_eligible"]
    native, chain = _row(1.), _row(1.000001)
    native["informative"] = False
    assert dual_fd_attribution(native, chain)["status"] == "ambiguous"
    native = _row(1.)
    chain["actual_secant_linear_prediction"] += .01
    assert dual_fd_attribution(native, chain)["status"] == "ambiguous"
    with pytest.raises(ValueError, match="identical endpoints"):
        dual_fd_attribution(_row(1.), _row(1., .8))


def test_source087_reference_rejects_changed_pinned_evidence_before_reading(monkeypatch):
    import latentmol.boltz.affinity_decoder_causal_diagnostic as diagnostic
    helper = SimpleNamespace(_sha=lambda path: "0" * 64,
        _read=lambda path: pytest.fail("Unauthenticated evidence must not be read"))
    monkeypatch.setattr(diagnostic, "observers", lambda:
        (SimpleNamespace(head_helpers=lambda: SimpleNamespace(_helper=lambda: helper)), None))
    with pytest.raises(ValueError, match="follow-up evidence changed"):
        source087_reference({}, {})


def test_only_original_prior5_fixed_mask_step20_is_selected():
    case = {"id": "prior-5", "step": 20, "arm": "fixed_initial_mask"}
    assert selected_case({"cases": [case]}) is case
    for edit in ({"step": 21}, {"arm": "dynamic"}, {"id": "prior-3"}):
        with pytest.raises(ValueError, match="prior-5"):
            selected_case({"cases": [{**case, **edit}]})
