"""Allocated-HPC regressions for deterministic chemistry event probabilities.

All numerical execution, including the CPU reference, belongs on DTU HPC.
The CUDA allocation must set LATENTMOL_REQUIRE_REPAIR_CUDA_FIXTURES=1 so a
missing device fails instead of silently skipping these production regressions.
These establish execution/loss/gradient parity, not chemical efficacy.
"""
from contextlib import contextmanager
import os

import pytest
import torch

from latentmol import repair_failure_guidance as guidance
from latentmol.representation import DecoderOutput, GraphSpec
from test_repair_failure_guidance import failed_batch, output_for


@pytest.fixture(params=["cpu", "cuda"])
def deterministic_device(request):
    device = request.param
    if device == "cuda" and not torch.cuda.is_available():
        if os.environ.get("LATENTMOL_REQUIRE_REPAIR_CUDA_FIXTURES") == "1":
            pytest.fail("Mandatory deterministic chemistry event fixtures require CUDA")
        pytest.skip("CUDA event regressions run in the allocated GPU precheck")
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        yield device
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
    finally:
        # Restore only after the check; every tested forward/backward runs with
        # strict determinism still enabled, never a warning-only workaround.
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _legacy_cpu_event(log_absent, log_present):
    """The sealed historical operation, deliberately restricted to CPU."""
    assert log_absent.device.type == log_present.device.type == "cpu"
    prefix = torch.cat((log_absent.new_zeros(1), log_absent.cumsum(0)[:-1]))
    return torch.logsumexp(prefix + log_present, 0)


def _tolerances(dtype):
    # The new fixed-order scan groups additions differently. FP32 tolerances
    # also include CPU/CUDA log_softmax and logsumexp rounding; there is no
    # claim of bitwise equality with the historical CPU cumsum operation.
    return dict(rtol=5e-5, atol=5e-6) if dtype == torch.float32 else dict(rtol=2e-11, atol=2e-12)


def _event_case(count, regime, dtype):
    index = torch.arange(count, dtype=dtype)
    if regime == "mixed":
        difference = ((index * 17).remainder(31) - 15) / 3
    elif regime == "confident_absent":
        difference = -2000 + (index.remainder(7) - 3) / 8
    elif regime == "confident_present":
        difference = 2000 + (index.remainder(7) - 3) / 8
    else:
        raise AssertionError(regime)
    # Columns are absence/presence. No random state is consumed by the fixture.
    return torch.stack((torch.zeros_like(difference), difference), -1)


def _event_value_and_gradient(logits, function):
    log_probabilities = torch.log_softmax(logits, -1)
    loss = -function(log_probabilities[:, 0], log_probabilities[:, 1])
    gradient = torch.autograd.grad(loss, logits)[0]
    return loss.detach().cpu(), gradient.detach().cpu()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("count", [1, 2, 3, 49, 50, 625])
@pytest.mark.parametrize("regime", ["mixed", "confident_absent", "confident_present"])
def test_deterministic_event_matches_historical_cpu_loss_and_gradient(
        deterministic_device, dtype, count, regime):
    original = _event_case(count, regime, dtype)
    reference = _event_value_and_gradient(original.clone().requires_grad_(), _legacy_cpu_event)
    observed = _event_value_and_gradient(
        original.to(deterministic_device).requires_grad_(), guidance._log_any_present)
    assert torch.isfinite(observed[0]) and torch.isfinite(observed[1]).all()
    for actual, expected in zip(observed, reference):
        torch.testing.assert_close(actual, expected, **_tolerances(dtype))
    if regime == "confident_absent":
        # A finite-tail log-space calculation must retain useful signal even
        # when FP32/FP64 absence probabilities have rounded to exactly one.
        assert observed[0] > 1900 and observed[1].norm() > 0


def test_deterministic_event_first_present_prefix_is_exclusive(deterministic_device):
    # Deliberately independent log inputs identify the prefix derivative:
    # the final absent event must NEVER enter any first-present probability.
    absent = torch.tensor([-.1, -.3, -.7], dtype=torch.float64,
                          device=deterministic_device, requires_grad=True)
    present = torch.tensor([-2., -1., -3.], dtype=torch.float64,
                           device=deterministic_device, requires_grad=True)
    value = guidance._log_any_present(absent, present)
    expected = torch.logsumexp(torch.stack((present[0], absent[0] + present[1],
                                           absent[0] + absent[1] + present[2])), 0)
    actual_gradients = torch.autograd.grad(value, (absent, present), retain_graph=True)
    expected_gradients = torch.autograd.grad(expected, (absent, present))
    torch.testing.assert_close(value, expected, rtol=1e-13, atol=1e-13)
    for actual, reference in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual, reference, rtol=1e-13, atol=1e-13)
    assert actual_gradients[0][-1] == 0


def test_single_event_retains_connected_zero_absence_gradient(deterministic_device):
    absent = torch.tensor([-1000.], dtype=torch.float64,
                          device=deterministic_device, requires_grad=True)
    present = torch.tensor([-2.], dtype=torch.float64,
                           device=deterministic_device, requires_grad=True)
    value = guidance._log_any_present(absent, present)
    # No allow_unused: even the size-one exclusive prefix must retain the
    # historical graph-connected zero derivative, rather than None/a detach.
    absence_gradient, present_gradient = torch.autograd.grad(value, (absent, present))
    assert value == present[0]
    assert torch.equal(absence_gradient, torch.zeros_like(absent))
    assert torch.equal(present_gradient, torch.ones_like(present))


def test_deterministic_event_rejects_an_empty_event_set(deterministic_device):
    empty = torch.empty(0, device=deterministic_device)
    with pytest.raises(ValueError, match="empty event set"):
        guidance._log_any_present(empty, empty)


def test_deterministic_event_forward_and_backward_repeat_bitwise(deterministic_device):
    original = _event_case(625, "mixed", torch.float32)
    first = _event_value_and_gradient(original.to(deterministic_device).requires_grad_(),
                                      guidance._log_any_present)
    second = _event_value_and_gradient(original.to(deterministic_device).requires_grad_(),
                                       guidance._log_any_present)
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])


@contextmanager
def _legacy_cpu_only(monkeypatch):
    with monkeypatch.context() as context:
        context.setattr(guidance, "_log_any_present", _legacy_cpu_event)
        yield


def _failure_fixture(kind, confidence):
    if kind == "empty":
        spec, decoded = output_for([], [], spec=GraphSpec(max_nodes=50), confidence=confidence)
        component = "nonempty_risk"
    elif kind == "disconnected":
        spec, decoded = output_for([9, 9, 9, 9],
            [(0, 1, "TRIPLE"), (2, 3, "TRIPLE")], confidence=confidence)
        component = "component_cut_risk"
    elif kind == "maximum_cut":
        # Two 25-node components produce the maximum 25*25 cut at N=50.
        # Overbonded fluorine chains cannot become valid with the original
        # one-edit/one-check teacher, even if its single candidate is a bridge.
        edges = [(i, i + 1, "SINGLE") for i in range(49) if i != 24]
        spec, decoded = output_for([9] * 50, edges, confidence=confidence)
        component = "component_cut_risk"
    else:
        raise AssertionError(kind)
    prepared = guidance.prepare_failure_guidance(decoded, failed_batch(decoded, spec), spec)
    assert prepared.rows[0] is not None
    assert prepared.rows[0].witness is None
    return spec, decoded, prepared, component


def _loss_component_and_gradients(decoded, prepared, spec, component):
    value = guidance.failure_aware_repair_loss(decoded, prepared, spec)[component]
    gradients = torch.autograd.grad(value, (decoded.atoms, decoded.charges, decoded.bonds),
                                    allow_unused=True)
    return value.detach().cpu(), tuple(None if v is None else v.detach().cpu() for v in gradients)


def _live_copy(decoded, dtype, device):
    return DecoderOutput(*(value.detach().to(device=device, dtype=dtype).clone().requires_grad_()
                           for value in (decoded.atoms, decoded.charges, decoded.bonds)))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("confidence", [3., 1000.])
@pytest.mark.parametrize("kind", ["empty", "disconnected", "maximum_cut"])
def test_actual_failure_loss_event_branches_match_cpu_reference_and_repeat(
        monkeypatch, deterministic_device, dtype, confidence, kind):
    spec, original, prepared, component = _failure_fixture(kind, confidence)
    with _legacy_cpu_only(monkeypatch):
        reference = _loss_component_and_gradients(_live_copy(original, dtype, "cpu"),
                                                 prepared, spec, component)
    actual = _loss_component_and_gradients(_live_copy(original, dtype, deterministic_device),
                                          prepared, spec, component)
    repeated = _loss_component_and_gradients(_live_copy(original, dtype, deterministic_device),
                                            prepared, spec, component)
    assert torch.isfinite(actual[0]) and actual[0] > 0
    torch.testing.assert_close(actual[0], reference[0], **_tolerances(dtype))
    assert torch.equal(actual[0], repeated[0])
    nonzero = 0
    for observed, expected, replay in zip(actual[1], reference[1], repeated[1]):
        assert (observed is None) == (expected is None) == (replay is None)
        if observed is not None:
            assert torch.isfinite(observed).all()
            torch.testing.assert_close(observed, expected, **_tolerances(dtype))
            assert torch.equal(observed, replay)
            nonzero += int(torch.count_nonzero(observed))
    assert nonzero > 0
