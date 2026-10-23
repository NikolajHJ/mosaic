"""Allocated-only native AdamW trajectory/rollback fixtures; no model benchmark."""
from copy import deepcopy
import hashlib
import json
import math
import os

import pytest
import torch

from latentmol.chemistry_optimizer_step import (
    VERSION, apply_optimizer_step, validate_optimizer_step_control,
    authenticate_optimizer_step_evidence, optimizer_step_chain, validate_optimizer_step_receipt,
)


POLICY = {"version": VERSION, "factor": .25}


def _device(name):
    if name == "cuda" and not torch.cuda.is_available():
        if os.environ.get("LATENTMOL_REQUIRE_REPAIR_CUDA_FIXTURES") == "1":
            pytest.fail("Mandatory optimizer-displacement CUDA fixture has no CUDA device")
        pytest.skip("CUDA fixture runs in the allocated native gate")
    return torch.device(name)


def _assert_same(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and left.device == right.device
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            _assert_same(a, b)
    else:
        assert left == right


def _optimizer(device="cpu", dtype=torch.float32):
    parameters = [torch.nn.Parameter(torch.tensor([1., -2., 3.], dtype=dtype, device=device)),
                  torch.nn.Parameter(torch.tensor([.125, 8.], dtype=dtype, device=device))]
    optimizer = torch.optim.AdamW(parameters, lr=.001, weight_decay=1e-5)
    for p in parameters:
        p.grad = torch.full_like(p, .2)
    optimizer.step()  # Nonzero inherited first/second moments, not a fresh AdamW.
    for index, p in enumerate(parameters):
        p.grad = torch.full_like(p, -.3 if index else .4)
    return parameters, optimizer


def _clone(parameters, optimizer):
    copied = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
    result = torch.optim.AdamW(copied)
    result.load_state_dict(deepcopy(optimizer.state_dict()))
    for p, original in zip(copied, parameters):
        p.grad = None if original.grad is None else original.grad.detach().clone()
    return copied, result


@pytest.mark.parametrize("bad", [
    {}, {"version": VERSION}, {"version": VERSION, "factor": .25, "extra": True},
    {"version": "unknown", "factor": .25}, {"version": VERSION, "factor": 1.},
    {"version": VERSION, "factor": 0.}, {"version": VERSION, "factor": -.25},
    {"version": VERSION, "factor": True}, {"version": VERSION, "factor": "0.25"},
    {"version": VERSION, "factor": float("nan")},
    {"version": VERSION, "factor": float("inf")}, [], False,
])
def test_only_explicit_quarter_policy_is_authorized(bad):
    with pytest.raises(ValueError):
        validate_optimizer_step_control(bad)


def test_policy_absent_and_copy():
    assert validate_optimizer_step_control(None) is None
    copy = validate_optimizer_step_control(POLICY)
    assert copy == POLICY and copy is not POLICY


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_native_moments_advance_once_and_only_parameter_displacement_changes(device_name, dtype):
    device = _device(device_name)
    parameters, optimizer = _optimizer(device, dtype)
    full, full_optimizer = _clone(parameters, optimizer)
    old = [p.detach().clone() for p in parameters]
    gradient_objects = [p.grad for p in parameters]
    gradients = [p.grad.detach().clone() for p in parameters]
    full_optimizer.step()
    calls = []
    native_step = optimizer.step
    optimizer.step = lambda: (calls.append("step"), native_step())[1]
    receipt = apply_optimizer_step(optimizer, iter(parameters), POLICY)
    assert calls == ["step"]
    for p, before, raw, grad_ref, grad_value in zip(parameters, old, full, gradient_objects, gradients):
        expected = (before.double() + .25 * (raw.detach().double() - before.double())).to(dtype)
        assert torch.equal(p, expected)
        assert p.grad is grad_ref and torch.equal(p.grad, grad_value)
    _assert_same(optimizer.state_dict(), full_optimizer.state_dict())
    assert receipt["factor"] == .25 and receipt["optimizer_steps"] == 1
    raw_norm = math.sqrt(math.fsum(float((p.detach().double() - before.double()).square().sum())
                                  for p, before in zip(full, old)))
    applied_norm = math.sqrt(math.fsum(float((p.detach().double() - before.double()).square().sum())
                                      for p, before in zip(parameters, old)))
    assert receipt["raw_displacement_norm"] == raw_norm
    assert receipt["applied_displacement_norm"] == applied_norm
    assert receipt["moment_state"] == "one_native_adamw_step_retained"


def test_absent_policy_is_bitwise_native_and_does_not_snapshot_or_consume_parameters(monkeypatch):
    parameters, optimizer = _optimizer()
    full, full_optimizer = _clone(parameters, optimizer)
    full_optimizer.step()
    class ForbiddenIterable:
        def __iter__(self):
            raise AssertionError("Legacy step consumed parameters")
    with monkeypatch.context() as patch:
        patch.setattr(optimizer, "state_dict", lambda: pytest.fail("Legacy step took a snapshot"))
        assert apply_optimizer_step(optimizer, ForbiddenIterable()) is None
    for p, native in zip(parameters, full):
        assert torch.equal(p, native)
    _assert_same(optimizer.state_dict(), full_optimizer.state_dict())


@pytest.mark.parametrize("failure", ["parameter_nonfinite", "moment_nonfinite", "exception", "interrupt",
                                     "gradient_values", "gradient_storage"])
def test_partial_or_nonfinite_step_rolls_back_all_parameters_moments_groups_and_gradients(failure):
    parameters, optimizer = _optimizer()
    before = [p.detach().clone() for p in parameters]
    before_state = deepcopy(optimizer.state_dict())
    gradient_objects = [p.grad for p in parameters]
    before_gradients = [p.grad.detach().clone() for p in parameters]
    native_step = optimizer.step
    calls = []
    def bad_step():
        calls.append("step")
        native_step()
        if failure == "parameter_nonfinite":
            with torch.no_grad():
                parameters[0].fill_(float("inf"))
        elif failure == "moment_nonfinite":
            optimizer.state[parameters[0]]["exp_avg"].fill_(float("nan"))
        elif failure == "gradient_values":
            parameters[0].grad.mul_(2)
        elif failure == "gradient_storage":
            parameters[0].grad = None
        else:
            optimizer.param_groups[0]["lr"] = .9
            if failure == "interrupt":
                raise KeyboardInterrupt("Interrupted after native step")
            raise RuntimeError("Partial native failure")
    optimizer.step = bad_step
    expected = KeyboardInterrupt if failure == "interrupt" else (FloatingPointError if "nonfinite" in failure else RuntimeError)
    with pytest.raises(expected):
        apply_optimizer_step(optimizer, parameters, POLICY)
    assert calls == ["step"]
    for p, old, grad_ref, grad_value in zip(parameters, before, gradient_objects, before_gradients):
        assert torch.equal(p, old)
        assert p.grad is grad_ref and torch.equal(p.grad, grad_value)
    _assert_same(optimizer.state_dict(), before_state)


@pytest.mark.parametrize("bad_parameters", ["missing", "duplicate", "foreign"])
def test_parameter_coverage_cannot_leave_an_unrolled_back_optimizer_parameter(bad_parameters):
    parameters, optimizer = _optimizer()
    provided = {"missing": parameters[:1], "duplicate": [parameters[0], parameters[0]],
                "foreign": [parameters[0], torch.nn.Parameter(parameters[1].detach().clone())]}[bad_parameters]
    optimizer.step = lambda: pytest.fail("Invalid ownership called AdamW")
    with pytest.raises(ValueError):
        apply_optimizer_step(optimizer, provided, POLICY)


def test_unrepresentable_delta_is_honest_zero_with_one_moment_update():
    p = torch.nn.Parameter(torch.tensor([1e20], dtype=torch.float32))
    p.grad = torch.ones_like(p)
    optimizer = torch.optim.AdamW([p], lr=.001, weight_decay=0.)
    old = p.detach().clone()
    receipt = apply_optimizer_step(optimizer, [p], POLICY)
    assert torch.equal(p, old)
    assert receipt["raw_displacement_norm"] == receipt["applied_displacement_norm"] == 0.
    assert optimizer.state[p]["step"].item() == 1


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_quarter_step_does_not_draw_rng_or_change_module_modes(device_name):
    device = _device(device_name)
    model = torch.nn.Linear(3, 2).to(device).eval()
    optimizer = torch.optim.AdamW(model.parameters())
    for p in model.parameters():
        p.grad = torch.full_like(p, .1)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == "cuda" else None
    modes = [module.training for module in model.modules()]
    apply_optimizer_step(optimizer, model.parameters(), POLICY)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    if cuda_rng is not None:
        assert torch.equal(torch.cuda.get_rng_state(device), cuda_rng)
    assert [module.training for module in model.modules()] == modes


def test_invalid_input_state_is_rejected_before_step():
    parameters, optimizer = _optimizer()
    optimizer.state[parameters[0]]["exp_avg_sq"].fill_(float("nan"))
    optimizer.step = lambda: pytest.fail("Nonfinite inherited state called AdamW")
    with pytest.raises(FloatingPointError):
        apply_optimizer_step(optimizer, parameters, POLICY)


def _receipt():
    return dict(version=VERSION, factor=.25, optimizer_steps=1, raw_displacement_norm=.1,
                applied_displacement_norm=.025, parameter_arithmetic="float64_delta_then_native_dtype",
                moment_state="one_native_adamw_step_retained")


@pytest.mark.parametrize("change", [
    lambda value: value.update(factor=1.), lambda value: value.update(optimizer_steps=True),
    lambda value: value.update(optimizer_steps=2), lambda value: value.update(raw_displacement_norm=float("nan")),
    lambda value: value.update(applied_displacement_norm=-.1),
    lambda value: value.update(applied_displacement_norm=.2),
    lambda value: value.update(moment_state="moments_reset"), lambda value: value.update(extra=True),
])
def test_step_receipt_cannot_claim_wrong_policy_or_nonfinite_or_multiple_steps(change):
    value = _receipt()
    change(value)
    with pytest.raises(ValueError):
        validate_optimizer_step_receipt(value, POLICY)


def _epoch_evidence(tmp_path, defect=None):
    rows = [dict(epoch=37, update=237456+i, receipt=_receipt()) for i in range(1, 4)]
    if defect == "missing": rows.pop()
    elif defect == "duplicate": rows[1]["update"] = rows[0]["update"]
    elif defect == "epoch": rows[1]["epoch"] = 38
    elif defect == "policy": rows[1]["receipt"]["factor"] = .5
    chain = "0" * 64
    for row in rows:
        chain = optimizer_step_chain(chain, row)
    path = tmp_path / "steps.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), updates=3,
                start_update=237456, end_update=237459, chain_sha256=chain)


@pytest.mark.parametrize("defect", [None, "missing", "duplicate", "epoch", "policy", "changed_bytes", "chain", "escaped"])
def test_complete_epoch_receipts_are_bound_even_if_untrusted_metadata_hashes_are_recomputed(tmp_path, defect):
    evidence = _epoch_evidence(tmp_path, defect)
    root = tmp_path
    if defect == "changed_bytes":
        from pathlib import Path
        with Path(evidence["path"]).open("a") as stream:
            stream.write("{}\n")
    elif defect == "chain": evidence["chain_sha256"] = "0" * 64
    elif defect == "escaped":
        root = tmp_path / "other-run"
        root.mkdir()
    if defect is None:
        assert authenticate_optimizer_step_evidence(evidence, POLICY, root=root,
            epoch=37, start_update=237456, end_update=237459) == evidence
    else:
        with pytest.raises(ValueError):
            authenticate_optimizer_step_evidence(evidence, POLICY, root=root,
                epoch=37, start_update=237456, end_update=237459)
