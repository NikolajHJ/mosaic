"""Allocated-CUDA regressions for the disposable chemistry resume gate.

These exercise real CUDA AdamW persistence and replay, without a large model or
repair search. Only the chemistry-update body is replaced by a cheap stochastic
parameter loss. The production save/CPU-load/restore/comparison/finally path is
unchanged. This establishes checkpoint bookkeeping, not chemical training.

CPU preflights skip this file. The same GPU worker as the native benchmark must
set LATENTMOL_REQUIRE_CUDA_RESUME_FIXTURES=1, making missing CUDA a hard failure.
All numerical execution belongs on allocated DTU HPC.
"""
from copy import deepcopy
from dataclasses import replace
import os
import random

import numpy as np
import pytest
import torch

from latentmol import chemistry_continuation as continuation
from latentmol.train import capture_rng_state, restore_rng_state
from test_chemistry_continuation import _FixedTeacher, _hard, _tiny_models


@pytest.fixture(autouse=True)
def allocated_cuda():
    if not torch.cuda.is_available():
        if os.environ.get("LATENTMOL_REQUIRE_CUDA_RESUME_FIXTURES") == "1":
            pytest.fail("Mandatory allocated chemistry resume fixtures require available CUDA")
        pytest.skip("CUDA checkpoint replay fixtures run in the allocated GPU precheck")
    initial = capture_rng_state({}, include_cuda=True)
    try:
        yield
    finally:
        # Fixtures must not leak their seeds into another allocated check.
        restore_rng_state(initial, {})


def _snapshot(model, reference, optimizer, generators):
    return {"model": deepcopy(model.state_dict()), "reference": deepcopy(reference.state_dict()),
            "optimizer": deepcopy(optimizer.state_dict()),
            "rng": capture_rng_state(generators, include_cuda=True)}


def _assert_restored(snapshot, model, reference, optimizer, generators):
    observed = _snapshot(model, reference, optimizer, generators)
    for key, original in snapshot.items():
        assert continuation.nested_equal(original, observed[key]), key
    assert set(observed["rng"]) == {"python", "numpy", "torch", "cuda", "loaders"}
    assert observed["rng"]["cuda"]
    assert set(observed["rng"]["loaders"]) == {"train", "validation", "fresh_prior"}
    assert all(parameter.grad is None for parameter in reference.parameters())


def _case(monkeypatch, *, failure=None):
    model, reference, config, protocol, composition = _tiny_models()
    model, reference = model.cuda(), reference.cuda()
    config = replace(config, device="cuda")
    teacher = _FixedTeacher(model.spec, accepted=False)
    hard = _hard(model.spec, ("CO", "C", "CCO", "CC"))
    data = [{"atoms": hard.atoms[index], "charges": hard.charges[index], "bonds": hard.bonds[index]}
            for index in range(4)]
    generators = {"train": torch.Generator().manual_seed(101),
                  "validation": torch.Generator().manual_seed(102),
                  "fresh_prior": torch.Generator().manual_seed(103)}
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay, foreach=False, fused=False, capturable=False)
    placements, call_count = [], 0

    def actual_stochastic_adamw(model, reference, optimizer, hard, config, p, composition,
                                teacher, generators, update, epoch, phase_start_epoch):
        """Consume every persisted RNG and perform a real CUDA AdamW update."""
        nonlocal call_count
        call_count += 1
        assert config.device == "cuda" and hard.atoms.shape[0] == 4
        hard = hard.to(config.device)
        assert hard.atoms.is_cuda and not any(parameter.requires_grad for parameter in reference.parameters())
        if optimizer.state:
            placements.append({state["step"].device.type for state in optimizer.state.values()})
            assert all(state["exp_avg"].is_cuda and state["exp_avg_sq"].is_cuda
                       for state in optimizer.state.values())
        optimizer.zero_grad(set_to_none=True)
        # Saving both NumPy's Gaussian cache and Python's complete state matters;
        # a test consuming only Torch would leave those restoration bugs hidden.
        noise = random.random() + float(np.random.normal()) + float(torch.rand(()))
        noise += sum(float(torch.randn((), generator=generator)) for generator in generators.values())
        device_noise = torch.randn((), device="cuda") + noise
        loss = sum(parameter.square().mean() + .01 * device_noise * parameter.mean()
                   for parameter in model.parameters())
        loss.backward()
        optimizer.step()
        if failure == "update_exception" and call_count == 4:
            raise RuntimeError("Injected CUDA replay update failure")
        if failure == "value_mismatch" and call_count == 4:
            with torch.no_grad():
                next(model.parameters()).add_(.125)
        return {"real_points": 4, "fresh_prior_points": 12}

    monkeypatch.setattr(continuation, "chemistry_update", actual_stochastic_adamw)
    # Establish genuine nonzero moments first. Legacy parent checkpoints can
    # carry scalar step tensors on CUDA, despite noncapturable AdamW expecting
    # those counters on CPU when loading a state_dict. Force precisely that
    # supported storage difference, without changing any numeric value.
    actual_stochastic_adamw(model, reference, optimizer, hard, config, protocol,
                            composition, teacher, generators, 17, 35, 35)
    assert optimizer.state
    for state in optimizer.state.values():
        assert state["step"].numel() == 1 and int(state["step"].item()) == 1
        state["step"] = state["step"].cuda()
    return {"model": model, "reference": reference, "optimizer": optimizer, "data": data,
            "config": config, "p": protocol, "composition": composition, "teacher": teacher,
            "generators": generators, "epoch": 35, "updates": 18}, placements


def test_native_cuda_probe_cpu_loaded_adamw_counter_replay_is_exact_and_disposable(monkeypatch, tmp_path):
    case, placements = _case(monkeypatch)
    initial = _snapshot(case["model"], case["reference"], case["optimizer"], case["generators"])
    checkpoint = tmp_path / "disposable_cuda_resume.pt"
    result = continuation.native_resume_probe(**case, output=checkpoint)
    assert result["status"] == "passed"
    assert result["checks"] == {"model": True, "optimizer": True, "rng": True}
    assert result["real_points"] == 4 and result["fresh_prior_points"] == 12
    # First probe update and uninterrupted next update use legacy CUDA counters;
    # CPU-loaded replay uses CPU counters. Moments and model remain on CUDA.
    assert placements == [{"cuda"}, {"cuda"}, {"cpu"}]
    persisted = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert all(not value.is_cuda for value in persisted["model"].values())
    assert all(not state["step"].is_cuda and not state["exp_avg"].is_cuda
               for state in persisted["optimizer"]["state"].values())
    _assert_restored(initial, case["model"], case["reference"], case["optimizer"], case["generators"])


@pytest.mark.parametrize("failure", ["value_mismatch", "update_exception"])
def test_native_cuda_probe_failure_restores_parent_moments_and_every_rng(monkeypatch, tmp_path, failure):
    case, placements = _case(monkeypatch, failure=failure)
    initial = _snapshot(case["model"], case["reference"], case["optimizer"], case["generators"])
    exception, message = ((ValueError, "next-step model/AdamW/RNG checkpoint replay failed")
                          if failure == "value_mismatch" else (RuntimeError, "Injected CUDA replay update failure"))
    with pytest.raises(exception, match=message):
        continuation.native_resume_probe(**case, output=tmp_path / "failed_disposable_resume.pt")
    assert placements[:2] == [{"cuda"}, {"cuda"}]
    assert placements[-1] == {"cpu"}
    _assert_restored(initial, case["model"], case["reference"], case["optimizer"], case["generators"])


def test_cuda_cpu_tensor_comparison_keeps_exact_shape_dtype_and_value_contract():
    value = torch.tensor([1., 2.], device="cuda", requires_grad=True)
    cpu = value.detach().cpu()
    assert continuation.nested_equal({"state": [value]}, {"state": [cpu]})
    assert not continuation.nested_equal(value, cpu.double())
    assert not continuation.nested_equal(value, cpu.reshape(1, 2))
    changed = cpu.clone(); changed[1] = torch.nextafter(changed[1], torch.tensor(float("inf")))
    assert not continuation.nested_equal(value, changed)
    assert value.grad is None
