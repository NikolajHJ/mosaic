"""Allocated native DiffSBDD qualification and cross-runtime replay fixtures.

Same-runtime replay is exact. Original-compatible-runtime parity is a separate
receipt and is never inferred from successful extraction in the combined runtime.
"""
from __future__ import annotations

import hashlib
import math
from typing import Sequence

import torch
from torch import Tensor

from .native import (GaussianStream, NativeState, NativeStepper,
                     assert_native_replay)


def state_fingerprint(core) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(core.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def normalization_check(core) -> dict:
    identity = torch.eye(core.atom_nf, device=next(core.parameters()).device)
    x = torch.arange(core.atom_nf * 3, device=identity.device,
                     dtype=identity.dtype).reshape(-1, 3) * 0.125
    _, normalized = core.normalize(pocket={"x": x.clone(), "one_hot": identity.clone()})
    returned_x, returned_h = core.unnormalize(normalized["x"], normalized["one_hot"])
    expected_h = (identity - core.norm_biases[1]) / core.norm_values[1]
    checks = {
        "loaded_bias_and_scale": torch.equal(normalized["one_hot"], expected_h),
        "onehot_roundtrip": torch.equal(returned_h, identity),
        "coordinate_roundtrip": torch.equal(returned_x, x),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Native normalization contract failed: {checks}")
    return {"passed": True, "checks": checks, "norm_values": list(core.norm_values),
            "norm_biases": list(core.norm_biases)}


@torch.no_grad()
def exact_boundary_check(stepper: NativeStepper, state: NativeState) -> dict:
    """Substitute independently evaluated exact atom-encoder output at boundary."""
    core = stepper.core
    t = torch.full((state.batch_size, 1), state.level,
                   device=state.ligand.device) / stepper.timesteps
    expected_embedding = core.dynamics.atom_encoder(state.ligand[:, 3:])
    native, native_pocket = core.dynamics(state.ligand, state.pocket, t,
                                          state.ligand_mask, state.pocket_mask)
    calls = []

    def replace_exact(_module, inputs, output):
        if not torch.equal(inputs[0], state.ligand[:, 3:]):
            raise RuntimeError("Native atom encoder received unexpected state features")
        if not torch.equal(output, expected_embedding):
            raise RuntimeError("Independent native atom-encoder output differed")
        calls.append(True)
        return expected_embedding.clone()

    hook = core.dynamics.atom_encoder.register_forward_hook(replace_exact)
    try:
        substituted, substituted_pocket = core.dynamics(
            state.ligand, state.pocket, t, state.ligand_mask, state.pocket_mask)
    finally:
        hook.remove()
    checks = {"single_ligand_boundary": len(calls) == 1,
              "ligand_noise_exact": torch.equal(native, substituted),
              "pocket_noise_exact": torch.equal(native_pocket, substituted_pocket)}
    if not all(checks.values()):
        raise RuntimeError(f"Exact native boundary substitution failed: {checks}")
    return {"passed": True, "level": state.level, "checks": checks,
            "embedding_shape": list(expected_embedding.shape)}


def chemical_denoiser_derivative_check(stepper: NativeStepper,
                                       state: NativeState) -> dict:
    """Independent forward/reverse AD plus finite differences of native features.

    This tests the denoiser's chemical input boundary only. It is not a native
    Boltz/context derivative certificate or a learned-head gradient certificate.
    Geometry/pocket/masks/time are fixed, including the native cutoff graph.
    """
    core = stepper.core
    t = torch.full((state.batch_size, 1), state.level,
                   device=state.ligand.device) / stepper.timesteps
    x = state.ligand[:, :3].detach()
    base = state.ligand[:, 3:].detach()
    indices = torch.arange(base.numel(), dtype=base.dtype,
                           device=base.device).reshape_as(base)
    direction = torch.cos(indices * 0.37 + 0.11)
    direction = direction / direction.norm()

    def noise(h):
        return core.dynamics(torch.cat((x, h), dim=-1), state.pocket.detach(), t,
                             state.ligand_mask, state.pocket_mask)[0]

    with torch.enable_grad():
        leaf = base.clone().requires_grad_(True)
        prediction = noise(leaf)
        weights = torch.sin(torch.arange(prediction.numel(), device=prediction.device,
                                         dtype=prediction.dtype) * 0.19 + 0.3).reshape_as(prediction)
        weights = weights / weights.norm()
        scalar = (prediction * weights).sum()
        reverse = torch.autograd.grad(scalar, leaf)[0]
        ad = float((reverse * direction).sum().detach())
        # torch.func.jvp is forward AD, independently of reverse autograd above.
        # Unsupported native operations raise and stop; no silent AD fallback.
        primal, tangent = torch.func.jvp(noise, (base,), (direction,))
        jvp = float((tangent * weights).sum().detach())
    allowed = 1e-6 + 0.02 * max(abs(ad), abs(jvp))
    if not math.isfinite(ad) or not math.isfinite(jvp) or abs(ad - jvp) > allowed:
        raise RuntimeError(f"Native denoiser forward/reverse AD disagreement: {ad}, {jvp}")
    if not torch.equal(prediction.detach(), primal.detach()):
        raise RuntimeError("Native denoiser independent primal evaluations disagree")
    finite_differences = []
    with torch.no_grad():
        for scale in (0.01, 0.003, 0.001):
            plus = float((noise(base + scale * direction) * weights).sum())
            minus = float((noise(base - scale * direction) * weights).sum())
            fd = (plus - minus) / (2 * scale)
            bound = 1e-6 + 0.02 * max(abs(ad), abs(fd))
            resolved = math.isfinite(fd) and plus != minus
            finite_differences.append({"scale": scale, "plus": plus, "minus": minus,
                                       "AD": ad, "FD": fd, "allowed_error": bound,
                                       "resolved": resolved,
                                       "passed": resolved and abs(ad - fd) <= bound})
    adjacent = any(a["passed"] and b["passed"]
                   for a, b in zip(finite_differences, finite_differences[1:]))
    if not adjacent:
        raise RuntimeError(f"Native denoiser FD is failed/inconclusive: {finite_differences}")
    # A wrong derivative cannot pass this observed, resolved directional signal.
    wrong_ad = -ad if ad != 0 else 1.0
    wrong_rejected = any(
        abs(wrong_ad - item["FD"]) > 1e-6 + 0.02 * max(abs(wrong_ad), abs(item["FD"]))
        for item in finite_differences if item["resolved"])
    if not wrong_rejected:
        raise RuntimeError("Native denoiser wrong-gradient control was not discriminating")
    return {"passed": True, "level": state.level,
            "scope": "native_denoiser_chemical_boundary_only",
            "reverse_AD": ad, "forward_AD": jvp, "allowed_error": allowed,
            "primal_exact": True, "wrong_gradient_rejected": wrong_rejected,
            "finite_differences": finite_differences}


def qualify_native_core(core, pocket: dict, num_nodes_lig: Tensor, *,
                        seeds: Sequence[int] = (17, 23, 29),
                        check_derivatives: bool = True) -> dict:
    """Run bounded same-runtime native contracts, with explicit provenance scope."""
    stepper = NativeStepper(core)
    before = state_fingerprint(core)
    normalization = normalization_check(core)
    replays = [assert_native_replay(stepper, pocket, num_nodes_lig, int(seed))
               for seed in seeds]
    stream = GaussianStream(int(seeds[0]), namespace="diffsbdd-native-checks-v1")
    state = stepper.initialize(pocket, num_nodes_lig, stream)
    check_levels = {0, int(core.T * 0.04), int(core.T * 0.10), int(core.T * 0.20)}
    boundaries, derivatives, noops = [], [], []
    while True:
        if state.level in check_levels:
            boundaries.append(exact_boundary_check(stepper, state))
            if check_derivatives:
                derivatives.append(chemical_denoiser_derivative_check(stepper, state))
            clean = stepper.clean_estimate(state)
            count_before = len(stream.receipts)
            noop = stepper.correct(state, clean.normalized, clean.normalized.clone())
            exact = noop is state and len(stream.receipts) == count_before
            if not exact:
                raise RuntimeError("Equal-z native correction changed state/RNG")
            noops.append({"level": state.level, "state_exact": exact,
                          "gaussian_draws_consumed": 0})
        if state.level == 0:
            break
        state = stepper.reverse_step(state, stream)
    after = state_fingerprint(core)
    if before != after:
        raise RuntimeError("Native qualification mutated frozen core tensors")
    return {"schema": "diffsbdd-native-qualification-v1", "passed": True,
            "normalization": normalization, "same_runtime_replays": replays,
            "exact_boundary": boundaries, "denoiser_derivatives": derivatives,
            "derivative_checks_requested": bool(check_derivatives),
            "equal_z_noop": noops, "frozen_before": before, "frozen_after": after,
            "original_compatible_runtime_parity": "separate_required_gate_not_established"}


@torch.no_grad()
def export_native_reference(core, pocket: dict, num_nodes_lig: Tensor,
                            seed: int = 17) -> dict:
    """CPU-saveable native high-level output for a separately pinned runtime."""
    stepper = NativeStepper(core)
    stream = GaussianStream(seed)
    ligand, pocket_out, lm, pm = stepper.sample_reference(pocket, num_nodes_lig, stream)
    state_stream = stream.fork()
    state = stepper.initialize(pocket, num_nodes_lig, state_stream)
    levels = {0, int(core.T * 0.04), int(core.T * 0.10), int(core.T * 0.20)}
    frames = {}
    while True:
        if state.level in levels:
            clean = stepper.clean_estimate(state)
            frames[str(state.level)] = {
                "ligand": state.ligand.cpu(), "pocket": state.pocket.cpu(),
                "frame_offset": state.frame_offset.cpu(),
                "clean_normalized": clean.normalized.cpu(),
                "predicted_noise": clean.predicted_noise.cpu()}
        if state.level == 0:
            break
        state = stepper.reverse_step(state, state_stream)
    return {"schema": "diffsbdd-native-cross-runtime-fixture-v1",
            "torch_version": str(torch.__version__), "seed": seed,
            "frozen_state_sha256": state_fingerprint(core),
            "native_output": {"ligand": ligand.cpu(), "pocket": pocket_out.cpu(),
                              "ligand_mask": lm.cpu(), "pocket_mask": pm.cpu()},
            "frames": frames, "draw_receipts": stream.receipts}


def compare_native_references(original: dict, combined: dict, *,
                              atol: float = 2e-5, rtol: float = 2e-5) -> dict:
    """Cross-runtime tolerance gate; same-runtime remains exact elsewhere."""
    schema = "diffsbdd-native-cross-runtime-fixture-v1"
    if original.get("schema") != schema or combined.get("schema") != schema:
        raise RuntimeError("Cross-runtime fixture schema mismatch")
    if original["seed"] != combined["seed"]:
        raise RuntimeError("Cross-runtime reference seed mismatch")
    if original["frozen_state_sha256"] != combined["frozen_state_sha256"]:
        raise RuntimeError("Cross-runtime loaded core weights/buffers differ")
    # Device strings can differ in a CPU-port qualification. Seed/key/shape/dtype
    # must agree. A CPU->CUDA RNG comparison is not claimed by this gate.
    def draws(fixture):
        return [{k: v for k, v in receipt.items() if k != "device"}
                for receipt in fixture["draw_receipts"]]

    if draws(original) != draws(combined):
        raise RuntimeError("Cross-runtime native Gaussian indexing differs")
    checks = []

    def compare(name, a, b):
        if a.shape != b.shape or a.dtype != b.dtype:
            raise RuntimeError(f"Cross-runtime shape/dtype mismatch: {name}")
        if a.dtype in (torch.long, torch.int, torch.bool):
            passed = torch.equal(a, b)
            maximum = 0.0 if passed else None
        else:
            passed = torch.allclose(a, b, atol=atol, rtol=rtol)
            maximum = float((a - b).abs().max())
        checks.append({"name": name, "passed": bool(passed),
                       "maximum_absolute_error": maximum})
        if not passed:
            raise RuntimeError(f"Cross-runtime native output mismatch: {checks[-1]}")

    for key in original["native_output"]:
        compare(f"output/{key}", original["native_output"][key], combined["native_output"][key])
    if set(original["frames"]) != set(combined["frames"]):
        raise RuntimeError("Cross-runtime reference frame levels differ")
    for level, frame in original["frames"].items():
        for key, tensor in frame.items():
            compare(f"level{level}/{key}", tensor, combined["frames"][level][key])
    return {"schema": "diffsbdd-cross-runtime-parity-v1", "passed": True,
            "original_torch_version": original["torch_version"],
            "combined_torch_version": combined["torch_version"],
            "atol": atol, "rtol": rtol, "checks": checks}
