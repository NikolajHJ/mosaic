"""Explicit parameter-displacement experiment around one native AdamW step.

The ordinary path is unchanged. The opt-in policy keeps inherited moments and
advances them once, then applies one quarter of the resulting parameter delta.
This changes the original optimizer trajectory while retaining its moments.
For a fixed moment update the delta has the mathematical effect of a smaller
step learning rate, with explicit float64 displacement arithmetic and unchanged
stored learning rates; native floating-point operation order can differ.
Validation of the JSON policy is safe at a scheduler/metadata boundary.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


VERSION = "adamw-displacement-scale-v1"


def validate_optimizer_step_control(value):
    """Return a fresh exact policy, or None for the historical native step."""
    if value is None:
        return None
    if (type(value) is not dict or set(value) != {"version", "factor"}
            or value["version"] != VERSION or type(value["factor"]) is not float
            or value["factor"] != .25):
        raise ValueError("Optimizer step control requires the explicit quarter-displacement policy")
    return dict(version=VERSION, factor=.25)


def validate_optimizer_step_receipt(value, control):
    """Authenticate observational step metadata without importing Torch."""
    policy = validate_optimizer_step_control(control)
    if policy is None:
        if value is not None:
            raise ValueError("An ordinary native step cannot claim a displacement amendment")
        return None
    if (type(value) is not dict or set(value) != {"version", "factor", "optimizer_steps",
            "raw_displacement_norm", "applied_displacement_norm", "parameter_arithmetic", "moment_state"}
            or value.get("version") != VERSION or type(value.get("factor")) is not float
            or value["factor"] != policy["factor"]
            or type(value.get("optimizer_steps")) is not int or value["optimizer_steps"] != 1
            or value.get("parameter_arithmetic") != "float64_delta_then_native_dtype"
            or value.get("moment_state") != "one_native_adamw_step_retained"
            or any(type(value.get(key)) not in (int, float) or not math.isfinite(value[key]) or value[key] < 0
                   for key in ("raw_displacement_norm", "applied_displacement_norm"))
            or value["applied_displacement_norm"] > value["raw_displacement_norm"]):
        raise ValueError("Optimizer step lost its exact one-step quarter-displacement receipt")
    return dict(value)


def optimizer_step_chain(previous, record):
    return hashlib.sha256(json.dumps(dict(previous=previous, record=record),
                                    sort_keys=True, allow_nan=False).encode()).hexdigest()


def authenticate_optimizer_step_evidence(evidence, control, *, root, epoch, start_update, end_update):
    """Bind every applied step of a completed epoch to its exclusive JSONL file."""
    if validate_optimizer_step_control(control) is None:
        if evidence is not None:
            raise ValueError("Historical epochs cannot claim amended optimizer evidence")
        return None
    if (type(evidence) is not dict or set(evidence) != {"path", "sha256", "updates", "start_update",
            "end_update", "chain_sha256"} or type(epoch) is not int
            or type(start_update) is not int or type(end_update) is not int or end_update <= start_update
            or type(evidence.get("start_update")) is not int or type(evidence.get("end_update")) is not int
            or evidence.get("start_update") != start_update or evidence.get("end_update") != end_update
            or type(evidence.get("updates")) is not int or evidence["updates"] != end_update - start_update
            or type(evidence.get("path")) is not str or not Path(evidence["path"]).is_absolute()
            or any(type(evidence.get(key)) is not str or len(evidence[key]) != 64
                   or any(char not in "0123456789abcdef" for char in evidence[key])
                   for key in ("sha256", "chain_sha256"))):
        raise ValueError("Completed epoch lacks its complete amended-step evidence")
    path = Path(evidence["path"]).resolve(strict=True)
    if not path.is_relative_to(Path(root).resolve(strict=True)) or hashlib.sha256(path.read_bytes()).hexdigest() != evidence["sha256"]:
        raise ValueError("Optimizer step evidence escaped its run or changed")
    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise ValueError("Duplicate optimizer-step evidence key")
            result[key] = item
        return result
    chain, count = "0" * 64, 0
    for line in path.read_text().splitlines():
        if not line.strip():
            raise ValueError("Empty optimizer-step record")
        row = json.loads(line, object_pairs_hook=pairs,
                         parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        count += 1
        if (type(row) is not dict or set(row) != {"epoch", "update", "receipt"}
                or type(row.get("epoch")) is not int or row["epoch"] != epoch
                or type(row.get("update")) is not int or row["update"] != start_update + count):
            raise ValueError("Optimizer step evidence is missing, duplicated or out of order")
        validate_optimizer_step_receipt(row["receipt"], control)
        chain = optimizer_step_chain(chain, row)
    if count != end_update - start_update or chain != evidence["chain_sha256"]:
        raise ValueError("Optimizer step evidence is partial or its chain changed")
    return dict(evidence)


def _finite_state(value, torch):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite_state(item, torch) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_state(item, torch) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def apply_optimizer_step(optimizer, parameters, control=None):
    """Step once; return an observational receipt only for the opt-in policy.

    ``parameters`` must cover the optimizer exactly, without aliases. The native
    path neither consumes this iterable nor snapshots/scans any state. The
    quarter path rolls parameters, gradients and complete AdamW state back on
    any failed/partial/nonfinite step. No target, RNG, mode or gradient is changed.
    Delta arithmetic and norm accumulation use float64 before the final cast.
    """
    policy = validate_optimizer_step_control(control)
    if policy is None:
        optimizer.step()
        return None

    import torch
    if type(optimizer) is not torch.optim.AdamW:
        raise ValueError("Quarter displacement requires the native AdamW optimizer")
    parameters = tuple(parameters)
    owned = tuple(p for group in optimizer.param_groups for p in group["params"])
    if (not parameters or len({id(p) for p in parameters}) != len(parameters)
            or len(owned) != len(parameters) or {id(p) for p in owned} != {id(p) for p in parameters}
            or any(not isinstance(p, torch.nn.Parameter) or not p.is_floating_point()
                   or p.is_complex() or not p.is_leaf for p in parameters)):
        raise ValueError("Displacement parameters must cover native AdamW exactly once")
    if (not all(bool(torch.isfinite(p).all()) for p in parameters)
            or not _finite_state(optimizer.state_dict(), torch)
            or any(p.grad is not None and (p.grad.is_sparse or not bool(torch.isfinite(p.grad).all()))
                   for p in parameters)):
        raise FloatingPointError("Quarter displacement requires finite native input state")

    previous = tuple(p.detach().clone() for p in parameters)
    gradients = tuple((p.grad, None if p.grad is None else p.grad.detach().clone()) for p in parameters)
    optimizer_state = deepcopy(optimizer.state_dict())
    try:
        optimizer.step()
        if (not all(bool(torch.isfinite(p).all()) for p in parameters)
                or not _finite_state(optimizer.state_dict(), torch)):
            raise FloatingPointError("Native AdamW produced nonfinite state")
        if any(p.grad is not original or (original is not None and not torch.equal(original, saved))
               for p, (original, saved) in zip(parameters, gradients)):
            raise RuntimeError("Native AdamW changed gradient storage or values")
        raw_squares, applied_squares = [], []
        with torch.no_grad():
            for parameter, old in zip(parameters, previous):
                old64 = old.double()
                delta64 = parameter.detach().double() - old64
                raw_squares.append(float(delta64.square().sum()))
                parameter.copy_((old64 + policy["factor"] * delta64).to(parameter.dtype))
                applied_squares.append(float((parameter.detach().double() - old64).square().sum()))
        raw_norm = math.sqrt(math.fsum(raw_squares))
        applied_norm = math.sqrt(math.fsum(applied_squares))
        if (not math.isfinite(raw_norm) or not math.isfinite(applied_norm)
                or not all(bool(torch.isfinite(p).all()) for p in parameters)):
            raise FloatingPointError("Scaled AdamW displacement is nonfinite")
    except BaseException:
        with torch.no_grad():
            for parameter, old, (original, saved) in zip(parameters, previous, gradients):
                parameter.copy_(old)
                parameter.grad = original
                if original is not None:
                    original.copy_(saved)
        optimizer.load_state_dict(optimizer_state)
        raise
    return dict(version=VERSION, factor=policy["factor"], optimizer_steps=1,
                raw_displacement_norm=raw_norm, applied_displacement_norm=applied_norm,
                parameter_arithmetic="float64_delta_then_native_dtype",
                moment_state="one_native_adamw_step_retained")
