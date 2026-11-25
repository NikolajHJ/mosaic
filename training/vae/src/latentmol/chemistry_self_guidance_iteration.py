"""Closed, disposable-only coefficient iterations from the original epoch36.

This metadata marker does not change the self-target loss or authorize training.
The shared short-plan validator still authenticates ancestry, the original
objective/search/ratio, fixed panels and the complete disposable proof. Absence
retains the historical alpha0.01 launch policy; stronger weights need an explicit
new immutable plan and can never promote their proof state.
"""
from __future__ import annotations


VERSION = "epoch36-valid-self-iteration-v1"
WEIGHTS = (.1, 1.0)
SELF_VERSION = "valid-prior-self-nll-v1"


def validate_self_guidance_iteration(plan):
    """Return a fresh declared marker, or None; import no scientific libraries.

    The allowed coefficient set is intentionally closed. The current campaign
    may choose one member; this reader is not a scheduler, sweep or selection
    rule. The explicit quarter-step policy is retained in every iteration.
    """
    if type(plan) is not dict:
        raise ValueError("Self-guidance iteration requires an immutable plan object")
    if "self_guidance_iteration" not in plan:
        return None
    value = plan["self_guidance_iteration"]
    if (type(value) is not dict or set(value) != {"version", "weight"}
            or value.get("version") != VERSION or type(value.get("weight")) is not float
            or value["weight"] not in WEIGHTS):
        raise ValueError("Require the explicit closed self-guidance iteration coefficient")
    short = plan.get("short_continuation")
    protocol = plan.get("protocol")
    if (type(short) is not dict or short.get("purpose") != "self_guidance_test"
            or type(protocol) is not dict):
        raise ValueError("Self-guidance iteration is a disposable test and cannot train or promote")
    self_policy = protocol.get("valid_prior_self_guidance")
    if (type(self_policy) is not dict or set(self_policy) != {"version", "weight"}
            or self_policy.get("version") != SELF_VERSION
            or type(self_policy.get("weight")) is not float or self_policy["weight"] != value["weight"]):
        raise ValueError("Self-guidance iteration must match the actual self-target loss coefficient")
    from .chemistry_optimizer_step import validate_optimizer_step_control
    if validate_optimizer_step_control(plan.get("optimizer_step_control")) is None:
        raise ValueError("Self-guidance iteration retains the explicit quarter-step policy")
    return dict(value)
