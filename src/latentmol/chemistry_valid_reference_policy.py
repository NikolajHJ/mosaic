"""Closed metadata scope for a frozen original epoch-36 valid-target experiment.

The primitive loss reader checks a schema, not run authorization. This full
plan boundary binds the reference checkpoint, separate 200-update probe and
validation-only purpose, and rejects coexistence with historical self/weighted
policies. Absent-marker historical plans keep their exact launch behavior.
"""
from copy import deepcopy
import json


VERSION = "epoch36-frozen-valid-anchor-test-v1"
START_SHA = "3179ef01f84a2c714ee031bc3e6a6a4dc2cedc767d30e700c887aab1abbd706f"
POLICY = dict(version=VERSION, reference_epoch=36, reference_sha256=START_SHA)
LOSS_POLICY = dict(version="frozen-valid-prior-reference-v1", weight=.1)
PROBE_VERSION = "valid-reference-probe-v1"
PROBE_POLICY = dict(version=PROBE_VERSION, updates=200,
                    gradient_updates=[1, 39, 96, 100, 199, 200])


def _exact(value, expected):
    try:
        return (json.dumps(value, sort_keys=True, allow_nan=False)
                == json.dumps(expected, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError):
        return False


def validate_valid_reference_test(plan):
    """Return a fresh marker or None; never allow a training/proof-state promotion.

    Shared short-plan validation still checks complete ancestry/data/runtime
    identities. Loss/probe fields without a top marker, explicit nulls, old
    simultaneous objectives and nonexact checkpoint/alpha/probe aliases fail.
    """
    if type(plan) is not dict:
        raise ValueError("Valid-reference testing requires an immutable plan object")
    if "valid_reference_continuation" in plan:
        # A separate through40 authorization is not a test-marker exemption:
        # validate its complete closed policy/adoption metadata before dispatch.
        from .chemistry_valid_reference_training_policy import validate_valid_reference_continuation
        validate_valid_reference_continuation(plan)
        return None
    protocol = plan.get("protocol")
    has_loss = type(protocol) is dict and "valid_prior_reference_guidance" in protocol
    if "valid_reference_test" not in plan:
        if has_loss or "valid_reference_probe" in plan:
            raise ValueError("Frozen-reference loss/probe requires its explicit test marker")
        return None
    if not _exact(plan["valid_reference_test"], POLICY):
        raise ValueError("Require the exact frozen original epoch-36 valid-reference marker")
    if not has_loss:
        raise ValueError("Frozen-reference test requires its explicit loss policy")
    if (any(key in plan for key in ("self_guidance_iteration", "self_strength_probe",
                                   "target_weighting_test", "target_weighting_probe"))
            or any(key in protocol for key in ("valid_prior_self_guidance", "chemistry_target_weighting"))):
        raise ValueError("Frozen-reference testing cannot mix old self or target-weighting fields")
    from .chemistry_valid_reference_guidance import validate_valid_prior_reference_guidance
    if not _exact(validate_valid_prior_reference_guidance(protocol["valid_prior_reference_guidance"]), LOSS_POLICY):
        raise ValueError("Frozen-reference testing fixes its full-NLL guidance weight at 0.1")
    if not _exact(plan.get("short_continuation"), dict(version="three-epoch-baseline-v1",
            start_epoch=36, end_epoch=39, purpose="self_guidance_test")):
        raise ValueError("Frozen-reference testing requires the disposable original epoch-36 test purpose")
    from .chemistry_optimizer_step import validate_optimizer_step_control
    if validate_optimizer_step_control(plan.get("optimizer_step_control")) is None:
        raise ValueError("Frozen-reference testing retains the explicit quarter displacement policy")
    if not _exact(plan.get("valid_reference_probe"), PROBE_POLICY):
        raise ValueError("Require the exact 200-update frozen valid-reference probe")
    return deepcopy(POLICY)
