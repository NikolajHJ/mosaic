"""Closed metadata authorization for the disposable source134 target test.

The primitive protocol reader validates a loss schema, not permission to run it.
This full-plan boundary requires the separate top-level marker, alpha0.1,
quarter displacement and exact200-update probe. Historical absent-marker plans
retain their behavior; no marker or successful proof authorizes training.
"""
from copy import deepcopy
import json


VERSION = "epoch36-bond-weighting-test-v1"
POLICY = {"version": VERSION}
LOSS_POLICY = {"version": "bond-weighted-target-ce-v1", "none_bond_weight": .1}
PROBE_VERSION = "chemistry-target-weighting-probe-v1"
PROBE_POLICY = dict(version=PROBE_VERSION, updates=200,
                    gradient_updates=[1, 39, 96, 100, 199, 200])


def _exact(value, expected):
    """Keep bool/int/float, null, unknown fields and nonfinite aliases distinct."""
    try:
        return (json.dumps(value, sort_keys=True, allow_nan=False)
                == json.dumps(expected, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError):
        return False


def validate_target_weighting_test(plan):
    """Return a fresh marker, or None; require the complete declared test scope.

    Original parent/data/guard identities remain checked by the shared short
    validator after this boundary. This function imports no numerical runtime
    and neither schedules work nor accepts a training/promotion purpose.
    """
    if type(plan) is not dict:
        raise ValueError("Target weighting requires an immutable plan object")
    protocol = plan.get("protocol")
    has_loss = type(protocol) is dict and "chemistry_target_weighting" in protocol
    if "target_weighting_test" not in plan:
        if has_loss or "target_weighting_probe" in plan:
            raise ValueError("Weighted loss/probe requires its explicit target-weighting test marker")
        return None
    if not _exact(plan["target_weighting_test"], POLICY):
        raise ValueError("Require the exact disposable target-weighting test marker")
    if not has_loss:
        raise ValueError("Target-weighting test requires its explicit loss policy")
    from .chemistry_target_weighting import validate_chemistry_target_weighting
    loss = validate_chemistry_target_weighting(protocol["chemistry_target_weighting"])
    if not _exact(loss, LOSS_POLICY):
        raise ValueError("The declared target test fixes NONE bond weight at0.1")
    if (not _exact(plan.get("short_continuation"), dict(version="three-epoch-baseline-v1",
            start_epoch=36, end_epoch=39, purpose="self_guidance_test"))
            or not _exact(plan.get("self_guidance_iteration"),
                          dict(version="epoch36-valid-self-iteration-v1", weight=.1))
            or not _exact(protocol.get("valid_prior_self_guidance"),
                          dict(version="valid-prior-self-nll-v1", weight=.1))):
        raise ValueError("Bond weighting retains alpha0.1 and a disposable original36 test purpose")
    from .chemistry_optimizer_step import validate_optimizer_step_control
    if validate_optimizer_step_control(plan.get("optimizer_step_control")) is None:
        raise ValueError("Bond weighting retains the explicit quarter AdamW displacement policy")
    if (not _exact(plan.get("target_weighting_probe"), PROBE_POLICY)
            or "self_strength_probe" in plan):
        raise ValueError("Require the new200-update target probe, not an old coefficient probe")
    return deepcopy(POLICY)
