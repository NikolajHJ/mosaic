"""Metadata authorization for validated frozen-reference training through epoch40.

This distinct marker never changes the historical epoch39/test contracts. It
admits only the unchanged source135 scientific objective from original epoch36,
with its passed disposable proof separately authenticated by the adoption
reader. A syntactically valid policy alone is not readiness, a submission or
permission to promote the disposable proof weights into training.
"""
from copy import deepcopy
import json
from pathlib import PurePosixPath
import re


VERSION = "epoch36-valid-reference-training-v1"
POLICY = dict(version=VERSION, start_epoch=36, end_epoch=40, purpose="validated_reference_training")
ADOPTION_VERSION = "validated-frozen-reference-adoption-v1"
SOURCE_PLAN_SHA = "ad7533a6ce9011c17ba8c962fe939f1e3391b77b73c4ae7574bdbebaf203c609"
SOURCE_STABILITY_SHA = "218a3c9e0318286f9122017cf9c4507885202ed7981bea61852d0c16886a39b0"
WORK_ROOT = PurePosixPath("/work3/s234818/latentmol")
RETIRED_MARKERS = ("short_continuation", "valid_reference_test", "valid_reference_probe",
    "self_guidance_iteration", "self_strength_probe", "target_weighting_test", "target_weighting_probe")


def _exact(value, expected):
    try:
        return json.dumps(value, sort_keys=True, allow_nan=False) == json.dumps(expected, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return False


def _sha(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _path(value):
    return (type(value) is str and PurePosixPath(value).is_absolute()
        and ".." not in PurePosixPath(value).parts and PurePosixPath(value).is_relative_to(WORK_ROOT))


def validate_adoption_metadata(value):
    """Check a closed proof-source binding; bytes and successful gates come later."""
    if (type(value) is not dict or set(value) != {"version", "source_plan", "source_plan_sha256",
            "source_stability_sha256", "artifacts"}
            or value.get("version") != ADOPTION_VERSION or not _path(value.get("source_plan"))
            or value.get("source_plan_sha256") != SOURCE_PLAN_SHA
            or value.get("source_stability_sha256") != SOURCE_STABILITY_SHA
            or type(value.get("artifacts")) is not dict or not value["artifacts"]
            or any(not _path(path) or not _sha(sha) for path, sha in value["artifacts"].items())):
        raise ValueError("Require the explicit passed source135 proof adoption metadata")
    return deepcopy(value)


def validate_valid_reference_continuation(plan):
    """Return the exact through40 policy, or None for historical absent scope.

    Source/model/AdamW/RNG/readiness/health/runtime/quota authentication remains
    mandatory in the existing readers and source135 adoption module. No legacy
    validation marker can be relabeled by editing its purpose or endpoint.
    """
    if type(plan) is not dict:
        raise ValueError("Valid-reference continuation requires an immutable plan object")
    if "valid_reference_continuation" not in plan:
        if "valid_reference_adoption" in plan:
            raise ValueError("Source135 proof adoption needs the explicit new training marker")
        return None
    if (not _exact(plan["valid_reference_continuation"], POLICY)
            or any(key in plan for key in RETIRED_MARKERS)):
        raise ValueError("Require the separate original36 through40 validated-reference training policy")
    protocol = plan.get("protocol")
    if (type(protocol) is not dict
            or any(key in protocol for key in ("valid_prior_self_guidance", "chemistry_target_weighting", "failure_guidance"))):
        raise ValueError("Validated-reference training retains full NLL and original live repair only")
    from .chemistry_valid_reference_guidance import validate_valid_prior_reference_guidance
    if validate_valid_prior_reference_guidance(protocol.get("valid_prior_reference_guidance")) is None:
        raise ValueError("Validated-reference training requires the unchanged alpha0.1 frozen36 guidance")
    from .chemistry_optimizer_step import validate_optimizer_step_control
    if validate_optimizer_step_control(plan.get("optimizer_step_control")) is None:
        raise ValueError("Validated-reference training retains quarter AdamW displacement and inherited moments")
    from .chemistry_short_continuation import validate_baseline_continuation, _validate_origin_bindings
    if validate_baseline_continuation(protocol) is None:
        raise ValueError("Validated-reference training requires the original source121 objective ancestry")
    _validate_origin_bindings(plan, protocol)
    if (not _exact(plan.get("worker_startup"), dict(version="chemistry-worker-startup-v1"))
            or not _exact(plan.get("classification_execution"), dict(version="repair-classification-pool-v1"))):
        raise ValueError("Validated-reference training preserves source135 worker startup and classifier execution")
    validate_adoption_metadata(plan.get("valid_reference_adoption"))
    return deepcopy(POLICY)
