"""Metadata-only bounded original36 continuation and separate test policies.

The scientific maximum remains 100 total epochs. This separately declared
operational endpoint prevents automatic progression beyond three new epochs;
it is never an epoch100 completion or permission to skip readiness/stability.
The optional weak self-target experiment is validation-only, never training.
A distinct validated-reference training marker may end at40; it cannot amend
historical39 plans. Absence preserves every historical plan's behavior.
"""
from copy import deepcopy
import hashlib
import json
import math


VERSION = "three-epoch-baseline-v1"
BASELINE_VERSION = "epoch36-original-repair-v1"
SELF_VERSION = "valid-prior-self-nll-v1"
START_EPOCH, END_EPOCH, START_UPDATES = 36, 39, 237456
START_SHA = "3179ef01f84a2c714ee031bc3e6a6a4dc2cedc767d30e700c887aab1abbd706f"
SOURCE_PLAN_SHA = "238bb5627cf7f4bbc2fd65a7d2d7bb0c860c7872c56bd1e51399c52ee0e98e8e"
SOURCE_COMPLETION_SHA = "4d9aab0216d574f1551a8b2ecd5e1c78f873726bcd5fab134773bbd6412c29b5"
ORIGINAL_IDENTITIES = dict(
    parent_epoch=35, parent_updates=230860,
    parent_checkpoint_sha256="19f438222e454ef53169f6d81082335ee7e24902b1b7cf5e0f18eced4b8e92ed",
    parent_archive_sha256="08f55c8b81f82f3952caae06f57735ccd03785922832ce3c2242c425e3cde792",
    parent_manifest_sha256="e8fe9b66df2224672c1c0ef69ad71c29aefab1ebfa7e353ea28536fc746bc405",
    dataset_sha256="54f37815d5901f7d76b855f69e237d97172c72b027e98616f949c08178f9cae3",
    split_sha256="356499265847be6af63f4e70c809ff4b82967fd971b4ffabe94f3aa095adf739",
    config_sha256="b6704c80c4f3308b5f20623eb0ef21fc39c4b7b1d0cd60d519a9990670b1b254",
    training_molecules=1688539)
# Exact successful source121 protocol, excluding only declared new monitoring
# metadata and the separate validation-only self-guidance field below.
ORIGINAL_PROTOCOL = dict(
    benchmark_panel_count=256, benchmark_updates=8, benchmark_wall_seconds=6300,
    charge_js_weight=.05, composition_prior_weight=1.5, element_js_weight=.1,
    evaluation_seed=904017, maximum_segments=80, microbatch_size=128,
    minimum_prior_target_fraction=.01, monitor_every=5, monitor_prior_count=1024,
    occupancy_anchor_weight=1.0, parameter_scope="joint_encoder_decoder", precision="fp32",
    prior_ratio=2, prior_repair_weight=.75, prior_seed=104017, real_batch_size=256,
    real_repair_weight=.25, repair_ramp_epochs=3, repair_weight=.5,
    search=dict(beam_width=4, checks_per_depth=32, max_edits=3,
                probability_floor=1e-12, probability_weight=1.0),
    segment_wall_seconds=82800, total_epochs=100, validation_count=256,
    version="joint-atom-preserving-chemistry-v2", vram_fraction_limit=.85)
EARLY_POLICY = dict(version="chemistry-early-health-v1", first_check_updates=32,
    interval_updates=100, max_examples_per_failure=4, max_strict_validity_drop=.02)
GUARDED_POLICY = dict(version="guarded-chemistry-restart-v1", validation_updates=2500,
                      validation_wall_seconds=82800)
EXECUTION = dict(backend="optimized", gpu_cpus=16, overlap=True, torch_threads=4, workers=12)


def _exact(value, expected):
    """Reject bool/int or int/float substitution, unknown fields and nonfinites."""
    return (json.dumps(value,sort_keys=True,allow_nan=False)
            == json.dumps(expected,sort_keys=True,allow_nan=False))


def validate_valid_prior_self_guidance(value):
    """Shared nonnumerical loss schema; launch policy separately fixes weak .01."""
    if value is None:
        return None
    if (type(value) is not dict or set(value) != {"version","weight"}
            or value["version"] != SELF_VERSION or type(value["weight"]) not in (int,float)
            or not math.isfinite(value["weight"]) or value["weight"] < 0):
        raise ValueError("Require explicit finite nonnegative valid-prior self guidance")
    return dict(value)


def validate_baseline_continuation(protocol):
    """Validate a protocol marker without inspecting a plan or calling guards.

    This small boundary is also used by the early-panel policy reader. It
    cannot authorize a run or training. A primitive weighted-loss schema may
    be read here; the full plan below separately requires its exact top marker,
    test purpose and probe before any execution entrypoint accepts it.
    """
    if type(protocol) is not dict:
        raise ValueError("Baseline continuation requires a protocol object")
    if "baseline_continuation" not in protocol:
        if any(key in protocol for key in ("valid_prior_self_guidance", "chemistry_target_weighting",
                                           "valid_prior_reference_guidance")):
            raise ValueError("Self/target guidance needs explicit baseline continuation")
        return None
    marker = protocol["baseline_continuation"]
    if not _exact(marker,{"version":BASELINE_VERSION}) or "failure_guidance" in protocol:
        raise ValueError("Original baseline cannot include failure-guidance objectives")
    excluded = {"baseline_continuation", "early_health", "valid_prior_self_guidance"}
    if "valid_prior_reference_guidance" in protocol:
        from .chemistry_valid_reference_guidance import validate_valid_prior_reference_guidance
        from .chemistry_valid_reference_policy import LOSS_POLICY as REFERENCE_LOSS_POLICY
        value = validate_valid_prior_reference_guidance(protocol["valid_prior_reference_guidance"])
        if (not _exact(value, REFERENCE_LOSS_POLICY)
                or "valid_prior_self_guidance" in protocol or "chemistry_target_weighting" in protocol):
            raise ValueError("Frozen-reference diagnostic view requires its exact separate loss policy")
        # Schema inspection only. Full-plan launch validation below requires
        # the frozen36 marker, source ancestry and nonpromotable test purpose.
        excluded.add("valid_prior_reference_guidance")
    if "chemistry_target_weighting" in protocol:
        from .chemistry_target_weighting import validate_chemistry_target_weighting
        from .chemistry_target_weighting_policy import LOSS_POLICY
        value = validate_chemistry_target_weighting(protocol["chemistry_target_weighting"])
        if not _exact(value, LOSS_POLICY):
            raise ValueError("Baseline diagnostic view requires the exact declared target weighting")
        excluded.add("chemistry_target_weighting")
    inherited = {key:value for key,value in protocol.items() if key not in excluded}
    if not _exact(inherited,ORIGINAL_PROTOCOL):
        raise ValueError("Original source121 repair/search/ratio/weights/protocol changed")
    if "valid_prior_self_guidance" in protocol:
        if validate_valid_prior_self_guidance(protocol["valid_prior_self_guidance"]) is None:
            raise ValueError("An explicit self-guidance field cannot be null")
    return dict(marker)


def validate_short_continuation(plan):
    """Require exact current36 ancestry, guards and a bounded declared purpose.

    File/source/model/optimizer authentication is still performed by existing
    read_plan/setup/readiness readers. These fixed identities restrict this
    authorization to the actual source121 checkpoint and successful objective.
    """
    if type(plan) is not dict:
        raise ValueError("Short continuation requires an immutable plan object")
    from .chemistry_valid_reference_training_policy import validate_valid_reference_continuation
    training = validate_valid_reference_continuation(plan)
    if training is not None:
        from .chemistry_epoch45_extension import validate_extension
        extension = validate_extension(plan)
        if extension is not None:
            return extension
        return training
    from .chemistry_valid_reference_policy import validate_valid_reference_test
    reference = validate_valid_reference_test(plan)
    from .chemistry_target_weighting_policy import validate_target_weighting_test
    validate_target_weighting_test(plan)
    from .chemistry_optimizer_step import validate_optimizer_step_control
    step = validate_optimizer_step_control(plan.get("optimizer_step_control"))
    if "optimizer_step_control" in plan and step is None:
        raise ValueError("An explicit optimizer-step amendment cannot be null")
    from .chemistry_self_guidance_iteration import validate_self_guidance_iteration
    iteration = validate_self_guidance_iteration(plan)
    protocol = plan.get("protocol",{})
    if "short_continuation" not in plan:
        if step is not None:
            raise ValueError("Quarter-displacement experiment requires the guarded bounded epoch36 branch")
        if isinstance(protocol,dict) and any(key in protocol for key in
                ("baseline_continuation", "valid_prior_self_guidance", "valid_prior_reference_guidance")):
            raise ValueError("Baseline continuation requires its bounded plan policy")
        return None
    value = plan["short_continuation"]
    if (type(value) is not dict or set(value) != {"version","start_epoch","end_epoch","purpose"}
            or value["version"] != VERSION or type(value["start_epoch"]) is not int
            or value["start_epoch"] != START_EPOCH or type(value["end_epoch"]) is not int
            or value["end_epoch"] != END_EPOCH
            or value["purpose"] not in ("baseline_training","self_guidance_test")):
        raise ValueError("Require the explicit three-epoch baseline/test endpoint policy")
    if validate_baseline_continuation(protocol) is None:
        raise ValueError("Short continuation requires the exact original repair objective marker")
    self_policy = protocol.get("valid_prior_self_guidance")
    self_weight = .01 if iteration is None else iteration["weight"]
    if ((value["purpose"] == "baseline_training" and "valid_prior_self_guidance" in protocol)
            or (value["purpose"] == "baseline_training" and "valid_prior_reference_guidance" in protocol)
            or (value["purpose"] == "self_guidance_test" and reference is None
                and not _exact(self_policy,{"version":SELF_VERSION,"weight":self_weight}))):
        raise ValueError("Training baseline and weak validation-only self guidance must stay separate")
    _validate_origin_bindings(plan, protocol)
    return deepcopy(value)


def _validate_origin_bindings(plan, protocol):
    """Shared exact36/reference35/data/guard boundary, without endpoint dispatch.

    Kept separate so the new through40 metadata policy can reuse the complete
    historical ancestry checks without recursively calling its launch reader.
    No model or saved numerical evidence is loaded by this function.
    """
    from .chemistry_runtime import validate_handoff_policy
    start = validate_handoff_policy(plan)
    if (start is None or start["epoch"] != START_EPOCH or start["updates"] != START_UPDATES
            or start["sha256"] != START_SHA or start["source_plan_sha256"] != SOURCE_PLAN_SHA
            or start["completion_sha256"] != SOURCE_COMPLETION_SHA
            or hashlib.sha256(json.dumps(start["algorithm"]["files"],sort_keys=True).encode()).hexdigest()
                != start["algorithm"]["sha256"]
            or any(type(plan.get(key)) is not type(expected) or plan[key] != expected
                   for key,expected in ORIGINAL_IDENTITIES.items())
            or not _exact(plan.get("execution"),EXECUTION)
            or not _exact(plan.get("guarded_restart"),GUARDED_POLICY)
            or plan.get("runtime_adoption") is not None):
        raise ValueError("Short continuation changed original36/reference35/data/execution/proof bindings")
    from .chemistry_early_health import validate_early_health_policy
    if not _exact(validate_early_health_policy(protocol),EARLY_POLICY):
        raise ValueError("Short continuation requires unchanged32/100 fixed1024/256 early checks")


def require_training_branch(plan):
    """A disposable weak-self experiment can never become a training successor."""
    policy = validate_short_continuation(plan)
    if policy is not None and policy["purpose"] not in ("baseline_training", "validated_reference_training"):
        raise ValueError("Self-guidance test is disposable and cannot train or promote")
    return policy


def continuation_end_epoch(plan):
    """Operational bound; callers executing training must also require purpose."""
    policy = validate_short_continuation(plan)
    return policy["end_epoch"] if policy is not None else plan["protocol"]["total_epochs"]


def expected_stop_reason(plan, epoch):
    """Match bounded/ordinary completion without fabricating an epoch100 result."""
    require_training_branch(plan)
    endpoint = continuation_end_epoch(plan)
    if type(epoch) is not int or not 1 <= epoch <= endpoint:
        raise ValueError("Completed epoch exceeds the declared operational endpoint")
    if epoch == endpoint:
        return "bounded_horizon" if endpoint != 100 else "planned_horizon"
    return "wall_budget_epoch_boundary"
