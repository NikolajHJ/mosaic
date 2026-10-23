"""Shared metadata policies for chemistry execution, runtime and prior mix.

The internal duration comes from the immutable protocol, while LSF receives
one additional hour for setup, checkpoint flushing and process shutdown. This
changes available time, not the historical 1.5x epoch estimate plus 1,800-second
fit guard, the objective or sampling counts. A separate, explicitly authorized
top-level runtime amendment may use 1.25x with the same 1,800-second reserve
only inside a 23-hour internal / 24-hour LSF allocation. It changes no protocol
or objective bytes. The explicit upper
bound corresponds to DTU's 72-hour gpua100 allocation ceiling. Historical
22-hour internal protocols still request exactly 23 hours from LSF.

No scientific package, model or scheduler is imported here. Numerical and
controller fixtures must run on allocated DTU HPC before promotion.
The separate explicit 1:2 policy preserves expected composition counts using
importance weights; it does not claim equal finite-sample JS or draw variance.
"""
from __future__ import annotations

import math
import re


SCHEDULER_MAX_SECONDS = 72 * 3600
ALLOCATION_MARGIN_SECONDS = 3600
SEGMENT_MAX_SECONDS = SCHEDULER_MAX_SECONDS - ALLOCATION_MARGIN_SECONDS
BENCHMARK_MAX_SECONDS = 105 * 60
LEGACY_EXECUTION = {"backend": "optimized", "workers": 12, "torch_threads": 4, "gpu_cpus": 16}
EXECUTION_WORKERS = (12, 20, 28)
RATIO2_VERSION = "joint-atom-preserving-chemistry-v2"
FAILURE_GUIDANCE_VERSION = "chemistry-failure-guidance-v1"
CHEMISTRY_HANDOFF_VERSION = "chemistry-objective-comparison-handoff-v1"
EPOCH_FIT_AMENDMENT_VERSION = "chemistry-epoch-fit-amendment-v1"


def validate_failure_guidance(value):
    """Strict optional objective metadata; absence preserves historical plans.

    This reader deliberately imports no scientific packages, so the scheduler
    and allocated worker reject the same undeclared coefficients or budgets.
    It does not establish that a configured loss has passed native checks.
    """
    if value is None:
        return None
    if type(value) is not dict or set(value) != {"version", "loss", "progress_search"} or value["version"] != FAILURE_GUIDANCE_VERSION:
        raise ValueError("Require a complete versioned failure-guidance policy")
    loss = value["loss"]
    names = ("extra_valid_weight", "partial_weight", "local_bond_weight", "component_cut_weight", "nonempty_weight")
    if (type(loss) is not dict or set(loss) != {"version", *names}
            or loss["version"] not in ("repair-failure-guidance-v1", "repair-failure-guidance-v2")):
        raise ValueError("Require complete failure-guidance loss coefficients")
    for name in names:
        number = loss[name]
        if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
            raise ValueError("Invalid failure-guidance coefficient: " + name)
    if any(loss[name] <= 0 for name in names[2:]):
        raise ValueError("All failure-risk terms must be explicitly positive")
    search = value["progress_search"]
    if search is not None:
        integers = {"max_edits": 12, "checks_per_depth": 4096, "beam_width": 128, "max_atom_edits": 2}
        if (type(search) is not dict or set(search) != {"version", *integers, "probability_weight", "allow_atom_edits", "atom_edit_penalty"}
                or search["version"] != "repair-progress-v1" or type(search["allow_atom_edits"]) is not bool):
            raise ValueError("Require a complete bounded progress-search policy")
        for name, maximum in integers.items():
            if type(search[name]) is not int or not 1 <= search[name] <= maximum:
                raise ValueError("Invalid progress-search bound: " + name)
        for name in ("probability_weight", "atom_edit_penalty"):
            number = search[name]
            if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
                raise ValueError("Invalid progress-search coefficient: " + name)
        if search["atom_edit_penalty"] < 1:
            raise ValueError("Atom edits require an additional positive cost")
    # JSON-safe independent copies prevent a reader from editing the plan.
    return {"version": value["version"], "loss": dict(loss),
            "progress_search": None if search is None else dict(search)}


def validate_starting_checkpoint(value):
    """Explicit objective-change handoff, separate from the frozen reference.

    A source121 chemistry checkpoint carries native AND fresh-prior RNG. This
    metadata declaration never licenses a cross-objective exact-resume claim;
    allocated setup independently authenticates its complete native payload.
    """
    if value is None:
        return None
    required = {"path", "sha256", "source_plan", "source_plan_sha256", "completion", "completion_sha256", "epoch", "updates", "algorithm"}
    if type(value) is not dict or set(value) != required:
        raise ValueError("Require the complete chemistry starting-checkpoint binding")
    for name in ("path", "source_plan", "completion"):
        if type(value[name]) is not str or not value[name]:
            raise ValueError("Starting checkpoint requires explicit artifact paths")
    for name in ("sha256", "source_plan_sha256", "completion_sha256"):
        if type(value[name]) is not str or re.fullmatch(r"[0-9a-f]{64}", value[name]) is None:
            raise ValueError("Starting checkpoint requires complete artifact hashes")
    if (type(value["epoch"]) is not int or not 1 <= value["epoch"] < 100
            or type(value["updates"]) is not int or value["updates"] < 1):
        raise ValueError("Starting checkpoint must precede absolute epoch100")
    algorithm = value["algorithm"]
    if (type(algorithm) is not dict or set(algorithm) != {"files", "sha256"}
            or type(algorithm["files"]) is not dict or not algorithm["files"]
            or type(algorithm["sha256"]) is not str or re.fullmatch(r"[0-9a-f]{64}", algorithm["sha256"]) is None
            or any(type(name) is not str or not name or type(sha) is not str or re.fullmatch(r"[0-9a-f]{64}", sha) is None
                   for name, sha in algorithm["files"].items())):
        raise ValueError("Starting checkpoint requires its exact originating algorithm identity")
    return dict(value)


def validate_handoff_policy(plan):
    """Allow a matched old-objective control, but explicitly version the handoff.

    Legacy comparison candidates are benchmarkable, not promotable training
    successors. The training/controller boundary separately requires the new
    failure-guidance objective before submitting or executing a continuation.
    """
    starting = validate_starting_checkpoint(plan.get("starting_checkpoint"))
    if starting is not None:
        if plan.get("handoff_version") != CHEMISTRY_HANDOFF_VERSION:
            raise ValueError("Require the explicit chemistry objective-comparison handoff version")
    elif plan.get("handoff_version") is not None:
        raise ValueError("A handoff version requires a complete starting checkpoint")
    return starting


def validate_execution_policy(value):
    """Keep old resource receipts exact; new policies declare CPU/GPU overlap."""
    if value is None:
        return None
    if type(value) is not dict:
        raise ValueError("Unsupported explicit chemistry execution resources")
    if set(value) == set(LEGACY_EXECUTION):
        if any(type(value[key]) is not type(expected) or value[key] != expected
               for key, expected in LEGACY_EXECUTION.items()):
            raise ValueError("Unsupported explicit chemistry execution resources")
    elif set(value) == {*LEGACY_EXECUTION, "overlap"}:
        if (value["backend"] != "optimized" or type(value["backend"]) is not str
                or type(value["workers"]) is not int or value["workers"] not in EXECUTION_WORKERS
                or type(value["torch_threads"]) is not int or value["torch_threads"] != 4
                or type(value["gpu_cpus"]) is not int or value["gpu_cpus"] not in (16, 24, 32)
                or value["gpu_cpus"] < value["workers"] + 4
                or type(value["overlap"]) is not bool):
            raise ValueError("Unsupported explicit chemistry execution resources")
    else:
        raise ValueError("Unsupported explicit chemistry execution resources")
    return dict(value)


def protocol_config_path(plan):
    """Resolve only the historical or explicitly source-bound variant config."""
    value = plan.get("protocol_config", "configs/chemistry_triangle_1to3.json")
    if (type(value) is not str or (value != "configs/chemistry_triangle_1to3.json"
            and re.fullmatch(r"configs/chemistry_(?:speed|failure)_variants/[a-z0-9][a-z0-9_-]{0,63}\.json", value) is None)):
        raise ValueError("Unsupported source-bound chemistry protocol_config path")
    return value


def composition_prior_weight(protocol):
    """Keep the expected composition population at real:prior = 1:3.

    The prior branch categorical NLL and occupancy anchor still use their own
    all-attempt means. Only additive expected composition counts and their VJP
    receive this factor. Reducing the number of fresh Monte Carlo points to
    two per real point therefore requires an explicit 1.5 importance factor.
    """
    ratio = protocol.get("prior_ratio")
    value = protocol.get("composition_prior_weight", 1.0)
    if (type(ratio) is not int or ratio not in (2, 3)
            or type(value) not in (int, float) or not math.isfinite(value)
            or value != 3.0 / ratio
            or (ratio == 2 and "composition_prior_weight" not in protocol)):
        raise ValueError("Require prior_ratio 2/3 and explicit composition_prior_weight=3/prior_ratio")
    return float(value)


def _bounded_seconds(value, *, field, maximum):
    """Accept only positive integer seconds, including no bool/float coercion."""
    if type(value) is not int or not 0 < value <= maximum:
        raise ValueError(f"{field} requires integer seconds in 1..{maximum}")
    return value


def segment_runtime_policy(protocol):
    """Derive an LSF minute limit with at least one hour beyond internal time.

    Rounding the allocation upward, rather than rounding internal time down,
    preserves the full margin for protocols whose seconds are not a multiple
    of 60. The returned metadata is newly constructed; inputs are untouched.
    """
    if type(protocol) is not dict:
        raise ValueError("Chemistry runtime protocol must be a JSON object")
    internal = _bounded_seconds(protocol.get("segment_wall_seconds"),
                                field="segment_wall_seconds", maximum=SEGMENT_MAX_SECONDS)
    minutes = (internal + ALLOCATION_MARGIN_SECONDS + 59) // 60
    allocation = minutes * 60
    return {"internal_seconds": internal, "allocation_seconds": allocation,
            "allocation_margin_seconds": allocation - internal,
            "lsf_minutes": minutes, "lsf_limit": f"{minutes // 60}:{minutes % 60:02d}"}


def epoch_fit_policy(plan):
    """Return the exact legacy guard or the narrowly declared runtime amendment.

    The optional declaration belongs to the top-level immutable plan, not its
    unchanged chemistry protocol. Absence preserves historical calculations;
    an explicit null, partial declaration, arbitrary factor or longer allocation
    is rejected. The metadata/controller and numerical worker use this same
    reader, so an authorization cannot silently become a different fit guard.
    """
    if type(plan) is not dict:
        raise ValueError("Epoch-fit policy requires an immutable plan object")
    if "runtime_policy_amendment" not in plan:
        return {"version": "chemistry-epoch-fit-legacy-v1", "factor": 1.5,
                "reserve_seconds": 1800}
    value = plan["runtime_policy_amendment"]
    if (type(value) is not dict or set(value) != {"version", "factor", "reserve_seconds"}
            or type(value["version"]) is not str or value["version"] != EPOCH_FIT_AMENDMENT_VERSION
            or type(value["factor"]) is not float or value["factor"] != 1.25
            or type(value["reserve_seconds"]) is not int or value["reserve_seconds"] != 1800):
        raise ValueError("Require the exact declared 1.25x epoch-fit runtime amendment")
    protocol = plan.get("protocol")
    if (type(protocol) is not dict or type(protocol.get("segment_wall_seconds")) is not int
            or protocol["segment_wall_seconds"] != 23 * 3600):
        raise ValueError("Amended epoch-fit guard requires exactly 23 internal hours")
    allocated = segment_runtime_policy(protocol)
    if allocated["allocation_seconds"] != 24 * 3600 or allocated["lsf_limit"] != "24:00":
        raise ValueError("Amended epoch-fit guard requires exactly 24 allocated hours")
    return dict(value)


def epoch_fit_seconds(plan, estimate):
    """Compute declared runtime headroom without changing a scientific estimate."""
    policy = epoch_fit_policy(plan)
    if type(estimate) not in (int, float) or not math.isfinite(estimate) or estimate <= 0:
        raise ValueError("Epoch runtime estimate must be a positive finite number")
    result = policy["factor"] * estimate + policy["reserve_seconds"]
    if not math.isfinite(result):
        raise ValueError("Reserved epoch runtime must remain finite")
    return result


def validate_runtime_policy(protocol):
    """Use the same duration validation in native and metadata-only readers."""
    result = segment_runtime_policy(protocol)
    _bounded_seconds(protocol.get("benchmark_wall_seconds"),
                     field="benchmark_wall_seconds", maximum=BENCHMARK_MAX_SECONDS)
    return result
