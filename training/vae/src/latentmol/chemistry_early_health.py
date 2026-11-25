"""Opt-in fixed-panel observations and explicitly nonpromotable failure evidence.

The historical trainer has no early-health policy. New plans bind the cadence,
the epoch-36 classification baseline and each saved observation. These records
are diagnostics, never substitutes for completed-epoch health/checkpoints.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


VERSION = "chemistry-early-health-v1"
BASELINE_VERSION = "chemistry-early-health-baseline-v1"
DIAGNOSTIC_VERSION = "chemistry-early-health-diagnostic-v1"


def validate_early_health_policy(protocol):
    """Absent means the exact historical path; present policy is never coerced."""
    value = protocol.get("early_health")
    if value is None:
        if "early_health" in protocol:
            raise ValueError("early_health must be omitted or an explicit versioned policy")
        return None
    keys = {"version", "first_check_updates", "interval_updates", "max_examples_per_failure",
            "max_strict_validity_drop"}
    if type(value) is not dict or set(value) != keys or value.get("version") != VERSION:
        raise ValueError("Unsupported early-health policy")
    for key in ("first_check_updates", "interval_updates", "max_examples_per_failure"):
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError("Early-health cadence/example bounds require positive integers")
    if (not value["first_check_updates"] <= value["interval_updates"] <= 100
            or value["max_examples_per_failure"] > 8
            or type(value["max_strict_validity_drop"]) is not float
            or value["max_strict_validity_drop"] != .02):
        raise ValueError("Early-health cadence/strict retention bounds changed")
    guidance = protocol.get("failure_guidance")
    from .chemistry_short_continuation import validate_baseline_continuation
    baseline = validate_baseline_continuation(protocol)
    if (protocol.get("monitor_prior_count") != 1024 or protocol.get("validation_count") != 256
            or (baseline is None and (type(guidance) is not dict or type(guidance.get("loss")) is not dict
                or guidance["loss"].get("version") != "repair-failure-guidance-v2"))):
        raise ValueError("Early health requires declared v2 guidance and fixed 1024/256 panels")
    return deepcopy(value)


def early_health_due(policy, updates, starting_updates):
    """Cadence is relative to the fixed handoff, including later serial segments."""
    if policy is None:
        return False
    if (type(updates) is not int or type(starting_updates) is not int
            or starting_updates < 0 or updates < starting_updates):
        raise ValueError("Invalid early-health update lineage")
    relative = updates - starting_updates
    return relative > 0 and (relative == policy["first_check_updates"]
                              or relative % policy["interval_updates"] == 0)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exclusive_json(path, value):
    """Persist immutable evidence before a failure; a prior attempt is not replaced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return _sha(path)


def strict_teacher_validity(summary):
    count = summary.get("attempts")
    valid = summary.get("counts", {}).get("strict_teacher_valid")
    if type(count) is not int or count < 1 or type(valid) is not int or not 0 <= valid <= count:
        raise ValueError("Strict-teacher classification lost its all-attempt denominator")
    return valid / count


def _baseline_context(plan):
    policy = validate_early_health_policy(plan["protocol"])
    starting = plan.get("starting_checkpoint")
    if policy is None or not isinstance(starting, dict):
        raise ValueError("An early-health baseline requires the authenticated starting checkpoint")
    expected = plan.get("_plan_sha256")
    if type(expected) is not str or len(expected) != 64:
        raise ValueError("Early-health baseline requires the authenticated plan SHA")
    context = dict(version=BASELINE_VERSION, plan_sha256=expected,
        archive_sha256=plan["archive_sha256"], manifest_sha256=plan["manifest_sha256"],
        dataset_sha256=plan["dataset_sha256"], split_sha256=plan["split_sha256"],
        starting_checkpoint=deepcopy(starting), policy=policy,
        panel_seed=plan["protocol"]["evaluation_seed"], prior_count=1024, reconstruction_count=256)
    from .chemistry_short_continuation import validate_short_continuation
    short = validate_short_continuation(plan)
    if short is not None:
        if "valid_reference_continuation" in plan:
            # This additive context binds the new through40 authorization to
            # its exact source135 proof. The original observation is reused by
            # the authenticated adoption reader, never relabeled as a new draw.
            context["valid_reference_continuation"] = short
            context["valid_reference_adoption"] = deepcopy(plan["valid_reference_adoption"])
            context["valid_prior_reference_guidance"] = deepcopy(plan["protocol"]["valid_prior_reference_guidance"])
        else:
            context["short_continuation"] = short
        context["baseline_continuation"] = deepcopy(plan["protocol"]["baseline_continuation"])
        if "valid_prior_self_guidance" in plan["protocol"]:
            context["valid_prior_self_guidance"] = deepcopy(plan["protocol"]["valid_prior_self_guidance"])
        from .chemistry_self_guidance_iteration import validate_self_guidance_iteration
        iteration = validate_self_guidance_iteration(plan)
        if iteration is not None:
            context["self_guidance_iteration"] = iteration
        from .chemistry_target_weighting_policy import validate_target_weighting_test
        weighting_test = validate_target_weighting_test(plan)
        if weighting_test is not None:
            context["target_weighting_test"] = weighting_test
            context["chemistry_target_weighting"] = deepcopy(plan["protocol"]["chemistry_target_weighting"])
        from .chemistry_valid_reference_policy import validate_valid_reference_test
        reference_test = validate_valid_reference_test(plan)
        if reference_test is not None:
            context["valid_reference_test"] = reference_test
            context["valid_prior_reference_guidance"] = deepcopy(plan["protocol"]["valid_prior_reference_guidance"])
    return context


def validate_early_health_observation(value):
    if (type(value) is not dict or set(value) != {"population", "reconstruction", "classification"}
            or value["population"].get("attempts") != 1024
            or value["reconstruction"].get("molecules") != 256
            or value["classification"].get("attempts") != 1024):
        raise ValueError("Early-health observation lost its fixed 1024/256 panels")
    strict_teacher_validity(value["classification"])
    # Validate counters/assessment denominators, including negative/unknown stages.
    from .chemistry_classification import merge_classification_summaries
    merge_classification_summaries([value["classification"]],
        max_examples_per_failure=value["classification"]["max_examples_per_failure"])
    raw = value["population"].get("raw_argmax_valid_count")
    if (type(raw) is not int or not 0 <= raw <= 1024
            or value["population"].get("raw_argmax_validity") != raw / 1024
            or value["classification"]["counts"]["smiles_parse_valid"] != raw
            or value["classification"]["counts"]["strict_teacher_valid"] > raw):
        raise ValueError("Raw and strict classifications disagree with the same fixed population")
    json.dumps(value, allow_nan=False)


def materialize_early_health_baseline(plan, observation, *, completed_epoch, updates, model_sha256):
    """Called once by allocated validation at the exact authenticated handoff state."""
    context = _baseline_context(plan)
    starting = context["starting_checkpoint"]
    if (type(completed_epoch) is not int or completed_epoch != starting["epoch"]
            or type(updates) is not int or updates != starting["updates"]
            or type(model_sha256) is not str or len(model_sha256) != 64):
        raise ValueError("Early-health baseline is not the original completed starting state")
    validate_early_health_observation(observation)
    job_id = os.environ.get("LSB_JOBID")
    if type(job_id) is not str or not job_id.isdecimal() or int(job_id) < 1:
        raise ValueError("Starting baseline must be observed in an allocated LSF job")
    record = dict(context, status="authenticated_starting_state_observed",
        completed_epoch=completed_epoch, updates=updates, model_sha256=model_sha256,
        job_id=job_id, created_at=datetime.now(timezone.utc).isoformat(), observation=deepcopy(observation))
    path = Path(plan["campaign"]) / "early-health-baseline.json"
    return record, exclusive_json(path, record)


def require_early_health_baseline(plan, expected_sha):
    path = Path(plan["campaign"]) / "early-health-baseline.json"
    if type(expected_sha) is not str or len(expected_sha) != 64 or _sha(path) != expected_sha:
        raise ValueError("Early-health starting baseline changed or is not bound by readiness")
    record = json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    context = _baseline_context(plan)
    if (any(record.get(k) != v for k, v in context.items())
            or record.get("status") != "authenticated_starting_state_observed"
            or record.get("completed_epoch") != context["starting_checkpoint"]["epoch"]
            or record.get("updates") != context["starting_checkpoint"]["updates"]
            or type(record.get("model_sha256")) is not str or len(record["model_sha256"]) != 64):
        raise ValueError("Early-health starting baseline identity differs")
    if type(record.get("job_id")) is not str or not record["job_id"].isdecimal() or int(record["job_id"]) < 1:
        raise ValueError("Early-health baseline lost its actual allocated observer identity")
    validate_early_health_observation(record["observation"])
    return record


@contextmanager
def preserve_observation_state(model, generators):
    """Observers consume no training RNG and restore every module's original mode."""
    import torch
    from .train import capture_rng_state, restore_rng_state
    include_cuda = any(p.device.type == "cuda" for p in model.parameters())
    state = capture_rng_state(generators, include_cuda)
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        restore_rng_state(state, generators)
        for module, was_training in modes:
            module.training = was_training


class EarlyPanelCollector:
    """Classify all hard graphs; retain actual labels/probabilities only within caps."""

    def __init__(self, max_examples_per_failure):
        if type(max_examples_per_failure) is not int or not 1 <= max_examples_per_failure <= 8:
            raise ValueError("Bounded early-panel examples required")
        self.maximum = max_examples_per_failure
        self.records, self.retained = [], {}

    def add(self, atoms, charges, bonds, spec, *, probabilities, candidate_id):
        from .chemistry_classification import classify_repair_graph
        from .quality import graph_identity
        classification = classify_repair_graph(atoms, charges, bonds, spec)
        record = dict(candidate_id=candidate_id, classification=classification)
        kind = classification["failure_kind"]
        if kind != "none" and self.retained.get(kind, 0) < self.maximum:
            self.retained[kind] = self.retained.get(kind, 0) + 1
            record.update(raw_sha256=graph_identity(atoms, charges, bonds, spec),
                labels={name: value.tolist() for name, value in zip(("atoms", "charges", "bonds"), (atoms, charges, bonds))},
                categorical_probabilities={name: value.detach().cpu().tolist() for name, value in
                    zip(("atoms", "charges", "symmetric_bonds"), probabilities() if callable(probabilities) else probabilities)},
                schema=spec.to_dict(), probability_scope="Saved raw atom/charge softmax and symmetric bond softmax before occupancy/diagonal gating; fixed coordinates, no repair/resampling")
        self.records.append(record)

    def summary(self):
        from .chemistry_classification import classification_summary
        return classification_summary(self.records, max_examples_per_failure=self.maximum)


def reject_diagnostic_snapshot(payload):
    if (payload.get("diagnostic_kind") == DIAGNOSTIC_VERSION
            or payload.get("nonpromotable") is True or payload.get("promotable") is False):
        raise ValueError("Early-health diagnostic snapshot is nonpromotable and cannot resume training")


def save_failed_early_health_snapshot(path, model, optimizer, config, plan, run, *,
                                    completed_epoch, partial_epoch, updates, generators, health_sha256):
    """Exact failed state for diagnosis, deliberately outside the checkpoint format."""
    import torch
    from dataclasses import asdict
    from .train import capture_rng_state
    payload = dict(diagnostic_kind=DIAGNOSTIC_VERSION, nonpromotable=True, promotable=False,
        epoch_complete=False, completed_epoch=completed_epoch, partial_epoch=partial_epoch,
        updates=updates, model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
        graph_spec=model.spec.to_dict(), model_config=asdict(config.model), train_config=asdict(config),
        rng=capture_rng_state(generators, any(p.device.type == "cuda" for p in model.parameters())),
        source_archive_sha256=plan["archive_sha256"], source_sha256=plan["manifest_sha256"],
        plan_sha256=plan["_plan_sha256"], dataset_sha256=plan["dataset_sha256"],
        split_sha256=plan["split_sha256"], run_id=run.run_id, health_record_sha256=health_sha256,
        early_health_baseline_sha256=plan.get("_early_health_baseline_sha256"),
        failure_readiness_sha256=plan.get("_failure_readiness_sha256"),
        benchmark_gate_sha256=plan.get("_benchmark_gate_sha256"),
        starting_checkpoint=deepcopy(plan["starting_checkpoint"]),
        module_training={name: module.training for name, module in model.named_modules()},
        created_at=datetime.now(timezone.utc).isoformat(),
        scope="Failed early-health diagnostic only; partial DataLoader iterator is not a resumable checkpoint; no completed epoch or final evaluation")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    return _sha(path)
