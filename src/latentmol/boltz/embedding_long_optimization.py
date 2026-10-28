"""Guarded long learned-embedding trajectories; numerical execution is HPC-only.

The guard is detached eligibility, never an added DBMol loss. Historical paired
optimization and shared step selection are unchanged. One immutable state per
completed update plus append-only receipts makes history storage linear.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time

import numpy as np
import torch

from latentmol.chemistry_classification import classify_repair_graph
from latentmol.multimodel.study import tensor_digest
from latentmol.optimization import NonfiniteObjective, StepControl, backtracking_step
from latentmol.representation import HardGraphBatch
from .adapter import adapt_graph
from .embedding_comparison import _plain, _scalars
from .embedding_comparison_policy import comparison_graph
from .embedding_training import conditioned_hard


VERSION = "guarded-learned-embedding-long-optimization-v1"
_BINDING_SHA = ("source_sha256", "plan_sha256", "vae_checkpoint_sha256",
    "bridge_checkpoint_sha256", "calibration_sha256", "gate_sha256", "context_sha256",
    "comparison_graph_policy_sha256", "start_sha256", "gradient_sha256")


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _json_digest(value):
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _sync_directory(path):
    # Production runs use Linux. Windows fixtures need no directory descriptor.
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _atomic(path, value, *, tensor=False):
    """Publish a new immutable receipt only after complete saved bytes exist."""
    path = Path(path)
    if path.exists():
        raise FileExistsError("Preserve existing trajectory artifact: " + str(path))
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("xb") as stream:
        if tensor:
            torch.save(value, stream)
        else:
            stream.write((json.dumps(_plain(value), sort_keys=True, indent=2,
                allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _sync_directory(path.parent)
    return dict(path=str(path.resolve()), sha256=file_sha256(path))


def _append(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_plain(value), sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _rng_state():
    array = np.random.get_state()
    return dict(python=random.getstate(), numpy=[array[0], array[1].tolist(),
        array[2], array[3], array[4]], torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def _restore_rng(value):
    random.setstate(value["python"])
    array = value["numpy"]
    np.random.set_state((array[0], np.asarray(array[1], dtype=np.uint32), *array[2:]))
    torch.set_rng_state(value["torch"])
    if value["cuda"]:
        if not torch.cuda.is_available() or len(value["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume CUDA RNG owner/device count changed")
        torch.cuda.set_rng_state_all(value["cuda"])


class EligibilityRejected(NonfiniteObjective):
    """Finite candidate excluded by declared chemistry/fidelity domain guards."""

    def __init__(self, assessment):
        self.assessment = assessment
        super().__init__("Candidate ineligible: " + ",".join(assessment["rejection_reasons"]))


class GuardedEmbeddingObjective:
    """FrozenComparisonObjective plus a cheap, detached candidate-domain check.

    ``mse_caps`` must contain every native projection key exactly once. The
    existing group-balanced per-head MSE is used, including NONE/present groups.
    Strict validity applies to the fixed-occupancy graph actually being scored.
    Natural occupancy/validity is retained but does not change eligibility.
    """

    def __init__(self, base, mse_caps):
        if base.bank is None or base.bank.training or any(p.requires_grad for p in base.bank.parameters()):
            raise ValueError("Long optimization requires a frozen eval native projection bank")
        expected = {descriptor.key for descriptor in base.bank.descriptors}
        if (not isinstance(mse_caps, Mapping) or set(mse_caps) != expected
                or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                       for value in mse_caps.values())):
            raise ValueError("Require positive finite calibrated caps for every projection")
        self.base, self.mse_caps = base, dict(mse_caps)
        self.context_sha256 = base.context_sha256
        self.comparison_graph_policy_sha256 = base.comparison_graph_policy_sha256
        self.guard_record = dict(version=VERSION, mse_caps=dict(mse_caps),
            strict_conditioned_validity_required=True, natural_validity="diagnostic",
            cap_comparison="per-head group-balanced mse <= cap", guard_is_loss_penalty=False)
        self.guard_sha256 = _json_digest(self.guard_record)
        self.calls = dict(assess_attempted=0, assess_completed=0, rejected=0)

    def assert_unchanged(self, *, full=False):
        if (self.guard_record["mse_caps"] != self.mse_caps
                or _json_digest(self.guard_record) != self.guard_sha256):
            raise ValueError("Calibrated guard changed during optimization")
        return self.base.assert_unchanged(full=full)

    @torch.no_grad()
    def assess(self, z, *, include_tensors=False):
        self.assert_unchanged()
        self.calls["assess_attempted"] += 1
        decoded, hidden = self.base._decode(z, features=True)
        prepared = self.base.prepared
        hard = conditioned_hard(decoded, prepared.active[None], prepared.spec)
        natural = decoded.harden(prepared.spec)
        classifications = {name: classify_repair_graph(graph.atoms[0], graph.charges[0],
            graph.bonds[0], prepared.spec) for name, graph in (("conditioned", hard), ("natural", natural))}
        predictions = self.base.bridge(hidden.atom_hidden, hidden.bond_input,
            prepared.active, hidden.charge_input)
        _, metrics = self.base.bank.loss(predictions, self.base.bank.targets(hard, prepared.spec))
        metrics.update(self.base.bank.identity_metrics(predictions, hard, prepared.spec))
        features = {name: float(value) for name, value in metrics.items()}
        if any(not math.isfinite(value) for value in features.values()):
            raise ValueError("Nonfinite detached feature audit")
        reasons = []
        if not classifications["conditioned"]["strict_teacher_valid"]:
            reasons.append("strict_invalid:" + str(classifications["conditioned"]["reason"]))
        reasons.extend("mse_cap:" + key for key, cap in self.mse_caps.items()
                       if features[key + "/mse"] > cap)
        labels = {field: getattr(hard, field).detach().cpu().clone() for field in ("atoms", "charges", "bonds")}
        record = dict(eligible=not reasons, rejection_reasons=reasons,
            guard_sha256=self.guard_sha256,
            context_sha256=self.context_sha256, latent_sha256=tensor_digest({"z": z}),
            hard_graph_sha256=tensor_digest(labels), feature_audit=features,
            classifications=classifications,
            canonical_identities={name: value["smiles"] for name, value in classifications.items()})
        tensors = dict(z=z.detach().cpu().clone(), conditioned=labels,
            natural={field: getattr(natural, field).detach().cpu().clone() for field in ("atoms", "charges", "bonds")})
        if include_tensors:
            graph = comparison_graph(decoded, prepared.active, prepared.spec,
                policy=self.base.comparison_graph_policy)
            tensors["relaxed"] = {field: getattr(graph, field).detach().cpu().clone()
                                  for field in ("atoms", "charges", "bonds")}
        self.calls["assess_completed"] += 1
        self.assert_unchanged()
        return dict(record=record, tensors=tensors)

    def evaluate(self, z, *, require_gradient=False, assessment=None):
        assessment = self.assess(z) if assessment is None else assessment
        record = assessment["record"]
        if (record["latent_sha256"] != tensor_digest({"z": z})
                or record["context_sha256"] != self.context_sha256
                or record["guard_sha256"] != self.guard_sha256):
            raise ValueError("Guard assessment belongs to a different latent")
        if not record["eligible"]:
            self.calls["rejected"] += 1
            raise EligibilityRejected(record)
        return self.base.evaluate("learned", z, require_gradient=require_gradient)

    @torch.no_grad()
    def hard_score(self, assessment):
        self.assert_unchanged()
        labels = assessment["tensors"]["conditioned"]
        hard = HardGraphBatch(*(labels[field].to(self.base.prepared.coordinates.device)
                              for field in ("atoms", "charges", "bonds")))
        features, bonds = adapt_graph(hard.to_relaxed(self.base.prepared.spec), self.base.prepared)
        scores = _scalars(self.base._forward("hard", features, bonds))
        self.assert_unchanged()
        return scores

    def observe(self, z, assessment, hard_scores):
        learned = self.evaluate(z, assessment=assessment)
        relaxed = self.base.evaluate("relaxed", z)
        return dict(**assessment["record"], scores=dict(learned=learned["scores"],
            relaxed=relaxed["scores"], hard_fixed_context=hard_scores),
            learned_minus_hard_total_loss=learned["loss"] - hard_scores["dbmol_total_loss"],
            hard_score_scope="same exact hard slot graph, context, fixed pose and names; no native rebuild")


def _validate_binding(binding, objective, start):
    if (not isinstance(binding, Mapping) or set(binding) != {*_BINDING_SHA, "baseline_scores"}
            or any(not isinstance(binding[key], str) or re.fullmatch(r"[0-9a-f]{64}", binding[key]) is None
                   for key in _BINDING_SHA)
            or not isinstance(binding["baseline_scores"], Mapping)
            or "dbmol_total_loss" not in binding["baseline_scores"]
            or any(type(value) not in (int, float) or not math.isfinite(value)
                   for value in binding["baseline_scores"].values())):
        raise ValueError("Require exact source/plan/asset/start/native gate binding")
    if (binding["context_sha256"] != objective.context_sha256
            or binding["comparison_graph_policy_sha256"] != objective.comparison_graph_policy_sha256
            or binding["start_sha256"] != tensor_digest({"z": start})):
        raise ValueError("Native gate binding differs from actual start/context/policy")


def run_guarded_optimization(start, objective, output, *, case_id, binding,
        target_updates=10000, segment_updates=10000, learning_rate=400.,
        max_gradient_norm=1., trust_radius=3., control=None, observation_every=10,
        resume=None, deadline=None, reserve_seconds=300., max_consecutive_noop=25):
    """Run one bounded segment; target counts iterations, including honest no-ops.

    ``resume`` is exactly {path, sha256} for a committed state emitted here.
    Split runs use the original z0 radius and restore all RNG/cumulative counts.
    Fixed-output scientific identities, guard caps and settings must be identical.
    Stop only between atomic updates with a caller-chosen runtime reserve.
    The reserve is not a guarantee against a single unexpectedly slow update.
    """
    control = control or StepControl(version="best_trial_v2", max_trials=16)
    if (control.version != "best_trial_v2" or control.max_trials != 16
            or type(target_updates) is not int or target_updates < 1
            or type(segment_updates) is not int or segment_updates < 1
            or type(observation_every) is not int or observation_every < 1
            or type(max_consecutive_noop) is not int or max_consecutive_noop < 1
            or not isinstance(case_id, str) or re.fullmatch(r"[A-Za-z0-9_-]+", case_id) is None
            or not isinstance(start, torch.Tensor) or start.ndim != 2 or start.shape[0] != 1
            or not start.is_floating_point() or not bool(start.isfinite().all())
            or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                   for value in (learning_rate, max_gradient_norm, trust_radius, reserve_seconds))):
        raise ValueError("Invalid guarded trajectory settings")
    _validate_binding(binding, objective, start)
    output = Path(output).resolve()
    frozen = objective.assert_unchanged(full=True)
    signature = dict(version=VERSION, case_id=case_id, binding=dict(binding),
        target_updates=target_updates, learning_rate=learning_rate,
        max_gradient_norm=max_gradient_norm, trust_radius=trust_radius,
        observation_every=observation_every, step_control=control.record(),
        max_consecutive_noop=max_consecutive_noop,
        guard=objective.guard_record, frozen=frozen)
    signature_sha = _json_digest(signature)
    if resume is None:
        output.mkdir(parents=True, exist_ok=False)
        for directory in ("states", "segments", "observations", "hard-cache", "best"):
            (output / directory).mkdir()
        _atomic(output / "manifest.json", dict(signature=signature, sha256=signature_sha))
        state = dict(version=VERSION, signature_sha256=signature_sha, case_id=case_id,
            update=0, z0=start.detach().cpu().clone(), z=start.detach().cpu().clone(),
            counts=dict(iterations=0, applied=0, improving=0, forced_fallback=0,
                        full_step_nonincrease=0, no_op=0, all_ineligible_no_op=0,
                        consecutive_noop=0),
            best_valid=None, last_hard_graph_sha256=None, gate_binding_verified=False,
            baseline_observation=None, last_observation=None)
        state.update(hard_cache_chain=hashlib.sha256(b"").hexdigest(), hard_cache_entries=0)
    else:
        if set(resume) != {"path", "sha256"} or file_sha256(resume["path"]) != resume["sha256"]:
            raise ValueError("Resume checkpoint authentication failed")
        state = torch.load(resume["path"], map_location="cpu", weights_only=True)
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if (manifest != dict(signature=signature, sha256=signature_sha)
                or state["signature_sha256"] != signature_sha or not state["gate_binding_verified"]
                or not torch.equal(state["z0"], start.cpu())
                or tensor_digest({"z": state["z0"]}) != binding["start_sha256"]):
            raise ValueError("Resume source/plan/frozen state/guard/original start changed")
        if (output / "failed.json").exists():
            raise ValueError("Failed trajectory needs explicit investigation; no automatic resume")
        objective.calls = copy.deepcopy(state["guard_calls"])
        objective.base.calls = copy.deepcopy(state["model_calls"])
        _restore_rng(state["rng"])
    if not 0 <= state["update"] < target_updates:
        raise ValueError("Resume update is outside remaining target budget")
    if state["counts"]["consecutive_noop"] >= max_consecutive_noop:
        raise ValueError("Scientific stall is terminal for this declared plan")
    cache_index, chain = {}, hashlib.sha256(b"").hexdigest()
    index_path = output / "hard-cache/index.jsonl"
    if index_path.exists():
        with index_path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if (set(row) != {"key", "sha256", "previous"} or row["previous"] != chain
                        or row["key"] in cache_index
                        or any(re.fullmatch(r"[0-9a-f]{64}", row[key]) is None for key in ("key", "sha256"))):
                    raise ValueError("Hard-cache receipt chain is corrupt")
                cache_index[row["key"]] = row["sha256"]
                chain = _json_digest(row)
    if chain != state["hard_cache_chain"] or len(cache_index) != state["hard_cache_entries"]:
        raise ValueError("Hard-cache receipt chain differs from committed resume state")
    point = state["z"].to(start)
    original = state["z0"].to(start)
    if float((point - original).double().norm()) > trust_radius + 1e-5:
        raise ValueError("Resume latent escaped original global trust ball")
    segment = output / "segments" / ("from-%06d" % state["update"])
    segment.mkdir(exist_ok=False)
    _atomic(segment / "intent.json", dict(signature_sha256=signature_sha,
        resume=resume, starting_update=state["update"], segment_updates=segment_updates))
    committed = None

    def save_state():
        state.update(z=point.detach().cpu().clone(), rng=_rng_state(),
            guard_calls=copy.deepcopy(objective.calls), model_calls=copy.deepcopy(objective.base.calls))
        saved = _atomic(output / "states" / ("state-%06d.pt" % state["update"]), state, tensor=True)
        _append(segment / "commits.jsonl", dict(update=state["update"], current_state=saved))
        return saved

    def hard_and_best(assessment, update):
        record = assessment["record"]
        key = record["hard_graph_sha256"]
        cache = output / "hard-cache" / (key + ".json")
        if cache.exists():
            if key not in cache_index or file_sha256(cache) != cache_index[key]:
                raise ValueError("Hard-cache bytes differ from authenticated receipt")
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if (cached["context_sha256"] != objective.context_sha256
                    or cached["hard_graph_sha256"] != key
                    or cached["signature_sha256"] != signature_sha):
                raise ValueError("Hard cache context/slot-graph binding mismatch")
            scores = cached["scores"]
        else:
            if key in cache_index:
                raise ValueError("Authenticated hard-cache file is missing")
            scores = objective.hard_score(assessment)
            saved = _atomic(cache, dict(context_sha256=objective.context_sha256,
                hard_graph_sha256=key, signature_sha256=signature_sha, scores=scores))
            entry = dict(key=key, sha256=saved["sha256"], previous=state["hard_cache_chain"])
            _append(index_path, entry)
            cache_index[key] = saved["sha256"]
            state["hard_cache_chain"] = _json_digest(entry)
            state["hard_cache_entries"] += 1
        if (not isinstance(scores, Mapping) or "dbmol_total_loss" not in scores
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in scores.values())):
            raise ValueError("Hard-cache score schema or finite scalar contract failed")
        if record["eligible"] and (state["best_valid"] is None
                or scores["dbmol_total_loss"] < state["best_valid"]["hard_loss"]):
            tensors = _atomic(output / "best" / ("best-%06d.pt" % update),
                assessment["tensors"], tensor=True)
            state["best_valid"] = dict(update=update, hard_loss=scores["dbmol_total_loss"],
                hard_graph_sha256=key, canonical_identity=record["canonical_identities"]["conditioned"],
                context_sha256=objective.context_sha256, tensors=tensors,
                selection_scope="eligible valid observed or identity-change checkpoints; common hard fixed-context total")
        state["last_hard_graph_sha256"] = key
        return scores

    def observe(assessment, update, scores=None):
        # Reconstruct actual relaxed probabilities only for scheduled diagnostics.
        complete = objective.assess(point, include_tensors=True)
        if complete["record"] != assessment["record"]:
            raise ValueError("Detached assessment changed during scheduled observation")
        scores = hard_and_best(complete, update) if scores is None else scores
        record = objective.observe(point, complete, scores)
        tensors = _atomic(output / "observations" / ("update-%06d.pt" % update),
            complete["tensors"], tensor=True)
        saved = _atomic(output / "observations" / ("update-%06d.json" % update), dict(**record,
            update=update, tensors=tensors, signature_sha256=signature_sha))
        reference = dict(update=update, record=saved, tensors=tensors)
        state["last_observation"] = reference
        if update == 0:
            state["baseline_observation"] = reference

    try:
        if resume is None:
            assessment = objective.assess(point)
            if not assessment["record"]["eligible"]:
                raise EligibilityRejected(assessment["record"])
            evaluated = objective.evaluate(point, require_gradient=True, assessment=assessment)
            if (evaluated["scores"] != binding["baseline_scores"]
                    or tensor_digest({"gradient": evaluated["gradient"]}) != binding["gradient_sha256"]
                    or evaluated["context_sha256"] != binding["context_sha256"]
                    or evaluated["comparison_graph_policy_sha256"] != binding["comparison_graph_policy_sha256"]):
                raise ValueError("Exact learned score/gradient differs from original gate binding")
            state["gate_binding_verified"] = True
            _atomic(output / "production-binding.json", dict(passed=True, binding=dict(binding)))
            scores = hard_and_best(assessment, 0)
            observe(assessment, 0, scores)
            committed = save_state()
        else:
            committed = dict(resume)
        initial_update = state["update"]
        stop = min(target_updates, initial_update + segment_updates)
        while state["update"] < stop:
            if deadline is not None and time.monotonic() + reserve_seconds >= deadline:
                break
            update = state["update"] + 1
            base_receipt = dict(update=update, kind="gradient_baseline",
                latent_sha256=tensor_digest({"z": point}), latent=point.detach().cpu().tolist())
            _append(segment / "calls.jsonl", dict(**base_receipt, status="attempted"))
            try:
                baseline = objective.evaluate(point, require_gradient=True)
            except BaseException as error:
                _append(segment / "calls.jsonl", dict(**base_receipt, status="failed",
                    error_type=type(error).__name__, error=str(error)))
                raise
            _append(segment / "calls.jsonl", dict(**base_receipt, status="completed",
                loss=baseline["loss"], loss_dtype=baseline["loss_dtype"],
                gradient_sha256=tensor_digest({"gradient": baseline["gradient"]})))
            attempt = dict(update=update)
            trial_records = []

            def trial(candidate):
                receipt = dict(update=update, trial=len(trial_records),
                    latent=candidate.detach().cpu().tolist(), latent_sha256=tensor_digest({"z": candidate}))
                _append(segment / "calls.jsonl", dict(**receipt, status="attempted"))
                try:
                    assessment = objective.assess(candidate)
                    result = objective.evaluate(candidate, assessment=assessment)
                except EligibilityRejected as error:
                    receipt.update(status="eligibility_rejected_before_native", assessment=error.assessment,
                        rejected_graphs={name: assessment["tensors"][name] for name in ("conditioned", "natural")})
                    trial_records.append(receipt)
                    _append(segment / "calls.jsonl", receipt)
                    raise
                except BaseException as error:
                    receipt.update(status="failed", error_type=type(error).__name__, error=str(error))
                    trial_records.append(receipt)
                    _append(segment / "calls.jsonl", receipt)
                    raise
                receipt.update(status="completed", assessment=assessment["record"],
                    loss=result["loss"], loss_dtype=result["loss_dtype"])
                trial_records.append(receipt)
                _append(segment / "calls.jsonl", receipt)
                return {key: value for key, value in result.items() if key != "gradient"}

            try:
                selected = backtracking_step(point, original, baseline["gradient"],
                    baseline_loss=baseline["loss"], baseline_loss_dtype=baseline["loss_dtype"],
                    learning_rate=learning_rate, max_gradient_norm=max_gradient_norm,
                    trust_radius=trust_radius, control=control, score=trial, report=attempt)
            except NonfiniteObjective:
                if len(trial_records) != control.max_trials or any(
                        row["status"] != "eligibility_rejected_before_native" for row in trial_records):
                    raise
                attempt.update(status="all_ineligible_no_op", accepted=False, no_op=True,
                    acceptance_kind="all_ineligible_no_op", selected_loss=baseline["loss"],
                    monotone_descent=False, actual_displacement_norm=0.)
                state["counts"]["all_ineligible_no_op"] += 1
            else:
                if not selected.accepted:
                    raise RuntimeError("Shared best_trial_v2 returned unexpected unapplied selection")
                point = selected.point.detach().clone()
                state["counts"]["applied"] += int(not attempt["no_op"])
                state["counts"]["improving"] += int(attempt["monotone_descent"])
                state["counts"]["forced_fallback"] += int(attempt["acceptance_kind"] == "best_finite_fallback")
                state["counts"]["full_step_nonincrease"] += int(attempt["acceptance_kind"] == "full_step_nonincrease")
            state["counts"]["no_op"] += int(attempt["no_op"])
            attempt["eligibility_rejections"] = sum(row["status"] == "eligibility_rejected_before_native" for row in trial_records)
            attempt["native_nonfinite_trials"] = attempt["oracle_calls_nonfinite"] - attempt["eligibility_rejections"]
            state["counts"]["consecutive_noop"] = state["counts"]["consecutive_noop"] + 1 if attempt["no_op"] else 0
            state["counts"]["iterations"] += 1
            displacement = float((point - original).double().norm())
            if displacement > trust_radius + 1e-5:
                raise ValueError("Applied candidate escaped original global trust radius")
            assessment = objective.assess(point)
            if not assessment["record"]["eligible"]:
                raise EligibilityRejected(assessment["record"])
            scheduled = (update % observation_every == 0 or update == target_updates
                         or state["counts"]["consecutive_noop"] >= max_consecutive_noop)
            changed = assessment["record"]["hard_graph_sha256"] != state["last_hard_graph_sha256"]
            hard_scores = hard_and_best(assessment, update) if scheduled or changed else None
            if scheduled:
                observe(assessment, update, hard_scores)
            state["update"] = update
            record = dict(update=update, step=attempt, assessment=assessment["record"],
                displacement=displacement, hard_scores=hard_scores, identity_changed=changed,
                counts=copy.deepcopy(state["counts"]))
            _append(segment / "updates.jsonl", record)
            committed = save_state()
            if state["counts"]["consecutive_noop"] >= max_consecutive_noop:
                break
        stalled = state["counts"]["consecutive_noop"] >= max_consecutive_noop
        result = dict(version=VERSION, status="scientific_stall_target_incomplete" if stalled
            else "target_completed" if state["update"] == target_updates else "segment_complete_target_pending",
            case_id=case_id, starting_update=initial_update,
            completed_update=state["update"], target_updates=target_updates,
            stopped_for_deadline=not stalled and state["update"] < stop,
            scientific_stall=stalled, max_consecutive_noop=max_consecutive_noop, counts=state["counts"],
            current_state=committed, best_valid=state["best_valid"],
            guard_calls=objective.calls, actual_model_calls=objective.base.calls,
            signature_sha256=signature_sha, frozen_after=objective.assert_unchanged(full=True),
            no_binding_improvement_claim=True)
        _atomic(segment / "completion.json", result)
        return result
    except BaseException as error:
        try:
            final_frozen = objective.assert_unchanged(full=True)
        except BaseException as frozen_error:
            final_frozen = dict(frozen_state_unchanged=False,
                error_type=type(frozen_error).__name__, error=str(frozen_error))
        _atomic(output / "failed.json", dict(status="failed_not_resumable_without_investigation",
            last_committed=committed, error_type=type(error).__name__, error=str(error),
            guard_calls=objective.calls, actual_model_calls=objective.base.calls,
            frozen_after=final_frozen))
        raise
