"""Matched frozen-VAE latent optimization across two chemical input boundaries.

Both arms minimize their own DBMol total, with the same initial latent, fixed
occupancy, pose and step policy. Hard fixed-context rescoring is independent of
the optimized boundary; it does not rebuild atom names/poses or predict binding
improvement. Every applied checkpoint and attempted objective input is retained.
Numerical use and the accompanying tests belong on allocated DTU HPC.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import re
import time

import torch
from torch import Tensor

from latentmol.chemistry_classification import classify_repair_graph
from latentmol.graph_policy import fixed_occupancy_graph
from latentmol.optimization import NonfiniteObjective, StepControl, backtracking_step
from latentmol.train import state_digest
from .adapter import adapt_graph
from .embedding_injection import inject_native_embeddings
from .embedding_training import conditioned_hard
from .provenance import context_digest


VERSION = "epoch36-matched-embedding-relaxed-optimization-v1"
PATHS = ("relaxed", "learned")


def _deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Matched embedding optimization internal deadline reached")


def _tensor_identity(value):
    digest = hashlib.sha256()
    digest.update(f"{value.dtype}|{tuple(value.shape)}".encode())
    digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _versions(modules):
    result = []
    for role, module in modules.items():
        if module.training or any(p.requires_grad for p in module.parameters()):
            raise ValueError("Matched optimization requires frozen eval modules: " + role)
        result.extend((role, name, id(value), value._version) for name, value in
                      [*module.named_parameters(), *module.named_buffers()])
    return tuple(result)


def _plain(value):
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _write(path, value):
    Path(path).write_text(json.dumps(_plain(value), indent=2, sort_keys=True,
                                   allow_nan=False) + "\n", encoding="utf-8")


def _scalars(scores):
    if not isinstance(scores, Mapping) or "dbmol_total_loss" not in scores:
        raise ValueError("Matched optimization needs explicit DBMol total loss")
    result = {}
    for key, value in scores.items():
        if not isinstance(value, Tensor) or value.numel() != 1 or not value.is_floating_point():
            raise ValueError("DBMol output must be a reduced floating scalar: " + key)
        if not bool(value.isfinite().all()):
            raise NonfiniteObjective("Nonfinite DBMol output: " + key)
        result[key] = float(value.detach())
    return result


class FrozenComparisonObjective:
    """Two frozen scalar functions and common detached hard-graph audits.

    ``evaluate`` completes backward while native injection hooks remain active
    and returns detached values. It never returns a graph attached to closed
    hooks. Full model-state digests are checked at entry and experiment exit;
    cheap version checks and context identity also protect individual calls.
    """

    def __init__(self, scorer, prepared, bridge, vae, *, bank=None, deadline=None):
        prepared.validate()
        if vae.spec != prepared.spec:
            raise ValueError("Comparison VAE/prepared graph schemas differ")
        self.scorer, self.prepared, self.bridge, self.vae = scorer, prepared, bridge, vae
        self.bank, self.deadline = bank, deadline
        self.modules = dict(scorer=scorer, bridge=bridge, vae=vae)
        if bank is not None:
            self.modules["bank"] = bank
        self.versions = _versions(self.modules)
        self.state_digests = {key: state_digest(module.state_dict())
                              for key, module in self.modules.items()}
        self.context_sha256 = context_digest(prepared)
        self.calls = dict(vae_decode_attempted=0, vae_decode_completed=0,
            boltz_forward_attempted=0, boltz_forward_completed=0, boltz_forward_errors=0,
            backward_attempted=0, backward_completed=0, backward_errors=0,
            forward_by_boundary={key: 0 for key in (*PATHS, "hard")})

    def assert_unchanged(self, *, full=False):
        versions_ok = _versions(self.modules) == self.versions
        context_now = context_digest(self.prepared)
        context_ok = context_now == self.context_sha256
        states = {key: state_digest(module.state_dict()) for key, module in self.modules.items()} if full else None
        states_ok = None if states is None else states == self.state_digests
        if not versions_ok or not context_ok or states_ok is False:
            raise ValueError("Frozen comparison model or fixed context changed")
        return dict(frozen_versions_unchanged=versions_ok, context_unchanged=context_ok,
            context_sha256=context_now, full_state_check=full,
            frozen_state_sha256=self.state_digests if full else None,
            frozen_state_unchanged=states_ok)

    def _decode(self, z, *, features=False):
        if (z.shape != (1, self.vae.config.latent) or not z.is_floating_point()
                or not bool(z.isfinite().all())):
            raise ValueError("Comparison latent must be finite [1, VAE latent width]")
        _deadline(self.deadline)
        self.calls["vae_decode_attempted"] += 1
        result = self.vae.decode_with_features(z) if features else self.vae.decode(z)
        self.calls["vae_decode_completed"] += 1
        return result

    def _forward(self, boundary, features, bonds):
        _deadline(self.deadline)
        self.calls["boltz_forward_attempted"] += 1
        self.calls["forward_by_boundary"][boundary] += 1
        try:
            result = self.scorer(features, bonds, self.prepared.coordinates)
        except BaseException:
            self.calls["boltz_forward_errors"] += 1
            raise
        self.calls["boltz_forward_completed"] += 1
        return result

    def evaluate(self, path, z, *, require_gradient=False, include_hard_score=False):
        if path not in PATHS:
            raise ValueError("Unknown comparison chemical boundary")
        self.assert_unchanged()
        with torch.enable_grad() if require_gradient else torch.no_grad():
            point = z.detach().clone().requires_grad_(require_gradient)
            if path == "relaxed":
                decoded = self._decode(point)
                graph = fixed_occupancy_graph(decoded, self.prepared.active, self.prepared.spec, 1.0)
                features, bonds = adapt_graph(graph, self.prepared)
                scores = self._forward(path, features, bonds)
                values = _scalars(scores)
                gradient = self._gradient(scores, point) if require_gradient else None
                injection = None
            else:
                decoded, hidden = self._decode(point, features=True)
                with torch.no_grad():
                    hard = conditioned_hard(decoded, self.prepared.active[None], self.prepared.spec)
                    features, bonds = adapt_graph(hard.to_relaxed(self.prepared.spec), self.prepared)
                predictions = self.bridge(hidden.atom_hidden, hidden.bond_input,
                                          self.prepared.active, hidden.charge_input)
                with inject_native_embeddings(self.scorer, self.prepared, predictions) as receipt:
                    scores = self._forward(path, features, bonds)
                    values = _scalars(scores)
                    gradient = self._gradient(scores, point) if require_gradient else None
                injection = receipt.record()
        result = dict(path=path, scores=values, loss=values["dbmol_total_loss"],
            loss_dtype=str(scores["dbmol_total_loss"].dtype),
            gradient=None if gradient is None else gradient.detach(),
            gradient_norm=None if gradient is None else float(gradient.double().norm()),
            context_sha256=self.context_sha256, injection_receipt=injection)
        if include_hard_score:
            with torch.no_grad():
                hard = conditioned_hard(decoded, self.prepared.active[None], self.prepared.spec)
                features, bonds = adapt_graph(hard.to_relaxed(self.prepared.spec), self.prepared)
                result["hard_fixed_context_scores"] = _scalars(self._forward("hard", features, bonds))
        self.assert_unchanged()
        return result

    def _gradient(self, scores, point):
        _deadline(self.deadline)
        self.calls["backward_attempted"] += 1
        try:
            gradient = torch.autograd.grad(scores["dbmol_total_loss"].reshape(()), point)[0]
            if not bool(gradient.isfinite().all()):
                raise ValueError("Nonfinite latent derivative")
        except BaseException:
            self.calls["backward_errors"] += 1
            raise
        self.calls["backward_completed"] += 1
        return gradient.detach()

    @torch.no_grad()
    def audit(self, z):
        """Retain valid and invalid identities and three same-context scores."""
        self.assert_unchanged()
        decoded, hidden = self._decode(z, features=True)
        hard = conditioned_hard(decoded, self.prepared.active[None], self.prepared.spec)
        natural = decoded.harden(self.prepared.spec)
        classifications = {name: classify_repair_graph(graph.atoms[0], graph.charges[0],
            graph.bonds[0], self.prepared.spec) for name, graph in (("conditioned", hard), ("natural", natural))}
        features, bonds = adapt_graph(hard.to_relaxed(self.prepared.spec), self.prepared)
        hard_scores = _scalars(self._forward("hard", features, bonds))
        predictions = self.bridge(hidden.atom_hidden, hidden.bond_input,
                                  self.prepared.active, hidden.charge_input)
        with inject_native_embeddings(self.scorer, self.prepared, predictions) as receipt:
            learned_scores = _scalars(self._forward("learned", features, bonds))
        graph = fixed_occupancy_graph(decoded, self.prepared.active, self.prepared.spec, 1.0)
        features, bonds = adapt_graph(graph, self.prepared)
        relaxed_scores = _scalars(self._forward("relaxed", features, bonds))
        upper = torch.triu(torch.ones_like(hard.bonds, dtype=torch.bool), diagonal=1)
        record = dict(context_sha256=self.context_sha256, classifications=classifications,
            canonical_identities={name: value["smiles"] for name, value in classifications.items()},
            scores=dict(hard_fixed_context=hard_scores, relaxed=relaxed_scores, learned=learned_scores),
            learned_minus_hard_total_loss=learned_scores["dbmol_total_loss"] - hard_scores["dbmol_total_loss"],
            relaxed_minus_hard_total_loss=relaxed_scores["dbmol_total_loss"] - hard_scores["dbmol_total_loss"],
            graph_policy_difference=dict(natural_occupied_slots=int((natural.atoms != 0).sum()),
                conditioned_occupied_slots=int((hard.atoms != 0).sum()),
                atom_label_differences=int((natural.atoms != hard.atoms).sum()),
                charge_label_differences=int((natural.charges != hard.charges).sum()),
                unordered_bond_label_differences=int(((natural.bonds != hard.bonds) & upper).sum())),
            injection_receipt=receipt.record(), invalid_scores_retained_as_tensor_diagnostics=True,
            hard_score_scope="same-context-hard-chemistry-fixed-pose-and-names; no-native-rebuild",
            useful_optimization_validated=False)
        if self.bank is not None:
            targets = self.bank.targets(hard, self.prepared.spec)
            _, losses = self.bank.loss(predictions, targets)
            record["feature_audit"] = {key: float(value) for key, value in
                {**losses, **self.bank.identity_metrics(predictions, hard, self.prepared.spec)}.items()}
        tensors = {"z": z.detach().cpu().clone(), **{
            name: {field: getattr(graph, field).detach().cpu().clone() for field in ("atoms", "charges", "bonds")}
            for name, graph in (("conditioned", hard), ("natural", natural))}}
        self.assert_unchanged()
        return dict(record=record, tensors=tensors)


def run_matched_optimization(starts, objective, output, *, steps, learning_rate,
                             max_gradient_norm, trust_radius, control=None,
                             deadline=None, paths=PATHS):
    """Run equal update budgets; record unequal actual forward costs honestly.

    Objective protocol is ``evaluate(path,z,require_gradient=bool)`` returning
    scalar loss/dtype, scores and detached gradient; ``audit(z)`` returns JSON
    ``record`` plus tensor ``tensors``; ``assert_unchanged(full=bool)`` validates
    identity. Toy protocols test scheduling without native pretrained models.
    A fatal or all-nonfinite evaluation persists failure and then propagates.
    """
    if not isinstance(starts, Mapping) or not starts:
        raise ValueError("Matched comparison needs named starting latents")
    if type(steps) is not int or steps < 1 or tuple(paths) != PATHS:
        raise ValueError("Require positive matched update budget and relaxed/learned arms")
    control = control or StepControl(version="best_trial_v2")
    if control.version != "best_trial_v2":
        raise ValueError("New comparison uses authorized best_trial_v2")
    for key, value in (("learning_rate", learning_rate), ("max_gradient_norm", max_gradient_norm),
                       ("trust_radius", trust_radius)):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("Invalid matched optimization " + key)
    for name, start in starts.items():
        if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", name)
                or not isinstance(start, Tensor) or start.ndim != 2 or start.shape[0] != 1
                or not start.numel() or not start.is_floating_point() or not bool(start.isfinite().all())):
            raise ValueError("Named starts need safe names and finite [1, latent] tensors")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(version=VERSION, status="running_not_passed", step_control=control.record(),
        steps_requested=steps, paths=list(paths), learning_rate=learning_rate,
        max_gradient_norm=max_gradient_norm, trust_radius=trust_radius, starts={}, runs=[],
        claim="matched update budgets at one fixed context; objective totals are boundary-specific",
        total_loss_cross_boundary_ranking=False, existing_vae_continuation="untouched")
    report["frozen_before"] = objective.assert_unchanged(full=True)
    _write(output / "comparison.json", report)
    begun = time.monotonic()
    run = None
    try:
        for name, initial in starts.items():
            _deadline(deadline)
            initial = initial.detach().clone()
            initial_sha = _tensor_identity(initial)
            report["starts"][name] = dict(latent_sha256=initial_sha, shape=list(initial.shape), dtype=str(initial.dtype))
            torch.save({"z": initial.cpu()}, output / (name + "-start.pt"))
            for path in paths:
                directory = output / name / path
                directory.mkdir(parents=True)
                run = dict(start_id=name, path=path, start_latent_sha256=initial_sha,
                    status="running_not_passed", applied=0, improving=0, forced_fallback=0,
                    full_step_nonincrease=0, no_op=0, iterations=[], checkpoints=[],
                    callback_calls=dict(attempted=0, completed=0, failed=0), best_observed=None)
                report["runs"].append(run)
                point = initial.detach().clone()
                journal = directory / "objective-calls.jsonl"

                def persist():
                    report["elapsed_seconds"] = time.monotonic() - begun
                    report["actual_model_calls"] = _plain(getattr(objective, "calls", {}))
                    _write(output / "comparison.json", report)

                def call(kind, candidate, *, gradient=False):
                    _deadline(deadline)
                    index = run["callback_calls"]["attempted"]
                    run["callback_calls"]["attempted"] += 1
                    row = dict(call_id=index, kind=kind, path=path, latent=_plain(candidate),
                               latent_sha256=_tensor_identity(candidate), require_gradient=gradient)
                    with journal.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({**row, "status": "attempted"}, allow_nan=False) + "\n")
                    try:
                        result = objective.evaluate(path, candidate, require_gradient=gradient)
                        if (type(result.get("loss")) not in (int, float)
                                or not math.isfinite(result["loss"])):
                            raise NonfiniteObjective("Comparison callback returned nonfinite scalar")
                        derivative = result.get("gradient")
                        if gradient and (not isinstance(derivative, Tensor) or derivative.shape != candidate.shape
                                or not bool(derivative.isfinite().all())):
                            raise ValueError("Comparison callback lacks a finite matched latent gradient")
                    except BaseException as error:
                        run["callback_calls"]["failed"] += 1
                        with journal.open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps({**row, "status": "failed", "error_type": type(error).__name__,
                                "error": str(error)}, allow_nan=False) + "\n")
                        persist()
                        raise
                    run["callback_calls"]["completed"] += 1
                    with journal.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({**row, "status": "completed", "result": _plain(result)}, allow_nan=False) + "\n")
                    persist()
                    return result

                def checkpoint(update, selected=None):
                    _deadline(deadline)
                    record = dict(update=update, latent_sha256=_tensor_identity(point),
                        status="audit_running_not_passed",
                        initial_displacement_norm=float((point - initial).double().norm()))
                    if record["initial_displacement_norm"] > trust_radius + 1e-5:
                        raise ValueError("Applied checkpoint escaped the original-latent trust ball")
                    # Preserve an applied latent even if its downstream audit
                    # fails; partial evidence is never labelled a full audit.
                    from latentmol.experiments import file_sha256
                    latent_path = directory / f"checkpoint-{update:04d}-latent.pt"
                    torch.save({"z": point.detach().cpu().clone()}, latent_path)
                    record.update(latent_tensor_path=str(latent_path),
                                  latent_tensor_sha256=file_sha256(latent_path))
                    run["checkpoints"].append(record)
                    _write(directory / f"checkpoint-{update:04d}.json", record)
                    persist()
                    try:
                        # Audit forwards are included in actual_model_calls.
                        audit = objective.audit(point)
                    except BaseException as error:
                        record.update(status="failed_audit_not_passed", error_type=type(error).__name__, error=str(error))
                        _write(directory / f"checkpoint-{update:04d}.json", record)
                        persist()
                        raise
                    record.update(audit["record"], status="completed_audit")
                    loss = record["scores"][path]["dbmol_total_loss"]
                    if selected is not None:
                        record["selected_objective_loss"] = selected
                        record["audit_minus_selected_objective_loss"] = loss - selected
                    tensor_path = directory / f"checkpoint-{update:04d}.pt"
                    tensors = dict(audit["tensors"])
                    tensors["z"] = point.detach().cpu().clone()
                    torch.save(tensors, tensor_path)
                    record.update(tensor_path=str(tensor_path), tensor_sha256=file_sha256(tensor_path))
                    _write(directory / f"checkpoint-{update:04d}.json", record)
                    best = run["best_observed"]
                    if best is None or loss < best["loss"]:
                        run["best_observed"] = dict(update=update, loss=loss, objective_path=path,
                            selection_scope="same-path-applied-checkpoint-audit-forwards",
                            context_sha256=record["context_sha256"], latent_sha256=record["latent_sha256"],
                            tensor_path=str(tensor_path), tensor_sha256=record["tensor_sha256"])
                    persist()

                checkpoint(0)
                for update in range(1, steps + 1):
                    _deadline(deadline)
                    observed = call("gradient_baseline", point, gradient=True)
                    iteration = dict(update=update)
                    run["iterations"].append(iteration)
                    selected = backtracking_step(point, initial, observed["gradient"],
                        baseline_loss=observed["loss"], baseline_loss_dtype=observed["loss_dtype"],
                        learning_rate=learning_rate, max_gradient_norm=max_gradient_norm,
                        trust_radius=trust_radius, control=control,
                        score=lambda candidate: {key: value for key, value in call("trial", candidate).items()
                                                 if key != "gradient"}, report=iteration, persist=persist)
                    if not selected.accepted:
                        raise RuntimeError("best_trial_v2 returned an unapplied update")
                    point = selected.point.detach().clone()
                    run["applied"] += 1
                    run["improving"] += int(iteration["monotone_descent"])
                    run["forced_fallback"] += int(iteration["acceptance_kind"] == "best_finite_fallback")
                    run["full_step_nonincrease"] += int(iteration["acceptance_kind"] == "full_step_nonincrease")
                    run["no_op"] += int(iteration["no_op"])
                    checkpoint(update, iteration["selected_loss"])
                run["status"] = "completed"
                persist()
        report["frozen_after"] = objective.assert_unchanged(full=True)
        report.update(status="completed_comparison_not_binding_validated", matched_update_budget_completed=True)
    except BaseException as error:
        if run is not None:
            run.update(status="failed_not_passed", error_type=type(error).__name__, error=str(error))
        report.update(status="failed_not_passed", matched_update_budget_completed=False,
                      error_type=type(error).__name__, error=str(error))
        try:
            report["frozen_after"] = objective.assert_unchanged(full=True)
        except BaseException as identity_error:
            report["frozen_after_error"] = dict(error_type=type(identity_error).__name__, error=str(identity_error))
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - begun
        report["actual_model_calls"] = _plain(getattr(objective, "calls", {}))
        _write(output / "comparison.json", report)
    return report
