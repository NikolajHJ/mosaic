"""Shared-graph scalar objectives and measured per-model latent derivatives.

No molecule conversion occurs here. Branches execute sequentially on the same
graph object, with input derivatives retained and model parameters frozen. This
reduces overlapping scorer activation lifetimes, not resident model weights.
"""

from dataclasses import asdict
import time

import torch

from latentmol.scoring import check_branches, evaluate_branches


def scalar_loss(outputs, objective):
    """Require an already reduced scalar; never average arbitrary pair logits."""
    if objective.output not in outputs:
        raise ValueError(f"Missing native output {objective.output!r}")
    value = outputs[objective.output]
    if not isinstance(value, torch.Tensor) or value.numel() != 1 or not torch.isfinite(value).all():
        raise ValueError("Objective output must be a finite scalar, reduced explicitly by the adapter")
    raw = value.reshape(())
    sign = -1.0 if objective.direction == "maximize" else 1.0
    scaled = sign * (raw - objective.offset) / objective.scale
    return scaled, {"raw": float(raw.detach()), "signed_scaled": float(scaled.detach()),
                    "output": objective.output, "direction": objective.direction,
                    "scale": objective.scale, "offset": objective.offset}


def _sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


class Counters:
    """Attempted and completed oracle/autograd calls survive partial failures."""
    def __init__(self, persist=None, *, before_call=None):
        self.values = {}
        self.persist = persist
        self.before_call = before_call

    def bump(self, model, purpose, action, seconds=0.0):
        row = self.values.setdefault(model, {}).setdefault(purpose, {})
        row[action] = row.get(action, 0) + 1
        row["elapsed_seconds"] = row.get("elapsed_seconds", 0.0) + seconds
        if self.persist:
            self.persist(self.values)

    def call(self, model, purpose, action, function, device):
        if self.before_call is not None:
            self.before_call(model, purpose, action)
        self.bump(model, purpose, action + "_attempted")
        _sync(device)
        start = time.perf_counter()
        try:
            result = function()
            _sync(device)
        except Exception:
            self.bump(model, purpose, action + "_failed", time.perf_counter() - start)
            raise
        self.bump(model, purpose, action + "_completed", time.perf_counter() - start)
        return result


def gradient_cosines(gradients):
    result = {}
    names = list(gradients)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            a, b = gradients[left], gradients[right]
            denominator = float(a.norm() * b.norm())
            result[f"{left}|{right}"] = float((a * b).sum()) / denominator if denominator > 0 else None
    return result


def score_graph(graph, point, *, runtimes, models, weights, spec,
                counters, purpose, gradients):
    """Return detached diagnostics plus the exact weighted scalar-sum gradient.

    Weights of zero disable a branch completely. A missing/detached/nonfinite
    positive-weight term is an error, not silently dropped or renormalized.
    Distinct branch autograd graphs are released after use; only the common
    decoder graph is needed again. No per-gradient normalization changes the
    configured objective; optional global clipping happens in the update loop.
    """
    names = [name for name in weights if weights[name] > 0]
    branches = {name: runtimes[name].branch for name in names}
    versions = tuple(value._version for value in (graph.atoms, graph.charges, graph.bonds))
    supports = check_branches(graph, branches, spec=spec,
                              allow_lossy={name: models[name].allow_lossy for name in names})
    if versions != tuple(value._version for value in (graph.atoms, graph.charges, graph.bonds)):
        raise ValueError("A branch mutated the shared graph during eligibility assessment")
    if gradients:
        for name, branch in branches.items():
            if branch.manifest.capabilities.gradient_mode not in ("pathwise", "partial"):
                raise ValueError(f"{name}: no supported pathwise derivative; use a separate oracle experiment")
    rows, derivatives = {}, {}
    total = 0.0
    for name in names:
        branch, cfg = branches[name], models[name]
        output = counters.call(name, purpose, "forward", lambda: evaluate_branches(
            graph, {name: branch}, spec=spec, allow_lossy={name: cfg.allow_lossy})[name], point.device)
        loss, row = scalar_loss(output, cfg.objective)
        row.update(weight=weights[name], weighted_loss=weights[name] * float(loss.detach()),
                   support=asdict(supports[name]), gradient_norm=None, weighted_gradient_norm=None)
        if gradients:
            if not loss.requires_grad:
                raise ValueError(f"{name}: objective is detached from molecular z")
            derivative, = counters.call(name, purpose, "backward", lambda: torch.autograd.grad(
                loss, point, retain_graph=True, allow_unused=True), point.device)
            if derivative is None or not derivative.isfinite().all():
                raise ValueError(f"{name}: missing or nonfinite latent gradient")
            derivatives[name] = derivative.detach()
            row.update(gradient_norm=float(derivative.norm()),
                       weighted_gradient_norm=float((weights[name] * derivative).norm()),
                       gradient_status="zero_at_current_point" if float(derivative.norm()) <= 1e-12 else "nonzero")
        total += row["weighted_loss"]
        rows[name] = row
        del loss, output
        if versions != tuple(value._version for value in (graph.atoms, graph.charges, graph.bonds)):
            raise ValueError(f"{name}: branch mutated the shared graph")
    combined = sum((weights[name] * value for name, value in derivatives.items()),
                   torch.zeros_like(point)) if gradients else None
    if combined is not None and not combined.isfinite().all():
        raise ValueError("Nonfinite combined gradient")
    return {"models": rows, "weighted_loss": total,
            "gradient_cosines": gradient_cosines(derivatives),
            "combined_gradient_norm": float(combined.norm()) if combined is not None else None}, combined
