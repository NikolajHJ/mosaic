"""Local first-order model for a single fixed-pose Boltz2 gradient trajectory."""
from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter

import torch

from latentmol.representation import RelaxedGraph


@dataclass
class _Anchor:
    step: int
    value: torch.Tensor
    inputs: tuple[torch.Tensor, ...]
    gradients: tuple[torch.Tensor, ...]

    def predict(self, inputs):
        return self.value + sum(((v - a) * g).sum()
                                for v, a, g in zip(inputs, self.inputs, self.gradients, strict=True))

    @torch.no_grad()
    def distance(self, inputs, active):
        # Maximum total variation of any real atom/charge/occupied-pair row.
        # Padding can change with the decoder and is absent from the native input.
        rows = [(v - a).abs().sum(-1)[0] * 0.5 for v, a in zip(inputs, self.inputs, strict=True)]
        return max(float(rows[0][active].max()), float(rows[1][active].max()),
                   float(rows[2][active][:, active].max()))

    def tangent_gradient(self, active):
        # Ignore row-constant derivatives on categorical simplexes. Combine
        # the two derivatives of each symmetric, off-diagonal bond decision.
        atom, charge, bond = self.gradients
        sub = bond[0][active][:, active]
        left, right = torch.triu_indices(len(sub), len(sub), offset=1, device=sub.device)
        rows = (atom[0, active, 1:], charge[0, active], sub[left, right] + sub[right, left])
        return torch.cat([(x - x.mean(-1, keepdim=True)).flatten() for x in rows]).double()


class LocalBoltzGradient:
    """Refresh a relaxed DBMol graph gradient after bounded decoder microsteps.

    The first update uses the exact derivative. Subsequent updates within the
    block differentiate a local linear model through the current decoder. Each
    block ends with an exact value and derivative. Poor agreement shrinks the
    radius used to admit future proxy evaluations; the pilot does not roll back
    already observed states. Hard scoring always uses the native scorer.
    """

    def __init__(self, native, *, steps_per_refresh=2, max_graph_distance=0.25,
                 min_graph_distance=0.04, acceptance_ratio=0.25, absolute_tolerance=0.02):
        if type(steps_per_refresh) is not int or not 2 <= steps_per_refresh <= 8:
            raise ValueError("Require two to eight optimizer steps per gradient refresh")
        values = (max_graph_distance, min_graph_distance, acceptance_ratio, absolute_tolerance)
        if (any(type(x) not in (float, int) or not math.isfinite(x) for x in values)
                or not 0 < min_graph_distance <= max_graph_distance <= 1
                or not 0 < acceptance_ratio <= 1 or absolute_tolerance < 0):
            raise ValueError("Invalid gradient reuse radius or agreement tolerance")
        if native.execution["tf32"] or native.execution["autocast"] != "none":
            raise ValueError("Local-gradient pilot requires FP32 for relaxed and hard scoring")
        self.native = native
        self.identity = native.identity
        self.steps_per_refresh = steps_per_refresh
        self.max_radius = max_graph_distance
        self.min_radius = min_graph_distance
        self.radius = max_graph_distance
        self.acceptance_ratio = acceptance_ratio
        self.absolute_tolerance = absolute_tolerance
        self.branch = self.active = self.anchor = self.last_step = None
        self.counts = dict(full_gradient_calls=0, proxy_calls=0, exact_comparisons=0,
                           poor_agreement=0, radius_refreshes=0, hard_fp32_forwards=0, retries=0)

    def _new_anchor(self, inputs, active, step):
        # Differentiate on separate leaf probabilities, then attach their VJP
        # to the live decoder. Only detached CPU graph-size tensors are cached.
        leaves = tuple(v.detach().clone().requires_grad_(True) for v in inputs)
        value, aux = self.native.score_at(RelaxedGraph(*leaves), active)
        if value.numel() != 1 or not value.requires_grad or not bool(value.isfinite().all()):
            raise ValueError("Full DBMol scalar must be finite and attached")
        grads = torch.autograd.grad(value, leaves, allow_unused=True)
        if any(g is None or not bool(g.isfinite().all()) for g in grads):
            raise ValueError("Detached or nonfinite full DBMol graph derivative")
        self.counts["full_gradient_calls"] += 1
        return _Anchor(step, value.detach(), tuple(v.detach() for v in leaves),
                       tuple(g.detach() for g in grads)), aux

    def score_branch_step(self, graph, active, *, branch, step, terminal):
        started = perf_counter()
        inputs = (graph.atoms, graph.charges, graph.bonds)
        if any(not v.requires_grad for v in inputs):
            raise ValueError("The live decoder must supply attached atom, charge and bond probabilities")
        active = self.native.validate_graph(graph, active)
        if step == 0:
            self.branch, self.active, self.anchor = branch, active.detach().clone(), None
            self.last_step, self.radius = -1, self.max_radius
        if (branch != self.branch or self.last_step is None or step != self.last_step + 1
                or not torch.equal(active, self.active)):
            raise ValueError("Local gradient must follow consecutive states of one branch and pose mask")
        previous = self.anchor
        distance = 0.0 if previous is None else previous.distance(inputs, active)
        if previous is None:
            reason = "initial"
        elif terminal:
            reason = "terminal"
        elif distance > self.radius:
            reason = "radius"
        elif step - previous.step >= self.steps_per_refresh:
            reason = "interval"
        else:
            reason = None
        record = dict(anchor_step=None if previous is None else previous.step, distance=distance,
                      radius_before=self.radius, reason=reason, kind="exact" if reason else "linear")
        aux = {}
        if reason is not None:
            self.anchor, aux = self._new_anchor(inputs, active, step)
            if reason == "radius":
                self.counts["radius_refreshes"] += 1
            if previous is not None:
                predicted = float(previous.value - previous.predict(tuple(v.detach() for v in inputs)))
                actual = float(previous.value - self.anchor.value)
                error = abs(predicted - actual)
                material = max(abs(predicted), abs(actual)) > self.absolute_tolerance
                agreement = (not material or (predicted * actual >= 0 and error <=
                    self.absolute_tolerance + (1 - self.acceptance_ratio) * abs(predicted)))
                self.counts["exact_comparisons"] += 1
                if agreement:
                    self.radius = min(self.max_radius, self.radius * 1.25)
                else:
                    self.counts["poor_agreement"] += 1
                    self.radius = max(self.min_radius, self.radius * 0.5)
                record.update(predicted_reduction=predicted, actual_reduction=actual,
                              absolute_error=error, agreement=agreement)
                prior_gradient = previous.tangent_gradient(active)
                current_gradient = self.anchor.tangent_gradient(active)
                scale = float(prior_gradient.norm() * current_gradient.norm())
                record["graph_tangent_gradient_cosine"] = (
                    float(torch.dot(prior_gradient, current_gradient)) / scale if scale else None)
        else:
            self.counts["proxy_calls"] += 1
        value = self.anchor.predict(inputs)
        if not bool(torch.isfinite(value)):
            raise ValueError("Nonfinite local DBMol model value")
        self.last_step = step
        record.update(radius_after=self.radius, seconds=perf_counter() - started)
        if terminal:
            self.branch = self.active = self.anchor = self.last_step = None
        return value, {**aux, "gradient_reuse": record}

    def score_valid_records(self, root, records):
        results = self.native.score_valid_records(root, records)
        self.counts["hard_fp32_forwards"] += len({r["scoring_key"] for r in results})
        return results

    def diagnostics(self):
        return dict(self.counts)

    def close(self):
        return self.native.close()
