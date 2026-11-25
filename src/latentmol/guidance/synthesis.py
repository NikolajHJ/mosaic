"""Opt-in frozen synthesis gradients on the shared relaxed molecular graph.

The default retains once-per-child calibration. Optional schedules, conflict
projection and a total gradient cap are exploratory direction controllers; they
do not change the reported contact loss. Sharpness explicitly tests additional
fixed-occupancy probability views, rather than claiming native hard derivatives.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from time import perf_counter

import torch


def load_models(spec, assets):
    if set(assets) - {"synfrag", "dfr", "gasa", "defog"} or not assets:
        raise ValueError("Synthesis assets must name synfrag and/or dfr")
    models = {}
    for name, asset in assets.items():
        path = Path(asset["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != asset["sha256"]:
            raise ValueError(f"Frozen synthesis weights changed: {name}")
        if name == "synfrag":
            from .synthesis_synfrag import FrozenSynFrag
            model = FrozenSynFrag(spec, path)
        elif name == "dfr":
            from .synthesis_dfr import FrozenDFRscore
            model = FrozenDFRscore(spec, path)
        elif name == "gasa":
            from .synthesis_gasa import FrozenGASA
            model = FrozenGASA(spec, path)
        else:
            from .synthesis_defog import FrozenDeFoG
            model = FrozenDeFoG(spec, path, python=asset["python"], source_path=asset["source_path"],
                model_time=asset.get("model_time", .95), timeout=asset.get("timeout", 120),
                startup_timeout=asset.get("startup_timeout", 240))
        model.eval().requires_grad_(False)
        if any(p.grad is not None for p in model.parameters()):
            raise ValueError("Frozen synthesis weights carry gradients")
        models[name] = model
    return models


class SynthesisGuidance:
    """Per-child synthesis objectives, with opt-in bounded direction controls.

    ``schedule`` multiplies the initial model ratios: ``constant`` accepts a
    scale; ``linear`` interpolates initial/final between start/end; ``phases``
    accepts nonoverlapping [start,end) intervals with scale (a scalar or a
    model-name mapping) and a default; ``cosine_wave`` cycles low/high over a
    period, starting at low. Zero scale postpones calibration.

    ``controller`` caps the *combined* auxiliary gradient to ``max_ratio``
    times an EMA contact norm (``ema_decay``, default .9). An optional
    ``invalid_grace`` requires ``raw_valid`` at evaluation; after that many
    consecutive invalid observations, ``invalid_scale`` (default zero) applies.
    Hard validity is an observation supplied outside the differentiable path.

    ``sharpness`` specifies probability-view temperatures and mean/max
    reduction. Models exposing ``forward_latent(point)`` bypass graph views.
    ``anchor={model: ..., ratio: ...}`` is one such optional extra objective;
    its frozen reference bank is owned by the caller, not updated here.
    """

    def __init__(self, models, ratios, delay_updates=0, *, coefficient_cap=100.,
                 controller=None, schedule=None, sharpness=None, anchor=None,
                 project_binding=False):
        models, ratios = dict(models), dict(ratios)
        if anchor is not None:
            if (set(anchor) != {"model", "ratio"} or "anchor" in models
                    or not callable(getattr(anchor["model"], "forward_latent", None))):
                raise ValueError("Anchor requires an independent forward_latent model and ratio")
            models["anchor"], ratios["anchor"] = anchor["model"], anchor["ratio"]
        if (set(ratios) - models.keys() or type(delay_updates) is not int or delay_updates < 0
                or not math.isfinite(coefficient_cap) or coefficient_cap <= 0
                or any(isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in ratios.values())):
            raise ValueError("Invalid synthesis gradient ratios or delay")
        self.models = {k: models[k] for k, value in ratios.items() if value}
        self.ratios = {k: float(ratios[k]) for k in self.models}
        self.delay_updates = delay_updates
        self.coefficient_cap = coefficient_cap
        self.coefficients = {}
        self.calibrated_steps = {}
        if type(project_binding) is not bool:
            raise ValueError("project_binding must be Boolean")
        self.project_binding = project_binding
        self.controller = dict(controller or {})
        if set(self.controller) - {"max_ratio", "ema_decay", "invalid_grace", "invalid_scale", "scale_cap_with_schedule"}:
            raise ValueError("Unknown synthesis controller option")
        if self.controller:
            ratio = self.controller.get("max_ratio")
            decay = self.controller.get("ema_decay", .9)
            invalid_scale = self.controller.get("invalid_scale", 0.)
            grace = self.controller.get("invalid_grace")
            if (not _nonnegative(ratio) or not _nonnegative(decay) or decay >= 1
                    or not _nonnegative(invalid_scale) or invalid_scale > 1
                    or (grace is not None and (type(grace) is not int or grace < 0))):
                raise ValueError("Invalid synthesis controller settings")
            if type(self.controller.get("scale_cap_with_schedule", False)) is not bool:
                raise ValueError("Controller schedule-cap flag must be boolean")
        self.reference_ema = None
        self.invalid_streak = 0
        self.schedule = dict(schedule or {})
        self._validate_schedule()
        self.sharpness = dict(sharpness or {})
        if self.sharpness:
            temperatures = self.sharpness.get("temperatures")
            if (set(self.sharpness) - {"temperatures", "reduction"}
                    or not isinstance(temperatures, (list, tuple)) or not temperatures
                    or any(not _nonnegative(t) or t == 0 for t in temperatures)
                    or self.sharpness.get("reduction", "mean") not in {"mean", "max"}):
                raise ValueError("Invalid synthesis sharpness views")

    @property
    def requires_raw_valid(self):
        return "invalid_grace" in self.controller

    def _validate_scale(self, value):
        if isinstance(value, dict):
            if set(value) - self.models.keys() or any(not _nonnegative(x) for x in value.values()):
                raise ValueError("Invalid per-model synthesis schedule scale")
        elif not _nonnegative(value):
            raise ValueError("Invalid synthesis schedule scale")

    def _validate_schedule(self):
        s = self.schedule
        kind = s.get("kind", "constant")
        allowed = {"constant": {"scale"}, "linear": {"start", "end", "initial", "final"},
                   "phases": {"phases", "default"},
                   "cosine_wave": {"period", "low", "high", "offset"}}
        if kind not in allowed or set(s) - allowed[kind] - {"kind"}:
            raise ValueError("Unknown synthesis ratio schedule")
        if kind == "constant":
            self._validate_scale(s.get("scale", 1.))
        elif kind == "linear":
            if (type(s.get("start", 0)) is not int or type(s.get("end")) is not int
                    or s.get("start", 0) < 0 or s["end"] <= s.get("start", 0)):
                raise ValueError("Invalid synthesis linear interval")
            for value in (s.get("initial", 0.), s.get("final", 1.)):
                if not _nonnegative(value):
                    raise ValueError("Invalid synthesis linear scale")
        elif kind == "cosine_wave":
            if (not _nonnegative(s.get("period")) or s["period"] == 0
                    or not _nonnegative(s.get("offset", 0.))
                    or any(not _nonnegative(s.get(k, v)) for k, v in (("low", 0.), ("high", 1.)))
                    or s.get("low", 0.) > s.get("high", 1.)):
                raise ValueError("Invalid synthesis wave schedule")
        else:
            phases = s.get("phases")
            if not isinstance(phases, list) or not phases:
                raise ValueError("Synthesis phases require a nonempty list")
            self._validate_scale(s.get("default", 0.))
            end = 0
            for phase in phases:
                if (set(phase) != {"start", "end", "scale"}
                        or type(phase["start"]) is not int or type(phase["end"]) is not int
                        or phase["start"] < end or phase["end"] <= phase["start"]):
                    raise ValueError("Synthesis phases must be ordered and nonoverlapping")
                self._validate_scale(phase["scale"])
                end = phase["end"]

    def _scales(self, step):
        s, kind = self.schedule, self.schedule.get("kind", "constant")
        if kind == "linear":
            progress = max(0., min(1., (step - s.get("start", 0)) / (s["end"] - s.get("start", 0))))
            value = s.get("initial", 0.) * (1 - progress) + s.get("final", 1.) * progress
        elif kind == "cosine_wave":
            phase = (step - s.get("offset", 0.)) / s["period"]
            value = s.get("low", 0.) + (s.get("high", 1.) - s.get("low", 0.)) * (1 - math.cos(2 * math.pi * phase)) / 2
        elif kind == "phases":
            value = next((p["scale"] for p in s["phases"] if p["start"] <= step < p["end"]),
                         s.get("default", 0.))
        else:
            value = s.get("scale", 1.)
        return {name: float(value.get(name, 0.) if isinstance(value, dict) else value) for name in self.models}

    def _outputs(self, name, model, graph, active, point):
        latent = getattr(model, "forward_latent", None)
        if callable(latent):
            return [(1., latent(point))]
        if not self.sharpness:
            return [(1., model(graph, active))]
        return [(float(t), model(_temperature_view(graph, active, t), active))
                for t in self.sharpness["temperatures"]]

    def evaluate(self, graph, active, point, binding_gradient, step, *, raw_valid=None):
        started = perf_counter()
        combined = torch.zeros_like(point)
        details = {}
        if (type(step) is not int or step < 0 or binding_gradient.shape != point.shape
                or not bool(point.isfinite().all()) or not bool(binding_gradient.isfinite().all())):
            raise ValueError("Invalid synthesis point, contact gradient, or step")
        reference_norm = float(binding_gradient.detach().double().norm())
        validity_scale = 1.
        if self.controller:
            decay = self.controller.get("ema_decay", .9)
            self.reference_ema = (reference_norm if self.reference_ema is None
                                  else decay * self.reference_ema + (1 - decay) * reference_norm)
            if "invalid_grace" in self.controller:
                if type(raw_valid) is not bool:
                    raise ValueError("Validity controller requires an explicit raw_valid observation")
                self.invalid_streak = 0 if raw_valid else self.invalid_streak + 1
                if self.invalid_streak > self.controller["invalid_grace"]:
                    validity_scale = self.controller.get("invalid_scale", 0.)
        scales = self._scales(step)
        if step < self.delay_updates or not any(scales.values()):
            return combined, dict(active=False, delay_updates=self.delay_updates, models={},
                                  gradient_norm=0., seconds=perf_counter() - started,
                                  schedule_scale=scales, controller_reference_norm=self.reference_ema,
                                  invalid_streak=self.invalid_streak)
        # Finite float32 elements can still overflow a float32 sum of squares.
        # Calibrate in float64 so a large finite guide cannot silently get zero
        # weight from division by an infinite diagnostic norm.
        for name, model in self.models.items():
            if not scales[name]:
                continue
            views = self._outputs(name, model, graph, active, point)
            if any(not bool(out["difficulty"].isfinite().all()) or not out["difficulty"].requires_grad
                   for _, out in views):
                raise ValueError(f"Detached/nonfinite synthesis loss: {name}")
            losses = torch.stack([output["difficulty"].mean() for _, output in views])
            reduction = self.sharpness.get("reduction", "mean")
            loss = losses.max() if reduction == "max" else losses.mean()
            if loss.ndim != 0 or not bool(loss.isfinite()) or not loss.requires_grad:
                raise ValueError(f"Detached/nonfinite synthesis loss: {name}")
            gradient, = torch.autograd.grad(loss, point, retain_graph=True, allow_unused=True)
            if gradient is None or not bool(gradient.isfinite().all()):
                raise ValueError(f"Missing/nonfinite synthesis latent path: {name}")
            norm = float(gradient.detach().double().norm())
            if not math.isfinite(norm):
                raise ValueError(f"Nonfinite synthesis gradient norm: {name}")
            warning = None
            if name not in self.coefficients:
                if norm > 1e-12 and reference_norm > 1e-12:
                    requested = self.ratios[name] * reference_norm / norm
                    self.coefficients[name] = min(requested, self.coefficient_cap)
                    self.calibrated_steps[name] = step
                    if requested > self.coefficient_cap:
                        warning = "coefficient_capped"
                else:
                    # Flat derivatives are a finding. Wait for an informative
                    # point instead of assigning a huge arbitrary multiplier.
                    warning = "calibration_pending_flat_gradient"
            coefficient = self.coefficients.get(name, 0.) * scales[name]
            combined = combined + coefficient * gradient.detach()
            raw_values = torch.stack([output["raw_score"].detach().mean() for _, output in views])
            raw = raw_values[losses.detach().argmax()] if reduction == "max" else raw_values.mean()
            if not bool(raw.isfinite().all()):
                raise ValueError(f"Nonfinite synthesis raw score: {name}")
            details[name] = dict(difficulty=float(loss.detach()), raw_score=float(raw.mean()),
                gradient_norm=norm, coefficient=coefficient,
                weighted_gradient_norm=coefficient * norm, requested_initial_ratio=self.ratios[name],
                calibration_step=self.calibrated_steps.get(name), warning=warning,
                schedule_scale=scales[name], views=[dict(temperature=t,
                    difficulty=float(out["difficulty"].detach().mean()),
                    raw_score=float(out["raw_score"].detach().mean())) for t, out in views])
            if callable(getattr(model, "forward_latent", None)) and callable(getattr(model, "diagnostics", None)):
                details[name]["latent_geometry"] = {
                    key: float(value.detach().mean()) for key, value in model.diagnostics(point).items()}
        if not bool(combined.isfinite().all()):
            raise ValueError("Nonfinite combined synthesis gradient")
        pre_control_norm = float(combined.double().norm())
        dot = float((combined.double() * binding_gradient.detach().double()).sum())
        cosine = dot / (reference_norm * pre_control_norm) if reference_norm and pre_control_norm else None
        projected = self.project_binding and dot < 0 and reference_norm > 0
        if projected:
            combined = (combined.double() - dot / reference_norm**2 * binding_gradient.detach().double()).to(point)
        combined = combined * validity_scale
        cap_scale = 1.
        if self.controller:
            norm = float(combined.double().norm())
            cap = self.controller["max_ratio"] * self.reference_ema
            if self.controller.get("scale_cap_with_schedule", False):
                cap *= max(scales.values(), default=0.)
            # Scaling only the pre-cap vector is ineffective when that vector
            # still saturates the cap. Reduce the cap as well, so invalid-state
            # suppression applies equally to weak and dominating gradients.
            cap *= validity_scale
            cap_scale = min(1., cap / norm) if norm else 1.
            combined = combined * cap_scale
        if not bool(combined.isfinite().all()):
            raise ValueError("Nonfinite controlled synthesis gradient")
        final_norm = float(combined.double().norm())
        final_dot = float((combined.double() * binding_gradient.detach().double()).sum())
        return combined, dict(active=True, delay_updates=self.delay_updates, models=details,
            contact_gradient_norm=reference_norm, gradient_norm=final_norm,
            seconds=perf_counter() - started,
            weighting=("controlled_initial_calibration" if self.controller or self.schedule or self.project_binding
                       else "fixed_after_first_informative_child_point"),
            schedule_scale=scales, controller_reference_norm=self.reference_ema,
            pre_control_gradient_norm=pre_control_norm, binding_cosine=cosine,
            final_binding_cosine=(final_dot / (reference_norm * final_norm)
                                  if reference_norm and final_norm else None),
            actual_gradient_ratio=final_norm / reference_norm if reference_norm else None,
            binding_projection_applied=projected, controller_cap_scale=cap_scale,
            invalid_streak=self.invalid_streak, validity_scale=validity_scale,
            sharpness=self.sharpness or None)


def _nonnegative(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _temperature_view(graph, active, temperature):
    """Explicit auxiliary view of a fixed-occupancy graph, preserving support.

    This is not an independent chemistry conversion. Re-tempering already
    occupancy-gated variable-size marginals would change their meaning, so that
    case is rejected. The unchanged graph remains the contact model input.
    """
    from latentmol.representation import RelaxedGraph

    if (not isinstance(graph, RelaxedGraph) or not isinstance(active, torch.Tensor)
            or active.dtype != torch.bool or active.shape != (graph.atoms.shape[1],)
            or not torch.equal(graph.occupancy.detach(), active.to(graph.atoms)[None].expand_as(graph.occupancy))):
        raise ValueError("Synthesis sharpness requires a fixed-occupancy RelaxedGraph")

    def temper(value):
        if (not bool(value.isfinite().all()) or bool((value < 0).any())
                or bool((value.sum(-1) <= 0).any())):
            raise ValueError("Invalid synthesis probability view")
        if temperature == 1:
            return value
        support = value > 0
        logits = value.clamp_min(torch.finfo(value.dtype).tiny).log() / temperature
        logits = logits.masked_fill(~support, -torch.inf)
        return logits.softmax(-1)

    return RelaxedGraph(*(temper(getattr(graph, key)) for key in ("atoms", "charges", "bonds")))
