"""One-query loss preparation, independent of the optimization/projection code."""

import math
from dataclasses import dataclass


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def linear(start, end, steps):
    """Zero-based linear schedule, clamped after steps-1; endpoints included."""
    start, end = _finite(start, "start"), _finite(end, "end")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
        raise ValueError("A linear schedule needs at least two integer steps")
    return lambda step: start + (end - start) * min(max(step / (steps - 1), 0), 1)


def cosine(low, high, period):
    """Raised cosine: low at update zero, high halfway through each period."""
    low, high = _finite(low, "low"), _finite(high, "high")
    if (
        high < low
        or isinstance(period, bool)
        or not isinstance(period, int)
        or period < 2
    ):
        raise ValueError("Cosine needs high >= low and an integer period >= 2")
    return lambda step: (
        low + (high - low) * (1 - math.cos(math.tau * (step % period) / period)) / 2
    )


@dataclass
class Guidance:
    """Prepare a stopped-teacher loss once per evaluated optimization state.

    ``weight`` and ``time`` can be constants or callables of zero-based step.
    ``native(graph, key=key)`` returns ``(loss, aux)`` and is minimized.
    Call ``prepare`` outside JIT/autodiff; differentiate its returned loss.
    """

    teacher: object
    weight: object = 0.0625
    time: object = 0.8
    backend: str = "jax"

    def __post_init__(self):
        if self.backend not in ("jax", "torch"):
            raise ValueError("backend must be 'jax' or 'torch'")
        if not callable(self.weight):
            self._weight(0)
        if not callable(self.time):
            self._time(0)

    def _weight(self, step):
        value = _finite(
            self.weight(step) if callable(self.weight) else self.weight, "weight"
        )
        if value < 0:
            raise ValueError("DeFoG weight must be nonnegative")
        return value

    def _time(self, step):
        value = _finite(self.time(step) if callable(self.time) else self.time, "time")
        if not 0 <= value <= 1:
            raise ValueError("Teacher time must lie in [0,1]")
        return value

    def prepare(self, native, graph, step=0):
        """Query the current graph once and return its differentiable objective."""
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("step must be a nonnegative integer")
        weight, time = self._weight(step), self._time(step)
        target = self.teacher.predict(graph, time=time)
        if (
            target.X.shape != graph.X.shape
            or target.E_upper.shape != graph.E_upper.shape
        ):
            raise ValueError("Teacher changed graph slots or categories")
        if self.backend == "jax":
            from .losses import jax_terms as terms
        else:
            from .losses import torch_terms as terms

        def combined(current, *, key=None):
            value, aux = native(current, key=key)
            prior, metrics = terms(current, target)
            reserved = {
                *metrics,
                "defog_loss",
                "defog_weight",
                "defog_time",
                "defog_native_loss",
            }
            if reserved.intersection(aux):
                raise ValueError(
                    "Native auxiliary metrics collide with the defog_ namespace"
                )
            return value + weight * prior, {
                **aux,
                **metrics,
                "defog_loss": prior,
                "defog_weight": weight,
                "defog_time": time,
                "defog_native_loss": value,
            }

        return combined

    def loss_factory(self, native, steps=None):
        """Adapter for Mosaic prepare_loss(native, epochs) / loss_factory hooks."""
        return lambda graph, step: self.prepare(native, graph, step)
