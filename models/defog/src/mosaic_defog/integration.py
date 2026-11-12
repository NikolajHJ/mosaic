"""Optional configuration adapter for Mosaic's existing preparation hook."""

from pathlib import Path

from .guidance import Guidance, cosine, linear
from .teacher import DeFoGTeacher


def _schedule(value, steps):
    if not isinstance(value, dict):
        return value
    options = dict(value)
    kind = options.pop("kind", None)
    if kind == "linear":
        options.setdefault("steps", max(steps, 2))
        return linear(**options)
    if kind == "cosine":
        return cosine(**options)
    raise ValueError("Schedule kind must be linear or cosine")


def mosaic_prepare_loss(config, resources, *, directory, prepare_loss=None):
    """Attach an explicitly configured teacher; keep the caller's projection.

    ``resources`` is a contextlib.ExitStack. Paths are relative to ``directory``.
    Returns ``(prepare_loss_hook, serializable_identity)``. An existing native
    preparation hook (e.g. hardening) can be composed before DeFoG.
    """
    options = dict(config)
    weight, time = options.pop("weight", 0.0625), options.pop("time", 0.8)
    for name in ("source", "checkpoint", "python", "stderr"):
        if options.get(name) is not None:
            path = Path(options[name]).expanduser()
            options[name] = path if path.is_absolute() else Path(directory) / path
    teacher = resources.enter_context(DeFoGTeacher(**options))

    def factory(native, steps):
        guidance = Guidance(
            teacher, weight=_schedule(weight, steps), time=_schedule(time, steps)
        )
        previous = prepare_loss(native, steps) if prepare_loss is not None else None

        def for_state(graph, step):
            active = previous(graph, step) if previous is not None else native
            return guidance.prepare(active, graph, step)

        return for_state

    return factory, {
        "weight": weight,
        "time": time,
        "teacher": teacher.identity,
        "loss": "mean atom squared L2 + mean unordered-bond squared L2",
        "target": "detached, refreshed once per state",
    }
