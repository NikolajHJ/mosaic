"""Strict, versioned study configuration; parsing performs no model imports.

All weights multiply signed, explicitly scaled scalar objectives. No automatic
renormalization hides a missing branch. Paths resolve relative to the YAML file.
"""

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re

import yaml


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def identifier(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(f"Invalid identifier: {value!r}")


def number(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite numeric")
    if value <= 0 if positive else value < 0:
        raise ValueError(f"{label} must be {'positive' if positive else 'nonnegative'}")


@dataclass(frozen=True)
class Objective:
    output: str
    direction: str = "minimize"
    weight: float = 1.0
    scale: float = 1.0
    offset: float = 0.0

    def __post_init__(self):
        if not isinstance(self.output, str) or not self.output:
            raise ValueError("An explicit native scalar output name is required")
        if self.direction not in ("minimize", "maximize"):
            raise ValueError("direction must be minimize or maximize")
        number(self.weight, "weight")
        number(self.scale, "scale", positive=True)
        if isinstance(self.offset, bool) or not isinstance(self.offset, (int, float)) or not math.isfinite(self.offset):
            raise ValueError("offset must be finite")


@dataclass(frozen=True)
class Model:
    plugin: str
    objective: Objective
    options: dict = field(default_factory=dict)
    allow_lossy: bool = False

    def __post_init__(self):
        identifier(self.plugin)
        if not isinstance(self.options, dict) or not isinstance(self.allow_lossy, bool):
            raise ValueError("options must be a mapping and allow_lossy must be boolean")


@dataclass(frozen=True)
class AtomBudget:
    """Soft expected heavy-atom count and optional initial-slot cost."""

    target: int = 25
    tolerance: float = 2.0
    weight: float = 0.01
    outside_weight: float = 0.0

    def __post_init__(self):
        if type(self.target) is not int or self.target < 1:
            raise ValueError("Atom budget target must be a positive integer")
        number(self.tolerance, "atom budget tolerance")
        if self.tolerance >= self.target:
            raise ValueError("Atom budget tolerance must be below the target")
        number(self.weight, "atom budget weight")
        number(self.outside_weight, "atom budget outside_weight")
        if not (self.weight or self.outside_weight):
            raise ValueError("Atom budget requires a positive weight")


@dataclass(frozen=True)
class Arm:
    models: tuple[str, ...]
    control: str = "optimize"
    weights: dict[str, float] = field(default_factory=dict)
    steps: int | None = None
    atom_budget: AtomBudget | None = None

    def __post_init__(self):
        if not self.models or len(set(self.models)) != len(self.models):
            raise ValueError("Each arm needs unique model aliases")
        if self.control not in ("optimize", "no_update", "random_direction"):
            raise ValueError("Unknown control")
        if set(self.weights) - set(self.models):
            raise ValueError("Arm weight override names a model outside that arm")
        for weight in self.weights.values():
            number(weight, "arm weight")
        if self.steps is not None and (type(self.steps) is not int or self.steps < 1):
            raise ValueError("steps must be a positive integer")
        if self.atom_budget is not None and not isinstance(self.atom_budget, AtomBudget):
            raise ValueError("Expected an atom budget configuration")


@dataclass(frozen=True)
class Optimizer:
    steps: int = 10
    learning_rate: float = 0.05
    trust_radius: float = 1.0
    max_gradient_norm: float = 1.0

    def __post_init__(self):
        if type(self.steps) is not int or self.steps < 1:
            raise ValueError("steps must be positive integer")
        for key in ("learning_rate", "trust_radius", "max_gradient_norm"):
            number(getattr(self, key), key, positive=True)


@dataclass(frozen=True)
class Diagnostics:
    seed: int = 1729
    hard_samples: int = 8
    fd_epsilons: tuple[float, ...] = (0.001, 0.003, 0.01)
    fd_rtol: float = 0.05
    fd_atol: float = 0.000001
    parity_rtol: float = 0.0001
    parity_atol: float = 0.00001
    minimum_fd_passes: int = 2

    def __post_init__(self):
        for key in ("seed", "hard_samples", "minimum_fd_passes"):
            if type(getattr(self, key)) is not int or getattr(self, key) < (1 if key == "minimum_fd_passes" else 0):
                raise ValueError(f"Invalid {key}")
        if len(set(self.fd_epsilons)) < self.minimum_fd_passes:
            raise ValueError("Need distinct finite-difference scales")
        for epsilon in self.fd_epsilons:
            number(epsilon, "finite-difference epsilon", positive=True)
        for key in ("fd_rtol", "fd_atol", "parity_rtol", "parity_atol"):
            number(getattr(self, key), key)


@dataclass(frozen=True)
class Study:
    version: int
    name: str
    decoder: dict
    models: dict[str, Model]
    arms: dict[str, Arm]
    device: str = "cuda"
    occupancy: str = "fixed"
    temperature: float = 1.0
    optimizer: Optimizer = field(default_factory=Optimizer)
    diagnostics: Diagnostics = field(default_factory=Diagnostics)
    start: dict = field(default_factory=dict)
    plugins: dict[str, str] = field(default_factory=dict)
    native_evaluators: dict[str, dict] = field(default_factory=dict)

    def __post_init__(self):
        if self.version != 1:
            raise ValueError("Unsupported study version")
        identifier(self.name)
        if not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", self.device):
            raise ValueError("device must be cpu, cuda, or cuda:index")
        if self.occupancy not in ("fixed", "variable"):
            raise ValueError("occupancy must be fixed or variable")
        if not self.models or not self.arms:
            raise ValueError("Study must define models and arms")
        if set(self.decoder) != {"checkpoint", "sha256"}:
            raise ValueError("decoder requires exactly checkpoint and sha256")
        if set(self.start) - {"path", "sha256"} or bool(self.start.get("path")) != bool(self.start.get("sha256")):
            raise ValueError("start is empty (encoded seed mean), or path plus sha256 for raw z")
        number(self.temperature, "temperature", positive=True)
        for name in self.models:
            identifier(name)
        for name, arm in self.arms.items():
            identifier(name)
            if arm.atom_budget is not None and self.occupancy != "variable":
                raise ValueError(f"{name}: atom budget requires variable occupancy")
            if set(arm.models) - set(self.models):
                raise ValueError(f"{name}: unknown model alias")
            if not any(self.weight(arm, m) > 0 for m in arm.models):
                raise ValueError(f"{name}: at least one positive objective weight required")
        for name, factory in self.plugins.items():
            identifier(name)
            if not isinstance(factory, str) or factory.count(":") != 1:
                raise ValueError("Plugin factory must be a trusted Python module:function")
        for name, evaluator in self.native_evaluators.items():
            identifier(name) # Hard-only independent evaluators need no fake gradient plugin.
            if set(evaluator) != {"command", "timeout_seconds", "identity", "candidates_per_endpoint"}:
                raise ValueError("Native evaluator requires command, timeout_seconds, identity, candidates_per_endpoint")
            argv = evaluator["command"]
            if not isinstance(argv, list) or not argv or any(not isinstance(x, str) for x in argv):
                raise ValueError("Native command must be a nonempty argv list, never a shell string")
            if not all(any(token in arg for arg in argv) for token in ("{input}", "{output}")):
                raise ValueError("Native command requires {input} and {output} placeholders")
            number(evaluator["timeout_seconds"], "timeout_seconds", positive=True)
            if type(evaluator["candidates_per_endpoint"]) is not int or evaluator["candidates_per_endpoint"] < 1:
                raise ValueError("candidates_per_endpoint must be a positive integer")
            if not isinstance(evaluator["identity"], dict) or not evaluator["identity"]:
                raise ValueError("Native evaluator must declare source/weights/config provenance")

    def weight(self, arm: Arm, model: str) -> float:
        return arm.weights.get(model, self.models[model].objective.weight)

    def to_dict(self):
        data = asdict(self)
        for arm in data["arms"].values():
            if arm["atom_budget"] is None:
                del arm["atom_budget"]
        return data

    def fingerprint(self):
        return digest(self.to_dict())

    @classmethod
    def from_dict(cls, raw: dict):
        data = dict(raw)
        data["models"] = {name: Model(**{**value, "plugin": value.get("plugin", name),
                         "objective": Objective(**value["objective"])})
                          for name, value in data["models"].items()}
        data["arms"] = {name: Arm(**{**value, "models": tuple(value["models"]),
                        "atom_budget": AtomBudget(**value["atom_budget"])
                        if value.get("atom_budget") is not None else None})
                        for name, value in data["arms"].items()}
        data["optimizer"] = Optimizer(**data.get("optimizer", {}))
        d = dict(data.get("diagnostics", {}))
        if "fd_epsilons" in d:
            d["fd_epsilons"] = tuple(d["fd_epsilons"])
        data["diagnostics"] = Diagnostics(**d)
        return cls(**data)

    @classmethod
    def load(cls, path: Path):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        base = path.resolve().parent
        def resolve(value):
            return str((base / value).resolve()) if value else value
        data["decoder"]["checkpoint"] = resolve(data["decoder"]["checkpoint"])
        if data.get("start", {}).get("path"):
            data["start"]["path"] = resolve(data["start"]["path"])
        for value in data["models"].values():
            for key, item in value.get("options", {}).items():
                if key in ("prepared", "checkpoint") or key.endswith(("_path", "_root")):
                    if not isinstance(item, str):
                        raise ValueError(f"Path option {key} must be text")
                    value["options"][key] = resolve(item)
            profile = value.get("options", {}).get("dbmol")
            if profile is not None:
                if not isinstance(profile, dict) or not isinstance(profile.get("confidence_checkpoint"), dict):
                    raise ValueError("DBMol model option requires a confidence checkpoint asset")
                asset = profile["confidence_checkpoint"]
                if not isinstance(asset.get("path"), str):
                    raise ValueError("DBMol confidence checkpoint path must be text")
                asset["path"] = resolve(asset["path"])
        return cls.from_dict(data)
