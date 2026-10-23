"""Strict opt-in alternating-guidance settings, absent from legacy identities."""

from dataclasses import asdict, dataclass
from pathlib import Path
import os
import re

from latentmol.multimodel.config import number


@dataclass(frozen=True)
class LinearRepairWeightSchedule:
    """Linear weights across the requested budget, without shortening on stalls.

    Progress is update_index/(requested_updates-1), so the last update receives
    end_weight and the endpoint retains it. A singleton budget uses end_weight.
    """

    version: str
    start_weight: float
    end_weight: float
    start_fraction: float
    end_fraction: float

    def __post_init__(self):
        if self.version != "linear_v1":
            raise ValueError("Unsupported repair weight schedule version")
        for key in ("start_weight", "end_weight", "start_fraction", "end_fraction"):
            number(getattr(self, key), f"repair schedule {key}")
        if self.end_weight < self.start_weight:
            raise ValueError("Repair schedule must have nondecreasing weights")
        if not 0 <= self.start_fraction < self.end_fraction <= 1:
            raise ValueError("Repair schedule requires 0 <= start_fraction < end_fraction <= 1")

    @classmethod
    def from_mapping(cls, raw):
        if not isinstance(raw, dict) or set(raw) != set(cls.__dataclass_fields__):
            raise ValueError("Repair schedule requires exactly version, weights and fractions")
        return cls(**raw)

    def at(self, update_index: int, requested_updates: int):
        if (type(update_index) is not int or update_index < 0
                or type(requested_updates) is not int or requested_updates < 1
                or update_index > requested_updates):
            raise ValueError("Repair schedule requires a valid update/endpoint index and requested budget")
        progress = 1.0 if requested_updates == 1 else min(update_index, requested_updates - 1) / (requested_updates - 1)
        if progress <= self.start_fraction:
            return self.start_weight
        if progress >= self.end_fraction:
            return self.end_weight
        fraction = (progress - self.start_fraction) / (self.end_fraction - self.start_fraction)
        return self.start_weight + fraction * (self.end_weight - self.start_weight)


@dataclass(frozen=True)
class AbsoluteRepairWeightSchedule:
    """A declared update-index ramp which does not stretch with the run budget.

    The 25-update pilot used indices 12 through 24 for its linear ramp. This
    version reproduces that arithmetic, then holds the final coefficient in a
    longer run. Indices describe objective states, not accepted-step counts.
    """

    version: str
    start_weight: float
    end_weight: float
    start_update: int
    end_update: int

    def __post_init__(self):
        if self.version != "linear_updates_v1":
            raise ValueError("Unsupported absolute repair weight schedule")
        for key in ("start_weight", "end_weight"):
            number(getattr(self, key), f"repair schedule {key}")
        if self.end_weight < self.start_weight:
            raise ValueError("Repair schedule must have nondecreasing weights")
        if (type(self.start_update) is not int or type(self.end_update) is not int
                or not 0 <= self.start_update < self.end_update):
            raise ValueError("Absolute repair ramp requires increasing nonnegative update indices")

    @classmethod
    def from_mapping(cls, raw):
        if not isinstance(raw, dict) or set(raw) != set(cls.__dataclass_fields__):
            raise ValueError("Absolute repair schedule requires exactly version, weights and update indices")
        return cls(**raw)

    def at(self, update_index: int, requested_updates: int):
        if (type(update_index) is not int or type(requested_updates) is not int
                or requested_updates < 1 or not 0 <= update_index <= requested_updates):
            raise ValueError("Repair schedule requires a valid update/endpoint index and requested budget")
        # Match the pilot's fractional arithmetic exactly on its initial 25
        # states, rather than replacing it with a differently rounded fraction.
        reference = LinearRepairWeightSchedule("linear_v1", self.start_weight,
            self.end_weight, self.start_update / self.end_update, 1.)
        return reference.at(min(update_index, requested_updates - 1, self.end_update), self.end_update + 1)


@dataclass(frozen=True)
class RepairGuidanceConfig:
    """One frozen provider, prescribed sampling budget, and fixed-mask NLL loss.

    Weight 0.01 is a prototype setting, not a fitted/calibrated recommendation.
    Sum is the user's exact -log q target loss. mean_per_decision is an explicit
    alternative objective, useful when comparing different fixed graph sizes.
    """

    provider: dict
    version: int = 1
    weight: float = 0.01
    refresh_interval: int = 5
    attempts_per_refresh: int = 1
    seed: int = 61000
    normalization: str = "sum"
    fallback: str = "boltz_only_clear_target"
    selection: str = "first_valid_prescribed_order"
    probability_policy: str = "stable_log_softmax_no_floor"
    weight_schedule: LinearRepairWeightSchedule | AbsoluteRepairWeightSchedule | None = None

    @classmethod
    def from_mapping(cls, raw, base):
        if not isinstance(raw, dict) or set(raw) - set(cls.__dataclass_fields__):
            raise ValueError("Unsupported repair-guidance fields")
        fields = dict(raw)
        if fields.get("weight_schedule") is not None:
            schedule = fields["weight_schedule"]
            schedule_type = (AbsoluteRepairWeightSchedule if isinstance(schedule, dict)
                and schedule.get("version") == "linear_updates_v1" else LinearRepairWeightSchedule)
            fields["weight_schedule"] = schedule_type.from_mapping(schedule)
        obj = cls(**fields)
        if type(obj.version) is not int or obj.version != 1:
            raise ValueError("Unsupported repair-guidance version")
        number(obj.weight, "repair guidance weight")
        for key, lower, upper in (("refresh_interval", 1, 10000), ("attempts_per_refresh", 1, 16),
                                  ("seed", 0, 2**31 - 1)):
            value = getattr(obj, key)
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"Invalid repair-guidance {key}")
        if obj.normalization not in ("sum", "mean_per_decision"):
            raise ValueError("Unknown repair NLL normalization")
        if (obj.fallback != "boltz_only_clear_target"
                or obj.selection != "first_valid_prescribed_order"
                or obj.probability_policy != "stable_log_softmax_no_floor"):
            raise ValueError("Unsupported repair refresh/fallback/probability policy")
        provider = obj.provider
        allowed = {"model", "python", "source", "checkpoint", "t0", "steps",
                   "initialization", "device", "sampling", "timeout_seconds", "charge_policy"}
        if (not isinstance(provider, dict) or set(provider) - allowed
                or not {"model", "python", "source", "checkpoint"} <= set(provider)):
            raise ValueError("Repair provider requires model/python/source/checkpoint")
        if provider["model"] != "defog_guacamol":
            raise ValueError("No registered hard-repair provider for this model")
        resolved = dict(provider)
        # Absence is the exact historical v1 protocol and stays absent from its
        # identity. The only explicit extension is a versioned lossy repair view.
        if "charge_policy" in provider and provider["charge_policy"] != "project_input_native_output_v1":
            raise ValueError("Unknown explicit GuacaMol charge policy")
        if not isinstance(provider["python"], str) or not provider["python"]:
            raise ValueError("Repair provider requires its Python executable")
        # A venv's bin/python is commonly a symlink to the system executable.
        # Preserve the launch path: resolving that symlink bypasses pyvenv.cfg
        # and loses the isolated native DeFoG dependencies. Assets below may
        # resolve their physical paths; interpreter selection must stay lexical.
        resolved["python"] = os.path.abspath(os.path.join(str(base), provider["python"]))
        for key, identity in (("source", "commit"), ("checkpoint", "sha256")):
            asset = provider[key]
            pattern = r"[0-9a-f]{40}" if identity == "commit" else r"[0-9a-f]{64}"
            if (not isinstance(asset, dict) or set(asset) != {"path", identity}
                    or not isinstance(asset["path"], str) or not asset["path"]
                    or not isinstance(asset[identity], str) or not re.fullmatch(pattern, asset[identity])):
                raise ValueError(f"Invalid repair provider {key} asset")
            resolved[key] = {**asset, "path": str((Path(base) / asset["path"]).resolve())}
        # These are the actual audited public assets, not placeholders
        # for an imagined DBMol charge-aware checkpoint. New model variants need
        # their own codec/native restart audit and registered provider.
        from latentmol.defog.assets import DEFOG_COMMIT, GUACAMOL_SHA256
        if (resolved["source"]["commit"] != DEFOG_COMMIT
                or resolved["checkpoint"]["sha256"] != GUACAMOL_SHA256):
            raise ValueError("defog_guacamol requires the pinned audited official source/checkpoint")
        resolved.setdefault("t0", .8)
        number(resolved["t0"], "repair start time", positive=True)
        if resolved["t0"] >= 1:
            raise ValueError("Repair start time must be below one")
        resolved.setdefault("steps", 1000)
        resolved.setdefault("timeout_seconds", 600)
        for key, minimum, maximum in (("steps", 2, 10000), ("timeout_seconds", 1, 7200)):
            if type(resolved[key]) is not int or not minimum <= resolved[key] <= maximum:
                raise ValueError(f"Invalid repair provider {key}")
        resolved.setdefault("initialization", "direct_candidate_no_corruption")
        if resolved["initialization"] not in ("direct_candidate_no_corruption", "checkpoint_forward_corruption"):
            raise ValueError("Declare the candidate corruption/insertion policy")
        resolved.setdefault("device", "cuda")
        if resolved["device"] not in ("cuda", "cpu"):
            raise ValueError("Repair worker device must be cpu or cuda")
        resolved.setdefault("sampling", {})
        sampling = resolved["sampling"]
        if not isinstance(sampling, dict) or set(sampling) - {"time_distortion", "eta", "omega", "rdb", "rdb_crit"}:
            raise ValueError("Repair sampling overrides must be a mapping")
        if "time_distortion" in sampling and sampling["time_distortion"] not in {
                "identity", "cos", "revcos", "polyinc", "polydec"}:
            raise ValueError("Unsupported native DeFoG time distortion")
        for key in ("eta", "omega"):
            if key in sampling:
                number(sampling[key], f"repair native {key}")
        if "rdb" in sampling and sampling["rdb"] not in {"general", "marginal", "column"}:
            raise ValueError("Unsupported native DeFoG RDB design")
        if "rdb_crit" in sampling and sampling["rdb_crit"] not in {
                "max_marginal", "x_t", "abs_state", "p_x1_g_xt", "x_1", "p_xt_g_x1", "xhat_t"}:
            raise ValueError("Unsupported native DeFoG RDB criterion")
        from dataclasses import replace
        return replace(obj, provider=resolved)

    def record(self):
        record = asdict(self)
        if self.weight_schedule is None:
            record.pop("weight_schedule")
        return record

    def weight_at(self, update_index: int, requested_updates: int):
        """Return a constant unchanged, or the explicitly scheduled state weight."""
        return (self.weight if self.weight_schedule is None
                else self.weight_schedule.at(update_index, requested_updates))

    @property
    def provider_activation_weight(self):
        """Enable a later positive phase without calling the oracle at zero.

        This is the existing provider's enable gate. The runner separately
        resolves and freezes the actual weight for each objective state.
        """
        return self.weight if self.weight_schedule is None else self.weight_schedule.end_weight
