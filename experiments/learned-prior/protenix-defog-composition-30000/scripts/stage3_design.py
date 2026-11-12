"""Deterministic Stage 3 ablations, seed blocks and nonoverlapping bundles.

This module imports only the standard library. Expansion neither loads models
nor submits jobs. Identical physical settings share one condition; comparison
memberships preserve why that condition belongs to several ablations.
"""

from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs/stage3-lr60-500.json"
INITIALIZATION_PARAMETERS = {
    "atom_distribution": [0.70, 0.14, 0.08, 0.04, 0.02, 0.02],
    "element_order": ["C", "N", "O", "S", "F", "Cl"],
    "atom_uniform_mixing": 0.02,
    "sparse_bond_uniform_mixing": 0.002,
    "dense_dirichlet_concentration": [1, 1, 1, 1, 1],
    "fragmented_deleted_tree_bonds": 4,
    "tree": "neutral_capacity_single_bonds",
    "capacity_fallback": "replace_terminal_category_with_carbon_when_needed",
}
REQUIREMENTS = {
    "6.1_compatibility": "Checkpoint vocabulary, fractional inputs and useful local denoising.",
    "6.2_one_query": "One refreshed frozen teacher prediction per current graph, stopped target.",
    "6.2_strength": "Moderate useful guidance without overwriting native-requested changes.",
    "6.2_projection": "Native plus prior gradient, simplex, valence, conditional connectivity/sparsity.",
    "6.3_validity": "Raw hard validity and fragmentation compared with matched Stage 2.",
    "6.3_gap": "Soft-hard discrepancy and fractional/adversarial states.",
    "6.3_exploration": "Useful native objective, graph changes and no premature freezing.",
    "6.3_revisit": "Strength and compatibility before relying on complex schedules.",
    "8_report": "Matched moving-teacher experiment and interpretable retained outputs.",
}


def constant(value):
    return {"kind": "constant", "value": float(value)}


def linear(start, end, *, begin=0, finish=499):
    return {"kind": "linear", "start": float(start), "end": float(end),
            "begin": begin, "finish": finish}


def cosine(low, high, period):
    return {"kind": "cosine", "low": float(low), "high": float(high), "period": period}


def schedule_value(schedule, update):
    """Read the schedule on zero-based gradient updates 0..updates-1."""
    if isinstance(schedule, (int, float)) and not isinstance(schedule, bool):
        return float(schedule)
    kind = schedule["kind"]
    if kind == "constant":
        return float(schedule["value"])
    if kind == "linear":
        begin, finish = schedule.get("begin", 0), schedule.get("finish", 499)
        fraction = min(1.0, max(0.0, (update - begin) / (finish - begin)))
        return float(schedule["start"] + fraction * (schedule["end"] - schedule["start"]))
    if kind == "cosine":
        lo, hi, period = schedule["low"], schedule["high"], schedule["period"]
        return float(lo + (hi - lo) * (1 - math.cos(math.tau * (update % period) / period)) / 2)
    raise ValueError(f"Unknown schedule kind {kind!r}")


def schedule_mean(schedule, updates=500):
    """Actual finite-grid mean, rather than continuous-ramp approximations."""
    return math.fsum(schedule_value(schedule, i) for i in range(updates)) / updates


def _base():
    return {
        "teacher": {"weight": constant(0.0625), "time": constant(0.8),
                    "input": "soft", "refresh": "current", "target": "teacher",
                    "loss": "mse", "atom_scale": 1.0, "bond_scale": 1.0,
                    "normalization": "row", "calibration": "none"},
        "projection": {"mode": "connected", "readout_caps": True,
                       "valence_damping": 1.0, "bridge_probability": 0.6,
                       "bridge_retry": False, "fluidity_threshold": 0.95,
                       "fluidity_strength": 0.05, "sparsity_amount": 0.01,
                       "sparsity_threshold": 0.5, "every": 1,
                       "start_update": 1, "final_valence": True},
        "cooling": {"schedule": constant(1), "placement": "view", "gradient": "exact"},
        "chemistry_weight": 0.0,
    }


def _canonical(settings):
    return json.dumps(settings, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _tag(value):
    return f"{value:g}".replace(".", "p").replace("-", "m")


class _Registry:
    def __init__(self):
        self.conditions = OrderedDict()
        self.physical = {}

    def add(self, condition_id, family, question, *, teacher=None, projection=None,
            cooling=None, chemistry_weight=0.0, requirements=(), comparison=None):
        condition_id = condition_id.replace("_", "-")
        settings = _base()
        for name, patch in (("teacher", teacher), ("projection", projection), ("cooling", cooling)):
            if patch:
                settings[name].update(deepcopy(patch))
        settings["chemistry_weight"] = chemistry_weight
        signature = _canonical(settings)
        if signature in self.physical:
            existing = self.conditions[self.physical[signature]]
            existing["comparisons"] = sorted(set(existing["comparisons"] + [comparison or family]))
            existing["requirements"] = sorted(set(existing["requirements"] + list(requirements)))
            return existing["id"]
        if condition_id in self.conditions:
            raise ValueError(f"Conflicting settings for condition {condition_id}")
        self.conditions[condition_id] = {
            "id": condition_id, "family": family, "question": question,
            "comparisons": [comparison or family], "requirements": sorted(set(requirements)),
            **settings,
        }
        self.physical[signature] = condition_id
        return condition_id

    def matched_controls(self):
        """A teacher-off control for every other-knob setup, shared exactly once."""
        for condition in list(self.conditions.values()):
            if not _teacher_enabled(condition):
                condition["paired_with"] = None
                continue
            settings = {k: deepcopy(condition[k]) for k in ("projection", "cooling", "chemistry_weight")}
            off = deepcopy(_base()["teacher"])
            off["weight"] = constant(0)
            digest = hashlib.sha256(_canonical(settings).encode()).hexdigest()[:10]
            control = self.add(
                f"control-{digest}", "matched_controls", "Matched teacher-off control.",
                teacher=off, **settings, requirements=condition["requirements"],
                comparison=condition["family"],
            )
            condition["paired_with"] = control
        for condition in self.conditions.values():
            condition.setdefault("paired_with", None)


def _teacher_enabled(condition):
    return max(schedule_value(condition["teacher"]["weight"], i) for i in range(500)) > 0


def _recipe():
    r = _Registry()
    core = ("6.2_strength", "6.3_validity", "6.3_gap", "6.3_exploration", "8_report")
    r.add("control", "reference", "Projection control without DeFoG.",
          teacher={"weight": constant(0)}, requirements=core)
    r.add("reference", "reference", "Moderate refreshed one-query DeFoG reference.",
          requirements=core + ("6.2_one_query",))

    for weight in (0, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5):
        r.add(f"weight-{_tag(weight)}", "weight", "Does moderate teacher strength improve useful valid candidates?",
              teacher={"weight": constant(weight)}, requirements=core + ("6.3_revisit",))
    for time in (0.2, 0.5, 0.8, 0.95):
        for weight in (0.03125, 0.0625, 0.125):
            r.add(f"time-{_tag(time)}-w-{_tag(weight)}", "time", "How does denoising time alter local useful guidance?",
                  teacher={"time": constant(time), "weight": constant(weight)},
                  requirements=core + ("6.1_compatibility", "6.3_revisit"))

    # Finite-grid mean controls separate scheduling from total coefficient.
    for peak in (0.0625, 0.125, 0.25):
        schedules = {"ramp": linear(0, peak), "reverse": linear(peak, 0),
                     "delayed": linear(0, peak, begin=250), "cos100": cosine(0, peak, 100)}
        if peak == 0.125:
            schedules["cos250"] = cosine(0, peak, 250)
        for name, schedule in schedules.items():
            condition_id = r.add(f"weight-{name}-{_tag(peak)}", "weight_schedule",
                  "Does varying teacher strength help beyond its mean weight?",
                  teacher={"weight": schedule}, requirements=core + ("6.3_revisit",))
            mean = schedule_mean(schedule)
            control = r.add(f"weight-mean-{_tag(round(mean, 12))}", "weight_schedule_controls",
                  "Constant teacher coefficient matched to a schedule's actual finite-grid mean.",
                  teacher={"weight": constant(round(mean, 12))}, requirements=core)
            r.conditions[condition_id]["mean_weight_control"] = control
            r.conditions[condition_id]["mean_weight"] = mean

    t_schedules = {"ramp": linear(0.2, 0.95), "reverse": linear(0.95, 0.2),
                   "late-ramp": linear(0.5, 0.95), "delayed": linear(0.2, 0.95, begin=250)}
    for name, schedule in t_schedules.items():
        for weight in (0.03125, 0.0625, 0.125):
            condition_id = r.add(f"time-{name}-w-{_tag(weight)}", "time_schedule",
                  "Does a changing denoiser time help at fixed moderate teacher strength?",
                  teacher={"time": schedule, "weight": constant(weight)}, requirements=core)
            mean = schedule_mean(schedule)
            control = r.add(f"time-mean-{_tag(round(mean, 12))}-w-{_tag(weight)}", "time_schedule_controls",
                  "Constant denoiser time matched to the finite-grid mean.",
                  teacher={"time": constant(round(mean, 12)), "weight": constant(weight)}, requirements=core)
            r.conditions[condition_id]["mean_time_control"] = control
            r.conditions[condition_id]["mean_time"] = mean
    for peak in (0.0625, 0.125, 0.25):
        for name in ("ramp", "delayed"):
            r.add(f"joint-{name}-{_tag(peak)}", "joint_schedule",
                  "Do jointly varying teacher strength and time interact?",
                  teacher={"weight": linear(0, peak, begin=250 if name == "delayed" else 0),
                           "time": t_schedules[name]}, requirements=core)

    projection_req = core + ("6.2_projection",)
    modes = ("none", "valence", "bridge_only", "connected", "sparse_only", "sparse")
    for mode in modes:
        for weight in (0, 0.0625):
            r.add(f"projection-{mode}-w-{_tag(weight)}", "projection",
                  "Which post-gradient operator combination helps DeFoG?",
                  teacher={"weight": constant(weight)}, projection={"mode": mode}, requirements=projection_req)
    knobs = {
        "caps-off": {"readout_caps": False},
        "damping-half": {"valence_damping": 0.5},
        "bridge-high": {"bridge_probability": 0.9},
        "bridge-retry": {"bridge_retry": True},
        "fluidity-off": {"fluidity_strength": 0.0},
        "shrink-high": {"mode": "sparse", "sparsity_amount": 0.03},
        "shrink-threshold": {"mode": "sparse", "sparsity_threshold": 0.7},
        "projection-every5": {"every": 5},
        "projection-delayed": {"start_update": 251},
        "omit-final-valence": {"final_valence": False},
    }
    for name, patch in knobs.items():
        for weight in (0, 0.0625):
            r.add(f"projection-{name}-w-{_tag(weight)}", "projection_settings",
                  "Which projection settings preserve useful connected candidates?",
                  teacher={"weight": constant(weight)}, projection=patch, requirements=projection_req)

    cooling_schedules = {"none": constant(1), "fixed-half": constant(0.5),
                         "ramp-half": linear(1, 0.5), "ramp-quarter": linear(1, 0.25),
                         "delayed-quarter": linear(1, 0.25, begin=250)}
    for mode in ("none", "valence", "connected"):
        for weight in (0, 0.0625):
            for name, schedule in cooling_schedules.items():
                r.add(f"cool-{name}-{mode}-w-{_tag(weight)}", "cooling",
                      "Does a shared cooled probability view improve hard feasibility and the score gap?",
                      teacher={"weight": constant(weight)}, projection={"mode": mode},
                      cooling={"schedule": schedule}, requirements=projection_req)
            for name in ("ramp-half", "delayed-quarter"):
                for variant in ("ste", "state"):
                    r.add(f"cool-{variant}-{name}-{mode}-w-{_tag(weight)}", "cooling_mechanism",
                          "Separate the cooled forward graph, its derivative and incremental state cooling.",
                          teacher={"weight": constant(weight)}, projection={"mode": mode},
                          cooling={"schedule": cooling_schedules[name],
                                   "placement": "state" if variant == "state" else "view",
                                   "gradient": "ste" if variant == "ste" else "exact"},
                          requirements=projection_req)
    for name in ("ramp-half", "delayed-quarter"):
        for weight in (0, 0.0625):
            r.add(f"cool-fluidity-off-{name}-w-{_tag(weight)}", "cooling_interactions",
                  "Do the fluidity floor and cooling oppose each other?",
                  teacher={"weight": constant(weight)}, projection={"fluidity_strength": 0},
                  cooling={"schedule": cooling_schedules[name]}, requirements=projection_req)
        for time in (0.2, 0.95):
            r.add(f"cool-time-{_tag(time)}-{name}", "cooling_interactions",
                  "Does teacher time interact with the graph temperature?",
                  teacher={"time": constant(time)}, cooling={"schedule": cooling_schedules[name]},
                  requirements=projection_req + ("6.1_compatibility",))

    controls = {
        "identity": {"target": "identity"},
        "stale": {"refresh": "initial"},
        "hard-input": {"input": "hard"},
        "atoms-only": {"bond_scale": 0},
        "bonds-only": {"atom_scale": 0},
        "bonds-boost": {"bond_scale": 4},
        "entry-mean": {"normalization": "entry"},
        "entry-mean-calibrated": {"normalization": "entry", "calibration": "initial_gradient"},
        "kl-calibrated": {"loss": "kl", "calibration": "initial_gradient"},
    }
    for name, patch in controls.items():
        if patch.get("calibration") == "initial_gradient":
            patch["calibration_reference"] = {"loss": "mse", "normalization": "row",
                                              "atom_scale": 1, "bond_scale": 1, "weight": 0.0625}
        r.add(f"mechanism-{name}", "teacher_mechanism",
              "Which information and live target refresh are responsible for useful teacher guidance?",
              teacher=patch, requirements=core + ("6.1_compatibility", "6.2_one_query"))
    for chem_weight in (0.01, 0.1):
        for weight in (0, 0.0625):
            r.add(f"soft-valence-{_tag(chem_weight)}-w-{_tag(weight)}", "simple_chemistry",
                  "Does DeFoG add useful information beyond an explicit simple_soft_valence baseline?",
                  teacher={"weight": constant(weight)}, chemistry_weight=chem_weight,
                  requirements=core + ("6.2_projection",))
    r.matched_controls()
    return list(r.conditions.values())


def _validate_schedule(schedule, label, *, lower=None, upper=None, positive=False):
    if not isinstance(schedule, dict):
        raise ValueError(f"{label} must be a schedule dictionary")
    kind = schedule.get("kind")
    if kind == "linear":
        begin, finish = schedule.get("begin", 0), schedule.get("finish", 499)
        if (isinstance(begin, bool) or isinstance(finish, bool)
                or not isinstance(begin, int) or not isinstance(finish, int) or not 0 <= begin < finish <= 499):
            raise ValueError(f"{label} linear endpoints must satisfy 0 <= begin < finish <= 499")
    elif kind == "cosine":
        period = schedule.get("period")
        if isinstance(period, bool) or not isinstance(period, int) or period < 2:
            raise ValueError(f"{label} cosine period must be an integer >=2")
        if schedule.get("high", -1) < schedule.get("low", 0):
            raise ValueError(f"{label} cosine high must be >= low")
    elif kind != "constant":
        raise ValueError(f"{label} unknown schedule kind {kind!r}")
    for update in range(500):
        try:
            value = schedule_value(schedule, update)
        except (KeyError, TypeError, ZeroDivisionError) as error:
            raise ValueError(f"{label} malformed schedule") from error
        if (not math.isfinite(value) or positive and value <= 0
                or lower is not None and value < lower or upper is not None and value > upper):
            raise ValueError(f"{label} has an out-of-range value at update {update}")


def validate_design(config, conditions=None):
    """Reject accidental protocol changes and malformed or colliding run settings."""
    for name, expected in (("model", "protenix-tiny"), ("learning_rate", 60.0),
                           ("updates", 500), ("n_atoms", 25), ("native_weight", 1.0)):
        if config.get(name) != expected:
            raise ValueError(f"This campaign fixes {name}={expected!r}")
    seeds = config.get("seeds", [])
    if len(seeds) != 5 or len(set(seeds)) != 5 or any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError("Exactly five distinct nonnegative integer seeds are required")
    if config.get("initializations") != ["sparse", "dense", "fragmented"]:
        raise ValueError("Initialization blocks must be sparse, dense, fragmented")
    if config.get("initialization_parameters") != INITIALIZATION_PARAMETERS:
        raise ValueError("Initialization parameters must match the implemented matched-start policy")
    if conditions is None:
        return
    ids = [c["id"] for c in conditions]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate condition IDs")
    seen = set()
    for c in conditions:
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", c["id"]):
            raise ValueError(f"Unsafe condition ID {c['id']!r}")
        signature = _canonical({k: c[k] for k in ("teacher", "projection", "cooling", "chemistry_weight")})
        if signature in seen:
            raise ValueError("Duplicate physical settings must share a condition")
        seen.add(signature)
        t, p, cooling = c["teacher"], c["projection"], c["cooling"]
        _validate_schedule(t["weight"], "weight", lower=0)
        _validate_schedule(t["time"], "time", lower=0, upper=1)
        _validate_schedule(cooling["schedule"], "temperature", positive=True)
        for field, choices in (("input", ("soft", "hard")), ("refresh", ("current", "initial")),
                               ("target", ("teacher", "identity")), ("loss", ("mse", "kl")),
                               ("normalization", ("row", "entry")), ("calibration", ("none", "initial_gradient"))):
            if t[field] not in choices:
                raise ValueError(f"Invalid teacher {field}")
        for field in ("atom_scale", "bond_scale"):
            if not math.isfinite(t[field]) or t[field] < 0:
                raise ValueError(f"Invalid {field}")
        if p["mode"] not in ("none", "valence", "bridge_only", "connected", "sparse_only", "sparse"):
            raise ValueError("Invalid projection mode")
        if (isinstance(p["every"], bool) or not isinstance(p["every"], int) or p["every"] < 1
                or not isinstance(p["start_update"], int) or not 1 <= p["start_update"] <= 500):
            raise ValueError("Invalid projection cadence")
        for field, lo, hi, open_low in (("valence_damping", 0, 1, True), ("bridge_probability", 0, 1, True),
                                      ("fluidity_threshold", 0, 1, True), ("fluidity_strength", 0, 1, False),
                                      ("sparsity_amount", 0, 1, False), ("sparsity_threshold", 0, 1, False)):
            value = p[field]
            if not math.isfinite(value) or not lo <= value <= hi or open_low and value == lo:
                raise ValueError(f"Invalid projection {field}")
        if cooling["placement"] not in ("view", "state") or cooling["gradient"] not in ("exact", "ste"):
            raise ValueError("Invalid cooling policy")
        if cooling["placement"] == "state" and cooling["gradient"] != "exact":
            raise ValueError("State cooling has no forward-view STE variant")
        if not math.isfinite(c["chemistry_weight"]) or c["chemistry_weight"] < 0:
            raise ValueError("Invalid chemistry weight")
        if any(req not in REQUIREMENTS for req in c.get("requirements", [])):
            raise ValueError("Unknown Stage 3 requirement")
        paired = c.get("paired_with")
        if paired is not None and paired not in ids:
            raise ValueError(f"Missing matched control {paired!r}")


def load_design(path=None):
    config = json.loads(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
    validate_design(config)
    return config


def expand_conditions(config):
    validate_design(config)
    if "conditions" in config:
        conditions = deepcopy(config["conditions"])
    elif config.get("recipe") == "stage3-lr60-v1":
        conditions = _recipe()
    else:
        raise ValueError(f"Unknown recipe {config.get('recipe')!r}")
    validate_design(config, conditions)
    return conditions


def _run_id(condition_id, seed, initialization):
    return f"{condition_id}__{initialization}__seed-{seed:05d}"


def expand_runs(config):
    conditions = expand_conditions(config)
    runs = []
    for condition in conditions:
        for initialization in config["initializations"]:
            for seed in config["seeds"]:
                control = condition.get("paired_with")
                runs.append({
                    "run_id": _run_id(condition["id"], seed, initialization),
                    "condition_id": condition["id"], "family": condition["family"],
                    "seed": seed, "initialization": initialization,
                    "paired_with_run_id": _run_id(control, seed, initialization) if control else None,
                    "model": config["model"], "native_weight": config["native_weight"],
                    "learning_rate": config["learning_rate"], "updates": config["updates"],
                    "n_atoms": config["n_atoms"], "condition": deepcopy(condition),
                    "initialization_parameters": deepcopy(config["initialization_parameters"]),
                })
    return runs


def make_bundles(config, max_arms=None):
    """Group compatible arms without serializing seeds or duplicating controls.

    A newly encountered matched control is placed with its treatment when it
    fits. A control already assigned to another bundle is referenced by run ID;
    consumers must not execute it again or count it as another replicate.
    """
    if max_arms is None:
        max_arms = config.get("bundling", {}).get("max_arms", 4)
    if isinstance(max_arms, bool) or not isinstance(max_arms, int) or not 2 <= max_arms <= 4:
        raise ValueError("Bundle size must be an integer from 2 to 4")
    conditions = expand_conditions(config)
    by_id = {c["id"]: c for c in conditions}
    runs = {r["run_id"]: r for r in expand_runs(config)}
    bundles = []
    families = list(dict.fromkeys(c["family"] for c in conditions if c["family"] != "matched_controls"))
    for initialization in config["initializations"]:
        for seed in config["seeds"]:
            assigned = set()
            for family in families:
                active = []
                def emit():
                    if not active:
                        return
                    run_ids = [_run_id(cid, seed, initialization) for cid in active]
                    refs = sorted({_run_id(by_id[cid]["paired_with"], seed, initialization)
                                   for cid in active if by_id[cid].get("paired_with")
                                   and by_id[cid]["paired_with"] not in active})
                    digest = hashlib.sha256("|".join(run_ids).encode()).hexdigest()[:10]
                    bundles.append({"bundle_id": f"{family}-{initialization}-seed-{seed:05d}-{digest}",
                                    "family": family, "wave": 0 if family == "reference" else 1,
                                    "seed": seed, "initialization": initialization,
                                    "condition_ids": list(active), "run_ids": run_ids,
                                    "reference_run_ids": refs, "runs": [runs[rid] for rid in run_ids]})
                    active.clear()
                for c in conditions:
                    if c["family"] != family or c["id"] in assigned:
                        continue
                    needed = []
                    control = c.get("paired_with")
                    if control and control not in assigned:
                        needed.append(control)
                    needed.append(c["id"])
                    if len(active) + len(needed) > max_arms:
                        emit()
                    active.extend(needed)
                    assigned.update(needed)
                emit()
            # Every matched control has an owning treatment; no orphans exist.
            if assigned != set(by_id):
                raise ValueError(f"Bundle coverage error: {sorted(set(by_id) - assigned)}")
    return bundles
