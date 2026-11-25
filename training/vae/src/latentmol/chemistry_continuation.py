"""Joint triangle-VAE chemistry phase, separate from the real-only campaign.

All numerical entrypoints require an allocated DTU LSF job. The parent remains
immutable. An objective-changing phase inherits model/AdamW/epoch/RNG state;
later resumes require this phase's exact source, protocol, data and runtime.
The base teacher searches outside autograd and preserves element/charge/mask.
A separately versioned optional extension adds failed-row guidance and may
explicitly permit atom edits; every target remains a bounded witness rather
than a globally nearest-valid distance. Historical plans omit that extension.
UNTESTED until the sealed CPU fixtures and native GPU benchmark have passed.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from copy import deepcopy
from dataclasses import asdict, fields
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from .atom_preserving_chemistry import (AtomPreservingRepairTeacher, composition_statistics,
                                      population_js, population_js_cotangent, repair_loss)
from .models.autoencoder import comparable_model_config, comparable_train_config
from .chemistry_runtime import (RATIO2_VERSION, composition_prior_weight, epoch_fit_policy, epoch_fit_seconds,
                                protocol_config_path, validate_execution_policy, validate_failure_guidance,
                                validate_runtime_policy, validate_handoff_policy)
from .chemistry_early_health import (EarlyPanelCollector, early_health_due, exclusive_json,
    preserve_observation_state, reject_diagnostic_snapshot, require_early_health_baseline,
    save_failed_early_health_snapshot, strict_teacher_validity, validate_early_health_observation,
    validate_early_health_policy)
from .data import GraphDataset, collate_graphs, validate_large_cache
from .evaluate import load_model
from .experiments import ExperimentRun, file_sha256, source_manifest, write_json
from .large_vae import gradient_norm, peak_host_memory_bytes
from .large_vae_evaluation import evaluate_panel
from .losses import LossConfig, latent_kl
from .models import ModelConfig
from .probability_repair import ProbabilityRepairConfig
from .representation import DecoderOutput, HardGraphBatch
from .train import (TrainConfig, algorithm_identity, capture_rng_state, compact_cohort_identity,
                    learning_rate_at_epoch, restore_rng_state, runtime_identity, save_checkpoint,
                    seed_everything, state_digest, training_cohorts)


VERSION = "joint-atom-preserving-chemistry-v1"
OPTIMIZED_EXECUTION = {"backend": "optimized", "workers": 12, "torch_threads": 4, "gpu_cpus": 16}


def execution_settings(plan, *, allocated=False):
    """Validate execution resources separately from the unchanged loss policy.

    Historical plans without this stanza keep their serial reference teacher.
    New stanzas explicitly select 12/20/28 search workers, four trainer threads,
    overlap and enough allocated slots, without changing the loss protocol.
    Benchmark overreservation up to32 slots stays explicit in the receipt.
    """
    value = validate_execution_policy(plan.get("execution"))
    if value is None:
        return None
    if allocated:
        slots = os.environ.get("LSB_DJOB_NUMPROC", "")
        if not slots.isdecimal() or int(slots) != value["gpu_cpus"]:
            raise ValueError("Optimized chemistry requires its declared allocated CPU slots")
        if (torch.get_num_threads() != value["torch_threads"]
                or any(os.environ.get(key) != str(value["torch_threads"])
                       for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"))):
            raise ValueError("Main Torch/OMP/MKL threads differ from authenticated parent runtime")
    return dict(value)


def load_parent_config(path: Path) -> TrainConfig:
    """Read the complete canonical JSON parent config without YAML coercion.

    The producer writes ``json.dumps(asdict(config))``. In particular, its
    numeric ``1e-05`` is valid JSON, while the YAML 1.1 resolver used by the
    historical YAML reader interprets that spelling as a string. Reading this
    artifact as JSON preserves the parent's numeric value and optimizer
    identity; accepting numeric-looking strings would conceal corrupt input.
    Only JSON arrays representing the two declared tuple fields are rebuilt.
    All checkpoint, source, runtime, optimizer and RNG gates remain separate.
    """
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate parent configuration key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"Nonfinite parent configuration value: {value}")

    raw = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique,
                     parse_constant=reject_constant)

    def schema(value, cls, prefix):
        if type(value) is not dict:
            raise ValueError(f"Parent configuration {prefix} must be a JSON object")
        required = {field.name for field in fields(cls)}
        missing, extra = sorted(required - value.keys()), sorted(value.keys() - required)
        if missing or extra:
            raise ValueError(f"Parent configuration {prefix} fields differ: missing={missing}, extra={extra}")

    def integer(value, prefix, minimum=0, optional=False):
        if optional and value is None:
            return
        if type(value) is not int or value < minimum:
            raise ValueError(f"Parent configuration {prefix} requires an integer >= {minimum}")

    def number(value, prefix, positive=False):
        try:
            finite = type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite or value < 0 or (positive and value == 0):
            qualifier = "positive" if positive else "nonnegative"
            raise ValueError(f"Parent configuration {prefix} requires a finite {qualifier} JSON number")

    def text(value, prefix):
        if type(value) is not str or not value:
            raise ValueError(f"Parent configuration {prefix} requires a nonempty JSON string")

    if type(raw) is dict:
        raw.setdefault("kl_warmup_molecules", None)  # older parents use epoch-based annealing
    schema(raw, TrainConfig, "train")
    raw["model"] = comparable_model_config(raw["model"])
    schema(raw["model"], ModelConfig, "model")
    # Every checkpoint predating graph matching used raw ordered targets.
    if type(raw["loss"]) is dict:
        raw["loss"].setdefault("atom_order", "ordered")
    schema(raw["loss"], LossConfig, "loss")
    positive_ints = ("epochs", "batch_size", "validation_limit", "chemistry_limit", "checkpoint_every")
    nonnegative_ints = ("seed", "kl_warmup_epochs", "num_workers", "full_chemistry_every")
    optional_ints = ("train_limit", "train_probe_limit", "kl_warmup_molecules")
    numbers = ("learning_rate", "weight_decay", "clip_grad", "beta")
    # New dataclass fields must receive an explicit parser rather than inherit
    # an unchecked default when an older parent is resumed by newer code.
    covered = set(positive_ints + nonnegative_ints + optional_ints + numbers +
                  ("mode", "device", "checkpoint_epochs", "lr_milestones", "model", "loss"))
    if covered != {field.name for field in fields(TrainConfig)}:
        raise RuntimeError("Parent TrainConfig schema changed; update its explicit JSON reader")
    for key in positive_ints:
        integer(raw[key], key, minimum=1)
    for key in nonnegative_ints:
        integer(raw[key], key)
    for key in optional_ints:
        integer(raw[key], key, minimum=1, optional=True)
    for key in numbers:
        number(raw[key], key, positive=key in ("learning_rate", "clip_grad"))
    for key in ("mode", "device"):
        text(raw[key], key)
    for key, value in raw["model"].items():
        if key in ("decoder_variant", "encoder_pooling", "encoder_aggregation"):
            text(value, "model." + key)
        elif key == "encoder_degree_features":
            if type(value) is not bool:
                raise ValueError("Parent configuration model.encoder_degree_features requires a boolean")
        elif key == "encoder_sum_scale":
            number(value, "model.encoder_sum_scale", positive=True)
        elif key == "atom_refinement_dropout":
            number(value, "model.atom_refinement_dropout")
        else:
            integer(value, "model." + key,
                    minimum=0 if key in ("refinement_rounds", "encoder_residual_layers") else 1,
                    optional=key in ("global_hidden", "trunk_hidden"))
    for key, value in raw["loss"].items():
        if key == "atom_order":
            text(value, "loss.atom_order")
        else:
            number(value, "loss." + key)
    for key in ("checkpoint_epochs", "lr_milestones"):
        if type(raw[key]) is not list:
            raise ValueError(f"Parent configuration {key} requires a JSON array")
    for index, value in enumerate(raw["checkpoint_epochs"]):
        integer(value, f"checkpoint_epochs[{index}]", minimum=1)
    for index, value in enumerate(raw["lr_milestones"]):
        prefix = f"lr_milestones[{index}]"
        if type(value) is not list or len(value) != 2:
            raise ValueError(f"Parent configuration {prefix} requires [epoch, learning_rate]")
        integer(value[0], prefix + "[0]", minimum=1)
        number(value[1], prefix + "[1]", positive=True)
    value = dict(raw, model=ModelConfig(**raw["model"]), loss=LossConfig(**raw["loss"]),
                 checkpoint_epochs=tuple(raw["checkpoint_epochs"]),
                 lr_milestones=tuple(tuple(pair) for pair in raw["lr_milestones"]))
    return TrainConfig(**value)


class ParentCheckpointMismatch(ValueError):
    """Keep every failed parent boundary, rather than hiding it in one OR guard."""

    def __init__(self, differences, actual, expected):
        self.differences, self.actual, self.expected = differences, actual, expected
        detail = "; ".join(f"{row['field']}: checkpoint={row.get('actual')!r}, required={row.get('expected')!r}"
                           for row in differences)
        super().__init__("Parent checkpoint authentication failed: " + detail)


def metadata_differences(actual, expected, prefix=""):
    """Exact JSON-value differences, allowing only tuple/list serialization.

    These comparisons have no tolerance, omitted-key defaults or inferred
    replacement values. Boolean/numeric distinctions and extra/missing fields
    remain visible. Checkpoint tensors never belong in the compared metadata.
    """
    if isinstance(actual, dict) and isinstance(expected, dict):
        result = []
        for key in sorted(set(actual) | set(expected)):
            field = (prefix + "." if prefix else "") + str(key)
            if key not in actual or key not in expected:
                result.append({"field": field, "actual_present": key in actual, "expected_present": key in expected,
                               "actual": actual.get(key), "expected": expected.get(key)})
            else:
                result.extend(metadata_differences(actual[key], expected[key], field))
        return result
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        result = []
        if len(actual) != len(expected):
            result.append({"field": prefix + ".length", "actual": len(actual), "expected": len(expected)})
        for index, (left, right) in enumerate(zip(actual, expected)):
            result.extend(metadata_differences(left, right, f"{prefix}[{index}]"))
        return result
    try:
        same = json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(expected, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        # Malformed tensor/nonfinite metadata must fail without serializing a
        # scientific tensor into diagnostics or silently converting its dtype.
        return [{"field": prefix, "actual": {"unsupported_type": type(actual).__name__}, "expected": expected}]
    return [] if same else [{"field": prefix, "actual": actual, "expected": expected}]


def authenticate_parent_metadata(checkpoint, config, plan, current_runtime, parent_run):
    """Require the exact completed native parent before restoring moments/RNG.

    The stopped parent run may legitimately retain a running/failed status;
    only its pinned completed checkpoint establishes the resumable boundary.
    No runtime, architecture, objective, source, epoch or optimizer-update
    mismatch is repaired by this reader. A mismatch names every actual field
    so the launcher/configuration can be corrected in a fresh sealed attempt.
    """
    state = checkpoint.get("continuation")
    state = state if isinstance(state, dict) else {}
    actual = {"format_version": checkpoint.get("format_version"),
        "continuation": {key: state[key] for key in
                         ("version", "epoch_complete", "scheduler_state", "updates", "split_sha256", "runtime") if key in state},
        "epoch": checkpoint.get("epoch"), "train_config": comparable_train_config(checkpoint.get("train_config")),
        "model_config": comparable_model_config(checkpoint.get("model_config")), "loss_config": checkpoint.get("loss_config"),
        "dataset_sha256": checkpoint.get("dataset_sha256"), "run_id": checkpoint.get("run_id"),
        "source_archive_sha256": checkpoint.get("source_archive_sha256"),
        "source_sha256": checkpoint.get("source_sha256")}
    expected = {"format_version": 1,
        "continuation": {"version": 1, "epoch_complete": True, "scheduler_state": None,
                         "updates": plan["parent_updates"], "split_sha256": plan["split_sha256"], "runtime": current_runtime},
        "epoch": plan["parent_epoch"], "train_config": asdict(config),
        "model_config": asdict(config.model), "loss_config": asdict(config.loss),
        "dataset_sha256": plan["dataset_sha256"], "run_id": parent_run.get("run_id"),
        "source_archive_sha256": plan["parent_archive_sha256"], "source_sha256": plan["parent_manifest_sha256"]}
    # Preserve the historical wire format when verifying a pre-matching parent.
    # The independently pinned originating JSON/checkpoint are still compared verbatim.
    old_train = checkpoint.get("train_config")
    if isinstance(old_train, dict) and "kl_warmup_molecules" not in old_train and config.kl_warmup_molecules is None:
        expected["train_config"].pop("kl_warmup_molecules")
    if config.loss.atom_order == "ordered":
        old_losses = (checkpoint.get("loss_config"), old_train.get("loss") if isinstance(old_train, dict) else None)
        for old, current in zip(old_losses, (expected["loss_config"], expected["train_config"]["loss"])):
            if isinstance(old, dict) and "atom_order" not in old:
                current.pop("atom_order")
    differences = metadata_differences(actual, expected)
    # The checkpoint's own runtime/config also have an independently pinned
    # originating run record. Do not confuse an allocation's resources with a
    # permission to change the parent's authenticated numerical settings.
    differences.extend(metadata_differences(state.get("runtime"), parent_run.get("runtime_identity"),
                                             "parent_run.runtime_identity"))
    differences.extend(metadata_differences(checkpoint.get("train_config"), parent_run.get("config"),
                                             "parent_run.config"))
    if differences:
        raise ParentCheckpointMismatch(differences, actual, expected)
    return {"passed": True, "parent_epoch": checkpoint["epoch"], "parent_updates": state["updates"],
            "runtime": current_runtime, "parent_run_id": checkpoint["run_id"],
            "parent_checkpoint_sha256": plan["parent_checkpoint_sha256"],
            "parameter_scope": "joint_encoder_decoder", "optimizer_scope": "authenticated_parent_AdamW_moments",
            "rng_scope": "all_authenticated_parent_streams_plus_separate_fresh_prior", "differences": []}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _continuation_endpoint(plan):
    """An explicit bounded branch does not rewrite its absolute100 config."""
    from .chemistry_short_continuation import continuation_end_epoch
    return continuation_end_epoch(plan)


def authenticated(path, expected):
    if file_sha256(Path(path)) != expected:
        raise ValueError(f"Bound artifact changed: {path}")


def require_failure_readiness(plan):
    """New training cannot bypass its supplementary native readiness gate."""
    if (plan["protocol"].get("failure_guidance") is None and plan.get("short_continuation") is None
            and plan.get("valid_reference_continuation") is None):
        return None
    expected = plan.get("_failure_readiness_sha256")
    if type(expected) is not str or len(expected) != 64:
        raise ValueError("New chemistry training/final requires the immutable failure-readiness SHA")
    from .failure_guidance_selection import authenticate_readiness
    receipt = authenticate_readiness(plan, expected)
    if validate_early_health_policy(plan["protocol"]) is not None:
        baseline_sha = receipt.get("early_health_baseline_sha256")
        require_early_health_baseline(plan, baseline_sha)
        plan["_early_health_baseline_sha256"] = baseline_sha
    corrected = (plan["protocol"].get("failure_guidance") or {}).get("loss", {}).get("version") == "repair-failure-guidance-v2"
    if "guarded_restart" in plan or corrected:
        from .chemistry_restart_policy import authenticate_stability
        authenticate_stability(plan)
    return receipt


def read_plan(path, expected):
    authenticated(path, expected)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate scientific key: {key}")
            result[key] = value
        return result
    plan = json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
    execution_settings(plan)
    from .repair_worker_pool import validate_classification_execution
    validate_classification_execution(plan.get("classification_execution"))
    from .chemistry_worker_startup import validate_worker_startup
    validate_worker_startup(plan.get("worker_startup"))
    from .chemistry_optimizer_step import validate_optimizer_step_control
    validate_optimizer_step_control(plan.get("optimizer_step_control"))
    p = plan["protocol"]
    validate_failure_guidance(p.get("failure_guidance"))
    validate_early_health_policy(p)
    from .chemistry_restart_policy import validate_guarded_restart
    validate_guarded_restart(plan)
    from .chemistry_short_continuation import validate_short_continuation
    validate_short_continuation(plan)
    starting = validate_handoff_policy(plan)
    try:
        composition_prior_weight(p)
    except ValueError as error:
        raise ValueError("Require the declared joint FP32 ratio and composition importance policy") from error
    if (p["version"] != (RATIO2_VERSION if p["prior_ratio"] == 2 else VERSION)
            or p["total_epochs"] != 100
            or p["real_batch_size"] != 256 or p["real_repair_weight"] != .25
            or p["prior_repair_weight"] != .75 or p["precision"] != "fp32"
            or p["parameter_scope"] != "joint_encoder_decoder"):
        raise ValueError("Require the declared joint FP32 1:3 or composition-reweighted 1:2 phase through100 total epochs")
    for key in ("microbatch_size", "benchmark_updates", "benchmark_panel_count", "monitor_every",
                "monitor_prior_count", "validation_count", "maximum_segments", "repair_ramp_epochs"):
        if type(p[key]) is not int or p[key] < 1:
            raise ValueError(f"Positive integer required: {key}")
    if p["microbatch_size"] not in (128, 256) or p["validation_count"] > 4096:
        raise ValueError("Unsupported native microbatch or monitoring size")
    for key in ("repair_weight", "element_js_weight", "charge_js_weight", "occupancy_anchor_weight"):
        if isinstance(p[key], bool) or not math.isfinite(p[key]) or p[key] < 0:
            raise ValueError(f"Finite nonnegative weight required: {key}")
    if min(p["repair_weight"], p["element_js_weight"], p["occupancy_anchor_weight"]) <= 0:
        raise ValueError("Repair, composition and size protection must all be enabled")
    validate_runtime_policy(p)
    epoch_fit_policy(plan)
    if not 0 < p["vram_fraction_limit"] < 1 or not 0 < p["minimum_prior_target_fraction"] < 1:
        raise ValueError("Invalid coverage or bounded allocation budgets")
    ProbabilityRepairConfig(**p["search"])
    root = Path(plan["work_root"]).resolve()
    if not str(root).startswith("/work3/"):
        raise ValueError("Require declared DTU work3 root")
    for key in ("release", "campaign", "data", "parent_checkpoint", "parent_release", "parent_run_record"):
        if not Path(plan[key]).is_absolute() or not Path(plan[key]).resolve().is_relative_to(root):
            raise ValueError(f"Artifact outside authorized work root: {key}")
    authenticated(str(plan["release"]) + ".tar.gz", plan["archive_sha256"])
    declared = json.loads((Path(plan["release"]) / protocol_config_path(plan)).read_text())
    if p != declared:
        raise ValueError("Protocol differs from the source-bound chemistry configuration")
    if Path(__file__).resolve() != Path(plan["release"]).resolve() / "src/latentmol/chemistry_continuation.py":
        raise ValueError("Execute the chemistry phase from its authenticated immutable release")
    if source_manifest(Path(plan["release"]))["sha256"] != plan["manifest_sha256"]:
        raise ValueError("Scientific source differs from immutable phase release")
    authenticated(plan["parent_checkpoint"], plan["parent_checkpoint_sha256"])
    authenticated(plan["parent_run_record"], plan["parent_run_record_sha256"])
    authenticated(str(plan["parent_release"]) + ".tar.gz", plan["parent_archive_sha256"])
    if starting is not None:
        for key, sha in (("path", "sha256"), ("source_plan", "source_plan_sha256"), ("completion", "completion_sha256")):
            path = Path(starting[key])
            if not path.is_absolute() or not path.resolve().is_relative_to(root):
                raise ValueError("Starting checkpoint escaped the declared work root")
            authenticated(path, starting[sha])
    for artifact, sha in plan.get("readiness_artifacts", {}).items():
        authenticated(artifact, sha)
    return plan


def hard_slice(hard, start, stop):
    return HardGraphBatch(*(v[start:stop] for v in (hard.atoms, hard.charges, hard.bonds)))


def chunks(count, size):
    return [(start, min(start + size, count)) for start in range(0, count, size)]


def reconstruction_denominators(hard, loss, *, dtype=torch.float32):
    """Global weighted-CE denominators; per-chunk means are not equivalent."""
    active = hard.atoms != 0
    upper = torch.ones(hard.bonds.shape[-2:], dtype=torch.bool, device=hard.atoms.device).triu(1)
    pair = active[:, :, None] & active[:, None, :] & upper
    weights = torch.ones_like(hard.atoms, dtype=dtype).masked_fill(~active, loss.padding_weight)
    bond_weights = torch.ones_like(hard.bonds[pair], dtype=dtype).masked_fill(hard.bonds[pair] == 0, loss.no_bond_weight)
    bond_total = bond_weights.sum()
    return {"atoms": weights.sum(),
            "charges": active.sum().clamp_min(1),
            "bonds": torch.where(bond_total > 0, bond_total, torch.ones_like(bond_total)),
            "rows": len(hard.atoms)}


def reconstruction_chunk(decoded, hard, loss, denominators):
    """A sum contribution to the exact logical batch reconstruction objective."""
    if loss.atom_order == "graph":
        from latentmol.training.matching import align_reconstruction_targets

        hard = align_reconstruction_targets(decoded, hard, loss, denominators=denominators)
    active = hard.atoms != 0
    weights = decoded.atoms.new_ones(decoded.atoms.shape[-1]); weights[0] = loss.padding_weight
    atom = F.cross_entropy(decoded.atoms.flatten(0, 1), hard.atoms.flatten(), weight=weights,
                           reduction="sum") / denominators["atoms"]
    charge = F.cross_entropy(decoded.charges.flatten(0, 1), hard.charges.flatten(),
                             reduction="none").reshape_as(active)
    charge = (charge * active).sum() / denominators["charges"]
    upper = torch.ones(hard.bonds.shape[-2:], dtype=torch.bool, device=hard.atoms.device).triu(1)
    pair = active[:, :, None] & active[:, None, :] & upper
    weights = decoded.bonds.new_ones(decoded.bonds.shape[-1]); weights[0] = loss.no_bond_weight
    bond = (F.cross_entropy(decoded.symmetric_bond_logits[pair], hard.bonds[pair], weight=weights,
                            reduction="sum") / denominators["bonds"] if bool(pair.any())
            else decoded.bonds.reshape(-1)[0] * 0)
    size = ((decoded.probabilities().expected_size - active.sum(-1)) / hard.atoms.shape[1]).square().sum() / denominators["rows"]
    return atom + loss.charge_weight * charge + loss.bond_weight * bond + loss.size_weight * size


def training_composition(dataset):
    """Empirical occupied counts from TRAIN rows only; never use held-out labels."""
    if dataset.split != "train" or dataset._indices is None:
        raise ValueError("Require the authenticated disk-backed training split")
    elements = np.zeros(len(dataset.spec.atomic_numbers), dtype=np.int64)
    charges = np.zeros(len(dataset.spec.formal_charges), dtype=np.int64)
    sizes = np.zeros(dataset.spec.max_nodes + 1, dtype=np.int64)
    for start in range(0, len(dataset), 8192):
        indices = dataset._indices[start:start + 8192]
        a, q = np.asarray(dataset._atoms[indices]), np.asarray(dataset._charges[indices])
        active = a != 0
        elements += np.bincount(a[active], minlength=len(elements))
        charges += np.bincount(q[active], minlength=len(charges))
        sizes += np.bincount(active.sum(-1), minlength=len(sizes))
    return {"split": "train", "dataset_sha256": dataset.manifest["dataset_sha256"],
            "split_sha256": dataset.manifest["split_sha256"], "molecules": len(dataset),
            "element_counts": elements[1:].tolist(), "charge_counts": charges.tolist(),
            "heavy_atom_histogram": sizes.tolist(), "schema": dataset.spec.to_dict()}


def authenticate_chemistry_handoff(plan, config, base_checkpoint, indices, runtime):
    """Authenticate a completed former-objective checkpoint without repinning z/reference.

    The original real-only parent remains the frozen occupancy anchor. This
    new branch inherits the latest completed chemistry model, AdamW moments,
    absolute update/epoch and ALL native plus fresh-prior random streams. No
    disposable benchmark checkpoint is an admissible training ancestor.
    """
    starting = validate_handoff_policy(plan)
    if starting is None:
        return None
    for key, sha in (("path", "sha256"), ("source_plan", "source_plan_sha256"), ("completion", "completion_sha256")):
        authenticated(starting[key], starting[sha])
    source = json.loads(Path(starting["source_plan"]).read_text())
    completion = json.loads(Path(starting["completion"]).read_text())
    root = Path(plan["work_root"]).resolve()
    for key in ("release", "campaign"):
        if not Path(source[key]).resolve().is_relative_to(root):
            raise ValueError("Starting objective source escaped the declared work root")
    authenticated(str(source["release"]) + ".tar.gz", source["archive_sha256"])
    if source_manifest(Path(source["release"]))["sha256"] != source["manifest_sha256"]:
        raise ValueError("Starting objective source manifest changed")
    # This declared observer adds no loss exemption to the source121 handoff.
    # Validate it before excluding only these explicitly versioned additions.
    validate_early_health_policy(plan["protocol"])
    from .chemistry_short_continuation import validate_short_continuation
    short = validate_short_continuation(plan)
    declared_additions = {"failure_guidance", "early_health"}
    if short is not None:
        declared_additions.update(("baseline_continuation", "valid_prior_self_guidance"))
    from .chemistry_target_weighting_policy import validate_target_weighting_test
    if validate_target_weighting_test(plan) is not None:
        # Only the explicitly declared new loss may differ from original36's
        # protocol; source/data/AdamW/RNG/reference checks below are unchanged.
        declared_additions.add("chemistry_target_weighting")
    from .chemistry_valid_reference_policy import validate_valid_reference_test
    if validate_valid_reference_test(plan) is not None or plan.get("valid_reference_continuation") is not None:
        declared_additions.add("valid_prior_reference_guidance")
    inherited = {key: value for key, value in plan["protocol"].items()
                 if key not in declared_additions}
    if (source["protocol"] != inherited or source.get("execution") != plan.get("execution")
            or any(source[key] != plan[key] for key in ("parent_checkpoint_sha256", "parent_epoch", "parent_updates",
                "parent_archive_sha256", "parent_manifest_sha256", "dataset_sha256", "split_sha256", "config_sha256", "training_molecules"))):
        raise ValueError("Starting objective data/reference/protocol/execution lineage changed")
    run_path = Path(completion["run_dir"]).resolve()
    checkpoint_path = Path(completion["checkpoint"]).resolve()
    if (not run_path.is_relative_to(Path(source["campaign"]).resolve() / "segments")
            or not checkpoint_path.is_relative_to(run_path)):
        raise ValueError("Only a persisted training segment may supply the new objective checkpoint")
    record = json.loads((run_path / "run.json").read_text())
    row = json.loads((run_path / "latest_epoch.json").read_text())
    gate_path = Path(source["campaign"]) / "benchmark_gate.json"
    gate = json.loads(gate_path.read_text())
    if (completion.get("status") != "passed" or completion.get("epoch_complete") is not True
            or completion.get("plan_sha256") != starting["source_plan_sha256"]
            or completion.get("completed_epoch") != starting["epoch"]
            or completion.get("updates") != starting["updates"]
            or completion.get("checkpoint_sha256") != starting["sha256"]
            or record.get("status") != "completed"
            or record.get("source_archive_sha256") != source["archive_sha256"]
            or record.get("source", {}).get("sha256") != source["manifest_sha256"]
            or record.get("config", {}).get("_plan_sha256") != starting["source_plan_sha256"]
            or row.get("epoch") != starting["epoch"] or row.get("updates") != starting["updates"]
            or row.get("checkpoint_sha256") != starting["sha256"]
            or not row.get("health") or not all(type(value) is bool and value for value in row["health"].values())
            or row.get("real_points") != plan["training_molecules"]
            or row.get("fresh_prior_points") != plan["protocol"]["prior_ratio"] * plan["training_molecules"]
            or gate.get("status") != "passed" or gate.get("plan_sha256") != starting["source_plan_sha256"]
            or not gate.get("gates") or not all(type(value) is bool and value for value in gate["gates"].values())):
        raise ValueError("Starting chemistry checkpoint lacks persisted completed native health/source evidence")
    authenticated(checkpoint_path, starting["sha256"])
    payload = torch.load(starting["path"], map_location="cpu", weights_only=True)
    cont, phase = payload.get("continuation", {}), payload.get("chemistry_phase", {})
    expected_phase = {"version": VERSION, "plan_sha256": starting["source_plan_sha256"],
        "parent_sha256": plan["parent_checkpoint_sha256"], "benchmark_gate_sha256": file_sha256(gate_path),
        "execution": source.get("execution"), "parent_epoch": plan["parent_epoch"], "protocol": source["protocol"]}
    expected_updates = plan["parent_updates"] + (starting["epoch"] - plan["parent_epoch"]) * math.ceil(plan["training_molecules"] / 256)
    if (payload.get("format_version") != 1 or phase != expected_phase or payload.get("epoch") != starting["epoch"]
            or not plan["parent_epoch"] < starting["epoch"] < 100 or starting["updates"] != expected_updates
            or cont.get("updates") != starting["updates"] or cont.get("epoch_complete") is not True
            or cont.get("scheduler_state") is not None or cont.get("version") != 1
            or cont.get("runtime") != runtime or cont.get("indices") != indices
            or cont.get("algorithm") != starting["algorithm"] or cont.get("split_sha256") != plan["split_sha256"]
            or payload.get("dataset_sha256") != plan["dataset_sha256"]
            or payload.get("source_archive_sha256") != source["archive_sha256"]
            or payload.get("source_sha256") != source["manifest_sha256"]
            or comparable_train_config(payload.get("train_config")) != comparable_train_config(base_checkpoint["train_config"])
            or payload.get("graph_spec") != base_checkpoint["graph_spec"]
            or comparable_model_config(payload.get("model_config")) != comparable_model_config(base_checkpoint["model_config"])
            or payload.get("loss_config") != base_checkpoint["loss_config"]
            or payload.get("run_id") != record.get("run_id")
            or "fresh_prior" not in cont.get("rng", {}).get("loaders", {})):
        raise ValueError("Starting chemistry native checkpoint lineage differs")
    files = starting["algorithm"]["files"]
    if hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest() != starting["algorithm"]["sha256"]:
        raise ValueError("Starting chemistry algorithm digest differs")
    for name, sha in files.items():
        path = (Path(source["release"]) / "src/latentmol" / name).resolve()
        if not path.is_relative_to(Path(source["release"]).resolve() / "src/latentmol"):
            raise ValueError("Starting chemistry algorithm escaped its package")
        authenticated(path, sha)
    return payload


def setup(plan, *, resume=None, parent_authentication=None):
    """Authenticate the phase boundary; carry parent moments rather than reset AdamW."""
    seed_everything(plan["seed"])
    if not os.environ.get("LSB_JOBID") or not torch.cuda.is_available():
        raise RuntimeError("Numerical phase requires an allocated DTU CUDA job")
    execution = execution_settings(plan, allocated=True)
    properties = torch.cuda.get_device_properties(0)
    if "A100" not in properties.name or properties.total_memory < 75 * 1024**3:
        raise RuntimeError("Require A10080GB hardware")
    authenticated(plan["config"], plan["config_sha256"])
    config = load_parent_config(Path(plan["config"]))
    if config.mode != "vae" or config.epochs != 100 or config.batch_size != 256 or config.num_workers:
        raise ValueError("Parent training configuration changed")
    cache = validate_large_cache(Path(plan["data"]))
    data = {split: GraphDataset(plan["data"], split, validated_cache=cache) for split in ("train", "val")}
    if (data["train"].manifest["dataset_sha256"] != plan["dataset_sha256"]
            or data["train"].manifest["split_sha256"] != plan["split_sha256"]):
        raise ValueError("Dataset or split identity differs")
    parent, checkpoint = load_model(Path(plan["parent_checkpoint"]), config.device, data["train"].spec)
    state = checkpoint.get("continuation", {})
    parent_run = json.loads(Path(plan["parent_run_record"]).read_text())
    proof = authenticate_parent_metadata(checkpoint, config, plan, runtime_identity(config.device), parent_run)
    if parent_authentication is not None:
        parent_authentication.update(proof)
    # Added chemistry files deliberately change algorithm_identity. Every old
    # executable must still be byte-identical to its authenticated source098 file.
    old_files = state["algorithm"]["files"]
    if hashlib.sha256(json.dumps(old_files, sort_keys=True).encode()).hexdigest() != state["algorithm"]["sha256"]:
        raise ValueError("Parent executable manifest digest differs")
    package = Path(__file__).resolve().parent
    for path, sha in old_files.items():
        authenticated(Path(plan["parent_release"]) / "src/latentmol" / path, sha)
        authenticated(package / path, sha)
    cohorts = training_cohorts(config, len(data["train"]), len(data["val"]))
    identity = compact_cohort_identity(cohorts)
    if state["indices"] != identity:
        raise ValueError("Parent training/validation cohort differs")
    reference = parent
    reference.eval().requires_grad_(False)
    model = deepcopy(reference).requires_grad_(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    generators = {name: torch.Generator() for name in state["rng"]["loaders"]}
    restore_rng_state(state["rng"], generators)
    generators["fresh_prior"] = torch.Generator().manual_seed(plan["protocol"]["prior_seed"])
    parent_epoch, updates = checkpoint["epoch"], state["updates"]
    algorithm, runtime = algorithm_identity(), runtime_identity(config.device)
    starting = authenticate_chemistry_handoff(plan, config, checkpoint, identity, runtime)
    if starting is not None:
        model.load_state_dict(starting["model_state"], strict=True)
        optimizer.load_state_dict(starting["optimizer_state"])
        restore_rng_state(starting["continuation"]["rng"], generators)
        parent_epoch, updates = starting["epoch"], starting["continuation"]["updates"]
        if parent_authentication is not None:
            parent_authentication["objective_change_handoff"] = {"passed": True,
                "starting_checkpoint": plan["starting_checkpoint"], "frozen_reference_epoch": plan["parent_epoch"],
                "phase_start_epoch": plan["parent_epoch"], "fresh_prior_rng_reseeded": False}
    # Pin the anchor BEFORE a later segment resume. It must always come from
    # original36, while reference35 retains its original occupancy-only role.
    valid_anchor = None
    if plan["protocol"].get("valid_prior_reference_guidance") is not None:
        from .chemistry_valid_reference_anchor import FrozenValidReferenceAnchor
        if starting is None or parent_epoch != 36:
            raise ValueError("Valid-reference targets require the authenticated epoch36 handoff")
        valid_anchor = FrozenValidReferenceAnchor(model,reference_epoch=36,
            checkpoint_sha256=plan["starting_checkpoint"]["sha256"])
        object.__setattr__(model,"_valid_reference_anchor",valid_anchor)
    if resume is not None:
        require_failure_readiness(plan)
        path, sha = resume
        authenticated(path, sha)
        resumed = torch.load(path, map_location="cpu", weights_only=True)
        reject_diagnostic_snapshot(resumed)
        phase = resumed.get("chemistry_phase", {})
        cont = resumed.get("continuation", {})
        identity_plan, identity_algorithm = plan, algorithm
        if "epoch45_extension" in plan:
            from .chemistry_epoch45_extension import resume_identity
            identity_plan, identity_algorithm = resume_identity(plan, path, sha, algorithm)
        if (phase.get("plan_sha256") != identity_plan["_plan_sha256"] or phase.get("version") != VERSION
                or phase.get("parent_sha256") != identity_plan["parent_checkpoint_sha256"]
                or phase.get("benchmark_gate_sha256") != identity_plan["_benchmark_gate_sha256"]
                or cont.get("algorithm") != identity_algorithm or cont.get("runtime") != runtime
                or phase.get("protocol") != identity_plan["protocol"]
                or phase.get("runtime_policy_amendment") != identity_plan.get("runtime_policy_amendment")
                or phase.get("early_health_baseline_sha256") != identity_plan.get("_early_health_baseline_sha256")
                or phase.get("execution") != identity_plan.get("execution")
                or phase.get("classification_execution") != identity_plan.get("classification_execution")
                or phase.get("worker_startup") != identity_plan.get("worker_startup")
                or phase.get("short_continuation") != identity_plan.get("short_continuation")
                or phase.get("valid_reference_continuation") != identity_plan.get("valid_reference_continuation")
                or phase.get("valid_reference_adoption") != identity_plan.get("valid_reference_adoption")
                or phase.get("epoch45_extension") != identity_plan.get("epoch45_extension")
                or phase.get("optimizer_step_control") != identity_plan.get("optimizer_step_control")
                or ((plan["protocol"].get("failure_guidance") is not None or plan.get("short_continuation") is not None
                     or plan.get("valid_reference_continuation") is not None)
                    and phase.get("failure_readiness_sha256") != identity_plan.get("_failure_readiness_sha256"))
                or resumed.get("source_archive_sha256") != identity_plan["archive_sha256"]
                or resumed.get("source_sha256") != identity_plan["manifest_sha256"]
                or not cont.get("epoch_complete") or cont.get("indices") != identity
                or cont.get("scheduler_state") is not None
                or type(resumed.get("epoch")) is not int or not plan["parent_epoch"] < resumed["epoch"] <= _continuation_endpoint(plan)
                or (starting is not None and resumed["epoch"] <= starting["epoch"])
                or cont.get("updates") != plan["parent_updates"] + (resumed["epoch"] - plan["parent_epoch"]) * math.ceil(len(data["train"]) / 256)
                or resumed["graph_spec"] != model.spec.to_dict()
                or resumed["dataset_sha256"] != plan["dataset_sha256"]
                or cont["split_sha256"] != plan["split_sha256"]
                or comparable_train_config(resumed["train_config"]) != comparable_train_config(checkpoint["train_config"])):
            raise ValueError("Exact chemistry continuation identity differs")
        model.load_state_dict(resumed["model_state"], strict=True)
        optimizer.load_state_dict(resumed["optimizer_state"])
        if plan.get("optimizer_step_control") is not None:
            from .chemistry_optimizer_step import authenticate_optimizer_step_evidence
            epoch_updates = math.ceil(len(data["train"]) / 256)
            authenticate_optimizer_step_evidence(cont.get("optimizer_step_evidence"), plan["optimizer_step_control"],
                root=Path(path).resolve().parent, epoch=resumed["epoch"],
                start_update=cont["updates"] - epoch_updates, end_update=cont["updates"])
        restore_rng_state(cont["rng"], generators)
        parent_epoch, updates = resumed["epoch"], cont["updates"]
    from .failure_guidance_training import build_repair_teacher
    teacher = build_repair_teacher(model.spec, plan["protocol"], execution,
        classification_execution=plan.get("classification_execution"))
    if valid_anchor is not None:
        teacher.valid_reference_anchor = valid_anchor
    return model, reference, optimizer, config, data, cohorts, generators, teacher, parent_epoch, updates, algorithm, runtime


def chemistry_update(model, reference, optimizer, hard, config, protocol, composition, teacher,
                     generators, update, epoch, phase_start_epoch, *, do_step=True, microbatch_size=None,
                     deadline=None, overlap=False, step_control=None, capture_context=None, component_observer=None,
                     additive_guidance=None, prior_sampler=None, from_canonical_cache=False,
                     allow_equal_prior_ratio=False, wait_observer=None):
    """Drain every detached task on failure; no partial optimizer step survives.

    The synchronous legacy path retains its ordering. Async CPU work can run
    while the frozen-weight GPU prepass and ready-chunk VJPs execute. Neither
    task completion order nor unavailable targets can change point membership.
    ``additive_guidance`` is a separate explicit pilot object, absent from old
    plans; it cannot replace existing repair, reference or optimizer semantics.
    """
    if type(overlap) is not bool:
        raise ValueError("Chemistry overlap must be an explicit boolean")
    try:
        return _chemistry_update_impl(model, reference, optimizer, hard, config, protocol,
            composition, teacher, generators, update, epoch, phase_start_epoch,
            do_step=do_step, microbatch_size=microbatch_size, deadline=deadline, overlap=overlap,
            step_control=step_control, capture_context=capture_context, component_observer=component_observer,
            additive_guidance=additive_guidance, prior_sampler=prior_sampler,
            from_canonical_cache=from_canonical_cache, allow_equal_prior_ratio=allow_equal_prior_ratio,
            wait_observer=wait_observer)
    except BaseException:
        if overlap:
            # Preserve the original computation/search exception. All tasks own
            # detached CPU snapshots, and must finish before state restoration.
            try:
                teacher.drain()
            except BaseException:
                pass
        raise


def _chemistry_update_impl(model, reference, optimizer, hard, config, protocol, composition, teacher,
                     generators, update, epoch, phase_start_epoch, *, do_step=True, microbatch_size=None,
                     deadline=None, overlap=False, step_control=None, capture_context=None, component_observer=None,
                     additive_guidance=None, prior_sampler=None, from_canonical_cache=False,
                     allow_equal_prior_ratio=False, wait_observer=None):
    """A logical real/prior update with fixed-target and population VJPs.

    Store posterior epsilon and fresh prior z once. Recomputing the deterministic
    decoder at unchanged weights yields the same teacher and global histogram.
    A global JS cotangent and globally weighted reconstruction denominators give
    the full-batch objective without retaining every graph's autograd tape.
    Optional additive guidance owns its preparation and complete-branch
    normalization; the historical path does not construct or evaluate it.
    """
    count = len(hard.atoms)
    if type(from_canonical_cache) is not bool:
        raise ValueError("Canonical cache use must be explicitly selected")

    def encode_real(target):
        if from_canonical_cache:
            return model.encoder(target, from_canonical_cache=True)
        return model.encode(target.to_relaxed(model.spec))

    if additive_guidance is not None:
        if capture_context is not None:
            raise ValueError("Historical mechanism capture does not include additive guidance")
        additive_guidance.begin_update(update=update)
    if step_control is not None:
        from .chemistry_optimizer_step import validate_optimizer_step_control
        step_control = validate_optimizer_step_control(step_control)
    prior_count = protocol["prior_ratio"] * count
    prior_composition_weight = composition_prior_weight(protocol, allow_equal_prior_ratio=allow_equal_prior_ratio)
    from .chemistry_short_continuation import validate_valid_prior_self_guidance
    self_policy = validate_valid_prior_self_guidance(protocol.get("valid_prior_self_guidance"))
    from .chemistry_target_weighting import validate_chemistry_target_weighting
    target_weighting = validate_chemistry_target_weighting(protocol.get("chemistry_target_weighting"))
    if "chemistry_target_weighting" in protocol and target_weighting is None:
        raise ValueError("An explicit chemistry target weighting cannot be null")
    if self_policy is not None:
        from .chemistry_valid_self_guidance import prepare_valid_self_guidance, valid_prior_self_nll
    from .chemistry_valid_reference_guidance import validate_valid_prior_reference_guidance
    reference_policy = validate_valid_prior_reference_guidance(protocol.get("valid_prior_reference_guidance"))
    if "valid_prior_reference_guidance" in protocol and reference_policy is None:
        raise ValueError("An explicit valid-reference policy cannot be null")
    valid_anchor = getattr(model,"_valid_reference_anchor",None)
    if reference_policy is not None:
        from .chemistry_valid_reference_guidance import valid_prior_reference_nll
        if valid_anchor is None or self_policy is not None or target_weighting is not None:
            raise ValueError("Frozen valid-reference guidance requires its separate anchor and full unweighted NLL")
    failure_policy = validate_failure_guidance(protocol.get("failure_guidance"))
    if reference_policy is not None and failure_policy is not None:
        raise ValueError("Valid-reference candidate keeps original live repair; no fallback objective")
    if target_weighting is not None and failure_policy is not None:
        raise ValueError("Weighted target CE does not implement failed-row fallback guidance")
    if capture_context is not None and (failure_policy is not None or self_policy is not None or target_weighting is not None or reference_policy is not None):
        raise ValueError("Mechanism capture is restricted to the original repair objective")
    if component_observer is not None and failure_policy is not None:
        raise ValueError("Component observation supports original repair and explicit valid-self guidance only")
    if failure_policy is not None:
        from .failure_guidance_training import FailureAwareRepairTeacher, FailureTeacherBatch
        from .repair_failure_guidance import FailureGuidanceConfig, failure_aware_repair_loss
        if (not isinstance(teacher, FailureAwareRepairTeacher)
                or teacher.failure_guidance != failure_policy):
            raise ValueError("New chemistry objective requires its matching versioned teacher")
        failure_loss_config = FailureGuidanceConfig(**failure_policy["loss"])
    size = microbatch_size or protocol["microbatch_size"]
    device = next(model.parameters()).device
    hard = hard.to(device)
    epsilon = torch.randn(count, model.config.latent, device=device)
    prior = torch.randn(prior_count, model.config.latent, generator=generators["fresh_prior"]).to(device)
    if prior_sampler is not None:
        # An explicit new study can replace a declared subset of the existing
        # prior budget. Draw once, detach sampling, and reuse exactly these z
        # values in both teacher preparation and the live decoder backward.
        prior = prior_sampler.sample(model, hard, prior, update=update, microbatch_size=size)
        if prior.shape != (prior_count, model.config.latent) or prior.requires_grad or not bool(torch.isfinite(prior).all()):
            raise ValueError("Prior sampler returned incompatible, attached or nonfinite coordinates")
    denominator = reconstruction_denominators(hard, config.loss, dtype=next(model.parameters()).dtype)
    targets, totals = {}, {"elements": torch.zeros(len(model.spec.atomic_numbers) - 1, dtype=torch.float64, device=device),
                          "charges": torch.zeros(len(model.spec.formal_charges), dtype=torch.float64, device=device)}
    accepted, eligible, checks = {"real": 0, "prior": 0}, {"real": 0, "prior": 0}, 0
    search_seconds = 0.
    receipt_chain = "0" * 64
    raw_statuses = {"real": {}, "prior": {}}
    failure_coverage = {"real": {}, "prior": {}}
    classifications = {"real": [], "prior": []}
    teacher_phases = {}
    pending = {}
    self_snapshots, self_targets = {}, {}
    self_eligible = 0
    reference_targets, reference_pending = {}, {}
    reference_eligible = reference_live_invalid = 0
    reference_chain = "0" * 64

    def resolve_target(future, name):
        if wait_observer is None:
            return future.result()
        with wait_observer(name):
            return future.result()

    def receive(branch, start, stop, batch):
        nonlocal receipt_chain, checks, search_seconds, self_eligible, reference_eligible, reference_live_invalid, reference_chain
        targets[branch, start] = batch
        if failure_policy is not None:
            if not isinstance(batch, FailureTeacherBatch):
                raise ValueError("Failure-aware targets lost their preparation metadata")
            receipt = batch.base.receipt
            extra = batch.prepared.receipt
            if len(batch.prepared.rows) != stop - start or batch.prepared.base is not batch.base:
                raise ValueError("Prepared references are not the original complete microbatch")
            coverage = failure_coverage[branch]
            for key in ("attempted_rows", "failed_rows", "fallback_rows"):
                coverage[key] = coverage.get(key, 0) + extra[key]
            for row in extra["rows"]:
                status = row["witness_kind"] or "no_witness"
                coverage[status] = coverage.get(status, 0) + 1
                for key in ("supported_local_atoms", "unsupported_local_atoms"):
                    coverage[key] = coverage.get(key, 0) + row[key]
                if row["extra_search"] is not None:
                    checks += row["extra_search"]["checks"]
                    coverage["extra_search_checks"] = coverage.get("extra_search_checks", 0) + row["extra_search"]["checks"]
            search_seconds += batch.preparation_seconds
            for name, seconds in batch.preparation_phases.items():
                # Detached coordinator/worker timing is observational only.
                # Worker sums and async phase walls may overlap GPU execution.
                if not math.isfinite(seconds) or seconds < 0:
                    raise ValueError("Invalid detached teacher phase timing")
                teacher_phases[name] = teacher_phases.get(name, 0.) + seconds
        else:
            receipt = batch.receipt
        classification = receipt.get("chemical_classification")
        if classification is not None:
            if classification.get("attempts") != stop - start:
                raise ValueError("Teacher classification lost its exact scheduled microbatch")
            classifications[branch].append(classification)
        elif failure_policy is not None and failure_policy["loss"]["version"] == "repair-failure-guidance-v2":
            raise ValueError("New guidance requires separate chemical/representation coverage")
        semantic = {key: value for key, value in receipt.items() if key not in ("elapsed_seconds", "cache_hits", "search_checks")}
        semantic["attempts"] = [{key: value for key, value in item.items() if key != "cache_hit"}
                                for item in receipt.get("attempts", [])]
        if failure_policy is not None:
            # The complete prepared reference retains timing in its identity
            # via its base receipt. A semantic execution comparison must not
            # mistake elapsed-time variation for changed scientific targets.
            semantic["failure_guidance"] = {key: value for key, value in batch.prepared.receipt.items()
                                             if key != "identity"}
        if self_policy is not None and branch == "prior":
            prepared_self = prepare_valid_self_guidance(self_snapshots.pop(start), batch, model.spec)
            self_targets[start] = prepared_self
            self_eligible += len(prepared_self.eligible_rows)
            semantic["valid_prior_self_guidance"] = prepared_self.receipt
        if target_weighting is not None:
            semantic["chemistry_target_weighting"] = dict(target_weighting)
        if reference_policy is not None and branch == "prior":
            if overlap:
                reference_targets[start] = resolve_target(reference_pending.pop(start), "reference_wait")
            prepared_reference = reference_targets[start]
            if prepared_reference.candidate_ids != batch.candidate_ids:
                raise ValueError("Frozen same-z targets lost live candidate-slot alignment")
            reference_eligible += len(prepared_reference.eligible_rows)
            reference_live_invalid += sum(
                receipt["attempts"][i]["chemical_classification"]["strict_teacher_valid"] is not True
                for i in prepared_reference.eligible_rows)
            semantic["valid_prior_reference_guidance"] = prepared_reference.receipt
            reference_chain = digest(dict(previous=reference_chain,receipt=prepared_reference.receipt))
        receipt_chain = digest({"previous": receipt_chain, "receipt": semantic})
        if prior_sampler is not None and branch == "prior":
            prior_sampler.observe_targets(receipt, start=start, stop=stop,
                reference_eligible=prepared_reference.eligible_rows if reference_policy is not None else ())
        if receipt["status_counts"].get("rejected_or_error", 0):
            raise RuntimeError("Repair provenance/representation check failed; do not train on silently rejected targets")
        accepted[branch] += receipt["accepted_targets"]
        eligible[branch] += stop - start - receipt["status_counts"].get("skipped_already_valid", 0)
        checks += receipt["search_checks"]
        search_seconds += receipt["elapsed_seconds"]
        for key, value in receipt["status_counts"].items():
            raw_statuses[branch][key] = raw_statuses[branch].get(key, 0) + value

    search_start = time.monotonic()
    model.train()
    with torch.no_grad():
        for branch, total in (("real", count), ("prior", prior_count)):
            for start, stop in chunks(total, size):
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError("Bounded teacher prepass budget exhausted")
                if branch == "real":
                    mu, logvar = encode_real(hard_slice(hard, start, stop))
                    z = mu + epsilon[start:stop] * (.5 * logvar).exp()
                else:
                    z = prior[start:stop]
                decoded = model.decode(z)
                if additive_guidance is not None:
                    # Freeze this original raw graph, never a repair witness.
                    additive_guidance.prepare(decoded, branch=branch, start=start, stop=stop)
                if self_policy is not None and branch == "prior":
                    # One frozen prepass snapshot binds strict validity and
                    # pseudo-labels; live logits are recomputed for backward.
                    # No discrete decision or chemistry validator is in autograd.
                    from .representation import DecoderOutput
                    self_snapshots[start] = DecoderOutput(*(value.detach().cpu().clone()
                        for value in (decoded.atoms, decoded.charges, decoded.bonds)))
                statistics = composition_statistics(decoded, model.spec)
                for key in totals:
                    contribution = getattr(statistics, key).double()
                    totals[key] += contribution if branch == "real" else contribution * prior_composition_weight
                # IDs are fresh per absolute update, branch and row, including tails.
                offset = 0 if branch == "real" else protocol["real_batch_size"]
                ids = [update * ((1 + protocol["prior_ratio"]) * protocol["real_batch_size"]) + offset + row for row in range(start, stop)]
                if reference_policy is not None and branch == "prior":
                    prepared_reference = valid_anchor.prepare(teacher,z,candidate_ids=ids,update=update,overlap=overlap)
                    if overlap:
                        reference_pending[start] = prepared_reference
                    else:
                        reference_targets[start] = prepared_reference
                if overlap:
                    pending[branch, start] = teacher.build_targets_async(decoded, candidate_ids=ids, update=update)
                else:
                    receive(branch, start, stop, teacher.build_targets(decoded, candidate_ids=ids, update=update))
    if additive_guidance is not None:
        additive_guidance.finalize_preparation()
    prepass_seconds = time.monotonic() - search_start
    js = {key: population_js_cotangent(value, composition[key + "_distribution"])
          for key, value in totals.items()}
    coefficient = protocol["repair_weight"] * min((epoch - phase_start_epoch) / protocol["repair_ramp_epochs"], 1.)
    beta = config.beta * min(epoch / max(config.kl_warmup_epochs, 1), 1.)
    optimizer.zero_grad(set_to_none=True)
    metrics = {"reconstruction": 0., "kl": 0., "repair_real": 0., "repair_prior": 0., "occupancy_anchor": 0.}
    if self_policy is not None:
        metrics.update(valid_prior_self_nll=0., valid_prior_self_weighted=0.)
        self_gradient = None
    if reference_policy is not None:
        metrics.update(valid_prior_reference_nll=0.,valid_prior_reference_weighted=0.)
        reference_gradient = None
    failure_components = {branch: {} for branch in ("real", "prior")}
    repair_gradient = failure_gradient = None
    for branch, total in (("real", count), ("prior", prior_count)):
        for start, stop in chunks(total, size):
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError("Bounded streamed update budget exhausted")
            fraction = (stop - start) / total
            if overlap:
                receive(branch, start, stop, resolve_target(pending.pop((branch, start)), f"{branch}_repair_wait"))
                if deadline is not None and time.monotonic() > deadline:
                    raise TimeoutError("Detached repair wait exhausted streamed update budget")
            rec = kl = torch.zeros((), device=device)
            if branch == "real":
                target = hard_slice(hard, start, stop)
                mu, logvar = encode_real(target)
                z = mu + epsilon[start:stop] * (.5 * logvar).exp()
            else:
                z = prior[start:stop]
            decoded = model.decode(z)
            self_term = decoded.atoms.reshape(-1)[0] * 0
            reference_term = decoded.atoms.reshape(-1)[0] * 0
            if self_policy is not None and branch == "prior":
                self_result = valid_prior_self_nll(decoded, self_targets.pop(start), model.spec, self_policy,
                    **({"target_weighting": target_weighting} if target_weighting is not None else {}))
                self_term = self_result["weighted_loss"] * fraction
                metrics["valid_prior_self_nll"] += float(self_result["loss"].detach()) * fraction
                metrics["valid_prior_self_weighted"] += float(self_term.detach())
                if self_gradient is None and self_result["eligible_rows"]:
                    values = torch.autograd.grad(self_term, tuple(model.decoder.parameters()),
                                                 retain_graph=True, allow_unused=True)
                    self_gradient = math.sqrt(sum(float(value.detach().double().square().sum())
                                                  for value in values if value is not None))
            if reference_policy is not None and branch == "prior":
                reference_result = valid_prior_reference_nll(decoded,reference_targets.pop(start),model.spec,reference_policy)
                reference_term = reference_result["weighted_loss"] * fraction
                metrics["valid_prior_reference_nll"] += float(reference_result["loss"].detach()) * fraction
                metrics["valid_prior_reference_weighted"] += float(reference_term.detach())
                if reference_gradient is None and reference_result["eligible_rows"]:
                    values = torch.autograd.grad(reference_term,tuple(model.decoder.parameters()),retain_graph=True,allow_unused=True)
                    reference_gradient = math.sqrt(sum(float(value.detach().double().square().sum()) for value in values if value is not None))
            if branch == "real":
                rec = reconstruction_chunk(decoded, target, config.loss, denominator)
                kl = latent_kl(mu, logvar) * fraction
            if failure_policy is None:
                chemical = repair_loss(decoded, targets[branch, start], model.spec,
                    **({"target_weighting": target_weighting} if target_weighting is not None else {})) * fraction
                valid_chemical = chemical
                has_guidance = any(t is not None for t in targets[branch, start].targets)
            else:
                prepared = targets[branch, start].prepared
                terms = failure_aware_repair_loss(decoded, prepared, model.spec, failure_loss_config)
                chemical = terms["total"] * fraction
                valid_chemical = terms["valid_full_nll"] * fraction
                for key in ("valid_full_nll", "extra_valid_nll", "partial_edited_nll", "local_bond_risk", "component_cut_risk", "nonempty_risk"):
                    failure_components[branch][key] = failure_components[branch].get(key, 0.) + float(terms[key].detach()) * fraction
                has_guidance = any(t is not None for t in prepared.base.targets)
                if branch == "prior" and failure_gradient is None and any(row is not None for row in prepared.rows):
                    values = torch.autograd.grad(chemical - valid_chemical, tuple(model.decoder.parameters()),
                                                 retain_graph=True, allow_unused=True)
                    failure_gradient = float(torch.stack([v.detach().double().square().sum() for v in values if v is not None]).sum().sqrt())
            if branch == "prior" and repair_gradient is None and has_guidance:
                values = torch.autograd.grad(valid_chemical, tuple(model.decoder.parameters()), retain_graph=True, allow_unused=True)
                repair_gradient = float(torch.stack([v.detach().double().square().sum() for v in values if v is not None]).sum().sqrt())
            statistics = composition_statistics(decoded, model.spec)
            composition_surrogate = (protocol["element_js_weight"] * (statistics.elements.double() * js["elements"][1]).sum()
                                     + protocol["charge_js_weight"] * (statistics.charges.double() * js["charges"][1]).sum())
            if branch == "prior":
                composition_surrogate = composition_surrogate * prior_composition_weight
            anchor = decoded.atoms.reshape(-1)[0] * 0
            if branch == "prior":
                with torch.no_grad():
                    old_occupancy = reference.decode(z).probabilities().occupancy
                anchor = (decoded.probabilities().occupancy - old_occupancy).square().mean() * fraction
            branch_weight = protocol["real_repair_weight"] if branch == "real" else protocol["prior_repair_weight"]
            loss = rec + beta * kl + coefficient * branch_weight * chemical + composition_surrogate + protocol["occupancy_anchor_weight"] * anchor
            if self_policy is not None and self_policy["weight"] > 0 and branch == "prior":
                loss = loss + self_term
            if reference_policy is not None and branch == "prior":
                loss = loss + reference_term
            if additive_guidance is not None:
                # The adapter owns exact branch-global center normalization;
                # no extra molecule-fraction weighting belongs at this call.
                additive_term = additive_guidance.loss(decoded, branch=branch, start=start, stop=stop)
                loss = loss + additive_term
            if component_observer is not None:
                observed_terms = dict(reconstruction=rec, kl=beta*kl,
                    repair=coefficient*branch_weight*chemical, composition=composition_surrogate,
                    occupancy=protocol["occupancy_anchor_weight"]*anchor)
                if self_policy is not None:
                    observed_terms["valid_self"] = self_term
                if reference_policy is not None:
                    observed_terms["valid_reference"] = reference_term
                if additive_guidance is not None:
                    observed_terms["additive_guidance"] = additive_term
                component_observer(branch, observed_terms)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite streamed phase loss")
            loss.backward()
            metrics["reconstruction"] += float(rec.detach()); metrics["kl"] += float(kl.detach())
            metrics["repair_" + branch] += float(chemical.detach()); metrics["occupancy_anchor"] += float(anchor.detach())
    if overlap:
        if pending:
            raise RuntimeError("Unconsumed scheduled chemistry targets remain before optimizer step")
        teacher.drain()
    if additive_guidance is not None:
        # Verify complete context consumption before an optimizer can step.
        metrics["additive_guidance"] = additive_guidance.summary()
    groups = {"encoder": gradient_norm(model.encoder.parameters()), "decoder": gradient_norm(model.decoder.parameters())}
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad, error_if_nonfinite=True)) if do_step else gradient_norm(model.parameters())
    if do_step:
        if step_control is None:
            optimizer.step()
        else:
            from .chemistry_optimizer_step import apply_optimizer_step
            step_receipt = apply_optimizer_step(optimizer, model.parameters(), step_control)
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise FloatingPointError("Nonfinite phase weights")
    metrics.update(element_js=float(js["elements"][0]), charge_js=float(js["charges"][0]), repair_coefficient=coefficient,
                   beta=beta, pre_clip_gradient_norm=norm, group_gradient_norms=groups,
                   clipped=norm > config.clip_grad, real_points=count, fresh_prior_points=prior_count,
                   repair_targets=accepted, repair_eligible=eligible, search_checks=checks,
                   search_seconds=search_seconds, prepass_seconds=prepass_seconds, repair_statuses=raw_statuses)
    metrics["repair_only_decoder_gradient_norm"] = repair_gradient
    if failure_policy is not None:
        metrics["failure_guidance_policy"] = failure_policy
        metrics["failure_guidance_coverage"] = failure_coverage
        metrics["failure_guidance_components"] = failure_components
        metrics["failure_only_decoder_gradient_norm"] = failure_gradient
    if any(classifications.values()):
        from .chemistry_classification import merge_classification_summaries
        metrics["teacher_classification"] = {branch: merge_classification_summaries(values)
                                             for branch, values in classifications.items()}
    metrics["ordered_teacher_receipt_chain_sha256"] = receipt_chain
    if target_weighting is not None:
        metrics["chemistry_target_weighting"] = dict(target_weighting)
        metrics["chemistry_target_score_kind"] = "weighted_target_cross_entropy_not_exact_graph_NLL"
    if self_policy is not None:
        if self_snapshots or self_targets:
            raise RuntimeError("Unconsumed valid-prior self-target snapshots remain")
        metrics["valid_prior_self_guidance"] = dict(policy=self_policy, attempted=prior_count,
            eligible=self_eligible, excluded=prior_count-self_eligible,
            scope="Detached strict-valid prior self-targets; separate from repair witnesses")
        metrics["valid_prior_self_decoder_gradient_norm"] = self_gradient
    if reference_policy is not None:
        if reference_targets or reference_pending:
            raise RuntimeError("Unconsumed frozen valid-reference targets remain")
        metrics["valid_prior_reference_guidance"] = dict(policy=reference_policy,
            reference_identity=valid_anchor.reference_identity,attempted=prior_count,eligible=reference_eligible,
            excluded=prior_count-reference_eligible,reference_valid_live_invalid=reference_live_invalid,
            target_receipt_chain_sha256=reference_chain,
            scope="Frozen original36 same-z strict-valid targets, independent of live validity; no reference repair")
        metrics["valid_prior_reference_decoder_gradient_norm"] = reference_gradient
    if teacher_phases:
        metrics["teacher_phase_seconds"] = teacher_phases
    metrics["total"] = (metrics["reconstruction"] + beta * metrics["kl"]
                        + coefficient * (.25 * metrics["repair_real"] + .75 * metrics["repair_prior"])
                        + protocol["element_js_weight"] * metrics["element_js"]
                        + protocol["charge_js_weight"] * metrics["charge_js"]
                        + protocol["occupancy_anchor_weight"] * metrics["occupancy_anchor"])
    if reference_policy is not None:
        metrics["total"] += metrics["valid_prior_reference_weighted"]
    if self_policy is not None:
        metrics["total"] += metrics["valid_prior_self_weighted"]
    if additive_guidance is not None:
        metrics["total"] += metrics["additive_guidance"]["weighted_loss"]
    if do_step and step_control is not None:
        metrics["optimizer_step_control"] = step_receipt
    if prior_sampler is not None:
        metrics["latent_neighborhood"] = prior_sampler.summary()
    if capture_context is not None:
        # Diagnostic ownership only: no tensor here becomes a training
        # checkpoint. Reuse the exact draws and complete frozen teacher batches
        # when evaluating the finite-step objective at counterfactual weights.
        capture_context.update(hard=hard, epsilon=epsilon.detach().clone(), prior=prior.detach().clone(),
            targets=dict(targets), denominator=denominator, coefficient=coefficient, beta=beta,
            size=size, real_points=count, prior_points=prior_count,
            prior_composition_weight=prior_composition_weight)
    return metrics


@torch.no_grad()
def population_panel(model, count, seed, batch_size, composition, *, classification=None):
    """Same fixed N(0,I) coordinates for parent and phase; no repair or filtering."""
    from .chemistry import inspect_graph
    from .large_vae_evaluation import checked_molecule
    generator = torch.Generator().manual_seed(seed)
    z = torch.randn(count, model.config.latent, generator=generator)
    device = next(model.parameters()).device
    elements = torch.zeros(len(model.spec.atomic_numbers) - 1, dtype=torch.float64, device=device)
    charges = torch.zeros(len(model.spec.formal_charges), dtype=torch.float64, device=device)
    occupied, nonempty, valid, smiles = 0., 0, 0, set()
    sizes = torch.zeros(model.spec.max_nodes + 1, dtype=torch.long)
    atom_entropy = 0.
    for start, stop in chunks(count, batch_size):
        decoded = model.decode(z[start:stop].to(device))
        stats = composition_statistics(decoded, model.spec)
        elements += stats.elements.double(); charges += stats.charges.double(); occupied += float(stats.occupied_mass)
        law = decoded.atoms.softmax(-1)
        atom_entropy += float(-(law * law.clamp_min(1e-30).log()).sum(-1).sum())
        hard = decoded.harden(model.spec).to("cpu")
        sizes += torch.bincount((hard.atoms != 0).sum(-1), minlength=model.spec.max_nodes + 1)
        nonempty += int((hard.atoms != 0).any(-1).sum())
        for index, (a, q, b) in enumerate(zip(hard.atoms, hard.charges, hard.bonds)):
            record = inspect_graph(a, q, b, model.spec)
            mol, _ = checked_molecule(record.smiles)
            if mol is not None:
                valid += 1; smiles.add(record.smiles)
            if classification is not None:
                classification.add(a, q, b, model.spec, candidate_id=start + index,
                    probabilities=lambda: (decoded.atoms[index].softmax(-1), decoded.charges[index].softmax(-1),
                                           decoded.symmetric_bond_logits[index].softmax(-1)))
    frequency = (elements / elements.sum().clamp_min(1)).cpu().tolist()
    sulfur = model.spec.atomic_numbers[1:].index(16) if 16 in model.spec.atomic_numbers else None
    return {"attempts": count, "raw_argmax_valid_count": valid, "raw_argmax_validity": valid / count,
            "unique_valid_count": len(smiles), "unique_valid_per_attempt": len(smiles) / count,
            "nonempty_fraction": nonempty / count, "expected_size": occupied / count,
            "element_frequencies": frequency, "sulfur_frequency": frequency[sulfur] if sulfur is not None else 0.,
            "argmax_size_histogram": sizes.tolist(), "mean_atom_slot_entropy_nats": atom_entropy / count / model.spec.max_nodes,
            "element_js": float(population_js(elements, composition["elements_distribution"])),
            "charge_js": float(population_js(charges, composition["charges_distribution"])),
            "sampling": "fixed_seed_standard_gaussian_raw_argmax_no_repair"}


def observe_early_health(model, data, config, protocol, composition, generators):
    """Fixed native panels with bounded actual examples; restore training RNG/modes."""
    policy = validate_early_health_policy(protocol)
    if policy is None:
        raise ValueError("Early observation requires its explicit versioned policy")
    collector = EarlyPanelCollector(policy["max_examples_per_failure"])
    with preserve_observation_state(model, generators):
        panel = population_panel(model, 1024, protocol["evaluation_seed"], 32, composition,
                                 classification=collector)
        reconstruction = validation(model, data["val"], config, 256, protocol["evaluation_seed"])
    observation = dict(population=panel, reconstruction=reconstruction, classification=collector.summary())
    validate_early_health_observation(observation)
    return observation


def assess_early_health(observation, reference_population, reference_reconstruction,
                        starting_population, starting_reconstruction, protocol, *, starting_classification):
    """Keep all old health checks and protect the exact handoff's strict validity."""
    policy = validate_early_health_policy(protocol)
    if policy is None:
        raise ValueError("Early assessment requires its explicit versioned policy")
    panel, reconstruction = observation["population"], observation["reconstruction"]
    if (panel.get("attempts") != 1024 or reconstruction.get("molecules") != 256
            or observation["classification"].get("attempts") != 1024
            or starting_classification.get("attempts") != 1024):
        raise ValueError("Early health requires the fixed native panel denominators")
    checks = retention_checks(reference_population, panel, reference_reconstruction, reconstruction, protocol)
    checks.update({"failure_start_" + key: value for key, value in
                   retention_checks(starting_population, panel, starting_reconstruction, reconstruction, protocol).items()})
    checks["failure_start_raw_prior_validity"] = panel["raw_argmax_validity"] >= starting_population["raw_argmax_validity"] - .02
    current_strict = strict_teacher_validity(observation["classification"])
    starting_strict = strict_teacher_validity(starting_classification)
    checks["failure_start_strict_teacher_validity"] = current_strict >= starting_strict - policy["max_strict_validity_drop"]
    return dict(status="passed" if all(checks.values()) else "failed", checks=checks,
                strict_teacher_validity=current_strict, starting_strict_teacher_validity=starting_strict)


def run_early_health_check(plan, run, model, optimizer, config, data, composition, generators, *,
                           completed_epoch, epoch, updates, reference_population, reference_reconstruction,
                           starting_population, starting_reconstruction, baseline):
    """Persist the complete observation before failing and retain the exact failed state."""
    observation, error = None, None
    try:
        observation = observe_early_health(model, data, config, plan["protocol"], composition, generators)
        assessment = assess_early_health(observation, reference_population, reference_reconstruction,
            starting_population, starting_reconstruction, plan["protocol"],
            starting_classification=baseline["observation"]["classification"])
    except Exception as exc:
        error = exc
        assessment = dict(status="failed", checks={"observation_completed": False},
                          error=f"{type(exc).__name__}: {exc}")
    record = dict(version="chemistry-early-health-observation-v1", **assessment, observation=observation,
        plan_sha256=plan["_plan_sha256"], archive_sha256=plan["archive_sha256"],
        manifest_sha256=plan["manifest_sha256"], dataset_sha256=plan["dataset_sha256"],
        split_sha256=plan["split_sha256"], starting_checkpoint_sha256=plan["starting_checkpoint"]["sha256"],
        baseline_sha256=plan["_early_health_baseline_sha256"], epoch=epoch, updates=updates,
        completed_epoch=completed_epoch, job_id=os.environ.get("LSB_JOBID"), created_at=datetime.now(timezone.utc).isoformat(),
        scope="Fixed held-out/prior observation, not a completed epoch or promotable training checkpoint")
    path = run.path / "early_health" / f"update_{updates:09d}.json"
    health_sha = exclusive_json(path, record)
    run.log(dict(event="early_health", epoch=epoch, updates=updates, status=record["status"],
                 checks=record["checks"], path=str(path), sha256=health_sha))
    if record["status"] != "passed":
        snapshot = path.with_suffix(".diagnostic.pt")
        try:
            snapshot_sha = save_failed_early_health_snapshot(snapshot, model, optimizer, config, plan, run,
                completed_epoch=completed_epoch, partial_epoch=epoch, updates=updates,
                generators=generators, health_sha256=health_sha)
            exclusive_json(path.with_suffix(".diagnostic.json"), dict(status="saved_nonpromotable_diagnostic",
                snapshot=str(snapshot), sha256=snapshot_sha, health_record_sha256=health_sha,
                epoch_complete=False, promotable=False))
        except Exception as exc:
            exclusive_json(path.with_suffix(".snapshot-error.json"), dict(status="diagnostic_snapshot_failed",
                error=f"{type(exc).__name__}: {exc}", health_record_sha256=health_sha, promotable=False))
            raise RuntimeError("Early health failed; diagnostic snapshot also failed; do not continue") from exc
        raise RuntimeError("Early fixed-panel chemistry health failed; diagnostic snapshot cannot resume") from error
    return record


def retention_checks(before, after, old_rec, new_rec, p):
    """Engineering health limits, not evidence of chemically useful generation."""
    return {"finite": all(math.isfinite(after[key]) for key in ("expected_size", "element_js", "charge_js", "sulfur_frequency")),
            "size_retention": .9 <= after["expected_size"] / max(before["expected_size"], 1e-8) <= 1.1,
            "nonempty_retention": after["nonempty_fraction"] >= before["nonempty_fraction"] - .02,
            "sulfur_no_inflation": after["sulfur_frequency"] <= before["sulfur_frequency"] * 1.5 + .005,
            "composition_no_deterioration": after["element_js"] <= before["element_js"] + .02,
            "charge_no_deterioration": after["charge_js"] <= before["charge_js"] + .02,
            "reconstruction_retention": new_rec["exact_reconstruction"] >= old_rec["exact_reconstruction"] - .02,
            "valid_reconstruction_retention": new_rec["validity"] >= old_rec["validity"] - .02}


@contextmanager
def phase_lock(campaign, stage):
    """No concurrent chemistry training/final allocations; keep a failed lock."""
    path = Path(campaign) / "active-phase.json"
    with path.open("x") as stream:
        json.dump({"job_id": os.environ.get("LSB_JOBID"), "stage": stage}, stream)
    try:
        yield
    except BaseException:
        raise  # Failed allocation retains its blocker and evidence for explicit recovery.
    else:
        path.unlink()


def validation(model, data, config, count, seed):
    from .evaluate import reconstruction_metrics
    from .sampling import subset_indices
    rows = subset_indices(len(data), count, seed, shuffle_full=True)
    loader = DataLoader(Subset(data, rows), batch_size=32, shuffle=False, collate_fn=collate_graphs,
                        generator=torch.Generator().manual_seed(seed + 1))
    model.eval()
    return reconstruction_metrics(model, loader, config.loss, config.device, len(rows), training_mode="vae")[0]


def composition_reference(data):
    reference = training_composition(data)
    return {"receipt": reference, "elements_distribution": reference["element_counts"],
            "charges_distribution": reference["charge_counts"]}


def nested_equal(left, right):
    """Exact typed checkpoint values, independent of serialization placement.

    Loading a CUDA checkpoint on CPU is deliberate. AdamW restores moments to
    parameter devices, but noncapturable/nonfused scalar ``step`` counters may
    remain on CPU. Compare detached host copies without moving live optimizer
    state, changing dtype, or introducing a numerical tolerance. Model/device
    runtime authentication and the real native next-update replay remain
    separate requirements; this function only compares persisted values.
    """
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
                and left.shape == right.shape and left.dtype == right.dtype
                and left.layout == right.layout
                and torch.equal(left.detach().cpu(), right.detach().cpu()))
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(nested_equal(v, right[k]) for k, v in left.items())
    if isinstance(left, (tuple, list)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(nested_equal(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def native_resume_probe(model, reference, optimizer, data, config, p, composition, teacher, generators, epoch, updates, output,
                        *, overlap=False, phase_start_epoch=None, step_control=None):
    """Disposable GPU updates; exact typed moments/RNG/native next-step parity.

    Probe is not promoted training. Restore the original parent and streams after
    measuring, and independently authenticate its byte checkpoint on setup.
    """
    initial = {"model": deepcopy(model.state_dict()), "optimizer": deepcopy(optimizer.state_dict()),
               "rng": capture_rng_state(generators, True)}
    hard = collate_graphs([data[i] for i in range(4)])
    execution_kwargs = {"overlap": True} if overlap else {}
    if step_control is not None:
        execution_kwargs["step_control"] = step_control
    phase_start = epoch if phase_start_epoch is None else phase_start_epoch
    try:
        chemistry_update(model, reference, optimizer, hard, config, p, composition, teacher, generators,
                         updates, epoch + 1, phase_start, **execution_kwargs)
        saved = {"model": deepcopy(model.state_dict()), "optimizer": deepcopy(optimizer.state_dict()),
                 "rng": capture_rng_state(generators, True)}
        torch.save(saved, output)
        chemistry_update(model, reference, optimizer, hard, config, p, composition, teacher, generators,
                         updates + 1, epoch + 1, phase_start, **execution_kwargs)
        expected = {"model": deepcopy(model.state_dict()), "optimizer": deepcopy(optimizer.state_dict()),
                    "rng": capture_rng_state(generators, True)}
        actual = torch.load(output, map_location="cpu", weights_only=True)
        model.load_state_dict(actual["model"]); optimizer.load_state_dict(actual["optimizer"])
        restore_rng_state(actual["rng"], generators)
        chemistry_update(model, reference, optimizer, hard, config, p, composition, teacher, generators,
                         updates + 1, epoch + 1, phase_start, **execution_kwargs)
        replay = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                  "rng": capture_rng_state(generators, True)}
        checks = {key: nested_equal(expected[key], replay[key]) for key in expected}
        if not all(checks.values()):
            raise ValueError("Native next-step model/AdamW/RNG checkpoint replay failed: "
                             + json.dumps(checks, sort_keys=True))
        result = {"status": "passed", "checks": checks, "real_points": 4, "fresh_prior_points": 4 * p["prior_ratio"],
                "checkpoint_sha256": file_sha256(output),
                "comparison": "exact shape/dtype/layout/values on detached host copies; no live state placement changes",
                "scope": "disposable exact native CUDA next-step replay after CPU checkpoint loading"}
        if step_control is not None:
            result["optimizer_step_control"] = dict(step_control)
        return result
    finally:
        model.load_state_dict(initial["model"]); optimizer.load_state_dict(initial["optimizer"])
        restore_rng_state(initial["rng"], generators)


def benchmark(plan):
    """Discard updates; gate teacher coverage, runtime, memory and retention."""
    campaign = Path(plan["campaign"]); campaign.mkdir(parents=True, exist_ok=True)
    if (campaign / "benchmark_gate.json").exists():
        raise ValueError("Retain prior benchmark outcome; use a new sealed attempt for recovery")
    p = plan["protocol"]
    overlap = bool(execution_settings(plan) and execution_settings(plan).get("overlap", False))
    report = {"status": "failed", "plan_sha256": plan["_plan_sha256"], "version": VERSION,
              "parent_sha256": plan["parent_checkpoint_sha256"]}
    if plan.get("optimizer_step_control") is not None:
        report["optimizer_step_control"] = dict(plan["optimizer_step_control"])
    if plan["protocol"].get("chemistry_target_weighting") is not None:
        report["chemistry_target_weighting"] = dict(plan["protocol"]["chemistry_target_weighting"])
    started = time.monotonic()
    with ExitStack() as resources, ExperimentRun(campaign / "benchmark", "chemistry-benchmark", plan, Path(plan["release"])) as run:
        try:
            parent_proof = {}
            model, reference, optimizer, config, data, cohorts, generators, teacher, epoch, updates, _, _ = setup(
                plan, parent_authentication=parent_proof)
            resources.callback(teacher.close)
            report["parent_checkpoint_authentication"] = parent_proof
            if p.get("valid_prior_reference_guidance") is not None:
                report["valid_prior_reference_guidance"] = dict(p["valid_prior_reference_guidance"])
                report["valid_reference_anchor_before"] = teacher.valid_reference_anchor.assert_unchanged()
            composition = composition_reference(data["train"])
            write_json(campaign / "training_composition.json", composition)
            baseline = population_panel(reference, p["benchmark_panel_count"], p["evaluation_seed"], 32, composition)
            old_rec = validation(reference, data["val"], config, p["validation_count"], p["evaluation_seed"])
            replay = native_resume_probe(model, reference, optimizer, data["train"], config, p, composition,
                                         teacher, generators, epoch, updates, run.path / "disposable_resume_probe.pt", overlap=overlap,
                                         **({"phase_start_epoch": plan["parent_epoch"]} if plan.get("starting_checkpoint") else {}),
                                         **({"step_control": plan["optimizer_step_control"]} if plan.get("optimizer_step_control") else {}))
            execution_parity = None
            if execution_settings(plan) is not None:
                from .chemistry_execution_validation import native_execution_parity
                execution_parity = native_execution_parity(
                    model, reference, optimizer, data["train"], cohorts, config, p, composition,
                    teacher, generators, epoch, updates, run.path / "native_execution_parity.json", overlap=overlap,
                    **({"phase_start_epoch": plan["parent_epoch"]} if plan.get("starting_checkpoint") else {}),
                    **({"step_control": plan["optimizer_step_control"]} if plan.get("optimizer_step_control") is not None else {}))
            # Replay is disposable scientific state. Empty its logical target
            # cache, while retaining warmed CPU workers for actual timing.
            teacher.clear_cache()
            rows = cohorts["train"][:p["benchmark_updates"] * config.batch_size]
            loader = DataLoader(Subset(data["train"], rows), batch_size=config.batch_size,
                                shuffle=False, collate_fn=collate_graphs,
                                generator=torch.Generator().manual_seed(plan["seed"] + 999))
            records = []
            torch.cuda.reset_peak_memory_stats()
            batches = iter(loader)
            for _ in range(len(loader)):
                if time.monotonic() - started > p["benchmark_wall_seconds"]:
                    raise TimeoutError("Benchmark exhausted safe budget; no training promotion")
                if "protocol_config" not in plan:
                    hard = next(batches)
                torch.cuda.synchronize(); tick = time.monotonic()
                if "protocol_config" in plan:
                    hard = next(batches)
                result = chemistry_update(model, reference, optimizer, hard, config, p, composition,
                                          teacher, generators, updates, epoch + 1, plan["parent_epoch"],
                                          deadline=started + p["benchmark_wall_seconds"], overlap=overlap,
                                          **({"step_control": plan["optimizer_step_control"]} if plan.get("optimizer_step_control") is not None else {}))
                if plan.get("optimizer_step_control") is not None:
                    from .chemistry_optimizer_step import validate_optimizer_step_receipt
                    validate_optimizer_step_receipt(result.get("optimizer_step_control"), plan["optimizer_step_control"])
                torch.cuda.synchronize()
                result["update_seconds"] = time.monotonic() - tick
                records.append(result); run.log({"stage": "benchmark_update", "index": len(records), **result})
                updates += 1
            if len(records) != p["benchmark_updates"]:
                raise ValueError("Native benchmark did not observe every declared measured update")
            after = population_panel(model, p["benchmark_panel_count"], p["evaluation_seed"], 32, composition)
            if p.get("valid_prior_reference_guidance") is not None:
                report["valid_reference_anchor_after"] = teacher.valid_reference_anchor.assert_unchanged()
                if report["valid_reference_anchor_after"] != report["valid_reference_anchor_before"]:
                    raise ValueError("Frozen valid-reference state changed during native benchmark")
            new_rec = validation(model, data["val"], config, p["validation_count"], p["evaluation_seed"])
            real = sum(x["real_points"] for x in records); prior = sum(x["fresh_prior_points"] for x in records)
            prior_targets = sum(x["repair_targets"]["prior"] for x in records)
            measured = sum(x["update_seconds"] for x in records)
            # Extrapolate REAL epochs; total-candidate throughput hides prior work.
            epoch_seconds = len(data["train"]) * measured / real + 600.
            vram = torch.cuda.max_memory_reserved()
            gates = retention_checks(baseline, after, old_rec, new_rec, p)
            gates.update(exact_ratio=prior == p["prior_ratio"] * real,
                         exact_native_resume=replay["status"] == "passed",
                         prior_target_coverage=prior_targets / prior >= p["minimum_prior_target_fraction"],
                         actual_repair_gradient=any((x["repair_only_decoder_gradient_norm"] or 0.) > 0 for x in records),
                         decoder_gradient=all(x["group_gradient_norms"]["decoder"] > 0 for x in records),
                         encoder_gradient=all(x["group_gradient_norms"]["encoder"] > 0 for x in records),
                         memory=vram < p["vram_fraction_limit"] * torch.cuda.get_device_properties(0).total_memory,
                         complete_epoch_fits=epoch_fit_seconds(plan, epoch_seconds) < p["segment_wall_seconds"],
                         no_search_errors=all(not x["repair_statuses"][branch].get("rejected_or_error", 0)
                                             for x in records for branch in ("real", "prior")))
            if execution_settings(plan) is not None:
                gates["exact_reference_execution"] = (execution_parity["passed"]
                    and execution_parity["restoration"]["passed"]
                    and all(execution_parity["checks"].values()))
            report.update(status="passed" if all(gates.values()) else "failed", gates=gates, updates=records,
                          protocol_version=p["version"], composition_prior_weight=composition_prior_weight(p),
                          parent_population=baseline, updated_population=after, parent_reconstruction=old_rec,
                          updated_reconstruction=new_rec, projected_epoch_seconds=epoch_seconds,
                          real_molecules_per_second=real / measured, all_candidates_per_second=(real + prior) / measured,
                          prior_target_fraction=prior_targets / prior, peak_reserved_vram_bytes=vram,
                          peak_host_resident_bytes=peak_host_memory_bytes(), run_dir=str(run.path),
                          composition_sha256=file_sha256(campaign / "training_composition.json"),
                          native_resume_probe=replay,
                          native_execution_parity=execution_parity,
                          execution=execution_settings(plan),
                          update_timing_scope=("synchronized wall including batch fetch/collation, teacher and full model update; setup and evaluation excluded"
                              if "protocol_config" in plan else "legacy synchronized model/teacher update wall; batch fetch/collation excluded"),
                          benchmark_weights_discarded=True, elapsed_seconds=time.monotonic() - started)
            write_json(run.path / "gate.json", report)
            write_json(campaign / "benchmark_gate.json", report)
            run.annotate(gate_status=report["status"], gates=gates)
            if report["status"] != "passed":
                raise RuntimeError("Chemistry benchmark failed explicit promotion gates")
        except Exception as exc:
            report.update(status="failed", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.monotonic() - started,
                          run_dir=str(run.path))
            if isinstance(exc, ParentCheckpointMismatch):
                report["parent_checkpoint_authentication"] = {"passed": False, "differences": exc.differences,
                                                              "actual": exc.actual, "expected": exc.expected}
            write_json(campaign / "benchmark_gate.json", report)
            raise


def phase_checkpoint(path, model, optimizer, config, data, epoch, run, state, plan):
    """Atomic checkpoint with distinct phase identity and all new RNG streams."""
    from .chemistry_short_continuation import require_training_branch
    require_training_branch(plan)
    require_failure_readiness(plan)
    if plan.get("optimizer_step_control") is not None:
        from .chemistry_optimizer_step import authenticate_optimizer_step_evidence
        epoch_updates = math.ceil(plan["training_molecules"] / 256)
        authenticate_optimizer_step_evidence(state.get("optimizer_step_evidence"), plan["optimizer_step_control"],
            root=run.path, epoch=epoch, start_update=state["updates"] - epoch_updates, end_update=state["updates"])
    save_checkpoint(path, model, optimizer, config, data, epoch, run, state)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["chemistry_phase"] = {"version": VERSION, "plan_sha256": plan["_plan_sha256"],
                                  "parent_sha256": plan["parent_checkpoint_sha256"],
                                  "benchmark_gate_sha256": plan["_benchmark_gate_sha256"],
                                  "execution": plan.get("execution"),
                                  "parent_epoch": plan["parent_epoch"], "protocol": plan["protocol"]}
    if plan.get("classification_execution") is not None:
        payload["chemistry_phase"]["classification_execution"] = dict(plan["classification_execution"])
    if plan.get("worker_startup") is not None:
        payload["chemistry_phase"]["worker_startup"] = dict(plan["worker_startup"])
    if (plan["protocol"].get("failure_guidance") is not None or plan.get("short_continuation") is not None
            or plan.get("valid_reference_continuation") is not None):
        payload["chemistry_phase"]["failure_readiness_sha256"] = plan["_failure_readiness_sha256"]
    if plan.get("short_continuation") is not None:
        payload["chemistry_phase"]["short_continuation"] = dict(plan["short_continuation"])
    if plan.get("valid_reference_continuation") is not None:
        payload["chemistry_phase"]["valid_reference_continuation"] = deepcopy(plan["valid_reference_continuation"])
        payload["chemistry_phase"]["valid_reference_adoption"] = deepcopy(plan["valid_reference_adoption"])
    if "epoch45_extension" in plan:
        payload["chemistry_phase"]["epoch45_extension"] = deepcopy(plan["epoch45_extension"])
    if plan.get("optimizer_step_control") is not None:
        payload["chemistry_phase"]["optimizer_step_control"] = dict(plan["optimizer_step_control"])
    if "runtime_policy_amendment" in plan:
        epoch_fit_policy(plan)
        payload["chemistry_phase"]["runtime_policy_amendment"] = dict(plan["runtime_policy_amendment"])
    if validate_early_health_policy(plan["protocol"]) is not None:
        payload["chemistry_phase"]["early_health_baseline_sha256"] = plan["_early_health_baseline_sha256"]
    temporary = path.with_suffix(".phase.tmp")
    torch.save(payload, temporary); temporary.replace(path)


def train_phase(plan, *, resume=None, segment=1):
    """Finish whole epochs inside allocation, serially resume toward absolute100."""
    campaign = Path(plan["campaign"]); p = plan["protocol"]
    from .chemistry_short_continuation import require_training_branch
    require_training_branch(plan)
    if (plan.get("starting_checkpoint") is not None and p.get("failure_guidance") is None
            and plan.get("short_continuation") is None and plan.get("valid_reference_continuation") is None):
        raise ValueError("Legacy handoff control is benchmark-only, not a training successor")
    failure_readiness = require_failure_readiness(plan)
    endpoint = _continuation_endpoint(plan)
    early_policy = validate_early_health_policy(p)
    early_baseline = (require_early_health_baseline(plan, plan["_early_health_baseline_sha256"])
                      if early_policy is not None else None)
    authenticated(campaign / "benchmark_gate.json", plan["_benchmark_gate_sha256"])
    gate = json.loads((campaign / "benchmark_gate.json").read_text())
    if gate.get("status") != "passed" or gate.get("plan_sha256") != plan["_plan_sha256"] or not all(gate["gates"].values()):
        raise ValueError("Native benchmark has not passed for this exact phase")
    authenticated(campaign / "training_composition.json", gate["composition_sha256"])
    composition = json.loads((campaign / "training_composition.json").read_text())
    start = time.monotonic()
    if not 1 <= segment <= p["maximum_segments"]:
        raise ValueError("Serial segment cap exceeded")
    with phase_lock(campaign, f"segment-{segment:03d}"), ExitStack() as resources, ExperimentRun(campaign / "segments" / f"{segment:03d}", "chemistry-train", plan, Path(plan["release"])) as run:
        model, reference, optimizer, config, data, cohorts, generators, teacher, completed, updates, algorithm, runtime = setup(plan, resume=resume)
        resources.callback(teacher.close)
        if early_baseline is not None and resume is None:
            if (completed != plan["starting_checkpoint"]["epoch"]
                    or updates != plan["starting_checkpoint"]["updates"]
                    or state_digest(model.state_dict()) != early_baseline["model_sha256"]):
                raise ValueError("Early-health baseline model differs from the authenticated handoff")
        overlap = bool(execution_settings(plan) and execution_settings(plan).get("overlap", False))
        if completed >= endpoint:
            raise ValueError("Checkpoint is already at the planned horizon")
        if plan.get("worker_startup") is not None:
            from .chemistry_worker_startup import warm_repair_workers
            startup_path = run.path / "worker-startup.json"
            startup = warm_repair_workers(plan, teacher, model, reference, optimizer,
                data["train"], cohorts, generators, startup_path)
            run.annotate(worker_startup_sha256=file_sha256(startup_path),
                         worker_startup_seconds=startup["elapsed_seconds"],
                         worker_startup_timing_scope="One-time allocation setup; included in allocation clock, not repeated per update")
        identity = compact_cohort_identity(cohorts)
        loader = DataLoader(Subset(data["train"], cohorts["train"]), batch_size=config.batch_size,
                            shuffle=True, generator=generators["train"], collate_fn=collate_graphs, num_workers=0)
        parent_panel = population_panel(reference, p["monitor_prior_count"], p["evaluation_seed"], 32, composition)
        parent_rec = validation(reference, data["val"], config, p["validation_count"], p["evaluation_seed"])
        estimate = gate["projected_epoch_seconds"]
        if "guarded_restart" in plan:
            from .chemistry_restart_policy import authenticate_stability
            # The longer disposable trajectory measures the new observer cost;
            # a short benchmark cannot understate the first real epoch budget.
            estimate = max(estimate, authenticate_stability(plan)["conservative_epoch_seconds"])
        checkpoint = None
        run.annotate(parent_epoch=plan["parent_epoch"], planned_epochs=100, joint_encoder_decoder=True,
                     execution=execution_settings(plan),
                     protocol=plan["protocol"], dataset_sha256=plan["dataset_sha256"], split_sha256=plan["split_sha256"],
                     inherited_optimizer="parent AdamW moments", generation_reference=parent_panel,
                     reconstruction_reference=parent_rec, starting_epoch=completed,
                     resume_checkpoint_sha256=(plan.get("starting_checkpoint", {}).get("sha256", plan["parent_checkpoint_sha256"])
                                               if resume is None else resume[1]),
                     benchmark_gate_sha256=plan["_benchmark_gate_sha256"])
        if plan.get("short_continuation") is not None:
            run.annotate(short_continuation=plan["short_continuation"], branch_endpoint_epoch=endpoint)
        if plan.get("valid_reference_continuation") is not None:
            run.annotate(valid_reference_continuation=plan["valid_reference_continuation"],
                         valid_reference_adoption=plan["valid_reference_adoption"], branch_endpoint_epoch=endpoint)
        if plan.get("optimizer_step_control") is not None:
            run.annotate(optimizer_step_control=plan["optimizer_step_control"])
        if early_baseline is not None:
            run.annotate(early_health_policy=early_policy,
                         early_health_baseline_sha256=plan["_early_health_baseline_sha256"])
        def observe_training_health(epoch, updates):
            return run_early_health_check(plan, run, model, optimizer, config, data, composition, generators,
                completed_epoch=completed, epoch=epoch, updates=updates,
                reference_population=parent_panel, reference_reconstruction=parent_rec,
                starting_population=failure_readiness["native_panel"]["starting_retention_population"],
                starting_reconstruction=failure_readiness["native_panel"]["starting_reconstruction"],
                baseline=early_baseline)
        stop = ("bounded_horizon" if plan.get("short_continuation") is not None
                or plan.get("valid_reference_continuation") is not None else "planned_horizon")
        for epoch in range(completed + 1, endpoint + 1):
            if time.monotonic() - start + epoch_fit_seconds(plan, estimate) > p["segment_wall_seconds"]:
                if checkpoint is None:
                    raise RuntimeError("Allocation cannot safely fit a complete chemistry epoch")
                stop = "wall_budget_epoch_boundary"; break
            tick = time.monotonic(); total_real = total_prior = 0; records = []
            statuses = {"real": {}, "prior": {}}; epoch_receipt_chain = "0" * 64; search_checks = 0
            failure_totals = {"real": {}, "prior": {}}
            epoch_classifications = {}
            step_control = plan.get("optimizer_step_control")
            step_evidence = None
            if step_control is not None:
                from .chemistry_optimizer_step import (optimizer_step_chain, validate_optimizer_step_receipt,
                    authenticate_optimizer_step_evidence)
                step_path = run.path / f"optimizer-steps-epoch-{epoch:03d}.jsonl"
                step_stream = resources.enter_context(step_path.open("x"))
                step_start, step_chain, step_count = updates, "0" * 64, 0
            for group in optimizer.param_groups:
                group["lr"] = learning_rate_at_epoch(config, epoch)
            for hard in loader:
                metrics = chemistry_update(model, reference, optimizer, hard, config, p, composition,
                                           teacher, generators, updates, epoch, plan["parent_epoch"],
                                           deadline=start + p["segment_wall_seconds"] - 900, overlap=overlap,
                                           **({"step_control": step_control} if step_control is not None else {}))
                total_real += metrics["real_points"]; total_prior += metrics["fresh_prior_points"]
                updates += 1
                if step_control is not None:
                    receipt = validate_optimizer_step_receipt(metrics.get("optimizer_step_control"), step_control)
                    step_record = dict(epoch=epoch, update=updates, receipt=receipt)
                    step_stream.write(json.dumps(step_record, allow_nan=False) + "\n")
                    step_stream.flush()
                    step_chain = optimizer_step_chain(step_chain, step_record)
                    step_count += 1
                epoch_receipt_chain = digest({"previous": epoch_receipt_chain,
                                              "update_receipt": metrics["ordered_teacher_receipt_chain_sha256"]})
                search_checks += metrics["search_checks"]
                for branch in statuses:
                    for key, value in metrics["repair_statuses"][branch].items():
                        statuses[branch][key] = statuses[branch].get(key, 0) + value
                    for key, value in metrics.get("failure_guidance_coverage", {}).get(branch, {}).items():
                        failure_totals[branch][key] = failure_totals[branch].get(key, 0) + value
                    if branch in metrics.get("teacher_classification", {}):
                        from .chemistry_classification import merge_classification_summaries
                        previous = [epoch_classifications[branch]] if branch in epoch_classifications else []
                        # Count EVERY scheduled row while keeping example payloads
                        # in sampled update/early-panel artifacts, not epoch RAM.
                        epoch_classifications[branch] = merge_classification_summaries(
                            previous + [metrics["teacher_classification"][branch]], max_examples_per_failure=0)
                # Persist scalar diagnostics, not millions of witness graphs each epoch.
                records.append({key: metrics[key] for key in ("total", "reconstruction", "kl", "repair_real", "repair_prior",
                                                              "element_js", "charge_js", "occupancy_anchor", "search_seconds",
                                                              "pre_clip_gradient_norm", "clipped", "repair_targets", "repair_eligible")})
                if updates % 100 == 0:
                    run.log({"epoch": epoch, "update": updates, **metrics})
                if early_health_due(early_policy, updates, plan.get("starting_checkpoint", {}).get("updates", 0)):
                    observe_training_health(epoch, updates)
                if time.monotonic() - start > p["segment_wall_seconds"] - 900:
                    raise TimeoutError("Unexpected epoch overrun; incomplete epoch is never promoted")
            if early_policy is not None and not early_health_due(early_policy, updates, plan["starting_checkpoint"]["updates"]):
                # Also protect the last <interval updates before an ordinary epoch checkpoint.
                observe_training_health(epoch, updates)
            if total_real != len(data["train"]) or total_prior != p["prior_ratio"] * total_real:
                raise ValueError("Whole-dataset or exact declared real/prior final-tail count differs")
            if step_control is not None:
                step_stream.close()
                step_evidence = dict(path=str(step_path), sha256=file_sha256(step_path), updates=step_count,
                    start_update=step_start, end_update=updates, chain_sha256=step_chain)
                authenticate_optimizer_step_evidence(step_evidence, step_control,
                    root=run.path, epoch=epoch, start_update=step_start, end_update=updates)
            reconstruction = validation(model, data["val"], config, p["validation_count"], p["evaluation_seed"])
            # Same coordinates at each completed epoch: stop on composition/size/reconstruction failure.
            panel = population_panel(model, p["monitor_prior_count"], p["evaluation_seed"], 32, composition)
            checks = retention_checks(parent_panel, panel, parent_rec, reconstruction, p)
            if failure_readiness is not None:
                # The authenticated handoff-start baseline stays fixed across
                # every serial segment. Keep the original source121 checks
                # AND protect gains achieved before this objective change;
                # never ratchet the baseline to a preceding chemistry epoch.
                start_panel = failure_readiness["native_panel"]["starting_retention_population"]
                start_rec = failure_readiness["native_panel"]["starting_reconstruction"]
                checks.update({"failure_start_" + key: value for key, value in
                    retention_checks(start_panel, panel, start_rec, reconstruction, p).items()})
                checks["failure_start_raw_prior_validity"] = (
                    panel["raw_argmax_validity"] >= start_panel["raw_argmax_validity"] - .02)
            checks["no_search_errors"] = all(not status.get("rejected_or_error", 0) for status in statuses.values())
            # More already-valid graphs are success, not a false drop in guidance coverage.
            prior_guided_or_valid = statuses["prior"].get("accepted", 0) + statuses["prior"].get("skipped_already_valid", 0)
            checks["prior_guidance_or_validity"] = prior_guided_or_valid / total_prior >= p["minimum_prior_target_fraction"]
            if not all(checks.values()):
                write_json(run.path / "failed_epoch_health.json", {"epoch": epoch, "checks": checks,
                           "population": panel, "reconstruction": reconstruction})
                raise RuntimeError("Completed epoch failed chemistry retention checks; do not continue")
            elapsed = time.monotonic() - tick; estimate = max(estimate, elapsed)
            continuation = {"version": 1, "epoch_complete": True, "updates": updates, "algorithm": algorithm,
                            "runtime": runtime, "indices": identity, "split_sha256": plan["split_sha256"],
                            "rng": capture_rng_state(generators, True), "scheduler_state": None,
                            "selections": {}, "phase_epoch_wall_seconds": elapsed,
                            "timing_scope": "this completed chemistry epoch including health evaluation; not cumulative parent training"}
            if step_control is not None:
                continuation["optimizer_step_evidence"] = step_evidence
            checkpoint = run.path / f"epoch_{epoch:04d}.pt"
            phase_checkpoint(checkpoint, model, optimizer, config, data["train"], epoch, run, continuation, plan)
            completed = epoch
            row = {"epoch": epoch, "updates": updates, "real_points": total_real, "fresh_prior_points": total_prior,
                   "population": panel, "validation": reconstruction, "health": checks, "epoch_seconds": elapsed,
                   "means": {key: sum(r[key] for r in records) / len(records) for key in
                             ("total", "reconstruction", "kl", "repair_real", "repair_prior", "element_js", "charge_js", "occupancy_anchor")},
                   "search_seconds": sum(r["search_seconds"] for r in records),
                   "search_checks": search_checks, "repair_status_counts": statuses,
                   "ordered_teacher_receipt_chain_sha256": epoch_receipt_chain,
                   "clipped_updates": sum(r["clipped"] for r in records),
                   "repair_target_counts": {branch: sum(r["repair_targets"][branch] for r in records) for branch in ("real", "prior")},
                   "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint)}
            if step_control is not None:
                row["optimizer_step_control"] = dict(step_control)
                row["optimizer_step_evidence"] = step_evidence
            if p.get("failure_guidance") is not None:
                row["failure_guidance_coverage"] = failure_totals
                row["failure_guidance_policy"] = p["failure_guidance"]
            if epoch_classifications:
                row["teacher_classification"] = epoch_classifications
            write_json(run.path / "latest_epoch.json", row); run.log(row)
            if epoch % p["monitor_every"] == 0 or epoch == endpoint:
                panel_protocol = {"evaluation_seed": p["evaluation_seed"], "evaluation_batch_size": 32,
                                  "prior_count": 4096, "categorical_samples_per_latent": 4,
                                  "local_geometry_count": 128, "local_geometry_scales": [0., .05, .1, .2],
                                  "interpolation_pairs": 32, "interpolation_alphas": [0., .25, .5, .75, 1.]}
                protected_rng = capture_rng_state(generators, True)
                deep = evaluate_panel(model, data["val"], panel_protocol, run.path / "monitor" / f"epoch{epoch:04d}",
                                      limit=4096, deep=True, loss=config.loss)
                write_json(run.path / "monitor" / f"epoch{epoch:04d}" / "metrics.json", deep)
                restore_rng_state(protected_rng, generators)
                estimate = max(estimate, time.monotonic() - tick)
            # Retain milestones plus two most recent OWN snapshots; ancestry and baseline untouched.
            snapshots = sorted(run.path.glob("epoch_*.pt"))
            if "epoch45_extension" in plan:
                snapshots = []  # Explicit extension keeps every completed epoch.
            for old in snapshots[:-2]:
                if int(old.stem.split("_")[1]) % 10 and old != checkpoint:
                    old.unlink()
        completion = {"status": "passed", "plan_sha256": plan["_plan_sha256"], "segment": segment,
                      "completed_epoch": completed, "stop_reason": stop, "checkpoint": str(checkpoint),
                      "checkpoint_sha256": file_sha256(checkpoint), "epoch_complete": True,
                      "epoch_seconds": estimate, "updates": updates, "run_dir": str(run.path)}
        run.annotate(completed_epoch=completed, stop_reason=stop, checkpoint=completion["checkpoint"])
        write_json(run.path / "completion.json", completion)
        write_json(campaign / f"segment-{segment:03d}-completion.json", completion)


def final_evaluation(plan, checkpoint, sha):
    """Reserved test access only after absolute epoch100; freeze before latent use."""
    require_failure_readiness(plan)
    # setup constructs a lazy teacher even though evaluation uses no repairs;
    # own its lifetime here as well, including failures before the final run.
    with ExitStack() as resources:
        return _final_evaluation(plan, checkpoint, sha, resources)


def _final_evaluation(plan, checkpoint, sha, resources):
    model, reference, _, config, data, _, _, teacher, epoch, _, _, _ = setup(plan, resume=(checkpoint, sha))
    resources.callback(teacher.close)
    if epoch != 100:
        raise ValueError("Final test remains reserved until absolute epoch100")
    p = plan["protocol"]
    authenticated(Path(plan["campaign"]) / "benchmark_gate.json", plan["_benchmark_gate_sha256"])
    protocol = {"evaluation_seed": p["evaluation_seed"], "evaluation_batch_size": 32, "prior_count": 4096,
                "categorical_samples_per_latent": 4, "local_geometry_count": 128,
                "local_geometry_scales": [0., .05, .1, .2], "interpolation_pairs": 32,
                "interpolation_alphas": [0., .25, .5, .75, 1.]}
    dataset = GraphDataset(plan["data"], "test", validated_cache=validate_large_cache(Path(plan["data"])))
    model.eval().requires_grad_(False); reference.eval().requires_grad_(False)
    with phase_lock(plan["campaign"], "final"), ExperimentRun(Path(plan["campaign"]) / "final", "chemistry-final", plan, Path(plan["release"])) as run:
        for label, candidate in (("parent", reference), ("chemistry_epoch100", model)):
            report = evaluate_panel(candidate, dataset, protocol, run.path / label, limit=len(dataset), deep=True, loss=config.loss)
            write_json(run.path / (label + ".json"), report)
        write_json(Path(plan["campaign"]) / "final_completion.json", {"status": "passed", "epoch": epoch,
                   "plan_sha256": plan["_plan_sha256"], "run_dir": str(run.path), "checkpoint_sha256": sha})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("benchmark", "train", "final"))
    parser.add_argument("--plan", type=Path, required=True); parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--resume", type=Path); parser.add_argument("--resume-sha256"); parser.add_argument("--segment", type=int, default=1)
    parser.add_argument("--benchmark-gate-sha256")
    parser.add_argument("--failure-readiness-sha256")
    args = parser.parse_args(); plan = read_plan(args.plan, args.plan_sha256); plan["_plan_sha256"] = args.plan_sha256
    if (args.resume is None) != (args.resume_sha256 is None):
        parser.error("Resume path and SHA are required together")
    resume = None if args.resume is None else (args.resume, args.resume_sha256)
    if args.stage != "benchmark":
        if args.benchmark_gate_sha256 is None:
            parser.error("Train/final require the passed immutable benchmark gate SHA")
        plan["_benchmark_gate_sha256"] = args.benchmark_gate_sha256
        if (plan["protocol"].get("failure_guidance") is not None
                or plan.get("short_continuation") is not None
                or plan.get("valid_reference_continuation") is not None):
            if args.failure_readiness_sha256 is None:
                parser.error("New chemistry train/final require the passed failure-readiness SHA")
            plan["_failure_readiness_sha256"] = args.failure_readiness_sha256
        elif args.failure_readiness_sha256 is not None:
            parser.error("Historical chemistry protocols have no failure-readiness override")
    elif args.failure_readiness_sha256 is not None:
        parser.error("Failure readiness is produced by a benchmark, not supplied to it")
    if args.stage == "benchmark":
        benchmark(plan)
    elif args.stage == "train":
        train_phase(plan, resume=resume, segment=args.segment)
    else:
        if resume is None:
            parser.error("Final evaluation requires a completed checkpoint")
        final_evaluation(plan, *resume)


if __name__ == "__main__":
    main()
