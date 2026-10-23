"""Metadata-only contract for the user-requested 1:1 chemistry coverage study.

No model imports: safe for source/config authentication before submission.
Historical integer-prior protocols and checkpoints are not rewritten.
"""
import json
from pathlib import Path
import re

from latentmol.experiments import file_sha256

FIELDS = {"version", "initializer", "initializer_sha256", "data", "dataset_sha256", "split_sha256",
          "fine_tune", "seeds", "coefficients", "train_limit", "validation_limit", "validation_prior_count",
          "final_prior_count", "final_repair_limit", "case_budget_seconds", "evaluation_reserve_seconds",
          "gate_budget_seconds", "historical_root", "historical_archive_sha256"}


def read_protocol(path, expected_sha):
    path = Path(path)
    if file_sha256(path) != expected_sha:
        raise ValueError("Density protocol hash mismatch")
    return validate_protocol(json.loads(path.read_text()))


def validate_protocol(value):
    """Pin counts, loss grid, lineage and a 23-hour allocation's headroom."""
    if set(value) != FIELDS or type(value["version"]) is not int or value["version"] != 1:
        raise ValueError("Unknown density protocol fields/version")
    for key in ("initializer_sha256", "dataset_sha256", "split_sha256", "historical_archive_sha256"):
        if not isinstance(value[key], str) or not re.fullmatch(r"[0-9a-f]{64}", value[key]):
            raise ValueError("Density study requires exact SHA256 lineage")
    for key in ("initializer", "data", "historical_root"):
        if not isinstance(value[key], str) or not value[key].startswith("/work3/"):
            raise ValueError("Density assets must be declared DTU work3 paths")
    if value["seeds"] != [17, 23] or any(type(seed) is not int for seed in value["seeds"]):
        raise ValueError("Prescribed training seeds are17/23")
    if value["coefficients"] != [0.0, 1.0] or any(type(v) not in (int, float) for v in value["coefficients"]):
        raise ValueError("Require matched control0 versus chemistry1")
    for key in ("train_limit", "validation_limit", "validation_prior_count", "final_prior_count", "final_repair_limit",
                "case_budget_seconds", "evaluation_reserve_seconds", "gate_budget_seconds"):
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError("Invalid density quota/budget: " + key)
    fine = value["fine_tune"]
    if (not isinstance(fine, dict) or fine.get("epochs") != 10 or fine.get("batch_size") != 64
            or fine.get("chemistry_reconstruction_batch") != 64 or "prior_batch_size" not in fine
            or fine["prior_batch_size"] is not None or value["train_limit"] != 215000):
        raise ValueError("Require full215k,10epochs and all-real chemistry with exact1:1 prior coverage")
    if any(type(fine[key]) is not int for key in ("epochs", "batch_size", "chemistry_reconstruction_batch")):
        raise ValueError("Density epoch and batch counts must be exact integers")
    if value["final_repair_limit"] > value["final_prior_count"]:
        raise ValueError("Repair audit exceeds final panel")
    if value["case_budget_seconds"] + value["evaluation_reserve_seconds"] > 21 * 3600:
        raise ValueError("Preserve two-hour scheduler headroom")
    if value["gate_budget_seconds"] > 90 * 60:
        raise ValueError("Keep gate/evaluation inside its two-hour allocation")
    return value


def cases(protocol):
    """One full case per serial allocation; no partial endpoint ranking."""
    return [{"index": index, "seed": seed, "coefficient": coefficient,
             "label": f"seed{seed}-a{coefficient:g}"}
            for index, (seed, coefficient) in enumerate(
                (seed, coefficient) for seed in protocol["seeds"] for coefficient in protocol["coefficients"])]
