"""Metadata-only full-search protocol; safe to authenticate before LSF submission.

This deliberately does not loosen the historical four-arm pilot reader. The
full comparison has two arms per training seed, with common search budgets and
anchors. Model/data execution and detailed loss validation remain on HPC.
"""
import json
import math
from pathlib import Path
import re

from latentmol.experiments import file_sha256


FIELDS = {"version", "initializer", "initializer_sha256", "data", "dataset_sha256", "split_sha256",
          "fine_tune", "seeds", "arms", "train_limit", "validation_limit", "validation_prior_count",
          "final_prior_count", "final_repair_limit", "training_prior_count", "training_prior_batch",
          "repair", "case_budget_seconds", "controller_budget_seconds", "evaluation_reserve_seconds",
          "smoke_budget_seconds", "preceding_job"}


def read_full_protocol(path, expected_sha):
    """Fail closed on altered data identity, partial budgets or unsupported arms."""
    path = Path(path)
    if file_sha256(path) != expected_sha:
        raise ValueError("Full search protocol hash mismatch")
    value = json.loads(path.read_text())
    return validate_full_protocol(value)


def validate_full_protocol(value):
    """Validate decoded metadata too, including saved per-seed protocol receipts."""
    if set(value) != FIELDS or type(value["version"]) is not int or value["version"] != 1:
        raise ValueError("Unknown full search protocol fields/version")
    for name in ("initializer_sha256", "dataset_sha256", "split_sha256"):
        if not isinstance(value[name], str) or not re.fullmatch(r"[0-9a-f]{64}", value[name]):
            raise ValueError("Full search requires declared SHA256 identities")
    for name in ("initializer", "data"):
        if not isinstance(value[name], str) or not value[name].startswith("/work3/"):
            raise ValueError("Full search assets must be declared DTU work3 paths")
    if value["seeds"] != [17, 23] or any(type(seed) is not int for seed in value["seeds"]):
        raise ValueError("Full comparison prespecifies training seeds17/23")
    for name in ("train_limit", "validation_limit", "validation_prior_count", "final_prior_count", "final_repair_limit",
                 "training_prior_count", "training_prior_batch", "case_budget_seconds", "controller_budget_seconds",
                 "evaluation_reserve_seconds", "smoke_budget_seconds"):
        if type(value[name]) is not int or value[name] < 1:
            raise ValueError("Invalid full protocol count/budget: " + name)
    if value["train_limit"] != 215000 or not isinstance(value["fine_tune"], dict) or value["fine_tune"].get("epochs") != 10:
        raise ValueError("Full comparison requires215000 graphs and10 epochs")
    if type(value["fine_tune"]["epochs"]) is not int:
        raise ValueError("Epoch count must be an integer")
    if value["training_prior_batch"] > value["training_prior_count"]:
        raise ValueError("Search minibatch exceeds fixed training pool")
    if value["final_repair_limit"] > value["final_prior_count"]:
        raise ValueError("Final repair audit exceeds the final panel")
    if value["controller_budget_seconds"] > 21 * 3600:
        raise ValueError("Keep two-hour LSF headroom inside23h allocation")
    if 2 * (value["case_budget_seconds"] + value["evaluation_reserve_seconds"] + value["smoke_budget_seconds"]) > value["controller_budget_seconds"]:
        raise ValueError("Segment cannot reserve both arms, evaluations and smoke")
    expected = [{"name": "control", "defect_weight": 0.0, "repair_weight": 0.0},
                {"name": "search", "defect_weight": 0.0, "repair_weight": 1.0}]
    if value["arms"] != expected:
        raise ValueError("Full comparison requires matched control versus search weight1")
    for arm in value["arms"]:
        for name in ("defect_weight", "repair_weight"):
            if isinstance(arm[name], bool) or not isinstance(arm[name], (int, float)) or not math.isfinite(arm[name]):
                raise ValueError("Arm weights require finite numeric values")
    if not isinstance(value["repair"], dict):
        raise ValueError("Repair policy must be an explicit mapping")
    if not isinstance(value["preceding_job"], str) or not value["preceding_job"].isdigit():
        raise ValueError("Preceding serial experiment must be an LSF job ID")
    return value


def segment_protocol(protocol, segment):
    """Separate full reader selects one seed; shared engine receives one pair."""
    if type(segment) is not int or segment not in (0, 1):
        raise ValueError("Only two declared seed segments")
    return {**protocol, "seed": protocol["seeds"][segment]}
