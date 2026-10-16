"""Prepared-context budget changes must not silently change scientific inputs."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


def _example():
    path = (
        Path(__file__).resolve().parents[1] / "examples/protenix_relaxed_gradients.py"
    )
    spec = importlib.util.spec_from_file_location("prepared_study_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configs():
    root = Path(__file__).resolve().parents[1] / "examples/configs"
    previous = json.loads((root / "carbon_chain_100.json").read_text())
    current = json.loads((root / "carbon_chain_500.json").read_text())
    return current, previous


def test_longer_budget_requires_explicit_policy_and_preserves_inputs():
    current, previous = _configs()
    original = deepcopy((current, previous))
    validate = _example()._validate_prepared_config
    with pytest.raises(ValueError, match="only matmul_precision"):
        validate(current, previous)
    validate(current, previous, allow_update_count_change=True)
    assert (current, previous) == original
    different_precision = deepcopy(previous)
    different_precision["matmul_precision"] = "default"
    validate(different_precision, previous)


def test_new_mixture_requires_separate_explicit_policy():
    current, previous = _configs()
    current["mixing_values"] = [0.5]
    validate = _example()._validate_prepared_config
    with pytest.raises(ValueError, match="unapproved"):
        validate(current, previous, allow_update_count_change=True)
    with pytest.raises(ValueError, match="explicit longer-budget"):
        validate(current, previous, allow_mixing_values_change=True)
    validate(
        current,
        previous,
        allow_update_count_change=True,
        allow_mixing_values_change=True,
    )


def test_mixture_policy_still_rejects_precision_changes():
    current, previous = _configs()
    current["mixing_values"] = [0.5]
    current["matmul_precision"] = "default"
    with pytest.raises(ValueError, match="unapproved"):
        _example()._validate_prepared_config(
            current,
            previous,
            allow_update_count_change=True,
            allow_mixing_values_change=True,
        )


@pytest.mark.parametrize("change", ["seed", "precision", "pocket", "eta", "not_longer"])
def test_longer_budget_rejects_unmatched_scientific_or_budget_settings(change):
    current, previous = _configs()
    if change == "seed":
        current["seed"] = 23
    elif change == "precision":
        current["matmul_precision"] = "default"
    elif change == "pocket":
        current["pocket_protein_ordinals"] = [0]
    elif change == "eta":
        current["checks"]["initial_step_size"] = 0.02
    else:
        current["checks"]["steps"] = 100
    with pytest.raises(ValueError, match="Update-count replay"):
        _example()._validate_prepared_config(
            current, previous, allow_update_count_change=True
        )
