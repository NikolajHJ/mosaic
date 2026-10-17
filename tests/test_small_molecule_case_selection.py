"""Allocated CPU checks for independent scheduling of sealed chain cases."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


def runner():
    path = Path(__file__).resolve().parents[1] / "examples/continue_brd4.py"
    spec = importlib.util.spec_from_file_location("continue_brd4", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("index", [0, 1, 2])
def test_selected_case_retains_recipe_and_resume_entry_without_mutating_plan(index):
    config = {
        "mixing_values": [0.15, 0.3, 0.5],
        "seed": 17,
        "checks": {
            "steps": 5000,
            "initial_step_size": 0.03,
            "acceptance_policy": "best_primal",
        },
    }
    before = deepcopy(config)
    entries = [{"last_step": 4998, "files": {"snapshot": "pinned"}}, None, None]
    selected, selected_entries, source_indices = runner()._select_cases(
        config, entries, index
    )
    assert selected["mixing_values"] == [before["mixing_values"][index]]
    assert selected["checks"] == before["checks"] and selected["seed"] == 17
    assert selected_entries == [entries[index]] and source_indices == [index]
    assert config == before
    selected["checks"]["steps"] = 1
    assert config == before


@pytest.mark.parametrize("index", [-1, 3, True])
def test_missing_or_invalid_case_is_rejected(index):
    with pytest.raises(ValueError):
        runner()._select_cases(
            {"mixing_values": [0.15, 0.3, 0.5]}, [{}, None, None], index
        )


def test_no_selection_preserves_full_plan_and_singleton_cannot_use_chain_index():
    config = {"mixing_values": [0.15, 0.3, 0.5]}
    assert runner()._select_cases(config, [{}, None, None]) == (
        config,
        [{}, None, None],
        [0, 1, 2],
    )
    with pytest.raises(ValueError):
        runner()._select_cases({"initialization": {"kind": "uniform"}}, [{}], 0)
