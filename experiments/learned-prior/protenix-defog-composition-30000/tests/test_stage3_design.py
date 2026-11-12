"""Protocol expansion checks: matching, independent seeds and schedule controls."""

from collections import Counter
from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest


_MODULE = Path(__file__).resolve().parents[1] / "scripts/stage3_design.py"
_SPEC = importlib.util.spec_from_file_location("stage3_design", _MODULE)
design = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(design)


@pytest.fixture(scope="module")
def config():
    return design.load_design()


@pytest.fixture(scope="module")
def conditions(config):
    return design.expand_conditions(config)


def test_expansion_is_deterministic_and_covers_protocol(config, conditions):
    assert conditions == design.expand_conditions(config)
    assert len(conditions) == 155
    assert set(design.REQUIREMENTS) == {r for c in conditions for r in c["requirements"]}
    assert {c["projection"]["mode"] for c in conditions} == {
        "none", "valence", "bridge_only", "connected", "sparse_only", "sparse"}
    assert {c["family"] for c in conditions} >= {
        "reference", "weight", "time", "weight_schedule", "time_schedule", "joint_schedule",
        "projection", "projection_settings", "cooling", "cooling_mechanism",
        "cooling_interactions", "teacher_mechanism", "simple_chemistry"}


def test_every_teacher_treatment_has_an_exact_other_knob_control(conditions):
    by_id = {c["id"]: c for c in conditions}
    for c in conditions:
        if design._teacher_enabled(c):
            control = by_id[c["paired_with"]]
            assert not design._teacher_enabled(control)
            for key in ("projection", "cooling", "chemistry_weight"):
                assert c[key] == control[key]
        else:
            assert c["paired_with"] is None


def test_five_seed_blocks_repeat_exactly_across_conditions(config, conditions):
    runs = design.expand_runs(config)
    assert len(runs) == 155 * 3 * 5
    assert len({r["run_id"] for r in runs}) == len(runs)
    for c in conditions:
        selected = [r for r in runs if r["condition_id"] == c["id"]]
        assert {(r["seed"], r["initialization"]) for r in selected} == {
            (s, i) for s in (17, 23, 42, 71, 101) for i in ("sparse", "dense", "fragmented")}
        assert all(r["learning_rate"] == 60 and r["updates"] == 500 and r["native_weight"] == 1
                   for r in selected)
        assert all(r["initialization_parameters"] == design.INITIALIZATION_PARAMETERS for r in selected)


def test_schedule_endpoints_delays_and_actual_mean_controls(conditions):
    linear = design.linear(0, 0.25, begin=250)
    assert [design.schedule_value(linear, s) for s in (0, 249, 250, 499, 800)] == [0, 0, 0, 0.25, 0.25]
    cyclic = design.cosine(0, 0.125, 100)
    assert [design.schedule_value(cyclic, s) for s in (0, 50, 100)] == [0, 0.125, 0]
    by_id = {c["id"]: c for c in conditions}
    for c in conditions:
        for variable in ("weight", "time"):
            field = f"mean_{variable}_control"
            if field in c:
                actual = design.schedule_mean(c["teacher"][variable])
                control = by_id[c[field]]["teacher"][variable]
                assert design.schedule_value(control, 0) == pytest.approx(actual, abs=5e-13)
                assert c[f"mean_{variable}"] == actual
    assert design.schedule_mean(linear) == pytest.approx(0.0625)


@pytest.mark.parametrize("max_arms", [2, 3, 4])
def test_bundles_have_one_writer_and_independent_seed_placement(config, max_arms):
    runs = design.expand_runs(config)
    bundles = design.make_bundles(config, max_arms=max_arms)
    expected = {r["run_id"] for r in runs}
    counts = Counter(rid for b in bundles for rid in b["run_ids"])
    assert set(counts) == expected
    assert set(counts.values()) == {1}
    assert len({b["bundle_id"] for b in bundles}) == len(bundles)
    first_wave = [b for b in bundles if b["wave"] == 0]
    assert len(first_wave) == 15
    assert all(b["condition_ids"] == ["control", "reference"] for b in first_wave)
    for b in bundles:
        assert 1 <= len(b["run_ids"]) <= max_arms
        assert {r["seed"] for r in b["runs"]} == {b["seed"]}
        assert {r["initialization"] for r in b["runs"]} == {b["initialization"]}
        assert set(b["reference_run_ids"]) <= expected
        assert not set(b["reference_run_ids"]) & set(b["run_ids"])


def test_expanded_records_are_independent_copies(config):
    runs = design.expand_runs(config)
    before = deepcopy(runs[1]["condition"])
    runs[0]["condition"]["projection"]["mode"] = "none"
    assert runs[1]["condition"] == before
    assert design.expand_conditions(config)[0]["projection"]["mode"] == "connected"
    runs[0]["initialization_parameters"]["atom_distribution"][0] = 1
    assert runs[1]["initialization_parameters"] == design.INITIALIZATION_PARAMETERS


def test_start_metadata_cannot_claim_unimplemented_parameters(config):
    changed = deepcopy(config)
    changed["initialization_parameters"]["fragmented_deleted_tree_bonds"] = 1
    with pytest.raises(ValueError, match="Initialization parameters"):
        design.expand_runs(changed)


@pytest.mark.parametrize("max_arms", [0, 1, 5, False, 3.5])
def test_invalid_bundle_sizes_do_not_fall_back_silently(config, max_arms):
    with pytest.raises(ValueError, match="Bundle size"):
        design.make_bundles(config, max_arms=max_arms)


@pytest.mark.parametrize("key,value", [
    ("learning_rate", 15), ("updates", 501), ("native_weight", 0.5),
    ("model", "boltz2"), ("n_atoms", 30),
    ("seeds", [17, 17, 23, 42, 101]),
    ("initializations", ["dense", "sparse", "fragmented"]),
])
def test_protocol_changes_fail_explicitly(config, key, value):
    changed = deepcopy(config)
    changed[key] = value
    with pytest.raises(ValueError):
        design.expand_runs(changed)


@pytest.mark.parametrize("schedule", [
    {"kind": "linear", "start": 0, "end": 1, "begin": 250, "finish": 250},
    {"kind": "cosine", "low": 0, "high": 1, "period": 0},
    {"kind": "constant", "value": float("nan")},
    {"kind": "constant", "value": -1},
])
def test_invalid_schedules_fail_before_model_loading(config, conditions, schedule):
    changed = deepcopy(conditions)
    changed[0]["teacher"]["weight"] = schedule
    with pytest.raises(ValueError):
        design.validate_design(config, changed)


def test_invalid_cooling_and_missing_control_are_not_silent(config, conditions):
    for patch in ({"schedule": design.constant(0)}, {"placement": "state", "gradient": "ste"}):
        changed = deepcopy(conditions)
        changed[0]["cooling"].update(patch)
        with pytest.raises(ValueError):
            design.validate_design(config, changed)
    changed = deepcopy(conditions)
    changed[1]["paired_with"] = "missing-control"
    with pytest.raises(ValueError, match="Missing matched control"):
        design.validate_design(config, changed)


def test_calibrated_mechanisms_fix_reference_before_updates(conditions):
    selected = [c for c in conditions if c["teacher"]["calibration"] == "initial_gradient"]
    assert {c["id"] for c in selected} == {"mechanism-entry-mean-calibrated", "mechanism-kl-calibrated"}
    for c in selected:
        assert c["teacher"]["calibration_reference"] == {
            "loss": "mse", "normalization": "row", "atom_scale": 1, "bond_scale": 1, "weight": 0.0625}


def test_best_valid_hard_candidate_is_primary_and_final_is_secondary(config):
    evaluation = config["evaluation"]
    assert evaluation["primary"] == "best_valid_connected_hard_contact_score"
    assert evaluation["candidate"] == "raw_argmax"
    assert evaluation["validity"] == "rdkit_sanitized_and_one_fragment"
    assert evaluation["final_state"] == "secondary_diagnostic"
    assert evaluation["include_initial"] is True
    assert evaluation["save_best_new_candidate_separately"] is True
