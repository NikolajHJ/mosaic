"""Operational policy fixtures; run on allocated DTU HPC, no live scheduler.

These checks establish time-limit wiring only. They cannot approve a failed
native epoch-fit/retention gate or establish whole-epoch throughput.
"""
from copy import deepcopy

import pytest

from latentmol.chemistry_runtime import (composition_prior_weight, protocol_config_path,
    segment_runtime_policy, validate_execution_policy, validate_runtime_policy)


@pytest.mark.parametrize("internal,lsf_limit,minutes", [(22 * 3600, "23:00", 1380),
    (71 * 3600, "72:00", 4320), (22 * 3600 + 1, "23:01", 1381), (1, "1:01", 61)])
def test_segment_time_preserves_internal_budget_and_full_scheduler_margin(internal, lsf_limit, minutes):
    value = segment_runtime_policy({"segment_wall_seconds": internal})
    assert value == {"internal_seconds": internal, "allocation_seconds": minutes * 60,
                     "allocation_margin_seconds": minutes * 60 - internal,
                     "lsf_minutes": minutes, "lsf_limit": lsf_limit}
    assert 3600 <= value["allocation_margin_seconds"] < 3660
    assert value["allocation_seconds"] <= 72 * 3600


@pytest.mark.parametrize("field,invalid", [(field, invalid)
    for field, maximum in (("segment_wall_seconds", 71 * 3600), ("benchmark_wall_seconds", 6300))
    for invalid in (None, True, False, 0, -1, 1.0, float("inf"), float("-inf"), float("nan"), "1", maximum + 1)])
def test_duration_validation_rejects_coercion_nonfinite_and_excess(field, invalid):
    protocol = {"segment_wall_seconds": 71 * 3600, "benchmark_wall_seconds": 6300}
    protocol[field] = invalid
    with pytest.raises(ValueError, match=field + " requires integer seconds"):
        validate_runtime_policy(protocol)


def test_runtime_derivation_never_mutates_science_or_input_metadata():
    protocol = {"segment_wall_seconds": 71 * 3600, "benchmark_wall_seconds": 6300,
                "prior_ratio": 3, "search": {"max_edits": 3}, "repair_weight": .5,
                "total_epochs": 100, "prior_seed": 104017}
    before = deepcopy(protocol)
    value = validate_runtime_policy(protocol)
    value["internal_seconds"] = 1
    assert protocol == before
    assert validate_runtime_policy(protocol)["internal_seconds"] == 71 * 3600


@pytest.mark.parametrize("protocol", [None, [], "79200", {}, {"benchmark_wall_seconds": 6300}])
def test_missing_or_malformed_runtime_metadata_fails_closed(protocol):
    with pytest.raises(ValueError):
        validate_runtime_policy(protocol)


@pytest.mark.parametrize("workers,slots", [(12, 16), (12, 24), (12, 32), (20, 24), (20, 32), (28, 32)])
@pytest.mark.parametrize("overlap", [False, True])
def test_new_execution_allows_declared_spare_benchmark_slots(workers, slots, overlap):
    value = {"backend": "optimized", "workers": workers, "torch_threads": 4, "gpu_cpus": slots, "overlap": overlap}
    before = deepcopy(value)
    result = validate_execution_policy(value)
    assert result == before and value == before
    result["workers"] = -1
    assert value == before


@pytest.mark.parametrize("field,value", [("workers", 32), ("workers", 12.), ("torch_threads", 8),
    ("gpu_cpus", 16), ("gpu_cpus", 33), ("overlap", 1), ("overlap", "true"), ("unknown", 0)])
def test_new_execution_rejects_undersubscribed_or_coerced_metadata(field, value):
    record = {"backend": "optimized", "workers": 28, "torch_threads": 4, "gpu_cpus": 32, "overlap": True}
    record[field] = value
    with pytest.raises(ValueError, match="explicit chemistry execution"):
        validate_execution_policy(record)


def test_legacy_execution_never_infers_new_overlap_or_worker_count():
    legacy = {"backend": "optimized", "workers": 12, "torch_threads": 4, "gpu_cpus": 16}
    assert validate_execution_policy(legacy) == legacy
    assert validate_execution_policy(None) is None
    with pytest.raises(ValueError):
        validate_execution_policy({**legacy, "workers": 20, "gpu_cpus": 24})


@pytest.mark.parametrize("ratio,weight", [(3, None), (3, 1), (2, 1.5)])
def test_composition_importance_preserves_expected_one_to_three_population(ratio, weight):
    protocol = {"prior_ratio": ratio}
    if weight is not None:
        protocol["composition_prior_weight"] = weight
    assert composition_prior_weight(protocol) * ratio == 3


@pytest.mark.parametrize("protocol", [{"prior_ratio": 2}, {"prior_ratio": 2, "composition_prior_weight": 1.},
    {"prior_ratio": 3, "composition_prior_weight": 1.5}, {"prior_ratio": 2, "composition_prior_weight": True},
    {"prior_ratio": 2, "composition_prior_weight": float("nan")}, {"prior_ratio": 3.}, {"prior_ratio": 1}])
def test_composition_weight_never_silently_changes_sampling_population(protocol):
    with pytest.raises(ValueError, match="composition_prior_weight"):
        composition_prior_weight(protocol)


@pytest.mark.parametrize("path", ["/tmp/evil.json", "../config.json", "configs/chemistry_speed_variants/../x.json",
    "configs/chemistry_speed_variants/a/b.json", "configs/chemistry_speed_variants/a.yaml", "configs/other.json", None])
def test_protocol_paths_cannot_escape_source_bound_config_directory(path):
    with pytest.raises(ValueError, match="protocol_config"):
        protocol_config_path({"protocol_config": path})


def test_protocol_variant_path_is_explicit_and_legacy_default_is_unchanged():
    assert protocol_config_path({}) == "configs/chemistry_triangle_1to3.json"
    assert protocol_config_path({"protocol_config": "configs/chemistry_speed_variants/r3-w28-m256.json"}) == "configs/chemistry_speed_variants/r3-w28-m256.json"
