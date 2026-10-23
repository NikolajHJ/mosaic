"""HPC fixtures for JSON configuration identity and strict legacy compatibility."""
from copy import deepcopy
import hashlib
import json

import pytest

from latentmol.config_identity import (
    CONFIG_HASH_VERSION, config_sha256, normalize_json, verify_config_identity,
)
from latentmol.experiments import ExperimentRun


def original_active_set_config():
    return {"contexts": {count: {"path": f"/count-{count:02d}/prepared.pt",
                                "sha256": f"{count:064x}"} for count in range(1, 51)},
            "optimizer": {"steps": 100}, "diagnostics": {"hard_samples": 8}}


def historical_record(config):
    """Reproduce old hashing before JSON coerced the integer object keys."""
    return json.loads(json.dumps({"config": config,
                                 "config_sha256": hashlib.sha256(
                                     json.dumps(config, sort_keys=True).encode()).hexdigest()}))


def test_new_count_bank_hash_survives_saved_loaded_json_and_records_version(tmp_path):
    original = original_active_set_config()
    loaded = json.loads(json.dumps(original))
    assert config_sha256(original) == config_sha256(loaded)
    with ExperimentRun(tmp_path, "count-bank", original, tmp_path) as run:
        pass
    record = json.loads((run.path / "run.json").read_text())
    assert record["config_hash_version"] == CONFIG_HASH_VERSION
    assert record["config"] == loaded
    assert record["config_sha256"] == config_sha256(loaded)
    assert verify_config_identity(record)["historical_context_keys_reconstructed"] is False
    assert set(original["contexts"]) == set(range(1, 51))  # Caller data is untouched.


def test_legacy_context_keys_require_explicit_exact_reconstruction_without_mutation():
    record = historical_record(original_active_set_config())
    before = deepcopy(record)
    assert config_sha256(record["config"]) != record["config_sha256"]
    with pytest.raises(ValueError, match="hash changed"):
        verify_config_identity(record)
    receipt = verify_config_identity(record, allow_legacy_active_set_contexts=True)
    assert receipt["scheme"] == "legacy-active-set-integer-contexts"
    assert receipt["stored_sha256"] == record["config_sha256"]
    assert receipt["historical_context_keys_reconstructed"] is True
    assert record == before


@pytest.mark.parametrize("change", ["path", "context_sha", "record_sha", "other_field"])
def test_legacy_compatibility_rejects_changed_values_or_hash(change):
    record = historical_record(original_active_set_config())
    if change == "path":
        record["config"]["contexts"]["7"]["path"] += ".changed"
    elif change == "context_sha":
        record["config"]["contexts"]["7"]["sha256"] = "0" * 64
    elif change == "record_sha":
        record["config_sha256"] = "0" * 64
    else:
        record["config"]["optimizer"]["steps"] = 99
    with pytest.raises(ValueError, match="hash changed"):
        verify_config_identity(record, allow_legacy_active_set_contexts=True)


def test_versioned_records_and_unrelated_integer_keys_cannot_use_legacy_fallback():
    record = historical_record(original_active_set_config())
    record["config_hash_version"] = CONFIG_HASH_VERSION
    with pytest.raises(ValueError, match="hash changed"):
        verify_config_identity(record, allow_legacy_active_set_contexts=True)
    record["config_hash_version"] = "unknown-version"
    with pytest.raises(ValueError, match="Unsupported"):
        verify_config_identity(record, allow_legacy_active_set_contexts=True)
    config = original_active_set_config()
    config["unrelated"] = {2: "two", 10: "ten"}
    with pytest.raises(ValueError, match="hash changed"):
        verify_config_identity(historical_record(config), allow_legacy_active_set_contexts=True)


@pytest.mark.parametrize("keys", [(1, "1"), (-2, "-2")])
def test_colliding_integer_string_keys_rejected_before_record_creation(tmp_path, keys):
    config = {"nested": {keys[0]: "first", keys[1]: "second"}}
    with pytest.raises(ValueError, match="Colliding"):
        normalize_json(config)
    with pytest.raises(ValueError, match="Colliding"):
        config_sha256(config)
    with pytest.raises(ValueError, match="Colliding"):
        ExperimentRun(tmp_path / "runs", "collision", config, tmp_path)
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("key", ["01", "0", "51"])
def test_legacy_reconstruction_rejects_noncanonical_or_unsupported_count_keys(key):
    # Hash the equivalent historical integer-key map but retain the unsupported
    # loaded spelling/count. This must not be normalized into a matching record.
    config = original_active_set_config()
    config["contexts"] = {int(key): {"path": "/example", "sha256": "a" * 64},
                          2: {"path": "/other", "sha256": "b" * 64},
                          10: {"path": "/ten", "sha256": "c" * 64}}
    record = historical_record(config)
    if key == "01":
        record["config"]["contexts"][key] = record["config"]["contexts"].pop("1")
    assert config_sha256(record["config"]) != record["config_sha256"]
    with pytest.raises(ValueError, match="hash changed"):
        verify_config_identity(record, allow_legacy_active_set_contexts=True)


def test_partial_trajectory_reader_accepts_exact_legacy_record_and_reports_scheme(tmp_path):
    from latentmol.boltz.trajectory_selection import _pilot_candidates
    record = historical_record(original_active_set_config())
    record.update(status="failed", source_archive_sha256="a" * 64)
    (tmp_path / "run.json").write_text(json.dumps(record))
    _, rows, proxies, warnings = _pilot_candidates(tmp_path, "a" * 64)
    assert rows == {} and proxies == {}
    assert warnings[0]["configuration_identity"]["scheme"] == "legacy-active-set-integer-contexts"
    record["config"]["contexts"]["7"]["path"] = "/tampered"
    (tmp_path / "run.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="hash changed"):
        _pilot_candidates(tmp_path, "a" * 64)
