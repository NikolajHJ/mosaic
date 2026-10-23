"""Large-cache identity, exact-count and mmap contracts (run on DTU HPC)."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

from latentmol.chemistry import encode_smiles
from latentmol.data import (GraphDataset, LARGE_DATA_FILES, large_manifest_digest,
                            prepare_dataset, sha256_file, validate_large_cache)
import latentmol.large_data as large_data
from latentmol.large_data import inventory_large_dataset, prepare_large_dataset
from latentmol.representation import GraphSpec


SMILES = ("C", "CC", "CCC", "CCCC", "CCCCC", "O", "CO", "CCO", "N", "CN",
          "CCN", "F", "CF", "CCl", "CBr", "c1ccccc1", "C=O", "CC=O")


def _make_protected(tmp_path, spec):
    source = tmp_path / "old.smi"
    source.write_text("\n".join(SMILES[:10]) + "\n", encoding="utf-8")
    cache = tmp_path / "protected"
    prepare_dataset(source, cache, spec=spec, train_count=6, val_count=2)
    return cache


def _all_roles(cache):
    return {name: set(GraphDataset(cache, name).smiles)
            for name in ("train", "val", "test")}


def test_v2_exact_counts_schema_parity_and_protected_roles(tmp_path):
    spec = GraphSpec(max_nodes=8)
    protected = _make_protected(tmp_path, spec)
    source = tmp_path / "new.smi"
    source.write_text("\n".join(reversed(SMILES)) + "\nC\n[SiH4]\nnotasmiles\n",
                      encoding="utf-8")
    cache = tmp_path / "large"
    manifest = prepare_large_dataset(source, cache, protected,
                                     train_count=11, val_count=3, test_count=2,
                                     seed=17, spec=spec)
    assert manifest["dataset_format_version"] == 2
    assert [manifest["counts"][name] for name in ("train", "val", "test")] == [11, 3, 2]
    assert manifest["counts"]["unique_accepted_before_selection"] == len(SMILES)
    assert manifest["counts"]["duplicate_rows"] == 1
    assert manifest["counts"]["excluded_supported_by_target"] == 2
    assert manifest["counts"]["rejected_rows"] == 2
    assert manifest["dataset_sha256"] == large_manifest_digest(manifest)
    assert set(manifest["files"]) == set(LARGE_DATA_FILES)
    assert set(manifest["preparation_identity"]["source_file_sha256"]) == {
        "large_data.py", "data.py", "chemistry.py", "representation.py"}
    assert manifest["provenance"]["reserved_cache"]["absent_protected"] == {
        "train": 0, "val": 0, "test": 0}
    receipt = validate_large_cache(cache)
    old_roles = _all_roles(protected)
    new_roles = {name: set(GraphDataset(cache, name, validated_cache=receipt).smiles)
                 for name in ("train", "val", "test")}
    for name in old_roles:
        assert old_roles[name] <= new_roles[name]
    assert not (new_roles["train"] & new_roles["val"]
                or new_roles["train"] & new_roles["test"]
                or new_roles["val"] & new_roles["test"])
    assert sum(len(values) for values in new_roles.values()) == 16
    dataset = GraphDataset(cache, "train", validated_cache=receipt)
    assert isinstance(dataset._bonds, np.memmap)
    assert len(dataset) == 11 and dataset._bonds.shape[0] == 16
    for index in (0, len(dataset) - 1):
        expected = encode_smiles(dataset.smiles[index], spec)
        actual = dataset[index]
        assert np.array_equal(actual["atoms"].numpy(), expected["atom_types"])
        assert np.array_equal(actual["charges"].numpy(), expected["charges"])
        assert np.array_equal(actual["bonds"].numpy(), expected["bond_types"])


def test_v2_split_is_source_order_independent_and_missing_protected_reported(tmp_path):
    spec = GraphSpec(max_nodes=8)
    protected = _make_protected(tmp_path, spec)
    old = _all_roles(protected)
    absent = next(iter(old["val"]))
    source_one = tmp_path / "one.smi"
    source_two = tmp_path / "two.smi"
    retained = [value for value in SMILES if value != absent]
    source_one.write_text("\n".join(retained), encoding="utf-8")
    source_two.write_text("\n".join(reversed(retained)), encoding="utf-8")
    one = prepare_large_dataset(source_one, tmp_path / "one", protected,
                                10, 3, 2, 29, spec=spec)
    two = prepare_large_dataset(source_two, tmp_path / "two", protected,
                                10, 3, 2, 29, spec=spec)
    assert one["split_sha256"] == two["split_sha256"]
    assert {name: one["files"][name]["sha256"] for name in LARGE_DATA_FILES} == {
        name: two["files"][name]["sha256"] for name in LARGE_DATA_FILES}
    assert one["provenance"]["reserved_cache"]["absent_protected"] == {
        "train": 0, "val": 1, "test": 0}
    roles = _all_roles(tmp_path / "one")
    assert old["train"] <= roles["train"]
    assert (old["val"] - {absent}) <= roles["val"]
    assert absent not in set.union(*roles.values())


def test_v2_exact_count_shortage_and_protected_overflow_do_not_publish(tmp_path):
    spec = GraphSpec(max_nodes=8)
    protected = _make_protected(tmp_path, spec)
    source = tmp_path / "small.smi"
    source.write_text("\n".join(SMILES[:10]), encoding="utf-8")
    shortage = tmp_path / "shortage"
    with pytest.raises(ValueError, match="only 10 unique supported identities"):
        prepare_large_dataset(source, shortage, protected, 10, 2, 1, spec=spec)
    assert not shortage.exists()
    overflow = tmp_path / "overflow"
    with pytest.raises(ValueError, match="smaller than present protected membership"):
        prepare_large_dataset(source, overflow, protected, 5, 3, 2, spec=spec)
    assert not overflow.exists()


def test_v2_manifest_and_file_integrity_are_checked(tmp_path):
    spec = GraphSpec(max_nodes=8)
    protected = _make_protected(tmp_path, spec)
    source = tmp_path / "new.smi"
    source.write_text("\n".join(SMILES), encoding="utf-8")
    cache = tmp_path / "large"
    prepare_large_dataset(source, cache, protected, 10, 3, 2, spec=spec)
    receipt = validate_large_cache(cache)
    with (cache / "smiles.utf8").open("ab") as stream:
        stream.write(b"X")
    with pytest.raises(ValueError, match="differs from its manifest"):
        validate_large_cache(cache)
    with pytest.raises(ValueError, match="changed since validation"):
        GraphDataset(cache, "train", validated_cache=receipt)
    manifest_path = cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["counts"]["train"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical manifest digest"):
        GraphDataset(cache, "train", validated_cache=receipt)


def test_v2_bounded_graph_semantics_check_even_with_updated_hashes(tmp_path):
    spec = GraphSpec(max_nodes=8)
    protected = _make_protected(tmp_path, spec)
    source = tmp_path / "new.smi"
    source.write_text("\n".join(SMILES), encoding="utf-8")
    cache = tmp_path / "large"
    prepare_large_dataset(source, cache, protected, 10, 3, 2, spec=spec)
    bonds = np.lib.format.open_memmap(cache / "bond_types.npy", mode="r+")
    bonds[0, 0, 0] = 1
    bonds.flush()
    del bonds
    manifest_path = cache / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["bond_types.npy"]["sha256"] = sha256_file(cache / "bond_types.npy")
    manifest["dataset_sha256"] = large_manifest_digest(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="bond symmetry, diagonal or padding"):
        validate_large_cache(cache)


def test_inventory_matches_full_preparation_filtering_and_composition_without_arrays(tmp_path):
    spec = GraphSpec(max_nodes=8)
    protected = _make_protected(tmp_path, spec)
    protected_hashes = {name: sha256_file(protected / name) for name in ("manifest.json", "dataset.npz")}
    source = tmp_path / "new.smi"
    source.write_text("\n".join(reversed(SMILES)) + "\nC\n[SiH4]\nnotasmiles\n", encoding="utf-8")
    destination = tmp_path / "inventory"
    report = inventory_large_dataset(source, destination, protected, seed=17, spec=spec)
    prepared = prepare_large_dataset(source, tmp_path / "large", protected,
                                     train_count=14, val_count=2, test_count=2, seed=17, spec=spec)
    assert report["status"] == "completed_inventory_only"
    assert report["counts"]["source_rows"] == len(SMILES) + 3
    assert report["counts"]["supported_unique"] == len(SMILES)
    assert report["counts"]["duplicate_rows"] == 1
    assert report["counts"]["rejected_rows"] == 2
    for name in ("source_rows", "duplicate_rows", "rejected_rows", "roundtrip_valid_rows"):
        assert report["counts"][name] == prepared["counts"][name]
    assert report["rejection_reasons"] == prepared["rejection_reasons"]
    assert report["protected_roles"] == {key: prepared["provenance"]["reserved_cache"][key]
                                         for key in ("total_protected", "present_protected", "absent_protected")}
    for key in ("atomic_number_counts", "formal_charge_counts", "active_upper_triangle_bond_counts"):
        assert report["composition"][key] == prepared["audit"][key]
    expected_histogram = [sum(group["heavy_atom_histogram"][index]
                              for group in prepared["audit"]["by_split"].values())
                          for index in range(spec.max_nodes + 1)]
    assert report["composition"]["heavy_atom_histogram"] == expected_histogram
    assert report["composition"]["molecules"] == len(SMILES)
    assert [path.name for path in destination.iterdir()] == ["inventory.json"]
    assert not report["provenance"]["graph_arrays_written"]
    assert report["provenance"]["selected_target_counts"] is None
    assert report["disk"]["all_supported_dense_array_payload_bytes"] == len(SMILES) * (8 * 8 + 2 * 8 + 1)
    assert report["disk"]["all_supported_smiles_utf8_bytes"] == sum(len(value.encode()) for value in SMILES)
    assert report["timing"]["canonical_scan_seconds"] > 0
    assert report["host_memory"]["peak_host_rss_bytes"] is None or report["host_memory"]["peak_host_rss_bytes"] > 0
    serialized = json.loads((destination / "inventory.json").read_text())
    digest = serialized.pop("inventory_sha256")
    assert digest == large_data._config_hash(serialized)
    assert protected_hashes == {name: sha256_file(protected / name) for name in protected_hashes}
    with pytest.raises(FileExistsError, match="Inventory destination already exists"):
        inventory_large_dataset(source, destination, protected, spec=spec)


def test_inventory_identity_order_independent_and_protected_absence_is_not_assigned(tmp_path):
    spec = GraphSpec(max_nodes=8)
    protected = _make_protected(tmp_path, spec)
    absent = next(iter(_all_roles(protected)["val"]))
    retained = [value for value in SMILES if value != absent]
    reports = []
    for name, values in (("one", retained), ("two", list(reversed(retained)))):
        source = tmp_path / (name + ".smi")
        source.write_text("\n".join(values), encoding="utf-8")
        reports.append(inventory_large_dataset(source, tmp_path / name, protected, spec=spec))
    assert reports[0]["supported_identity_sha256"] == reports[1]["supported_identity_sha256"]
    assert reports[0]["composition"] == reports[1]["composition"]
    assert reports[0]["protected_roles"]["absent_protected"] == {"train": 0, "val": 1, "test": 0}
    assert reports[0]["protected_roles"]["present_protected"]["val"] == 1
    assert reports[0]["counts"]["supported_unique"] == len(SMILES) - 1
    assert not (tmp_path / "one" / "manifest.json").exists()


def test_inventory_rechecks_source_and_rejects_protected_output_subdirectories(tmp_path, monkeypatch):
    spec = GraphSpec(max_nodes=8)
    protected = _make_protected(tmp_path, spec)
    source = tmp_path / "new.smi"
    source.write_text("\n".join(SMILES), encoding="utf-8")
    with pytest.raises(ValueError, match="Output must differ"):
        inventory_large_dataset(source, protected / "inventory", protected, spec=spec)
    original_scan = large_data._scan_supported
    def mutate_source(*args, **kwargs):
        result = original_scan(*args, **kwargs)
        source.write_text("C\n", encoding="utf-8")
        return result
    monkeypatch.setattr(large_data, "_scan_supported", mutate_source)
    with pytest.raises(RuntimeError, match="Source file changed during inventory"):
        inventory_large_dataset(source, tmp_path / "failed", protected, spec=spec)
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob(".failed.inventory-*"))


def test_inventory_cli_dispatch_does_not_prepare_or_use_requested_counts(tmp_path, monkeypatch, capsys):
    captured = {}
    def fake_inventory(*args):
        captured["args"] = args
        return {"status": "completed_inventory_only"}
    monkeypatch.setattr(large_data, "inventory_large_dataset", fake_inventory)
    monkeypatch.setattr(large_data, "prepare_large_dataset",
                        lambda *args, **kwargs: pytest.fail("inventory must not prepare arrays or assign splits"))
    monkeypatch.setattr(sys, "argv", ["large_data", "--mode", "inventory", "--input", "raw.csv",
                                      "--output", "inventory", "--reserve-splits", "protected",
                                      "--smiles-column", "SMILES", "--train-count", "1900000"])
    large_data.main()
    assert captured["args"] == ("raw.csv", "inventory", "protected", 17, "SMILES")
    assert json.loads(capsys.readouterr().out)["status"] == "completed_inventory_only"


def test_inventory_submission_is_serial_cpu_after_contracts(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path.cwd() / "scripts/hpc"))
    import submit_large_triangle as submission
    assert submission.STAGE_PREREQUISITES["inventory"] == ("cpu",)
    assert submission.STAGE_PREREQUISITES["benchmark"] == ("cpu", "prepare")
    plan = submission.workflow("inventory", tmp_path / "release", tmp_path, tmp_path / "campaign",
                               tmp_path / "bindings.json", "binding", "archive", "manifest", run_id="001")
    assert len(plan) == 1 and plan[0]["stage"] == "inventory"
    arguments = plan[0]["arguments"]
    assert arguments[arguments.index("-q") + 1] == "hpc"
    assert arguments[arguments.index("-W") + 1] == "12:00"
    assert "-gpu" not in arguments and "select[gpu80gb]" not in arguments
    assert "inventory" in plan[0]["job_command"]


@pytest.mark.parametrize("changed", ["raw", "protected"])
def test_inventory_dispatch_authenticates_actually_processed_inputs(tmp_path, monkeypatch, changed):
    monkeypatch.syspath_prepend(str(Path.cwd() / "scripts/hpc"))
    from large_triangle_stage import require_processed_input_identity
    report = {"raw_sha256": "raw", "provenance": {"reserved_cache": {"manifest_sha256": "protected"}}}
    binding = {"raw_input_sha256": "raw", "reserve_manifest_sha256": "protected"}
    require_processed_input_identity(report, binding)
    if changed == "raw":
        report["raw_sha256"] = "another raw file"
    else:
        report["provenance"]["reserved_cache"]["manifest_sha256"] = "another protected manifest"
    with pytest.raises(ValueError, match="Processed source/protected inputs differ"):
        require_processed_input_identity(report, binding)
