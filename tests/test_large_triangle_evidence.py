"""HPC-only metadata fixtures: synthetic receipts, no model or dataset work.

The tiny files stand in for already allocated evidence. Passing these fixtures
establishes authentication behavior, not molecule validity or training results.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shlex

import pytest


def _module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path.cwd() / "scripts/hpc"))
    import large_triangle_evidence
    return large_triangle_evidence


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _intent(identity, stage, binding_path):
    command = shlex.join(["python", "scripts/verify_stability_release.py",
                          "--archive-sha256", identity["archive_sha256"],
                          "--manifest-sha256", identity["manifest_sha256"], "--",
                          "python", "scripts/hpc/large_triangle_stage.py", stage,
                          "--bindings", str(binding_path), "--bindings-sha256", identity["bindings_sha256"]])
    return {"identity": identity, "cwd": identity["release"],
            "command": ["bsub", "-J", Path(identity["release"]).name + "-" + stage, command]}


def _stage_fixture(tmp_path, stage="prepare"):
    release = tmp_path / "releases/source095"
    release.mkdir(parents=True)
    campaign = tmp_path / "campaign"
    receipts = campaign / "receipts"
    identity = {"release": str(release), "campaign": str(campaign), "archive_sha256": "a" * 64,
                "manifest_sha256": "b" * 64, "bindings_sha256": "c" * 64}
    intent = _intent(identity, stage, tmp_path / "bindings.json")
    _write(receipts / (stage + "-intent.json"), intent)
    receipt = {"intent": intent, "job_id": "12345", "status": "submitted_not_passed"}
    receipt_sha = _write(receipts / (stage + "-receipt.json"), receipt)
    artifact = campaign / "data/manifest.json"
    artifact_sha = _write(artifact, {"fixture": "not a scientific cache"})
    completion = {"stage": stage, "status": "passed", "stage_returncode": 0, "job_id": "12345",
                  "bindings_sha256": identity["bindings_sha256"], "source_sha256": identity["manifest_sha256"],
                  "source_archive_sha256": identity["archive_sha256"],
                  "artifacts": {} if stage == "cpu" else {str(artifact): artifact_sha}}
    _write(campaign / "stage-completions" / (stage + ".json"), completion)
    return receipts, identity, receipt, receipt_sha, completion, artifact


def test_submission_identity_is_not_execution_success_and_completion_binds_artifacts(tmp_path, monkeypatch):
    evidence = _module(monkeypatch)
    receipts, identity, _, receipt_sha, _, artifact = _stage_fixture(tmp_path)
    assert evidence.require_submitted_stage(receipts, "prepare", identity) == ("12345", receipt_sha)
    assert evidence.require_stage_completion(receipts, "prepare", identity) == "12345"
    artifact.write_text("changed artifact", encoding="utf-8")
    with pytest.raises(ValueError, match="Completion artifact SHA256 mismatch"):
        evidence.require_stage_completion(receipts, "prepare", identity)
    # A submitted receipt still cannot establish a successful preparation.
    assert evidence.require_submitted_stage(receipts, "prepare", identity)[0] == "12345"


def test_cpu_completion_may_have_no_artifacts_but_prepare_may_not(tmp_path, monkeypatch):
    evidence = _module(monkeypatch)
    receipts, identity, *_ = _stage_fixture(tmp_path / "cpu", "cpu")
    assert evidence.require_stage_completion(receipts, "cpu", identity) == "12345"
    receipts, identity, _, _, completion, _ = _stage_fixture(tmp_path / "prepare")
    completion["artifacts"] = {}
    _write(Path(identity["campaign"]) / "stage-completions/prepare.json", completion)
    with pytest.raises(ValueError, match="lacks its artifact hashes"):
        evidence.require_stage_completion(receipts, "prepare", identity)


@pytest.mark.parametrize("bad", ["identity", "job", "status", "cwd", "label", "executed_stage", "command_sha"])
def test_submitted_receipt_requires_actual_stage_and_exact_intent(tmp_path, monkeypatch, bad):
    evidence = _module(monkeypatch)
    receipts, identity, receipt, _, _, _ = _stage_fixture(tmp_path)
    intent = receipt["intent"]
    if bad == "identity":
        intent["identity"] = {**identity, "archive_sha256": "d" * 64}
    elif bad == "job":
        receipt["job_id"] = 12345
    elif bad == "status":
        receipt["status"] = "unknown_not_retryable"
    elif bad == "cwd":
        intent["cwd"] = str(tmp_path)
    elif bad == "label":
        intent["command"][2] = "source095-cpu"
    elif bad == "executed_stage":
        intent["command"][-1] = intent["command"][-1].replace("large_triangle_stage.py prepare", "large_triangle_stage.py cpu")
    else:
        intent["command"][-1] = intent["command"][-1].replace(identity["archive_sha256"], "e" * 64)
    _write(receipts / "prepare-receipt.json", receipt)
    _write(receipts / "prepare-intent.json", intent)
    with pytest.raises(ValueError):
        evidence.require_submitted_stage(receipts, "prepare", identity)


@pytest.mark.parametrize("field,value", [("job_id", "54321"), ("source_sha256", "d" * 64),
                                         ("source_archive_sha256", "e" * 64), ("bindings_sha256", "f" * 64),
                                         ("stage", "cpu"), ("stage_returncode", True), ("status", "failed")])
def test_allocated_completion_cannot_be_borrowed_from_a_different_job_or_source(tmp_path, monkeypatch, field, value):
    evidence = _module(monkeypatch)
    receipts, identity, _, _, completion, _ = _stage_fixture(tmp_path)
    completion[field] = value
    _write(Path(identity["campaign"]) / "stage-completions/prepare.json", completion)
    with pytest.raises(ValueError, match="Allocated stage completion evidence differs"):
        evidence.require_stage_completion(receipts, "prepare", identity)


def _inventory_fixture(tmp_path, evidence):
    """Build complete synthetic history with real hashes, without RDKit/NumPy."""
    work = tmp_path / "work"
    old_release, release = work / "releases/source094", work / "releases/source095"
    for name in evidence.PREPROCESSING_FILES:
        for root in (old_release, release):
            _file(root / "src/latentmol" / name, ("# static fixture for " + name + "\n").encode())
    from latentmol.experiments import source_manifest
    old_source = source_manifest(old_release)
    _write(old_release / "release_manifest.json", old_source)
    archive_sha = _file(Path(str(old_release) + ".tar.gz"), b"synthetic immutable archive bytes")
    spec = {"version": 1, "max_nodes": 50, "atomic_numbers": [0, 6, 7, 8, 9, 15, 16, 17, 35, 53],
            "formal_charges": [-2, -1, 0, 1, 2], "bond_names": ["NONE", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]}
    spec_sha = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    raw = work / "raw/source.csv"
    raw_sha = _file(raw, b"SMILES\nC\nCC\n")
    reserve = work / "data/original"
    dataset_sha = _file(reserve / "dataset.npz", b"synthetic protected identity bytes, not an array")
    reserve_manifest = {"dataset_format_version": 1, "dataset_sha256": dataset_sha, "split_sha256": "d" * 64,
                        "spec_fingerprint": spec_sha, "rdkit_version": "2024.09.6",
                        "counts": {"train": 2, "val": 1, "test": 1}}
    reserve_sha = _write(reserve / "manifest.json", reserve_manifest)
    old_campaign = work / "runs/source094"
    old_binding = {"version": 1, "work_root": str(work), "campaign": str(old_campaign),
                   "raw_input": str(raw), "raw_input_sha256": raw_sha, "smiles_column": "SMILES",
                   "reserve_cache": str(reserve), "reserve_manifest_sha256": reserve_sha}
    binding_path = work / "submissions/source094/bindings.json"
    binding_sha = _write(binding_path, old_binding)
    old_identity = {"release": str(old_release), "campaign": str(old_campaign), "archive_sha256": archive_sha,
                    "manifest_sha256": old_source["sha256"], "bindings_sha256": binding_sha}
    config = {"mode": "inventory", "spec_fingerprint": spec_sha, "seed": 17,
              "smiles_column": "SMILES", "protected_dataset_sha256": dataset_sha}
    report = {"inventory_format_version": 1, "status": "completed_inventory_only", "spec": spec,
              "spec_fingerprint": spec_sha, "raw_sha256": raw_sha, "rdkit_version": "2024.09.6",
              "supported_identity_sha256": "e" * 64,
              "counts": {"source_rows": 6, "roundtrip_valid_rows": 5, "duplicate_rows": 1,
                         "rejected_rows": 1, "supported_unique": 4},
              "protected_roles": {"total_protected": {"train": 2, "val": 1, "test": 1},
                                  "present_protected": {"train": 1, "val": 1, "test": 0},
                                  "absent_protected": {"train": 1, "val": 0, "test": 1}},
              "inventory_identity": {"config": config, "config_sha256": _json_hash(config),
                                     "source_file_sha256": {name: old_source["files"]["src/latentmol/" + name]
                                                            for name in evidence.PREPROCESSING_FILES}},
              "provenance": {"source_path": str(raw), "smiles_column": "SMILES", "graph_arrays_written": False,
                             "selected_target_counts": None,
                             "reserved_cache": {"path": str(reserve), "manifest_sha256": reserve_sha,
                                                "dataset_sha256": dataset_sha, "split_sha256": "d" * 64}}}
    report["inventory_sha256"] = _json_hash(report)
    report_path = old_campaign / "inventory/inventory.json"
    report_sha = _write(report_path, report)
    intent = _intent(old_identity, "inventory", binding_path)
    intent_path = old_campaign / "receipts/inventory-intent.json"
    intent_sha = _write(intent_path, intent)
    receipt_path = old_campaign / "receipts/inventory-receipt.json"
    receipt_sha = _write(receipt_path, {"intent": intent, "job_id": "29680603", "status": "submitted_not_passed"})
    completion_path = old_campaign / "stage-completions/inventory.json"
    completion_sha = _write(completion_path, {"status": "passed", "stage": "inventory", "stage_returncode": 0,
                                             "job_id": "29680603", "bindings_sha256": binding_sha,
                                             "source_sha256": old_source["sha256"], "source_archive_sha256": archive_sha,
                                             "artifacts": {str(report_path): report_sha}})
    reference = {name: {"path": str(path), "sha256": digest} for name, path, digest in (
        ("report", report_path, report_sha), ("completion", completion_path, completion_sha),
        ("receipt", receipt_path, receipt_sha), ("intent", intent_path, intent_sha), ("bindings", binding_path, binding_sha))}
    value = {**old_binding, "campaign": str(work / "runs/source095"), "inventory_reference": reference}
    return {"value": value, "protocol": {"requested_counts": {"train": 2, "val": 1, "test": 1}},
            "release": release, "old_release": old_release, "report": report}


def _reseal_report(case):
    """Bind a deliberately changed report through all hashes to test semantics."""
    report = case["report"]
    report.pop("inventory_sha256", None)
    report["inventory_sha256"] = _json_hash(report)
    reference = case["value"]["inventory_reference"]
    report_sha = _write(Path(reference["report"]["path"]), report)
    reference["report"]["sha256"] = report_sha
    completion_path = Path(reference["completion"]["path"])
    completion = json.loads(completion_path.read_text())
    completion["artifacts"][reference["report"]["path"]] = report_sha
    reference["completion"]["sha256"] = _write(completion_path, completion)


def _prepared_fixture(tmp_path, evidence):
    """Metadata-only prior preparation; current cache still requires HPC arrays."""
    case = _inventory_fixture(tmp_path, evidence)
    value = case["value"]
    original = value["inventory_reference"]
    binding_path = Path(original["bindings"]["path"])
    binding = json.loads(binding_path.read_text())
    work = Path(value["work_root"])
    data = work / "data/prepared"
    value["data"] = binding["data"] = str(data)
    binding_sha = _write(binding_path, binding)
    source = json.loads((case["old_release"] / "release_manifest.json").read_text())
    archive_sha = evidence.file_sha256(Path(str(case["old_release"]) + ".tar.gz"))
    identity = {"release": str(case["old_release"]), "campaign": binding["campaign"],
                "archive_sha256": archive_sha, "manifest_sha256": source["sha256"],
                "bindings_sha256": binding_sha}
    protected = json.loads((Path(binding["reserve_cache"]) / "manifest.json").read_text())
    config = {"spec_fingerprint": protected["spec_fingerprint"], "seed": 17,
              "smiles_column": "SMILES", "train_count": 2, "val_count": 1, "test_count": 1,
              "protected_dataset_sha256": protected["dataset_sha256"]}
    manifest = {"dataset_format_version": 2, "spec": case["report"]["spec"],
                "spec_fingerprint": protected["spec_fingerprint"],
                "counts": {"train": 2, "val": 1, "test": 1, "total": 4},
                "raw_sha256": value["raw_input_sha256"], "dataset_sha256": "e" * 64,
                "split_sha256": "f" * 64,
                "preparation_identity": {"config": config, "config_sha256": _json_hash(config),
                                         "source_file_sha256": case["report"]["inventory_identity"]["source_file_sha256"]},
                "provenance": {"source_path": value["raw_input"], "smiles_column": "SMILES", "seed": 17,
                               "reserved_cache": {"path": value["reserve_cache"],
                                                  "manifest_sha256": value["reserve_manifest_sha256"],
                                                  "dataset_sha256": protected["dataset_sha256"],
                                                  "split_sha256": protected["split_sha256"]}}}
    manifest_path = data / "manifest.json"
    manifest_sha = _write(manifest_path, manifest)
    campaign = Path(binding["campaign"])
    intent_path = campaign / "receipts/prepare-intent.json"
    intent = _intent(identity, "prepare", binding_path)
    intent_sha = _write(intent_path, intent)
    receipt_path = campaign / "receipts/prepare-receipt.json"
    receipt_sha = _write(receipt_path, {"intent": intent, "job_id": "29683909", "status": "submitted_not_passed"})
    completion_path = campaign / "stage-completions/prepare.json"
    completion_sha = _write(completion_path, {"status": "passed", "stage": "prepare", "stage_returncode": 0,
                                              "job_id": "29683909", "bindings_sha256": binding_sha,
                                              "source_sha256": source["sha256"], "source_archive_sha256": archive_sha,
                                              "artifacts": {str(manifest_path): manifest_sha}})
    value["prepared_reference"] = {name: {"path": str(path), "sha256": digest} for name, path, digest in (
        ("manifest", manifest_path, manifest_sha), ("completion", completion_path, completion_sha),
        ("receipt", receipt_path, receipt_sha), ("intent", intent_path, intent_sha),
        ("bindings", binding_path, binding_sha))}
    case["manifest"] = manifest
    return case


def _reseal_prepared(case):
    reference = case["value"]["prepared_reference"]
    digest = _write(Path(reference["manifest"]["path"]), case["manifest"])
    reference["manifest"]["sha256"] = digest
    completion_path = Path(reference["completion"]["path"])
    completion = json.loads(completion_path.read_text())
    completion["artifacts"][reference["manifest"]["path"]] = digest
    reference["completion"]["sha256"] = _write(completion_path, completion)


def test_prepared_reference_reuses_only_authenticated_data_not_old_cpu_gate(tmp_path, monkeypatch):
    evidence = _module(monkeypatch)
    case = _prepared_fixture(tmp_path, evidence)
    before = {v["path"]: Path(v["path"]).read_bytes() for v in case["value"]["prepared_reference"].values()}
    report = evidence.authenticate_prepared_reference(case["value"], case["protocol"], case["release"])
    assert report["preparation_job_id"] == "29683909" and report["counts"]["train"] == 2
    assert before == {path: Path(path).read_bytes() for path in before}


@pytest.mark.parametrize("field", ["counts", "seed", "protected", "old_job", "current_source", "command_binding"])
def test_prepared_reference_rejects_changed_semantics_and_actual_stage_identity(tmp_path, monkeypatch, field):
    evidence = _module(monkeypatch)
    case = _prepared_fixture(tmp_path, evidence)
    reference = case["value"]["prepared_reference"]
    if field == "counts":
        case["manifest"]["counts"]["train"] = 3
    elif field == "seed":
        config = case["manifest"]["preparation_identity"]["config"]
        config["seed"] = 18
        case["manifest"]["preparation_identity"]["config_sha256"] = _json_hash(config)
    elif field == "protected":
        case["manifest"]["provenance"]["reserved_cache"]["split_sha256"] = "0" * 64
    elif field == "current_source":
        (case["release"] / "src/latentmol/chemistry.py").write_text("# changed preprocessing")
    elif field == "old_job":
        path = Path(reference["completion"]["path"])
        completion = json.loads(path.read_text())
        completion["job_id"] = "111"
        reference["completion"]["sha256"] = _write(path, completion)
    else:
        path = Path(reference["intent"]["path"])
        intent = json.loads(path.read_text())
        intent["command"][-1] = intent["command"][-1].replace(reference["bindings"]["path"], str(tmp_path / "other.json"))
        reference["intent"]["sha256"] = _write(path, intent)
        receipt_path = Path(reference["receipt"]["path"])
        receipt = json.loads(receipt_path.read_text())
        receipt["intent"] = intent
        reference["receipt"]["sha256"] = _write(receipt_path, receipt)
    if field in ("counts", "seed", "protected"):
        _reseal_prepared(case)
    with pytest.raises(ValueError):
        evidence.authenticate_prepared_reference(case["value"], case["protocol"], case["release"])


def test_reused_cache_requires_fresh_current_stages_and_cpu_resources(monkeypatch, tmp_path):
    _module(monkeypatch)
    import submit_large_triangle as submitter
    assert submitter.required_stages("cache", reuse_cache=True) == ("cpu",)
    assert submitter.required_stages("segment", reuse_cache=True) == ("cpu", "cache", "benchmark")
    assert submitter.required_stages("benchmark", reuse_cache=True) == ("cpu", "cache")
    assert submitter.required_stages("benchmark") == ("cpu", "prepare")
    row = submitter.workflow("cache", tmp_path / "release", tmp_path, tmp_path / "campaign",
                             tmp_path / "binding.json", "a" * 64, "b" * 64, "c" * 64, run_id="001")[0]
    assert row["arguments"][row["arguments"].index("-q") + 1] == "hpc"
    assert "-gpu" not in row["arguments"]


def test_prior_inventory_reuse_authenticates_source_inputs_and_never_edits_history(tmp_path, monkeypatch):
    evidence = _module(monkeypatch)
    case = _inventory_fixture(tmp_path, evidence)
    before = {row["path"]: Path(row["path"]).read_bytes() for row in case["value"]["inventory_reference"].values()}
    result = evidence.authenticate_inventory_reference(case["value"], case["protocol"], case["release"])
    assert result["job_id"] == "29680603" and result["supported_unique"] == 4
    assert result["requested_counts"] == case["protocol"]["requested_counts"]
    assert result["protected_roles"] == case["report"]["protected_roles"]
    assert result["report_sha256"] == case["value"]["inventory_reference"]["report"]["sha256"]
    assert result["preprocessing_source_sha256"] == case["report"]["inventory_identity"]["source_file_sha256"]
    assert before == {path: Path(path).read_bytes() for path in before}


@pytest.mark.parametrize("changed", ["old_source", "old_archive", "current_preprocessing", "raw_input", "protected_manifest"])
def test_prior_inventory_rejects_changed_executable_or_input_bytes(tmp_path, monkeypatch, changed):
    evidence = _module(monkeypatch)
    case = _inventory_fixture(tmp_path, evidence)
    path = {"old_source": case["old_release"] / "src/latentmol/data.py",
            "old_archive": Path(str(case["old_release"]) + ".tar.gz"),
            "current_preprocessing": case["release"] / "src/latentmol/chemistry.py",
            "raw_input": Path(case["value"]["raw_input"]),
            "protected_manifest": Path(case["value"]["reserve_cache"]) / "manifest.json"}[changed]
    with path.open("ab") as stream:
        stream.write(b"\nchanged\n")
    with pytest.raises(ValueError):
        evidence.authenticate_inventory_reference(case["value"], case["protocol"], case["release"])


@pytest.mark.parametrize("bad", ["graph_arrays", "selected_counts", "seed", "schema", "counts", "protected_roles"])
def test_even_resealed_inventory_must_preserve_semantic_scope(tmp_path, monkeypatch, bad):
    evidence = _module(monkeypatch)
    case = _inventory_fixture(tmp_path, evidence)
    report = case["report"]
    if bad == "graph_arrays":
        report["provenance"]["graph_arrays_written"] = True
    elif bad == "selected_counts":
        report["provenance"]["selected_target_counts"] = {"train": 2}
    elif bad == "seed":
        report["inventory_identity"]["config"]["seed"] = 18
        report["inventory_identity"]["config_sha256"] = _json_hash(report["inventory_identity"]["config"])
    elif bad == "schema":
        report["spec"]["max_nodes"] = 49
    elif bad == "counts":
        report["counts"]["supported_unique"] = 5
    else:
        report["protected_roles"]["present_protected"]["train"] = 3
    _reseal_report(case)
    with pytest.raises(ValueError):
        evidence.authenticate_inventory_reference(case["value"], case["protocol"], case["release"])


@pytest.mark.parametrize("requested", [{"train": 1, "val": 1, "test": 1},
                                        {"train": 3, "val": 1, "test": 1},
                                        {"train": True, "val": 1, "test": 2}])
def test_counts_must_explicitly_use_all_inventoried_supported_unique_rows(tmp_path, monkeypatch, requested):
    evidence = _module(monkeypatch)
    case = _inventory_fixture(tmp_path, evidence)
    with pytest.raises(ValueError, match="Explicit requested counts"):
        evidence.authenticate_inventory_reference(case["value"], {"requested_counts": requested}, case["release"])


def test_inventory_reference_rejects_foreign_completion_job_and_escaping_reference(tmp_path, monkeypatch):
    evidence = _module(monkeypatch)
    case = _inventory_fixture(tmp_path, evidence)
    reference = case["value"]["inventory_reference"]
    completion_path = Path(reference["completion"]["path"])
    completion = json.loads(completion_path.read_text())
    completion["job_id"] = "another_job"
    reference["completion"]["sha256"] = _write(completion_path, completion)
    with pytest.raises(ValueError, match="Allocated stage completion evidence differs"):
        evidence.authenticate_inventory_reference(case["value"], case["protocol"], case["release"])
    reference["report"] = {"path": str(tmp_path / "outside.json"), "sha256": "a" * 64}
    with pytest.raises(ValueError, match="contained path/SHA pair"):
        evidence.authenticate_inventory_reference(case["value"], case["protocol"], case["release"])


def test_ambiguous_evidence_json_and_unhashed_identity_are_rejected(tmp_path, monkeypatch):
    evidence = _module(monkeypatch)
    receipts, identity, *_ = _stage_fixture(tmp_path)
    (receipts / "prepare-receipt.json").write_text('{"job_id":"12345","job_id":"54321"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate evidence JSON key"):
        evidence.require_submitted_stage(receipts, "prepare", identity)
    receipts, identity, *_ = _stage_fixture(tmp_path / "other")
    identity["manifest_sha256"] = "not-a-source-hash"
    with pytest.raises(ValueError, match="Invalid campaign identity SHA256"):
        evidence.require_submitted_stage(receipts, "prepare", identity)
