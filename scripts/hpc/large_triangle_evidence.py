"""Metadata-only gates shared by large-corpus launchers and allocated stages.

These checks authenticate durable allocated receipts, never infer success from
missing scheduler history. They do not call LSF, import scientific libraries,
read graph tensors, assign splits or modify historical evidence. Scheduler
state/dependency checks remain the submitter's responsibility.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shlex

from latentmol.experiments import file_sha256, source_manifest


PREPROCESSING_FILES = ("large_data.py", "data.py", "chemistry.py", "representation.py")
IDENTITY_FIELDS = ("release", "campaign", "archive_sha256", "manifest_sha256", "bindings_sha256")


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate evidence JSON key: " + key)
        result[key] = value
    return result


def _read(path, expected=None):
    """Hash the same bytes that are parsed, rejecting ambiguous/nonfinite JSON."""
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected is not None and (not _hash(expected) or digest != expected):
        raise ValueError("Evidence file SHA256 mismatch: " + str(path))
    def invalid_constant(value):
        raise ValueError("Nonfinite evidence JSON constant: " + value)
    value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError("Evidence must be a JSON object: " + str(path))
    return value, digest


def _identity(identity):
    if not isinstance(identity, dict) or any(key not in identity for key in IDENTITY_FIELDS):
        raise ValueError("Require complete immutable campaign submission identity")
    for key in ("archive_sha256", "manifest_sha256", "bindings_sha256"):
        if not _hash(identity[key]):
            raise ValueError("Invalid campaign identity SHA256: " + key)
    for key in ("release", "campaign"):
        if not isinstance(identity[key], str) or not Path(identity[key]).is_absolute():
            raise ValueError("Require absolute campaign identity path: " + key)


def _stage(stage):
    if not isinstance(stage, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", stage):
        raise ValueError("Require a simple stage receipt name")


def _submitted(receipt, intent, identity, stage):
    """A pending receipt must identify the intended stage, not just campaign."""
    _stage(stage)
    _identity(identity)
    if (receipt.get("status") != "submitted_not_passed" or receipt.get("intent") != intent
            or intent.get("identity") != identity):
        raise ValueError("Stage submission receipt/intent differs from campaign")
    job = receipt.get("job_id")
    if not isinstance(job, str) or not re.fullmatch(r"[0-9]+", job):
        raise ValueError("Need an authenticated numeric stage job ID")
    command = intent.get("command")
    if (intent.get("cwd") != identity["release"] or not isinstance(command, list)
            or not command or command[0] != "bsub" or any(not isinstance(v, str) for v in command)
            or command.count("-J") != 1):
        raise ValueError("Stage intent lacks its actual structured bsub command/cwd")
    label_index = command.index("-J") + 1
    if label_index >= len(command) or command[label_index] != Path(identity["release"]).name + "-" + stage:
        raise ValueError("Submission command names a different stage")
    tokens = shlex.split(command[-1])
    script = "scripts/hpc/large_triangle_stage.py"
    expected_stage = stage.split("-", 1)[0] if stage.startswith(("segment-", "recover-", "final-")) else stage
    if (tokens.count(script) != 1 or tokens.index(script) + 1 >= len(tokens)
            or tokens[tokens.index(script) + 1] != expected_stage):
        raise ValueError("Submission command executes a different stage")
    for flag, expected in (("--archive-sha256", identity["archive_sha256"]),
                           ("--manifest-sha256", identity["manifest_sha256"]),
                           ("--bindings-sha256", identity["bindings_sha256"])):
        if tokens.count(flag) != 1 or tokens.index(flag) + 1 >= len(tokens) or tokens[tokens.index(flag) + 1] != expected:
            raise ValueError("Submission command differs from immutable identity: " + flag)
    return job


def _completion(completion, stage, identity, job):
    if (completion.get("status") != "passed" or type(completion.get("stage_returncode")) is not int
            or completion["stage_returncode"] != 0 or completion.get("job_id") != job
            or completion.get("stage") != stage or completion.get("bindings_sha256") != identity["bindings_sha256"]
            or completion.get("source_sha256") != identity["manifest_sha256"]
            or completion.get("source_archive_sha256") != identity["archive_sha256"]):
        raise ValueError("Allocated stage completion evidence differs")
    artifacts = completion.get("artifacts")
    if not isinstance(artifacts, dict) or (stage != "cpu" and not artifacts):
        raise ValueError("Allocated stage completion lacks its artifact hashes")
    for path, expected in artifacts.items():
        if not isinstance(path, str) or not Path(path).is_absolute() or not _hash(expected) or file_sha256(Path(path)) != expected:
            raise ValueError("Completion artifact SHA256 mismatch: " + str(path))


def require_submitted_stage(receipts, stage, identity):
    """Return ``(job_id, receipt_sha256)`` after exact submission checks.

    This proves submission identity only. It does not establish pending/running
    scheduler state or successful execution and cannot replace a completion gate.
    """
    _stage(stage)
    receipts = Path(receipts)
    receipt, digest = _read(receipts / (stage + "-receipt.json"))
    intent, _ = _read(receipts / (stage + "-intent.json"))
    return _submitted(receipt, intent, identity, stage), digest


def require_stage_completion(receipts, stage, identity):
    """Return job ID only after durable execution and artifact authentication."""
    job, _ = require_submitted_stage(receipts, stage, identity)
    completion, _ = _read(Path(identity["campaign"]) / "stage-completions" / (stage + ".json"))
    _completion(completion, stage, identity, job)
    return job


def _reference(reference, name, work):
    row = reference.get(name)
    if (not isinstance(row, dict) or set(row) != {"path", "sha256"}
            or not isinstance(row["path"], str) or not Path(row["path"]).is_absolute()
            or not Path(row["path"]).resolve().is_relative_to(work)):
        raise ValueError("Inventory reference needs a contained path/SHA pair: " + name)
    path = Path(row["path"]).resolve(strict=True)
    value, digest = _read(path, row["sha256"])
    return path, value, digest


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def authenticate_prepared_reference(value, protocol, release):
    """Authenticate a prior prepared cache without borrowing its CPU gate.

    A current allocated ``cache`` stage still validates every array/file and
    writes its own receipt. Only the existing dataset is reused: benchmark
    workers, reference evaluation and training gates belong to the new source.
    No old completion/source hash is rewritten to impersonate a new run.
    """
    reference = value.get("prepared_reference")
    names = {"manifest", "completion", "receipt", "intent", "bindings"}
    if not isinstance(reference, dict) or set(reference) != names:
        raise ValueError("Require all five authenticated prepared-cache reference assets")
    work = Path(value["work_root"]).resolve()
    release = Path(release).resolve(strict=True)
    if not release.is_relative_to(work / "releases"):
        raise ValueError("Current release must remain inside the work root")
    assets = {name: _reference(reference, name, work) for name in names}
    manifest_path, manifest, manifest_sha = assets["manifest"]
    _, receipt, _ = assets["receipt"]
    _, intent, _ = assets["intent"]
    old_identity = intent.get("identity")
    _identity(old_identity)
    old_release = Path(old_identity["release"]).resolve(strict=True)
    old_campaign = Path(old_identity["campaign"]).resolve(strict=True)
    if not old_release.is_relative_to(work / "releases") or not old_campaign.is_relative_to(work):
        raise ValueError("Historical preparation escaped the work root")
    if (assets["receipt"][0] != old_campaign / "receipts/prepare-receipt.json"
            or assets["intent"][0] != old_campaign / "receipts/prepare-intent.json"
            or assets["completion"][0] != old_campaign / "stage-completions/prepare.json"
            or manifest_path != Path(value["data"]).resolve(strict=True) / "manifest.json"):
        raise ValueError("Prepared reference assets differ from their actual stage/cache")
    job = _submitted(receipt, intent, old_identity, "prepare")
    completion = assets["completion"][1]
    _completion(completion, "prepare", old_identity, job)
    if completion["artifacts"].get(str(manifest_path)) != manifest_sha:
        raise ValueError("Preparation did not certify the referenced cache manifest")
    old_binding = assets["bindings"][1]
    command_tokens = shlex.split(intent["command"][-1])
    binding_flag = command_tokens.index("--bindings") if "--bindings" in command_tokens else -1
    if (assets["bindings"][2] != old_identity["bindings_sha256"] or old_binding.get("campaign") != str(old_campaign)
            or old_binding.get("version") != 1 or old_binding.get("work_root") != str(work)
            or command_tokens.count("--bindings") != 1 or binding_flag + 1 >= len(command_tokens)
            or command_tokens[binding_flag + 1] != str(assets["bindings"][0])):
        raise ValueError("Prepared reference binding differs from submitted identity")
    actual = source_manifest(old_release)
    sealed, _ = _read(old_release / "release_manifest.json")
    if (actual["sha256"] != old_identity["manifest_sha256"] or sealed.get("sha256") != actual["sha256"]
            or sealed.get("files") != actual["files"]
            or file_sha256(Path(str(old_release) + ".tar.gz")) != old_identity["archive_sha256"]):
        raise ValueError("Historical prepared source/archive changed")
    for name in ("raw_input", "raw_input_sha256", "reserve_cache", "reserve_manifest_sha256", "smiles_column", "data"):
        if old_binding.get(name) != value.get(name):
            raise ValueError("Current input differs from authenticated preparation: " + name)
    if file_sha256(Path(value["raw_input"])) != value["raw_input_sha256"]:
        raise ValueError("Raw preparation input changed")
    protected, _ = _read(Path(value["reserve_cache"]) / "manifest.json", value["reserve_manifest_sha256"])
    requested = protocol.get("requested_counts")
    if (not isinstance(requested, dict) or set(requested) != {"train", "val", "test"}
            or any(type(requested[name]) is not int or requested[name] < 1 for name in requested)
            or any(manifest.get("counts", {}).get(name) != requested[name] for name in requested)
            or manifest.get("counts", {}).get("total") != sum(requested.values())):
        raise ValueError("Prepared cache must match exact requested split counts")
    expected_spec = {"version": 1, "max_nodes": 50,
                     "atomic_numbers": [0, 6, 7, 8, 9, 15, 16, 17, 35, 53],
                     "formal_charges": [-2, -1, 0, 1, 2],
                     "bond_names": ["NONE", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]}
    spec_sha = hashlib.sha256(json.dumps(expected_spec, sort_keys=True).encode()).hexdigest()
    preparation = manifest.get("preparation_identity", {})
    config = {"spec_fingerprint": spec_sha, "seed": 17, "smiles_column": value["smiles_column"],
              "train_count": requested["train"], "val_count": requested["val"],
              "test_count": requested["test"], "protected_dataset_sha256": protected["dataset_sha256"]}
    provenance = manifest.get("provenance", {})
    reserved = provenance.get("reserved_cache", {})
    if (manifest.get("dataset_format_version") != 2 or manifest.get("spec") != expected_spec
            or manifest.get("spec_fingerprint") != spec_sha or protected.get("spec_fingerprint") != spec_sha
            or manifest.get("raw_sha256") != value["raw_input_sha256"]
            or preparation.get("config") != config or preparation.get("config_sha256") != _json_hash(config)
            or provenance.get("source_path") != value["raw_input"]
            or provenance.get("smiles_column") != value["smiles_column"] or provenance.get("seed") != 17
            or reserved.get("path") != value["reserve_cache"]
            or reserved.get("manifest_sha256") != value["reserve_manifest_sha256"]
            or reserved.get("dataset_sha256") != protected["dataset_sha256"]
            or reserved.get("split_sha256") != protected["split_sha256"]):
        raise ValueError("Prepared cache schema/config/protected provenance differs")
    current = {name: file_sha256(release / "src/latentmol" / name) for name in PREPROCESSING_FILES}
    if (preparation.get("source_file_sha256") != current
            or any(actual["files"].get("src/latentmol/" + name) != digest for name, digest in current.items())):
        raise ValueError("Preprocessing executable bytes differ from preparation")
    for key in ("dataset_sha256", "split_sha256"):
        if not _hash(manifest.get(key)):
            raise ValueError("Prepared cache lacks its canonical identity: " + key)
    return {"preparation_job_id": job, "manifest_sha256": manifest_sha,
            "dataset_sha256": manifest["dataset_sha256"], "split_sha256": manifest["split_sha256"],
            "counts": dict(requested), "preparation_source_sha256": actual["sha256"],
            "preprocessing_source_sha256": current,
            "scope": "Prior preparation authenticated; current allocated cache validation still required"}


def authenticate_inventory_reference(value, protocol, release):
    """Reuse only an allocated inventory with byte-identical preprocessing.

    ``inventory_reference`` has five ``{path, sha256}`` entries named report,
    completion, receipt, intent and bindings. The older complete source/archive
    remains authenticated, while only the four actual preprocessing files must
    match the new release. Whole-campaign CPU receipts are never grandfathered.
    The report chooses no split and is not modified or converted into a cache.
    """
    reference = value.get("inventory_reference")
    if not isinstance(reference, dict) or set(reference) != {"report", "completion", "receipt", "intent", "bindings"}:
        raise ValueError("Require all five authenticated inventory reference assets")
    work = Path(value["work_root"]).resolve()
    release = Path(release).resolve(strict=True)
    if not release.is_relative_to(work / "releases"):
        raise ValueError("Current release must remain inside the work root")
    assets = {name: _reference(reference, name, work) for name in reference}
    report_path, report, report_sha = assets["report"]
    _, receipt, _ = assets["receipt"]
    _, intent, _ = assets["intent"]
    old_identity = intent.get("identity")
    _identity(old_identity)
    old_release = Path(old_identity["release"]).resolve(strict=True)
    old_campaign = Path(old_identity["campaign"]).resolve(strict=True)
    if not old_release.is_relative_to(work / "releases") or not old_campaign.is_relative_to(work):
        raise ValueError("Historical inventory source/campaign escaped the work root")
    if (assets["receipt"][0].name != "inventory-receipt.json"
            or assets["intent"][0] != assets["receipt"][0].with_name("inventory-intent.json")
            or not assets["receipt"][0].is_relative_to(old_campaign)
            or assets["completion"][0] != old_campaign / "stage-completions/inventory.json"
            or report_path != old_campaign / "inventory/inventory.json"):
        raise ValueError("Historical inventory assets differ from their actual campaign stage")
    job = _submitted(receipt, intent, old_identity, "inventory")
    completion = assets["completion"][1]
    _completion(completion, "inventory", old_identity, job)
    if completion["artifacts"].get(str(report_path)) != report_sha:
        raise ValueError("Historical completion does not authenticate the inventory report")
    if assets["bindings"][2] != old_identity["bindings_sha256"]:
        raise ValueError("Historical binding identity differs from its submission")
    old_binding = assets["bindings"][1]
    if (old_binding.get("version") != 1 or old_binding.get("campaign") != old_identity["campaign"]
            or Path(old_binding["work_root"]).resolve() != work):
        raise ValueError("Historical inventory binding scope differs")
    tokens = shlex.split(intent["command"][-1])
    if (tokens.count("--bindings") != 1 or tokens.index("--bindings") + 1 >= len(tokens)
            or Path(tokens[tokens.index("--bindings") + 1]).resolve() != assets["bindings"][0]):
        raise ValueError("Historical submission did not use the declared binding file")
    if file_sha256(Path(str(old_release) + ".tar.gz")) != old_identity["archive_sha256"]:
        raise ValueError("Historical inventory archive changed")
    manifest, _ = _read(old_release / "release_manifest.json")
    actual = source_manifest(old_release)
    if manifest.get("sha256") != old_identity["manifest_sha256"] or actual["sha256"] != old_identity["manifest_sha256"] or manifest.get("files") != actual["files"]:
        raise ValueError("Historical inventory executable source changed")
    inventory_sha = report.get("inventory_sha256")
    if (report.get("status") != "completed_inventory_only" or type(report.get("inventory_format_version")) is not int
            or report["inventory_format_version"] != 1 or not _hash(inventory_sha)
            or _json_hash({k: v for k, v in report.items() if k != "inventory_sha256"}) != inventory_sha
            or report.get("provenance", {}).get("graph_arrays_written") is not False
            or report["provenance"].get("selected_target_counts") is not None
            or not _hash(report.get("supported_identity_sha256"))):
        raise ValueError("Historical report is not an intact inventory-only result")
    provenance = report["provenance"]
    protected = provenance["reserved_cache"]
    inventory_identity = report["inventory_identity"]
    inventory_config = inventory_identity["config"]
    spec = report["spec"]
    spec_sha = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    # Pin the actual canonical schema, not just a caller-supplied fingerprint.
    expected_spec = {"version": 1, "max_nodes": 50,
                     "atomic_numbers": [0, 6, 7, 8, 9, 15, 16, 17, 35, 53],
                     "formal_charges": [-2, -1, 0, 1, 2],
                     "bond_names": ["NONE", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]}
    expected_spec_sha = hashlib.sha256(json.dumps(expected_spec, sort_keys=True).encode()).hexdigest()
    if (spec != expected_spec or spec_sha != expected_spec_sha or report.get("spec_fingerprint") != spec_sha
            or inventory_config != {"mode": "inventory", "spec_fingerprint": spec_sha, "seed": 17,
                                    "smiles_column": value["smiles_column"],
                                    "protected_dataset_sha256": protected["dataset_sha256"]}
            or inventory_identity.get("config_sha256") != _json_hash(inventory_config)):
        raise ValueError("Historical inventory schema/seed/config identity differs")
    for key in ("raw_input", "raw_input_sha256", "reserve_cache", "reserve_manifest_sha256", "smiles_column"):
        if old_binding.get(key) != value.get(key):
            raise ValueError("Current input differs from authenticated inventory: " + key)
    if (provenance["source_path"] != value["raw_input"] or report["raw_sha256"] != value["raw_input_sha256"]
            or provenance["smiles_column"] != value["smiles_column"] or protected["path"] != value["reserve_cache"]
            or protected["manifest_sha256"] != value["reserve_manifest_sha256"]):
        raise ValueError("Actually inventoried source/protected inputs differ")
    if file_sha256(Path(value["raw_input"])) != value["raw_input_sha256"]:
        raise ValueError("Current raw input changed since inventory")
    reserve_manifest, _ = _read(Path(value["reserve_cache"]) / "manifest.json", value["reserve_manifest_sha256"])
    if (reserve_manifest.get("dataset_sha256") != protected["dataset_sha256"]
            or reserve_manifest.get("split_sha256") != protected["split_sha256"]
            or reserve_manifest.get("spec_fingerprint") != spec_sha
            or reserve_manifest.get("rdkit_version") != report.get("rdkit_version")):
        raise ValueError("Protected cache identity/schema/runtime differs from inventory")
    source_hashes = inventory_identity.get("source_file_sha256")
    current_hashes = {name: file_sha256(release / "src/latentmol" / name) for name in PREPROCESSING_FILES}
    if (source_hashes != current_hashes or any(actual["files"].get("src/latentmol/" + name) != digest
                                             for name, digest in current_hashes.items())):
        raise ValueError("Preprocessing executable bytes differ from inventory")
    counts = report["counts"]
    for name in ("source_rows", "roundtrip_valid_rows", "duplicate_rows", "rejected_rows", "supported_unique"):
        if type(counts.get(name)) is not int or counts[name] < 0:
            raise ValueError("Inventory needs nonnegative integer counts")
    if (counts["roundtrip_valid_rows"] + counts["rejected_rows"] != counts["source_rows"]
            or counts["roundtrip_valid_rows"] - counts["duplicate_rows"] != counts["supported_unique"]):
        raise ValueError("Inventory supported counts are inconsistent")
    requested = protocol.get("requested_counts")
    if (not isinstance(requested, dict) or set(requested) != {"train", "val", "test"}
            or any(type(requested[name]) is not int or requested[name] < 1 for name in requested)
            or sum(requested.values()) != counts["supported_unique"]):
        raise ValueError("Explicit requested counts must use exactly the inventoried supported unique set")
    roles = report["protected_roles"]
    for name in requested:
        total, present, absent = (roles[key].get(name) for key in ("total_protected", "present_protected", "absent_protected"))
        if (any(type(v) is not int or v < 0 for v in (total, present, absent))
                or total != present + absent or total != reserve_manifest["counts"][name]
                or present > requested[name]):
            raise ValueError("Requested split cannot preserve inventoried protected roles")
    return {"job_id": job, "supported_unique": counts["supported_unique"],
            "requested_counts": dict(requested), "report_sha256": report_sha,
            "inventory_sha256": inventory_sha, "supported_identity_sha256": report["supported_identity_sha256"],
            "protected_roles": roles, "preprocessing_source_sha256": current_hashes,
            "scope": "Authenticated prior inventory only; no imported CPU gate, split assignment or cache validation"}
