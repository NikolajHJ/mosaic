"""HPC-only storage tests: retain scientific bytes, quotas and legacy evidence.

The paired-evaluator fixture is numerical and must run in a DTU allocation.
The remaining fixtures exercise filesystem failure paths with temporary files.
None authorize pruning a historical experiment or replacing failed evidence.
"""
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from latentmol.decoder_checkpoint_storage import (
    archive_panel_rows, materialized_panel_rows, prune_owned_checkpoints,
)
from latentmol.experiments import file_sha256


def _panel_files(tmp_path):
    directory = tmp_path / "panel"
    directory.mkdir()
    # Preserve exact UTF-8 and newline bytes, including an unterminated last row.
    raw = '{"label":"molécula","valid":false}\n{"valid":true}'.encode("utf-8")
    (directory / "rows.jsonl").write_bytes(raw)
    metrics = {"artifacts": {"rows.jsonl": {"sha256": hashlib.sha256(raw).hexdigest()}}}
    (directory / "metrics.json").write_text(json.dumps(metrics))
    (directory / "panel.json").write_bytes(b'{"historical_manifest":true}\n')
    return directory, raw


def _rewrite_receipt(directory, **changes):
    path = directory / "row-storage.json"
    record = json.loads(path.read_text())
    record.update(changes)
    path.write_text(json.dumps(record))


def _checkpoints(tmp_path):
    run = tmp_path / "new-run"
    run.mkdir()
    originals = {}
    for epoch in list(range(1, 9)) + [10]:
        path = run / f"epoch-{epoch:03d}.pt"
        originals[path.name] = f"checkpoint-{epoch}".encode()
        path.write_bytes(originals[path.name])
    return run, originals


def test_lossless_archive_preserves_metrics_manifest_and_all_raw_bytes(tmp_path):
    directory, raw = _panel_files(tmp_path)
    unchanged = {name: (directory / name).read_bytes() for name in ("metrics.json", "panel.json")}
    record = archive_panel_rows(directory)
    assert not (directory / "rows.jsonl").exists()
    assert gzip.decompress((directory / "rows.jsonl.gz").read_bytes()) == raw
    assert record["raw_sha256"] == hashlib.sha256(raw).hexdigest()
    assert record["compressed_sha256"] == file_sha256(directory / "rows.jsonl.gz")
    assert record["raw_bytes"] == len(raw)
    assert record["compressed_bytes"] == (directory / "rows.jsonl.gz").stat().st_size
    assert json.loads((directory / "row-storage.json").read_text()) == record
    with materialized_panel_rows(directory) as restored:
        assert restored == directory.resolve()
        assert (restored / "rows.jsonl").read_bytes() == raw
        # Nested serial consumers must not remove the outer consumer's copy.
        with materialized_panel_rows(directory):
            assert (directory / "rows.jsonl").read_bytes() == raw
        assert (directory / "rows.jsonl").read_bytes() == raw
    assert not (directory / "rows.jsonl").exists()
    assert all((directory / name).read_bytes() == content for name, content in unchanged.items())


@pytest.mark.parametrize("archived", [False, True])
def test_context_exception_removes_only_its_own_materialized_copy(tmp_path, archived):
    directory, raw = _panel_files(tmp_path)
    if archived:
        archive_panel_rows(directory)
    with pytest.raises(RuntimeError, match="consumer failed"):
        with materialized_panel_rows(directory):
            assert (directory / "rows.jsonl").read_bytes() == raw
            raise RuntimeError("consumer failed")
    if archived:
        assert not (directory / "rows.jsonl").exists()
        assert (directory / "rows.jsonl.gz").is_file()
    else:
        assert (directory / "rows.jsonl").read_bytes() == raw
        assert not (directory / "row-storage.json").exists()


def test_changed_raw_or_existing_archive_never_removes_raw(tmp_path):
    directory, raw = _panel_files(tmp_path)
    (directory / "rows.jsonl").write_bytes(raw + b"\n{}\n")
    with pytest.raises(ValueError, match="Raw panel rows differ"):
        archive_panel_rows(directory)
    assert not (directory / "rows.jsonl.gz").exists()
    (directory / "rows.jsonl").write_bytes(raw)
    (directory / "rows.jsonl.gz").write_bytes(b"previous artifact")
    with pytest.raises(FileExistsError):
        archive_panel_rows(directory)
    assert (directory / "rows.jsonl").read_bytes() == raw
    assert (directory / "rows.jsonl.gz").read_bytes() == b"previous artifact"


def test_failed_archive_receipt_write_keeps_raw_and_verified_gzip(tmp_path, monkeypatch):
    directory, raw = _panel_files(tmp_path)
    def quota_failure(*args, **kwargs):
        raise OSError(122, "Disk quota exceeded")
    monkeypatch.setattr("latentmol.decoder_checkpoint_storage.write_json", quota_failure)
    with pytest.raises(OSError, match="quota"):
        archive_panel_rows(directory)
    assert (directory / "rows.jsonl").read_bytes() == raw
    assert gzip.decompress((directory / "rows.jsonl.gz").read_bytes()) == raw
    assert not (directory / "row-storage.json").exists()


@pytest.mark.parametrize("changes", [
    {"version": 2}, {"raw": "../rows.jsonl"},
    {"compressed": "../rows.jsonl.gz"}, {"raw_sha256": "0" * 64},
])
def test_materialization_rejects_identity_or_path_drift_before_creating_raw(tmp_path, changes):
    directory, _ = _panel_files(tmp_path)
    archive_panel_rows(directory)
    _rewrite_receipt(directory, **changes)
    with pytest.raises(ValueError, match="identity/path"):
        with materialized_panel_rows(directory):
            pytest.fail("Corrupt receipt must not reach the consumer")
    assert not (directory / "rows.jsonl").exists()


@pytest.mark.parametrize("rewrite_compressed_identity", [False, True])
def test_changed_compressed_or_decompressed_payload_is_rejected_and_cleaned(tmp_path, rewrite_compressed_identity):
    directory, _ = _panel_files(tmp_path)
    archive_panel_rows(directory)
    zipped = directory / "rows.jsonl.gz"
    zipped.write_bytes(gzip.compress(b"different raw rows"))
    if rewrite_compressed_identity:
        _rewrite_receipt(directory, compressed_sha256=file_sha256(zipped))
    message = "Restored panel differs" if rewrite_compressed_identity else "Archived panel bytes changed"
    with pytest.raises(ValueError, match=message):
        with materialized_panel_rows(directory):
            pytest.fail("Corrupt rows must not reach the consumer")
    assert not (directory / "rows.jsonl").exists()


@pytest.mark.parametrize("artifact", ["rows.jsonl", "rows.jsonl.gz", "row-storage.json"])
def test_archive_rejects_symlink_artifacts_without_touching_external_bytes(tmp_path, artifact):
    directory, raw = _panel_files(tmp_path)
    outside = tmp_path / "historical-artifact"
    outside.write_bytes(raw if artifact == "rows.jsonl" else b"historical bytes")
    path = directory / artifact
    if path.exists():
        path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="symlinks"):
        archive_panel_rows(directory)
    assert outside.read_bytes() == (raw if artifact == "rows.jsonl" else b"historical bytes")
    assert path.is_symlink()


def test_retention_keeps_milestones_latest_two_future_and_historical_sibling(tmp_path):
    run, originals = _checkpoints(tmp_path)
    parent = tmp_path / "historical-run"
    parent.mkdir()
    (parent / "epoch-001.pt").write_bytes(b"protected original")
    (run / "run.json").write_bytes(b"unchanged metadata")
    deleted = prune_owned_checkpoints(run, 8, [2, 5], 2)
    assert {path.name for path in run.glob("epoch-*.pt")} == {
        "epoch-002.pt", "epoch-005.pt", "epoch-007.pt", "epoch-008.pt", "epoch-010.pt",
    }
    assert {item["file"] for item in deleted} == {"epoch-001.pt", "epoch-003.pt", "epoch-004.pt", "epoch-006.pt"}
    for item in deleted:
        assert item["sha256"] == hashlib.sha256(originals[item["file"]]).hexdigest()
        assert item["bytes"] == len(originals[item["file"]])
    journal = [json.loads(line) for line in (run / "checkpoint-retention.jsonl").read_text().splitlines()]
    assert journal[0]["planned_deletions"] == deleted
    assert journal[0]["keep_recent"] == 2 and journal[0]["keep_milestones"] == [2, 5]
    assert journal[1]["deleted"] == deleted
    assert (parent / "epoch-001.pt").read_bytes() == b"protected original"
    assert (run / "run.json").read_bytes() == b"unchanged metadata"


@pytest.mark.parametrize("unsafe", ["run_symlink", "late_checkpoint_symlink", "journal_symlink", "unexpected_name"])
def test_retention_prevalidates_all_paths_before_any_deletion(tmp_path, unsafe):
    run, originals = _checkpoints(tmp_path)
    protected = tmp_path / "old-checkpoint"
    protected.write_bytes(b"old protected bytes")
    destination = run
    if unsafe == "run_symlink":
        destination = tmp_path / "alias"
        destination.symlink_to(run, target_is_directory=True)
    elif unsafe == "late_checkpoint_symlink":
        (run / "epoch-999.pt").symlink_to(protected)
    elif unsafe == "journal_symlink":
        (run / "checkpoint-retention.jsonl").symlink_to(protected)
    else:
        (run / "epoch-zbad.pt").write_bytes(b"unexpected")
    with pytest.raises(ValueError):
        prune_owned_checkpoints(destination, 8, [2, 5], 2)
    assert all((run / name).read_bytes() == content for name, content in originals.items())
    assert protected.read_bytes() == b"old protected bytes"


def test_quota_error_on_deletion_receipt_preserves_every_checkpoint(tmp_path, monkeypatch):
    run, originals = _checkpoints(tmp_path)
    original_open = Path.open
    def guarded_open(path, mode="r", *args, **kwargs):
        if path.name == "checkpoint-retention.jsonl" and mode == "a":
            raise OSError(122, "Disk quota exceeded")
        return original_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(OSError, match="quota"):
        prune_owned_checkpoints(run, 8, [2, 5], 2)
    assert all((run / name).read_bytes() == content for name, content in originals.items())


def test_partial_unlink_failure_retains_planned_sha_evidence_for_all_candidates(tmp_path, monkeypatch):
    run, originals = _checkpoints(tmp_path)
    original_unlink = Path.unlink
    def fail_second(path, *args, **kwargs):
        if path.name == "epoch-003.pt":
            raise OSError("fixture unlink failure")
        return original_unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", fail_second)
    with pytest.raises(OSError, match="unlink failure"):
        prune_owned_checkpoints(run, 8, [2, 5], 2)
    journal = [json.loads(line) for line in (run / "checkpoint-retention.jsonl").read_text().splitlines()]
    assert len(journal) == 1  # A plan is not an invented completion observation.
    planned = journal[0]["planned_deletions"]
    assert {item["file"] for item in planned} == {"epoch-001.pt", "epoch-003.pt", "epoch-004.pt", "epoch-006.pt"}
    assert all(item["sha256"] == hashlib.sha256(originals[item["file"]]).hexdigest() for item in planned)
    assert not (run / "epoch-001.pt").exists()
    assert (run / "epoch-003.pt").read_bytes() == originals["epoch-003.pt"]


@pytest.mark.parametrize("change", ["zero_epoch", "bool_epoch", "one_recent", "zero_recent",
                                    "bool_recent", "nonpositive_milestone", "bool_milestone", "duplicate_milestone"])
def test_retention_rejects_invalid_reusable_policy_before_mutation(tmp_path, change):
    run, originals = _checkpoints(tmp_path)
    epoch, milestones, recent = 8, [2, 5], 2
    if change == "zero_epoch": epoch = 0
    elif change == "bool_epoch": epoch = True
    elif change == "one_recent": recent = 1
    elif change == "zero_recent": recent = 0
    elif change == "bool_recent": recent = True
    elif change == "nonpositive_milestone": milestones = [0, 5]
    elif change == "bool_milestone": milestones = [True, 5]
    else: milestones = [2, 2]
    with pytest.raises(ValueError, match="Retention needs"):
        prune_owned_checkpoints(run, epoch, milestones, recent)
    assert all((run / name).read_bytes() == content for name, content in originals.items())
    assert not (run / "checkpoint-retention.jsonl").exists()


def _prefix_fixture(tmp_path):
    """Byte-authenticated metadata, deliberately not a torch model checkpoint."""
    campaign = tmp_path / "historical-campaign"
    case = campaign / "seed17-a0"
    run = case / "training" / "historical-run"
    run.mkdir(parents=True)
    checkpoint = run / "epoch-010.pt"
    checkpoint.write_bytes(b"synthetic completed-prefix identity")
    metrics = run / "metrics.jsonl"
    metrics.write_text("".join(json.dumps({"epoch": epoch}) + "\n" for epoch in range(1, 11)))
    source = "a" * 64
    record = {"status": "completed", "completed_epochs": 10, "source_archive_sha256": source}
    (run / "run.json").write_text(json.dumps(record))
    fit = {"run_dir": str(run), "checkpoint": str(checkpoint), "sha256": file_sha256(checkpoint)}
    summary = {"status": "completed", "fit": fit}
    (case / "summary.json").write_text(json.dumps(summary))
    protocol = {"base_campaign_root": str(campaign), "base_archive_sha256": source, "endpoint_epochs": 100}
    return protocol, case, run, fit, summary, record


@pytest.mark.parametrize("state", ["missing", "failed_without_fit", "incomplete_training"])
def test_missing_or_failed_prefix_restarts_original_without_using_partial_state(tmp_path, state):
    from latentmol.decoder_density_long_study import initial_prefix
    protocol, case, run, fit, summary, record = _prefix_fixture(tmp_path)
    if state == "missing":
        (case / "summary.json").unlink()
    else:
        summary["status"] = "failed_or_time_censored"
        if state == "failed_without_fit":
            summary["fit"] = None
        else:
            record.update(status="failed", completed_epochs=7)
            (run / "run.json").write_text(json.dumps(record))
        (case / "summary.json").write_text(json.dumps(summary))
    originals = {name: (run / name).read_bytes() for name in ("epoch-010.pt", "run.json", "metrics.jsonl")}
    envelope, metrics, extension, lineage = initial_prefix(protocol, "seed17-a0")
    assert envelope is metrics is extension is None
    assert lineage["actual_prefix_epochs"] == 0
    assert "restart original" in lineage["reason"]
    assert all((run / name).read_bytes() == content for name, content in originals.items())
    if state != "missing":
        assert lineage["parent_summary_sha256"] == file_sha256(case / "summary.json")


@pytest.mark.parametrize("evaluation_failed", [False, True])
def test_completed_ten_epoch_prefix_authenticates_bytes_even_if_evaluation_failed(tmp_path, evaluation_failed):
    from latentmol.decoder_density_long_study import initial_prefix
    protocol, case, run, fit, summary, _ = _prefix_fixture(tmp_path)
    if evaluation_failed:
        summary.update(status="failed_or_time_censored", failure_phase="paired_comparison")
        (case / "summary.json").write_text(json.dumps(summary))
    envelope, metrics, extension, lineage = initial_prefix(protocol, "seed17-a0")
    assert envelope["checkpoint"] == fit["checkpoint"]
    assert envelope["checkpoint_sha256"] == file_sha256(run / "epoch-010.pt")
    assert envelope["parent_record_sha256"] == file_sha256(run / "run.json")
    assert envelope["parent_metrics_sha256"] == file_sha256(run / "metrics.jsonl")
    assert envelope["parent_archive_sha256"] == protocol["base_archive_sha256"]
    assert metrics == run / "metrics.jsonl"
    assert extension == {"from_epochs": 10, "to_epochs": 100}
    assert lineage["actual_prefix_epochs"] == 10


@pytest.mark.parametrize("corruption", ["checkpoint", "source", "epoch_gap", "completed_record_conflict", "missing_fit"])
def test_declared_completed_prefix_corruption_fails_closed_without_original_fallback(tmp_path, corruption):
    from latentmol.decoder_density_long_study import initial_prefix
    protocol, case, run, fit, summary, record = _prefix_fixture(tmp_path)
    if corruption == "checkpoint":
        (run / "epoch-010.pt").write_bytes(b"changed after checkpoint receipt")
    elif corruption == "source":
        record["source_archive_sha256"] = "b" * 64
        (run / "run.json").write_text(json.dumps(record))
    elif corruption == "epoch_gap":
        (run / "metrics.jsonl").write_text("".join(json.dumps({"epoch": epoch}) + "\n" for epoch in range(2, 11)))
    elif corruption == "completed_record_conflict":
        record.update(status="failed", completed_epochs=7)
        (run / "run.json").write_text(json.dumps(record))
    else:
        summary["fit"] = None
        (case / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        initial_prefix(protocol, "seed17-a0")


def test_real_paired_evaluator_has_identical_results_after_lossless_materialization(tmp_path):
    # Reuse the frozen finite fixture; this is an engineering parity check,
    # not a new training experiment or evidence of chemistry improvement.
    from test_decoder_quality_evaluation import _DecoderFixture, _panel
    from latentmol.decoder_quality_evaluation import evaluate_random_panel, paired_compare
    reference, candidate = tmp_path / "original", tmp_path / "new-panel"
    evaluate_random_panel(_DecoderFixture(), _panel(), set(), reference, repair_limit=0)
    metrics = evaluate_random_panel(_DecoderFixture(always_valid=True), _panel(), set(), candidate, repair_limit=0)
    original_bytes = (reference / "rows.jsonl").read_bytes()
    candidate_bytes = (candidate / "rows.jsonl").read_bytes()
    assert metrics["all_attempts"]["attempted"] == 20
    assert len(candidate_bytes.splitlines()) == 20
    saved_metrics = (candidate / "metrics.json").read_bytes()
    saved_manifest = (candidate / "panel.json").read_bytes()
    before = paired_compare(reference, candidate, bootstrap_seed=47, bootstrap_draws=40)
    archive_panel_rows(candidate)
    with materialized_panel_rows(reference), materialized_panel_rows(candidate):
        assert (candidate / "rows.jsonl").read_bytes() == candidate_bytes
        after = paired_compare(reference, candidate, bootstrap_seed=47, bootstrap_draws=40)
    assert after == before
    assert (reference / "rows.jsonl").read_bytes() == original_bytes
    assert not (candidate / "rows.jsonl").exists()
    assert (candidate / "metrics.json").read_bytes() == saved_metrics
    assert (candidate / "panel.json").read_bytes() == saved_manifest
