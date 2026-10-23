"""Synthetic metadata fixtures, never proof of a real completed HPC segment."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/hpc"))
from failure_guidance_campaign import algorithm_from_manifest, healthy_handoff
from latentmol.experiments import file_sha256, write_json


def source_fixture(tmp_path):
    root = tmp_path / "source"; root.mkdir()
    release = tmp_path / "release"; release.mkdir()
    write_json(release / "release_manifest.json", dict(files={"src/latentmol/example.py": "e" * 64}))
    run = root / "segments/001/fixture"; run.mkdir(parents=True)
    plan = dict(parent_epoch=35, parent_updates=100, training_molecules=256, archive_sha256="a" * 64,
                manifest_sha256="b" * 64, parent_checkpoint_sha256="c" * 64)
    source = SimpleNamespace(root=root, release=release, path=tmp_path / "source-plan.json", sha="d" * 64, plan=plan,
                            receipt_job=lambda stage: "123", complete=lambda stage: "123")
    checkpoint = run / "epoch_0036.pt"; checkpoint.write_bytes(b"fixture bytes, not a model checkpoint")
    write_json(root / "benchmark_gate.json", dict(status="passed", gates=dict(example=True)))
    latest = dict(epoch=36, updates=101, real_points=256, fresh_prior_points=512, health=dict(example=True),
                  checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint))
    write_json(run / "latest_epoch.json", latest)
    (run / "metrics.jsonl").write_text(json.dumps(dict(latest, at="fixture")) + "\n")
    write_json(run / "run.json", dict(status="completed", environment=dict(job_id="123"),
        source_archive_sha256=plan["archive_sha256"], source=dict(sha256=plan["manifest_sha256"]),
        config=dict(_plan_sha256=source.sha), starting_epoch=35, resume_checkpoint_sha256=plan["parent_checkpoint_sha256"],
        benchmark_gate_sha256=file_sha256(root / "benchmark_gate.json")))
    completion = dict(status="passed", epoch_complete=True, plan_sha256=source.sha, segment=1,
        completed_epoch=36, updates=101, stop_reason="wall_budget_epoch_boundary", checkpoint=str(checkpoint),
        checkpoint_sha256=file_sha256(checkpoint), run_dir=str(run))
    write_json(run / "completion.json", completion)
    write_json(root / "segment-001-completion.json", completion)
    return source, run


def test_handoff_uses_only_the_complete_healthy_checkpoint(tmp_path):
    source, run = source_fixture(tmp_path)
    binding = healthy_handoff(source, 1, "123")
    assert binding["epoch"] == 36 and binding["updates"] == 101
    assert binding["source_plan_sha256"] == source.sha
    assert binding["algorithm"] == algorithm_from_manifest(source.release)


@pytest.mark.parametrize("change", ["active", "health", "checkpoint", "history", "completion"])
def test_partial_or_mismatched_handoff_evidence_stops(tmp_path, change):
    source, run = source_fixture(tmp_path)
    if change == "active":
        (source.root / "active-phase.json").write_text("{}")
    elif change == "health":
        value = json.loads((run / "metrics.jsonl").read_text()); value["health"] = dict(example=False)
        (run / "metrics.jsonl").write_text(json.dumps(value) + "\n")
    elif change == "checkpoint":
        (run / "epoch_0036.pt").write_bytes(b"changed")
    elif change == "history":
        (run / "metrics.jsonl").write_text("")
    else:
        value = json.loads((source.root / "segment-001-completion.json").read_text())
        value["status"] = "failed"; write_json(source.root / "segment-001-completion.json", value)
    with pytest.raises(ValueError):
        healthy_handoff(source, 1, "123")


def test_unknown_source_segment_or_job_cannot_be_rebound(tmp_path):
    source, _ = source_fixture(tmp_path)
    with pytest.raises(ValueError, match="declared"):
        healthy_handoff(source, 2, "123")
    with pytest.raises(ValueError, match="declared"):
        healthy_handoff(source, 1, "999")


@pytest.mark.parametrize("name", ["advance-after-segment-001-intent.json", "advance-after-segment-001.json",
                                 "receipts/segment-002-intent.json", "receipts/segment-002-receipt.json"])
def test_pending_successor_or_started_callback_blocks_rebinding(tmp_path, name):
    source, _ = source_fixture(tmp_path)
    path = source.root / name; path.parent.mkdir(exist_ok=True)
    path.write_text("{}")
    with pytest.raises(ValueError, match="already acted"):
        healthy_handoff(source, 1, "123")
