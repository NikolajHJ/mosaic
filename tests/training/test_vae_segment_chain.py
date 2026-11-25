import importlib.util
import json
from pathlib import Path

import pytest


path = Path(__file__).resolve().parents[2] / "scripts/hpc/vae_segment_chain.py"
spec = importlib.util.spec_from_file_location("vae_segment_chain", path)
chain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chain)


def test_partial_combined_epoch_resumes_before_sparse_benchmark(tmp_path, monkeypatch):
    initial = tmp_path / "initial"
    initial.mkdir()
    checkpoint = initial / "checkpoint-000000100.pt"
    checkpoint.touch()
    plan_path = tmp_path / "plan.json"
    plan = dict(initial_output=str(initial), run_root=str(tmp_path / "run"),
                combined=dict(dataset_sha256="combined", training_molecules=120,
                              benchmark_report="combined-benchmark.json", max_segments=5))
    cursor = dict(real_points=60, epoch=45, row=60, updates=100)
    report = dict(status="paused", stop_reason="allocation_budget_reached", dataset_sha256="combined",
                  training_molecules=120, segment_updates=5, **cursor)
    report["latest"] = dict(**cursor, checkpoint=str(checkpoint), sha256="a" * 64, completed_epoch=44)
    summary = initial / "summary.json"
    summary.write_text(json.dumps(report))
    monkeypatch.setattr(chain, "launch", lambda *args: args[2:])
    phase, segment, flags = chain.advance(plan_path, plan, "combined", 0)
    assert (phase, segment) == ("combined", 1)
    assert flags[-4:] == ["--resume", str(checkpoint), "--resume-sha256", "a" * 64]

    report.update(status="completed", stop_reason="epoch_target_reached")
    summary.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="whole epoch"):
        chain.advance(plan_path, plan, "combined", 0)

    final = dict(real_points=120, epoch=46, row=0, updates=105)
    report.update(final)
    report["latest"].update(final, completed_epoch=46)
    summary.write_text(json.dumps(report))
    assert chain.advance(plan_path, plan, "combined", 0) == ("benchmark", 0, ["--benchmark"])


def test_sparse_launch_requires_both_reports_for_the_same_recipe(tmp_path, monkeypatch):
    plan_path = tmp_path / "plan.json"
    plan = dict(run_root=str(tmp_path), timing_study="study.json",
                sparse=dict(parent_epoch=46, dataset_sha256="data", training_molecules=120,
                            config_sha256="recipe"))
    for stage in ("benchmark-00", "timing-00"):
        (tmp_path / stage).mkdir()
    report = dict(dataset_sha256="data", training_molecules=120, config_sha256="recipe")
    (tmp_path / "benchmark-00/benchmark.json").write_text(json.dumps(dict(report, status="passed")))
    timing = tmp_path / "timing-00/timing.json"
    timing.write_text(json.dumps(dict(report, status="completed", config_sha256="other-recipe")))
    monkeypatch.setattr(chain, "launch", lambda *args: args[2:])
    with pytest.raises(ValueError, match="Both A100 benchmarks"):
        chain.advance(plan_path, plan, "benchmark", 0)
    timing.write_text(json.dumps(dict(report, status="completed")))
    assert chain.advance(plan_path, plan, "benchmark", 0)[:2] == ("sparse", 1)


def test_forked_segment_uses_its_saved_batch_shape(tmp_path, monkeypatch):
    directory = tmp_path / "sparse-01"
    directory.mkdir()
    checkpoint = directory / "checkpoint-000000100.pt"
    checkpoint.touch()
    shape = dict(batch_size=240, microbatch_size=120)
    plan = dict(run_root=str(tmp_path), sparse=dict(parent_epoch=46, dataset_sha256="dataset",
                training_molecules=6000, max_segments=4, resume_shape=shape))
    cursor = dict(real_points=2400, epoch=46, row=2400, updates=100)
    summary = dict(status="paused", stop_reason="allocation_budget_reached",
                   dataset_sha256="dataset", training_molecules=6000, segment_updates=10,
                   latest=dict(**cursor, checkpoint=str(checkpoint), sha256="a" * 64), **shape, **cursor)
    path = directory / "summary.json"
    path.write_text(json.dumps(summary))
    monkeypatch.setattr(chain, "launch", lambda *args: args[2:])
    phase, segment, flags = chain.advance(tmp_path / "plan.json", plan, "sparse", 1)
    assert (phase, segment) == ("sparse", 2)
    assert flags[:4] == ["--batch-size", "240", "--microbatch-size", "120"]
    summary["microbatch_size"] = 240
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="pinned resume batch shape"):
        chain.advance(tmp_path / "plan.json", plan, "sparse", 1)


def test_handoff_blocks_on_predecessor_launch_output_or_changed_cursor(tmp_path):
    latest, receipt, output = (tmp_path / name for name in ("latest.json", "launch.json", "old-output"))
    cursor = dict(checkpoint="old.pt", updates=100)
    latest.write_text(json.dumps(cursor))
    plan = dict(superseded_segment=dict(latest_path=str(latest), latest=cursor,
                launch_receipt=str(receipt), output=str(output)))
    assert chain.verify_handoff(plan) == {"verified_checkpoint": "old.pt"}
    latest.write_text(json.dumps(dict(cursor, updates=101)))
    with pytest.raises(RuntimeError, match="cursor changed"):
        chain.verify_handoff(plan)
    latest.write_text(json.dumps(cursor))
    receipt.touch()
    with pytest.raises(RuntimeError, match="launched"):
        chain.verify_handoff(plan)
    receipt.unlink()
    output.mkdir()
    with pytest.raises(RuntimeError, match="created outputs"):
        chain.verify_handoff(plan)
