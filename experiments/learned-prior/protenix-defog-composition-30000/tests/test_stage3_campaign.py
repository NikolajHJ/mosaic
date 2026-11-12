"""End-to-end CPU execution, failure ownership and scheduler materialization."""

import csv
import gzip
import importlib
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
campaign = importlib.import_module("stage3_campaign")
design = importlib.import_module("stage3_design")
maths = importlib.import_module("stage3_math")
results = importlib.import_module("stage3_results")
dispatch = importlib.import_module("stage3_dispatch")
from stage3_native import mock_native


def make_plan(tmp_path, mock=True):
    path = tmp_path / "campaign"
    plan = campaign.prepare_plan(None, path, wave="reference", mock=mock)
    return path, plan


def test_reference_plan_has_full_paired_seed_blocks(tmp_path):
    path, plan = make_plan(tmp_path)
    assert len(plan["runs"]) == 18
    assert len(plan["bundles"]) == 9
    assert len(plan["conditions"]) == 2
    assert plan["mock"]
    assert {r["updates"] for r in plan["runs"]} == {500}
    with pytest.raises(FileExistsError):
        campaign.prepare_plan(None, path)


def test_mock_bundle_outputs_best_valid_and_tidy_tables(tmp_path):
    path, plan = make_plan(tmp_path)
    bid = plan["bundles"][0]["bundle_id"]
    record = campaign.run_bundle(path / "plan.json", bid, mock=True, mock_updates=3)
    assert record["status"] == "complete", record
    assert len(record["arms"]) == 2
    runs = [r for r in plan["runs"] if r["run_id"] in plan["bundles"][0]["run_ids"]]
    for config in runs:
        folder = path / "runs" / config["run_id"]
        status = json.loads((folder / "status.json").read_text())
        assert status["updates_completed"] == 3
        saved = json.loads((folder / "config.json").read_text())
        assert saved["updates"] == 3 and saved["planned_updates"] == 500
        summary = json.loads((folder / "summary.json").read_text())
        assert summary["best_valid_hard_score"] is not None
        with np.load(folder / "best_valid.npz") as best:
            assert best["X"].shape == (25, 6)
            assert np.all(np.logical_or(best["X"] == 0, best["X"] == 1))
        with np.load(folder / "trajectory.npz") as trajectory:
            assert trajectory["X"].shape == (4, 25, 6)
    result = results.write_campaign_tables(path)
    assert result["rows"]["runs"] == 18
    assert result["rows"]["states"] == 8
    assert result["rows"]["teacher"] == 3
    with (path / "tables/runs.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert sum(r["status"] == "complete" for r in rows) == 2
    assert sum(r["status"] == "missing" for r in rows) == 16
    with pytest.raises(FileExistsError):
        campaign.run_bundle(path / "plan.json", bid, mock=True)


def test_compressed_native_record_format_retains_all_mock_observations(tmp_path):
    config = next(r for r in design.expand_runs(design.load_design()) if r["condition_id"] == "reference")
    config = config | {"updates": 3, "mock": True, "compress_jsonl": True}
    native, _, _ = mock_native()
    teacher = campaign.MockTeacher()
    folder = tmp_path / "compressed"
    summary = campaign.run_arm(config, folder, native, teacher, {}, mock=True)
    assert summary["status"] == "complete" and teacher.queries == 3
    for name, count in (("states", 4), ("gradients", 3), ("teacher", 3), ("operators", 3)):
        assert not (folder / f"{name}.jsonl").exists()
        with gzip.open(folder / f"{name}.jsonl.gz", "rt", encoding="utf-8") as stream:
            direct = [json.loads(row) for row in stream]
        assert len(direct) == count
        assert results._read_jsonl(folder / f"{name}.jsonl") == direct
    assert (folder / "all_candidates.json.gz").exists()
    assert not (folder / "all_candidates.json").exists()
    with gzip.open(folder / "all_candidates.json.gz", "rt", encoding="utf-8") as stream:
        archive = json.load(stream)
    assert archive["candidates"]
    assert (folder / "top_valid.npz").exists()


def test_mock_native_separation_and_unallocated_preflight_dont_consume_claim(tmp_path, monkeypatch):
    path, plan = make_plan(tmp_path, mock=False)
    bid = plan["bundles"][0]["bundle_id"]
    monkeypatch.delenv("LSB_JOBID", raising=False)
    with pytest.raises(RuntimeError, match="allocated"):
        campaign.run_bundle(path / "plan.json", bid)
    assert not (path / "bundles" / f"{bid}.claim.json").exists()
    with pytest.raises(ValueError, match="separate"):
        campaign.run_bundle(path / "plan.json", bid, mock=True)


@pytest.mark.parametrize("condition_id", [
    "mechanism-identity", "mechanism-stale", "mechanism-hard-input",
    "mechanism-entry-mean-calibrated", "mechanism-kl-calibrated",
    "cool-fixed-half-valence-w-0p0625", "cool-ste-ramp-half-none-w-0p0625",
    "cool-state-ramp-half-connected-w-0p0625", "soft-valence-0p1-w-0p0625",
])
def test_interventions_execute_in_real_toy_optimizer(tmp_path, condition_id):
    cfg = design.load_design()
    config = next(r for r in design.expand_runs(cfg)
                  if r["condition_id"] == condition_id and r["seed"] == 17 and r["initialization"] == "sparse")
    config = config | {"updates": 2, "planned_updates": 500, "mock": True}
    native, _, _ = mock_native()
    teacher = campaign.MockTeacher()
    summary = campaign.run_arm(config, tmp_path / "arm", native, teacher, {}, mock=True)
    assert summary["status"] == "complete"
    assert np.isfinite(summary["calibration_multiplier"])
    expected_queries = 0 if condition_id == "mechanism-identity" else 1 if condition_id == "mechanism-stale" else 2
    assert teacher.queries == expected_queries


def test_early_failed_arm_is_retained_and_next_matched_arm_can_run(tmp_path, monkeypatch):
    path, plan = make_plan(tmp_path)
    original = campaign.run_arm
    calls = [0]
    def fail_first(config, output, *args, **kwargs):
        calls[0] += 1
        if calls[0] == 1:
            output.mkdir()
            campaign.write_json(output / "config.json", config)
            raise ValueError("Input shape failure before arm status creation")
        return original(config, output, *args, **kwargs)
    monkeypatch.setattr(campaign, "run_arm", fail_first)
    record = campaign.run_bundle(path / "plan.json", plan["bundles"][0]["bundle_id"], mock=True, mock_updates=1)
    assert [r["status"] for r in record["arms"]] == ["failed", "complete"]
    assert record["status"] == "failed"
    first = record["arms"][0]["run_id"]
    assert json.loads((path / "runs" / first / "status.json").read_text())["status"] == "failed"


def test_nonfinite_objective_stops_only_affected_arm_and_retains_failure(tmp_path):
    config = design.expand_runs(design.load_design())[0] | {"updates": 2, "mock": True}
    def nonfinite(graph, *, key):
        return jnp.sum(graph.X) * jnp.asarray(float("nan")), {}
    with pytest.raises(FloatingPointError):
        campaign.run_arm(config, tmp_path / "failed", nonfinite, campaign.MockTeacher(), {}, mock=True)
    status = json.loads((tmp_path / "failed/status.json").read_text())
    assert status["status"] == "failed" and status["updates_completed"] == 0


def test_bootstrap_failure_is_not_left_running(tmp_path, monkeypatch):
    path, plan = make_plan(tmp_path)
    import stage3_native
    def failed_load():
        raise FileNotFoundError("Missing model assets during bootstrap")
    monkeypatch.setattr(stage3_native, "mock_native", failed_load)
    bid = plan["bundles"][0]["bundle_id"]
    with pytest.raises(FileNotFoundError):
        campaign.run_bundle(path / "plan.json", bid, mock=True)
    record = json.loads((path / "bundles" / f"{bid}.json").read_text())
    assert record["status"] == "failed" and record["phase"] == "bootstrap"
    assert record["finished_utc"]


def test_device_startup_failure_has_failed_bundle_record(tmp_path, monkeypatch):
    path, plan = make_plan(tmp_path)
    bid = plan["bundles"][0]["bundle_id"]
    def missing_device():
        raise RuntimeError("CUDA initialization failure")
    monkeypatch.setattr(jax, "devices", missing_device)
    with pytest.raises(RuntimeError, match="CUDA"):
        campaign.run_bundle(path / "plan.json", bid, mock=True)
    record = json.loads((path / "bundles" / f"{bid}.json").read_text())
    assert record["status"] == "failed" and record["finished_utc"]


def test_frozen_failure_excludes_completed_arm_without_discarding_raw_data(tmp_path, monkeypatch):
    path, plan = make_plan(tmp_path)
    native, identity, _ = mock_native()
    import stage3_native
    def changed():
        raise RuntimeError("Frozen model mutated")
    monkeypatch.setattr(stage3_native, "mock_native", lambda: (native, identity, changed))
    bid = plan["bundles"][0]["bundle_id"]
    with pytest.raises(RuntimeError, match="mutated"):
        campaign.run_bundle(path / "plan.json", bid, mock=True, mock_updates=1)
    first = plan["bundles"][0]["run_ids"][0]
    folder = path / "runs" / first
    status = json.loads((folder / "status.json").read_text())
    assert status["scientific_usable"] is False and status["status"] == "failed"
    assert (folder / "states.jsonl").exists() and (folder / "best_valid.npz").exists()


def test_prior_and_native_evaluate_same_cooled_graph_with_attached_student():
    native, _, _ = mock_native()
    native_grad, _, prior_grad, _ = campaign.make_evaluators(native)
    graph = maths.make_initial_graph(17, "sparse")
    target = maths.make_initial_graph(23, "sparse")
    key = jax.random.key(17)
    temp = jnp.asarray(.5)
    (value, _), grad = native_grad(graph, key, temp, "view", "exact")
    expected = native(maths.cool_graph(graph, .5), key=key)[0]
    assert float(value) == pytest.approx(float(expected), abs=1e-6)
    penalty, pg = prior_grad(graph, target, temp, "view", "exact", "mse", 1., 1., "row")
    independent = jax.grad(lambda g: maths.teacher_distance(maths.cool_graph(g, .5), target))(graph)
    np.testing.assert_allclose(pg.X, independent.X, atol=1e-6)
    assert np.linalg.norm(grad.X) > 0 and np.linalg.norm(pg.X) > 0
    assert np.isfinite(float(penalty))


def test_dispatch_is_dry_by_default_and_keeps_block_on_same_profile(tmp_path, monkeypatch):
    path, plan = make_plan(tmp_path, mock=False)
    folder = tmp_path / "dispatch"
    result = dispatch.prepare(path / "plan.json", folder, "/remote/snapshot", "/env/bin/python",
                              "/remote/assets.json", ["a10", "a100", "l40s"])
    assert len(result["jobs"]) == 9
    def forbidden(*args, **kwargs):
        raise AssertionError("Dry run must never call bsub")
    monkeypatch.setattr(dispatch.subprocess, "run", forbidden)
    rows = dispatch.submit(folder / "dispatch.json")
    assert all(not r["submitted"] for r in rows)
    for job in result["jobs"]:
        assert "mode=exclusive_process:j_exclusive=yes:mps=no" in " ".join(job["bsub"])
    assert dispatch.assignment(17, "sparse", ["a10", "a100"]) == dispatch.assignment(17, "sparse", ["a10", "a100"])


def test_uncertain_submission_is_durable_and_cannot_be_repeated(tmp_path, monkeypatch):
    path, _ = make_plan(tmp_path, mock=False)
    folder = tmp_path / "dispatch"
    result = dispatch.prepare(path / "plan.json", folder, "/remote/snapshot", "/env/bin/python",
                              "/remote/assets.json", ["a10"])
    bid = result["jobs"][0]["bundle_id"]
    def lost(*args, **kwargs):
        raise TimeoutError("response lost after scheduler may have accepted")
    monkeypatch.setattr(dispatch.subprocess, "run", lost)
    with pytest.raises(TimeoutError):
        dispatch.submit(folder / "dispatch.json", [bid], execute=True)
    receipt = Path(result["jobs"][0]["receipt"])
    assert json.loads(receipt.read_text())["state"] == "unknown"
    with pytest.raises(FileExistsError):
        dispatch.submit(folder / "dispatch.json", [bid], execute=True)
    # Re-preparing scripts in another dispatch directory is not a queue race.
    other = tmp_path / "other_dispatch"
    dispatch.prepare(path / "plan.json", other, "/remote/snapshot", "/env/bin/python",
                     "/remote/assets.json", ["a100"])
    with pytest.raises(FileExistsError):
        dispatch.submit(other / "dispatch.json", [bid], execute=True)


def test_dispatch_rejects_script_changed_after_preparation(tmp_path):
    path, _ = make_plan(tmp_path, mock=False)
    folder = tmp_path / "dispatch"
    result = dispatch.prepare(path / "plan.json", folder, "/remote/snapshot", "/env/bin/python",
                              "/remote/assets.json", ["a10"])
    row = result["jobs"][0]
    Path(row["script"]).write_text("unexpected changed job command")
    with pytest.raises(ValueError, match="script changed"):
        dispatch.submit(folder / "dispatch.json", [row["bundle_id"]])
