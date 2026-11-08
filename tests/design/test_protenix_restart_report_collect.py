import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPORT = Path(__file__).resolve().parents[2] / "experiments/latent-design/protenix-vae/restart-report"


def report_module(name):
    spec = importlib.util.spec_from_file_location(name, REPORT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_defaults_and_explicit_job_selection(tmp_path):
    collector = report_module("collect")
    default = collector.select_runs(collector.DEFAULT_MANIFEST)
    all_runs = collector.select_runs(collector.DEFAULT_MANIFEST, all_runs=True)
    assert [job["id"] for job in default] == [
        29914616, 29914617, 29914618, 29914683, 29914684, 29914686,
        29914687, 29914688, 29914689, 29914690, 29914691,
    ]
    assert len(all_runs) > len(default)
    subset = collector.select_runs(collector.DEFAULT_MANIFEST, job_ids=[29926073, 29920256, 29924369])
    assert [job["id"] for job in subset] == [29926073, 29920256, 29924369]
    assert subset[0]["run_path"].endswith("combined-retry01")
    assert subset[1]["launch_path"].endswith("v100-requeue-20261107a/launch.json")
    collector.verify_snapshot_jobs({"jobs": subset}, subset)
    with pytest.raises(ValueError, match="differ from manifest"):
        collector.verify_snapshot_jobs({"jobs": subset[::-1]}, subset)
    with pytest.raises(ValueError, match="absent from the manifest"):
        collector.select_runs(collector.DEFAULT_MANIFEST, job_ids=[1])

    custom = tmp_path / "runs.json"
    custom.write_text(json.dumps({"runs": subset}), encoding="utf-8")
    assert collector.select_runs(custom) == subset


def test_tree_collector_keeps_exact_restart_anchor_and_every_loss_observation(tmp_path, monkeypatch):
    collector = report_module("collect")
    monkeypatch.setitem(sys.modules, "collect", collector)
    trees = report_module("collect_trees")
    helpers = collector.REMOTE.rpartition("\ncaptured = datetime.")[0]
    tree_helpers = trees.TREE_REMOTE.rpartition("\ncaptured = datetime.")[0]
    worker = {"JOBS_JSON": "[]", "INCLUDE_RAW": False}
    exec(helpers + "\n" + tree_helpers, worker)
    root = tmp_path / "run"
    first = root / "cycle-01" / "branch-01"
    second = root / "cycle-02" / "branch-01"
    first.mkdir(parents=True)
    second.mkdir(parents=True)

    def write_json(path, data):
        path.write_text(json.dumps(data), encoding="utf-8")

    def write_metrics(branch, molecules):
        rows = [dict(step=step, raw_valid=True, smiles=smiles, relaxed_loss=3.0 - 0.1 * step,
                     chemistry_loss=0.1, chemistry_weight=1.0, valence_excess=0.03)
                for step, smiles in enumerate(molecules)]
        (branch / "metrics.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        write_json(branch / "summary.json", {"updates": len(rows) - 1})

    write_metrics(first, ["CCO", "CCO", "CCN", "CCN"])
    write_metrics(second, ["CCO", "CCC", "CCC", "CCC"])
    write_json(first.parent / "archive.json", {
        "elites": [dict(smiles="CCN", fresh_loss=2.0, step=2, branch="cycle-01/branch-01")],
        "restart_parents": [dict(smiles="CCO", fresh_loss=2.1, step=1, branch="cycle-01/branch-01")],
    })
    write_json(second / "start.json", {"parent": "CCO", "parent_source": {"branch": "cycle-01/branch-01", "step": 1}})
    receipt = tmp_path / "alternate" / "launch.json"
    receipt.parent.mkdir()
    write_json(receipt, {"status": "worker_exited", "returncode": 0})
    spec = dict(id=9, task="replicate", campaign="custom", layout="cycles", queue="gpua100",
                run_path=str(root), launch_path=str(receipt), updates_per_branch=3,
                branches_per_parent=1, planned_branches=2, budget_updates=6, label="Replicate")
    job = worker["tree_job"](spec, {})
    assert job["status"] == "DONE"
    assert job["label"] == "Replicate"
    assert job["completed_updates"] == 6
    bootstrap, child = job["branches"]
    assert [row[0] for row in bootstrap["states"]] == [0, 1, 2]
    assert bootstrap["repeat_anchor_steps"] == [1]
    assert [row[0] for row in bootstrap["loss_series"]] == [0, 1, 2, 3]
    assert bootstrap["loss_series"][-1] == [3, 2.7, 0.1, 1.0, 0.03]
    assert child["source"] == {"branch": "branch-01", "cycle": 1, "step": 1, "smiles": "CCO", "fresh_loss": 2.1}
    assert child["root_type"] == "archive-fork"
    assert job["archives"][0]["elites"]["replicate"][0]["smiles"] == "CCN"
