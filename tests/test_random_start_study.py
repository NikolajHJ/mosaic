"""HPC fixtures for seed isolation, serial failure accounting and child provenance.

These do not establish real Boltz/VAE optimization performance. Run in the same
HPC allocation before the sequential production study, never as a local model run.
"""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

from latentmol.boltz import random_start_study as study
from latentmol.boltz.initialization_report import molecular_diversity, write_report
from latentmol.boltz.workflow_process import require_child, run_logged
from latentmol.experiments import file_sha256, write_json
from latentmol.initialization import LatentStart, sample_prior


def test_prior_is_exact_cpu_draw_without_global_rng_side_effects():
    torch.manual_seed(199)
    original = torch.random.get_rng_state().clone()
    actual = sample_prior(256, 7)
    assert torch.equal(torch.random.get_rng_state(), original)
    expected = torch.randn((1, 256), generator=torch.Generator().manual_seed(7))
    assert torch.equal(actual, expected)
    assert torch.equal(actual, sample_prior(256, 7))
    assert not torch.equal(actual, sample_prior(256, 8))
    assert actual.dtype == torch.float32 and actual.shape == (1, 256)


@pytest.mark.parametrize("kwargs", [{"kind": "filtered"}, {"seed": 1},
                                    {"kind": "standard_normal", "seed": True},
                                    {"kind": "standard_normal", "seed": -1}])
def test_ambiguous_start_policy_is_rejected(kwargs):
    with pytest.raises(ValueError):
        LatentStart(**kwargs)


def test_logged_child_failure_and_timeout_are_retained(tmp_path):
    failed = run_logged([sys.executable, "-c", "print('evidence'); raise SystemExit(3)"],
                        tmp_path / "failed.log", timeout_seconds=10)
    assert failed["returncode"] == 3 and not failed["timed_out"]
    assert "evidence" in Path(failed["log"]).read_text()
    with pytest.raises(RuntimeError):
        require_child(failed)
    timeout = run_logged([sys.executable, "-c", "import time; time.sleep(30)"],
                         tmp_path / "timeout.log", timeout_seconds=1)
    assert timeout["timed_out"] and timeout["returncode"] != 0


def test_completed_child_requires_record_and_exactly_one_path(tmp_path):
    child = tmp_path / "child"
    child.mkdir()
    write_json(child / "run.json", {"status": "completed"})
    result = {"returncode": 0, "timed_out": False, "run_paths": [str(child)]}
    assert require_child(result, experiment=True) == child
    result["run_paths"].append(str(child))
    with pytest.raises(ValueError, match="exactly one"):
        require_child(result, experiment=True)


def test_child_cannot_claim_another_source_or_output_root(tmp_path, monkeypatch):
    monkeypatch.setenv("LATENTMOL_SOURCE_SHA256", "archive")
    root = tmp_path / "pilot"
    child = root / "run"
    child.mkdir(parents=True)
    record = {"source": {"sha256": "source"}, "source_archive_sha256": "archive"}
    write_json(child / "run.json", record)
    result = {"run_paths": [str(child)]}
    assert study.child_path(result, root, "source") == child
    with pytest.raises(ValueError, match="source identity"):
        study.child_path(result, root, "other")
    with pytest.raises(ValueError, match="escaped"):
        study.child_path(result, tmp_path / "different", "source")


def test_failed_preparation_is_retained_and_other_counts_continue(tmp_path, monkeypatch):
    existing = tmp_path / "existing.pt"
    existing.write_bytes(b"fixture")
    base = SimpleNamespace(contexts={2: {"path": str(existing), "sha256": file_sha256(existing)}},
                           decoder={"path": "fixture-decoder.pt"})
    calls = []
    def child(command, log_path, **kwargs):
        command = list(map(str, command))
        calls.append(command)
        # First missing context fails; count3 must nevertheless be prepared.
        failed = "--count" in command and command[command.index("--count") + 1] == "1"
        if "snapshot" in command:
            output = Path(command[command.index("--output") + 1])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"new-fixture")
        return {"returncode": int(failed), "timed_out": False, "run_paths": []}
    sealed = {}
    def seal(paths, output, **kwargs):
        sealed.update(paths=paths, **kwargs)
        output.write_text("{}")
        return output
    monkeypatch.setattr(study, "run_logged", child)
    monkeypatch.setattr(study, "seal_count_catalog", seal)
    protocol = {"counts": [1, 2, 3], "example_assets": "assets", "cache": "cache",
                "preparation_timeout_seconds": 10}
    output = study.prepare_contexts(protocol, base, tmp_path / "contexts")
    states = json.loads((output.parent / "preparation.json").read_text())
    assert [states[str(n)]["status"] for n in (1, 2, 3)] == ["failed", "reused", "prepared"]
    assert set(sealed["paths"]) == {2, 3}
    assert set(sealed["failed_counts"]) == {"1"}
    assert len(calls) == 4  # Failed count1 once, reused count2, three count3 stages.


def test_failed_seed_is_not_redrawn_and_does_not_launch_rescore(tmp_path, monkeypatch):
    calls, persisted = [], []
    monkeypatch.setenv("LATENTMOL_SOURCE_SHA256", "archive")
    def config(base_path, catalog, root, seed, eps):
        root.mkdir(parents=True)
        path = root / "pilot.yaml"
        path.write_text(f"seed: {seed}\n")
        return path
    def child(command, log_path, **kwargs):
        calls.append(command)
        config_path = Path(command[-1])
        seed = yaml.safe_load(config_path.read_text())["seed"]
        pilot = config_path.parent / "pilot" / "fixture"
        pilot.mkdir(parents=True)
        write_json(pilot / "run.json", {"status": "failed", "source": {"sha256": "source"},
                   "source_archive_sha256": "archive", "error": "fixture derivative gate failed",
                   "config": {"config_file_sha256": file_sha256(config_path),
                              "start": {"kind": "standard_normal", "seed": seed}}})
        return {"returncode": 1, "timed_out": False, "run_paths": [str(pilot)], "log": str(log_path)}
    monkeypatch.setattr(study, "trial_config", config)
    monkeypatch.setattr(study, "run_logged", child)
    protocol = {"fd_epsilons": [.001, .003, .01], "optimization_timeout_seconds": 30}
    for seed in range(10):
        result = study.run_trial(protocol, tmp_path / "base", tmp_path / "catalog",
                                 tmp_path / f"seed-{seed}", seed, "source", lambda row: persisted.append(dict(row)))
        assert result["status"] == "optimization_failed" and result["seed"] == seed
    assert len(calls) == 10
    assert [r["seed"] for r in persisted if r["status"] == "optimization_failed"] == list(range(10))
    assert all("latentmol.boltz.active_set" in command for command in calls)


def test_pending_and_failed_seeds_stay_in_full_report_denominator(tmp_path):
    trials = [{"seed": n, "status": "optimization_failed" if n == 0 else "not_attempted"} for n in range(10)]
    result = write_report(tmp_path, trials)
    assert result["planned_seed_count"] == 10
    for arm in result["arms"].values():
        assert arm["planned_seeds"] == 10
        assert arm["completed_valid_endpoints_per_planned_seed"] == 0
        assert arm["diversity_among_completed_valid_endpoints"]["mean_pairwise_tanimoto"] is None


def test_endpoint_similarity_keeps_duplicates_and_absent_pairs():
    result = molecular_diversity([{"seed": 0, "smiles": "CC"}, {"seed": 1, "smiles": "CC"},
                                  {"seed": 2, "smiles": "N"}])
    assert result["pair_count"] == 3 and result["unique_smiles"] == 2
    assert result["pairs"][0]["tanimoto"] == 1.0
    assert result["pairs"][1]["tanimoto"] == 0.0
    assert molecular_diversity([])["mean_pairwise_tanimoto"] is None


def test_declared_protocol_rejects_duplicate_seed_and_unknown_field(tmp_path):
    # The repository configuration is static input; validation here does not
    # resolve or open its HPC model assets.
    example = Path(__file__).resolve().parents[1] / "configs/boltz_random_start_study.yaml"
    value = yaml.safe_load(example.read_text())
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(value))
    assert study.load_protocol(path)["seeds"] == list(range(10))
    value["seeds"] = [0, 0]
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="distinct"):
        study.load_protocol(path)
    value["seeds"] = [0, 1]
    value["redraw_invalid"] = True
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="documented fields"):
        study.load_protocol(path)


def test_controller_visits_all_declared_seeds_after_scientific_failure(tmp_path, monkeypatch):
    """Exercise the outer controller itself; no GPU/model/child process involved."""
    from latentmol.boltz import active_set_rescore
    protocol = {"name": "fixture-study", "output_root": str(tmp_path / "runs"),
                "base_config": {"path": "base.yaml", "sha256": "basehash"},
                "native_template": {"path": "template.yaml"}, "counts": [1],
                "seeds": list(range(10)), "reference_pilot": str(tmp_path / "absent-reference")}
    base = SimpleNamespace(mode="vae", start=LatentStart(), context_identity={
                           "target_sequence_sha256": "target", "msa_sha256": "msa"})
    monkeypatch.chdir(tmp_path)
    write_json(tmp_path / "release_manifest.json", {"sha256": "fixture-source"})
    monkeypatch.setenv("LSB_JOBID", "fixture")
    monkeypatch.setenv("LATENTMOL_SOURCE_SHA256", "fixture-archive")
    monkeypatch.setattr(study, "verified_file", lambda path, expected: Path(path))
    monkeypatch.setattr(study, "load_protocol", lambda path: protocol)
    monkeypatch.setattr(study.ActiveSetConfig, "load", lambda path: base)
    monkeypatch.setattr(study, "verify_assets", lambda *args: None)
    monkeypatch.setattr(study, "source_manifest", lambda root: {"sha256": "fixture-source"})
    monkeypatch.setattr(study, "installed_boltz_provenance", lambda: {})
    monkeypatch.setattr(active_set_rescore, "_target_identity", lambda *args: {"template": {}})
    def prepare(protocol, base, destination):
        destination.mkdir()
        path = destination / "contexts.json"
        write_json(path, {"contexts": {"1": {"path": "context.pt", "sha256": "fixture"}}, "failed_counts": {}})
        return path
    visited = []
    def trial(protocol, base, catalog, root, seed, source_sha, persist):
        visited.append(seed)
        row = {"seed": seed, "status": "optimization_failed" if seed == 0 else "completed"}
        persist(row)
        return row
    monkeypatch.setattr(study, "prepare_contexts", prepare)
    monkeypatch.setattr(study, "run_trial", trial)
    path = study.run(tmp_path / "protocol.yaml", "fixturehash")
    assert visited == list(range(10))
    trials = json.loads((path / "trials.json").read_text())
    assert len(trials) == 10 and trials[0]["status"] == "optimization_failed"
    record = json.loads((path / "run.json").read_text())
    assert record["all_planned_seeds_processed"] and record["successful_seed_workflows"] == 9
