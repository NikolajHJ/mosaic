"""Small report/process fixtures; no pretrained model, network or molecular run.

These regressions are authored for the next authorized HPC validation. They
must not be interpreted as prior-sampling or native Boltz evidence.
"""

import json
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest

from latentmol.boltz.initialization_report import (
    arm_report, hard_composition, pair_tanimoto, write_report,
)
from latentmol.boltz import workflow_process
from latentmol.experiments import file_sha256, write_json


def _pilot(tmp_path, *, status="completed", frozen=True, initial_smiles="CC"):
    """Three saved states with an atom-count excursion and changing elements."""
    pilot = tmp_path / "pilot"
    folder = pilot / "active_set"
    folder.mkdir(parents=True)
    config = {"start": {"kind": "standard_normal", "seed": 0},
              "optimizer": {"steps": 2, "max_gradient_norm": 0.000125, "trust_radius": 3.0}}
    write_json(pilot / "run.json", {"run_id": "fixture-pilot", "status": status, "config": config})
    counts, smiles = [2, 3, 2], [initial_smiles, "CCN", "CO"]
    atom_labels = [[1, 1, 0], [1, 1, 2], [1, 3, 0]]
    rows, candidates = [], []
    for step in range(3):
        selected = {"count": counts[step], "active_slots": list(range(counts[step]))}
        row = {"step": step, "selection": selected, "natural_selection": selected,
               "score": {"raw": .1 * (step + 1), "loss": -.1 * (step + 1),
                         "gradient_norm": [.0001, .0002, None][step]},
               "latent_displacement": step * .05, "argmax_smiles": smiles[step],
               "context_sha256": "fixture-context"}
        rows.append(row)
        write_json(folder / f"metrics-{step:04d}.json", row)
        candidates.append({"kind": "argmax", "step": step, "draw": 0,
                           "smiles": smiles[step],
                           "graph_spec": {"atomic_numbers": [0, 6, 7, 8]},
                           "labels": {"atoms": atom_labels[step]}})
    candidate_path = folder / "candidates.jsonl"
    candidate_path.write_text("".join(json.dumps(row) + "\n" for row in candidates))
    summary = {"endpoint": rows[-1], "steps": 2, "arm": "active_set"}
    write_json(folder / "summary.json", summary)
    write_json(pilot / "summary.json", {"frozen_state_unchanged": frozen,
               "arms": {"active_set": summary},
               "candidate_file_sha256": {"active_set": file_sha256(candidate_path)}})
    return pilot


def test_verified_report_tracks_count_excursion_and_atom_composition(tmp_path):
    report = arm_report(_pilot(tmp_path), "active_set")
    assert report["status"] == "completed" and report["verified_completed"]
    assert report["accepted_count_change_events"] == 2
    assert report["observed_count_range"] == [2, 3]
    assert report["maximum_absolute_count_deviation_from_start"] == 1
    assert report["final_count_minus_initial"] == 0
    assert report["initial_composition"]["carbon_fraction"] == 1.0
    assert report["final_composition"]["carbon_fraction"] == .5
    assert report["final_composition"]["atomic_number_counts"] == {"6": 1, "8": 1}
    assert 0 <= report["initial_final_tanimoto"] < 1
    assert report["gradient_clipped_updates"] == 1


@pytest.mark.parametrize("status,frozen", [("failed", True), ("running", True), ("completed", False)])
def test_arm_summary_alone_never_promotes_an_unverified_final(tmp_path, status, frozen):
    pilot = _pilot(tmp_path, status=status, frozen=frozen)
    report = arm_report(pilot, "active_set")
    assert report["status"] == "trajectory_complete_unverified"
    assert report["trajectory_completed"] and not report["verified_completed"]
    assert report["last_observed_smiles"] == "CO"
    assert report["last_observed_composition"]["carbon_fraction"] == .5
    assert report["final_smiles"] is None and report["final_count"] is None
    assert report["final_composition"] is None and report["initial_final_tanimoto"] is None
    trials = [{"seed": n, "status": "not_attempted"} for n in range(10)]
    trials[0].update(status="optimization_failed", pilot=str(pilot))
    aggregate = write_report(tmp_path, trials)
    assert aggregate["arms"]["active_set"]["planned_seeds"] == 10
    assert aggregate["arms"]["active_set"]["diversity_among_completed_valid_endpoints"]["valid_endpoint_count"] == 0


def test_invalid_start_does_not_get_invented_similarity(tmp_path):
    report = arm_report(_pilot(tmp_path, initial_smiles=None), "active_set")
    assert report["final_smiles"] == "CO" and report["verified_completed"]
    assert report["initial_final_tanimoto"] is None
    assert pair_tanimoto("CC", None) is None


def test_composition_uses_declared_vocabulary_and_excludes_absent_slots():
    composition = hard_composition({"graph_spec": {"atomic_numbers": [0, 8, 6]},
                                    "labels": {"atoms": [1, 2, 2, 0]}})
    assert composition["heavy_atoms"] == 3 and composition["carbon_atoms"] == 2
    assert composition["carbon_fraction"] == pytest.approx(2 / 3)
    empty = hard_composition({"graph_spec": {"atomic_numbers": [0, 6]},
                              "labels": {"atoms": [0, 0]}})
    assert empty["heavy_atoms"] == 0 and empty["carbon_fraction"] is None


def test_failed_trial_retains_rows_before_truncated_candidate_write(tmp_path):
    pilot = _pilot(tmp_path, status="failed")
    with (pilot / "active_set/candidates.jsonl").open("a") as handle:
        handle.write('{"kind": "categorical", "step":')
    report = arm_report(pilot, "active_set")
    assert report["hard_graph_diagnostics"]["argmax"] == {"attempted": 3, "valid": 3}
    assert report["unreadable_artifacts"][0]["line"] == 4
    assert report["last_observed_smiles"] == "CO" and report["final_smiles"] is None


def test_completed_trial_rejects_truncated_or_changed_candidate_data(tmp_path):
    pilot = _pilot(tmp_path)
    with (pilot / "active_set/candidates.jsonl").open("a") as handle:
        handle.write('{"kind":')
    with pytest.raises(ValueError):
        arm_report(pilot, "active_set")


def test_failed_trial_does_not_infer_transitions_across_unreadable_state(tmp_path):
    pilot = _pilot(tmp_path, status="failed")
    (pilot / "active_set/metrics-0001.json").write_text('{"step":')
    report = arm_report(pilot, "active_set")
    assert report["metrics_states"] == 2
    assert report["accepted_count_change_events"] == 0
    assert report["unreadable_artifacts"]
    assert not report["trajectory_completed"]


def test_report_rejects_relabelled_seed(tmp_path):
    pilot = _pilot(tmp_path)
    with pytest.raises(ValueError, match="seed differs"):
        write_report(tmp_path, [{"seed": 1, "status": "completed", "pilot": str(pilot)}])


@pytest.mark.skipif(not hasattr(signal, "SIGKILL"), reason="POSIX process groups are the HPC contract")
def test_termination_kills_workers_even_if_group_leader_exits_promptly(monkeypatch):
    signals, waits = [], []
    monkeypatch.setattr(workflow_process, "os", SimpleNamespace(
        name="posix", killpg=lambda pid, sig: signals.append((pid, sig))))
    process = SimpleNamespace(pid=4242, wait=lambda timeout=None: waits.append(timeout) or 0)
    workflow_process._terminate_group(process, grace_seconds=7)
    assert signals == [(4242, signal.SIGTERM), (4242, signal.SIGKILL)]
    assert waits == [7, None]


def test_launch_error_is_a_durable_process_attempt(tmp_path):
    result = workflow_process.run_logged([str(tmp_path / "does-not-exist")],
                                         tmp_path / "launch.log", timeout_seconds=10)
    assert result["returncode"] is None and result["failure"]["type"] == "FileNotFoundError"
    assert "CHILD_RESULT=" in Path(result["log"]).read_text()
    with pytest.raises(RuntimeError):
        workflow_process.require_child(result)


def test_interrupt_is_logged_after_cleanup_before_propagation(tmp_path, monkeypatch):
    cleaned = []
    class InterruptedChild:
        returncode = None
        def wait(self, timeout=None):
            raise KeyboardInterrupt()
    process = InterruptedChild()
    monkeypatch.setattr(workflow_process.subprocess, "Popen", lambda *a, **k: process)
    def cleanup(child):
        cleaned.append(child)
        child.returncode = -15
    monkeypatch.setattr(workflow_process, "_terminate_group", cleanup)
    log = tmp_path / "interrupt.log"
    with pytest.raises(KeyboardInterrupt):
        workflow_process.run_logged(["fixture-child"], log, timeout_seconds=10)
    assert cleaned == [process]
    result = json.loads(next(line.removeprefix("CHILD_RESULT=") for line in log.read_text().splitlines()
                             if line.startswith("CHILD_RESULT=")))
    assert result["interrupted"] and result["returncode"] == -15
