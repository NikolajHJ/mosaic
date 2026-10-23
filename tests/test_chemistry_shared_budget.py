"""Allocated contract checks for one budget across native/stability stages."""
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

from latentmol import chemistry_guarded_validation as validation


def configure_main(monkeypatch):
    clock = {"now": 1234.}
    plan = {"fixture": "same allocated campaign"}
    monkeypatch.setattr(sys, "argv", ["chemistry_guarded_validation", "--plan", "plan.json",
                                      "--plan-sha256", "a" * 64])
    monkeypatch.setattr(validation.continuation, "read_plan", lambda *_: plan)
    monkeypatch.setattr(validation, "validate_guarded_restart", lambda *_: {})
    monkeypatch.setattr(validation.time, "monotonic", lambda: clock["now"])
    return clock, plan


def test_native_readiness_time_cannot_become_free_time_for_the_stability_proof(monkeypatch):
    clock, plan = configure_main(monkeypatch)
    def native(received):
        assert received is plan
        clock["now"] += 321.
    stability = Mock()
    monkeypatch.setattr(validation, "benchmark_candidate", native)
    monkeypatch.setattr(validation, "run_stability", stability)
    validation.main()
    stability.assert_called_once_with(plan, allocation_started=1234.)
    assert clock["now"] - stability.call_args.kwargs["allocation_started"] == 321.


def test_a_failed_native_benchmark_does_not_start_the_longer_proof(monkeypatch):
    configure_main(monkeypatch)
    stability = Mock()
    monkeypatch.setattr(validation, "run_stability", stability)
    monkeypatch.setattr(validation, "benchmark_candidate", Mock(side_effect=RuntimeError("native gate failed")))
    with pytest.raises(RuntimeError, match="native gate failed"):
        validation.main()
    stability.assert_not_called()


def test_worker_startup_extends_all_existing_cpu_readiness_contracts(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/hpc"))
    from chemistry_guarded_restart import GuardedRestart, STARTUP_FIXTURES, cpu_command
    candidate = object.__new__(GuardedRestart)
    candidate.root = tmp_path
    candidate.plan = dict(worker_startup=dict(version="chemistry-worker-startup-v1"))
    command = candidate.cpu_contract_command("python")
    assert command[:len(cpu_command("python", tmp_path))] == cpu_command("python", tmp_path)
    assert all(path in command for path in STARTUP_FIXTURES)
    assert len(command) == len(set(command))
