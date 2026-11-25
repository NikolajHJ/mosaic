"""Allocated CLI regressions; fake dispatch is not native/scientific evidence.

The campaign additionally checks its exact generated argv against the real
sealed plan/readiness files using check_chemistry_launch before GPU submission.
"""
from copy import deepcopy
import importlib.util
import inspect
from pathlib import Path
import sys

import pytest

from latentmol import chemistry_continuation as continuation
from latentmol import chemistry_valid_reference_adoption as adoption


@pytest.fixture
def cli(monkeypatch):
    plan = {"protocol": {}, "valid_reference_continuation": {"end_epoch": 40}}
    seen = []
    monkeypatch.setattr(continuation, "read_plan", lambda path, sha: deepcopy(plan))
    monkeypatch.setattr(continuation, "train_phase", lambda p, **kwargs: seen.append((p, kwargs)))
    monkeypatch.setattr(continuation, "final_evaluation", lambda p, *args: seen.append((p, args)))
    monkeypatch.setattr(continuation, "benchmark", lambda p: seen.append((p, "benchmark")))
    command = ["chemistry", "train", "--plan", "/plan.json", "--plan-sha256", "a" * 64,
               "--benchmark-gate-sha256", "b" * 64, "--failure-readiness-sha256", "c" * 64,
               "--segment", "1"]
    return plan, seen, command


@pytest.mark.parametrize("marker", ["valid_reference_continuation", "short_continuation", "failure_guidance"])
def test_real_cli_accepts_required_readiness_for_each_supported_policy(cli, monkeypatch, marker):
    plan, seen, command = cli
    plan.pop("valid_reference_continuation")
    if marker == "failure_guidance":
        plan["protocol"][marker] = {"version": "fixture"}
    else:
        plan[marker] = {"version": "fixture"}
    monkeypatch.setattr(sys, "argv", command)
    continuation.main()
    assert seen[0][0]["_failure_readiness_sha256"] == "c" * 64
    assert seen[0][0]["_benchmark_gate_sha256"] == "b" * 64
    assert seen[0][1] == {"resume": None, "segment": 1}


@pytest.mark.parametrize("mode", ["missing_readiness", "missing_benchmark", "historical_override",
                                  "benchmark_override", "unpaired_resume"])
def test_real_cli_still_rejects_missing_or_inapplicable_guard_arguments(cli, monkeypatch, mode):
    plan, seen, command = cli
    if mode == "missing_readiness":
        del command[command.index("--failure-readiness-sha256"):command.index("--failure-readiness-sha256") + 2]
    elif mode == "missing_benchmark":
        del command[command.index("--benchmark-gate-sha256"):command.index("--benchmark-gate-sha256") + 2]
    elif mode == "historical_override":
        plan.pop("valid_reference_continuation")
    elif mode == "benchmark_override":
        command[1] = "benchmark"
    else:
        command += ["--resume", "/epoch.pt"]
    monkeypatch.setattr(sys, "argv", command)
    with pytest.raises(SystemExit) as error:
        continuation.main()
    assert error.value.code == 2 and seen == []


def test_final_cli_retains_readiness_and_paired_resume(cli, monkeypatch):
    _, seen, command = cli
    command[1] = "final"
    command += ["--resume", "/epoch.pt", "--resume-sha256", "d" * 64]
    monkeypatch.setattr(sys, "argv", command)
    continuation.main()
    assert seen[0][0]["_failure_readiness_sha256"] == "c" * 64
    assert seen[0][1] == (Path("/epoch.pt"), "d" * 64)


def test_adoption_allows_exact_cli_predicate_patch_but_no_other_main_change(tmp_path):
    current = inspect.getsource(continuation.main)
    expanded = ('if (plan["protocol"].get("failure_guidance") is not None\n'
                '                or plan.get("short_continuation") is not None\n'
                '                or plan.get("valid_reference_continuation") is not None):')
    assert expanded in current
    historical = current.replace(expanded, 'if plan["protocol"].get("failure_guidance") is not None:')
    name = "src/latentmol/chemistry_continuation.py"
    old, new = tmp_path / "old", tmp_path / "new"
    for root, text in ((old, historical), (new, current)):
        path = root / name
        path.parent.mkdir(parents=True)
        path.write_text(text)
    assert adoption.compare_source(old, new, {name: "a"}, {name: "b"}) == [name]
    (new / name).write_text(current.replace('default=1', 'default=2'))
    with pytest.raises(ValueError, match="more than readiness"):
        adoption.compare_source(old, new, {name: "a"}, {name: "b"})


def _harness():
    path = Path(__file__).resolve().parents[1] / "scripts/hpc/check_chemistry_launch.py"
    spec = importlib.util.spec_from_file_location("launch_harness_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_harness_requires_allocation_and_exact_module_command(monkeypatch):
    module = _harness()
    monkeypatch.delenv("LSB_JOBID", raising=False)
    with pytest.raises(ValueError, match="allocated"):
        module.check_launch(["python", "-m", "latentmol.chemistry_continuation", "train"])
    monkeypatch.setenv("LSB_JOBID", "123")
    with pytest.raises(ValueError, match="exact generated"):
        module.check_launch(["python", "other.py"])


def test_harness_propagates_real_cli_failure_and_restores_process_state(monkeypatch):
    module = _harness()
    monkeypatch.setenv("LSB_JOBID", "123")
    before_argv, before_trace = sys.argv, sys.gettrace()
    def rejected(*args, **kwargs):
        assert sys.argv == ["latentmol.chemistry_continuation", "train", "--unknown"]
        raise SystemExit(2)
    monkeypatch.setattr(module.runpy, "run_module", rejected)
    with pytest.raises(SystemExit) as error:
        module.check_launch(["python", "-m", "latentmol.chemistry_continuation", "train", "--unknown"])
    assert error.value.code == 2
    assert sys.argv is before_argv and sys.gettrace() is before_trace


def test_harness_stops_before_training_body_only_after_readiness(monkeypatch):
    module = _harness()
    from latentmol import chemistry_valid_reference_training_policy as policy
    from latentmol import chemistry_short_continuation as short
    monkeypatch.setenv("LSB_JOBID", "123")
    monkeypatch.setattr(policy, "validate_valid_reference_continuation", lambda p: {"end_epoch": 40})
    monkeypatch.setattr(short, "continuation_end_epoch", lambda p: 40)
    calls = []
    plan = dict(_plan_sha256="a" * 64, _benchmark_gate_sha256="b" * 64,
                _failure_readiness_sha256="c" * 64, starting_checkpoint={"epoch": 36})
    def readiness(p):
        assert p is plan
        calls.append("authenticated")
        return {"status": "passed"}
    def entry(*args, **kwargs):
        scope = {"__name__": "__main__", "require_failure_readiness": readiness}
        exec(compile('def train_phase(plan, *, resume=None, segment=1):\n'
                     '    raise AssertionError("training body must not execute")\n',
                     "chemistry_continuation.py", "exec"), scope)
        scope["train_phase"](plan, segment=1)
    monkeypatch.setattr(module.runpy, "run_module", entry)
    result = module.check_launch(["python", "-m", "latentmol.chemistry_continuation", "train"])
    assert calls == ["authenticated"]
    assert result["status"] == "passed_dispatch_not_training"
    assert result["model_setup_started"] is False and result["updates"] == 0
