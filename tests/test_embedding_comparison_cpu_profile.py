"""CPU-only execution profile; fixtures never submit jobs or load a model."""

import importlib.util
import os
from pathlib import Path
import runpy
import sys
from types import ModuleType

import pytest


def operator():
    path = Path(__file__).resolve().parents[1] / "scripts/hpc/embedding_comparison_operator.py"
    spec = importlib.util.spec_from_file_location("comparison_cpu_profile_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


@pytest.mark.parametrize("arguments", [
    ["verify-run", "--stage", "cpu"],
    ["verify-run", "--release", "/sealed/source", "--stage=cpu"],
    ["verify-run", "--stage", "gpu", "--stage", "cpu"],
])
def test_only_actual_cpu_execution_selects_compatible_without_mutating_input(arguments):
    module, _ = operator()
    supplied = {"MKL_CBWR": "AUTO", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                "LSB_JOBID": "fixture", "UNRELATED": "retained"}
    before = dict(supplied)
    result = module.configure_cpu_environment(arguments, supplied)
    assert module.CPU_KERNEL_PROFILE == "mkl_compatible_four_threads_v1"
    assert result == {**before, "MKL_CBWR": "COMPATIBLE"}
    assert supplied == before and result is not supplied


@pytest.mark.parametrize("arguments", [
    [], ["submit", "--stage", "cpu"], ["collect", "--stage", "cpu"],
    ["verify-run", "--stage", "gpu"], ["verify-run", "--stage=gpu"],
    ["verify-run", "--stage", "cpu", "--stage", "gpu"],
    ["verify-run", "--stage"], ["verify-run", "--stage", "invalid"],
    ["verify-run", "--", "--stage", "cpu"],
])
def test_gpu_scheduler_collection_and_malformed_stages_do_not_change_environment(arguments):
    module, _ = operator()
    supplied = {"MKL_CBWR": "AVX2", "OMP_NUM_THREADS": "4", "KEEP": "same"}
    result = module.configure_cpu_environment(arguments, supplied)
    assert result == supplied and result is not supplied


@pytest.mark.parametrize("stage,expected", [("cpu", "COMPATIBLE"), ("gpu", "fixture-original")])
def test_wrapper_configures_before_shared_dispatch_and_exec_inherits_profile(monkeypatch, stage, expected):
    _, path = operator()
    calls = []
    shared = ModuleType("embedding_operator")

    def fake_exec(executable, command):
        calls.append(dict(kind="exec", executable=executable, command=command,
                          kernel=os.environ.get("MKL_CBWR"), marker=os.environ.get("KEEP")))

    def fake_main(**kwargs):
        calls.append(dict(kind="dispatch", kernel=os.environ.get("MKL_CBWR"), settings=kwargs))
        # Exercise the actual inheritance path used by the stdlib operator.
        os.execv(sys.executable, [sys.executable, "-m", "fixture.experiment", stage])

    shared.main = fake_main
    monkeypatch.setitem(sys.modules, "embedding_operator", shared)
    monkeypatch.setattr(os, "execv", fake_exec)
    monkeypatch.setattr(sys, "argv", [str(path), "verify-run", "--stage", stage])
    monkeypatch.setenv("MKL_CBWR", "fixture-original")
    monkeypatch.setenv("KEEP", "sentinel")
    runpy.run_path(str(path), run_name="__main__")
    assert [row["kind"] for row in calls] == ["dispatch", "exec"]
    assert all(row["kernel"] == expected for row in calls)
    assert calls[-1]["marker"] == "sentinel"
    assert calls[0]["settings"]["experiment_module"] == "latentmol.boltz.embedding_comparison_experiment"
