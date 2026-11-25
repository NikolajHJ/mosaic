"""Allocated CPU fixtures for process lifetime ownership; no live CUDA here."""

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from latentmol import cuda_process as subject


def test_coordinator_inspection_does_not_import_or_query_devices(monkeypatch):
    fake = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False))
    monkeypatch.setitem(subject.sys.modules, "torch", fake)
    result = subject.require_cuda_free_coordinator()
    assert result["cuda_initialized"] is False and result["torch_imported"] is True


def test_initialized_coordinator_is_rejected_before_launch(monkeypatch, tmp_path):
    monkeypatch.setitem(subject.sys.modules, "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True)))
    monkeypatch.setattr(subject.subprocess, "Popen", lambda *a, **k: pytest.fail("No child may launch"))
    with pytest.raises(RuntimeError, match="parent already initialized CUDA"):
        subject.run_cuda_child([sys.executable, "-c", "pass"], tmp_path / "not-created.log", timeout_seconds=5)
    assert not (tmp_path / "not-created.log").exists()


def test_child_exits_before_next_phase_and_logs_are_exclusive(monkeypatch, tmp_path):
    monkeypatch.setattr(subject, "require_cuda_free_coordinator", lambda: {"cuda_initialized": False})
    marker = tmp_path / "child.txt"
    program = f"from pathlib import Path;Path({str(marker)!r}).write_text('complete')"
    result = subject.run_cuda_child([sys.executable, "-c", program], tmp_path / "child.log", timeout_seconds=10)
    subject.require_cuda_child(result)
    assert result["returncode"] == 0 and marker.read_text() == "complete"
    with pytest.raises(FileExistsError):
        subject.run_cuda_child([sys.executable, "-c", "pass"], tmp_path / "child.log", timeout_seconds=10)


def test_failed_or_timed_out_child_never_releases_next_phase(monkeypatch, tmp_path):
    monkeypatch.setattr(subject, "require_cuda_free_coordinator", lambda: {"cuda_initialized": False})
    result = subject.run_cuda_child([sys.executable, "-c", "raise SystemExit(7)"],
                                   tmp_path / "failed.log", timeout_seconds=10)
    with pytest.raises(RuntimeError, match="failed"):
        subject.require_cuda_child(result)
    assert result["returncode"] == 7
    timed = subject.run_cuda_child([sys.executable, "-c", "import time;time.sleep(30)"],
                                  tmp_path / "timeout.log", timeout_seconds=1)
    assert timed["timed_out"] is True
    with pytest.raises(RuntimeError):
        subject.require_cuda_child(timed)
