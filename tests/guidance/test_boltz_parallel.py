"""Real spawned CPU workers test orchestration, not native CUDA performance."""
import os
from pathlib import Path
import time

import pytest

from latentmol.guidance.boltz_parallel import ParallelWorkerError, run_spawn_group


def _cpu_worker(payload, context):
    (context.output / "value.txt").write_text(str(payload["value"]))
    context.ready(dict(marker=payload["value"]))
    started = time.perf_counter()
    if payload.get("fail"):
        raise ValueError("intentional worker failure")
    deadline = started + payload.get("seconds", .5)
    while time.perf_counter() < deadline:
        if not payload.get("ignore_abort"):
            context.check_abort()
        time.sleep(.01)
    return dict(pid=os.getpid(), value=payload["value"], started=started, ended=time.perf_counter())


def test_spawned_workers_overlap_and_have_isolated_outputs(tmp_path):
    output = tmp_path / "overlap"
    result = run_spawn_group("test_boltz_parallel:_cpu_worker", [dict(value=1), dict(value=2)], output,
                            startup_timeout=45, run_timeout=5)
    assert result["status"] == "completed" and result["cleanup_complete"] and not result["requires_job_stop"]
    workers = [row["result"] for row in result["results"].values()]
    assert len({row["pid"] for row in workers}) == 2
    assert max(row["started"] for row in workers) < min(row["ended"] for row in workers)
    assert (output / "worker-00/value.txt").read_text() == "1"
    assert (output / "worker-01/value.txt").read_text() == "2"
    with pytest.raises(FileExistsError):
        run_spawn_group("test_boltz_parallel:_cpu_worker", [dict(value=1)], output)


def test_failed_child_stops_its_peer_and_retains_failure(tmp_path):
    with pytest.raises(ParallelWorkerError, match="intentional worker failure") as caught:
        run_spawn_group("test_boltz_parallel:_cpu_worker",
            [dict(value=1, fail=True), dict(value=2, seconds=20)], tmp_path / "failure",
            startup_timeout=45, run_timeout=5, cleanup_grace=2)
    report = caught.value.report
    assert report["status"] == "failed" and report["cleanup_complete"] and not report["requires_job_stop"]
    assert not any(row["alive"] for row in report["processes"])
    assert (tmp_path / "failure/worker-00/error.json").exists()


def test_run_timeout_stops_workers(tmp_path):
    with pytest.raises(ParallelWorkerError, match="run timeout") as caught:
        run_spawn_group("test_boltz_parallel:_cpu_worker", [dict(value=1, seconds=20)], tmp_path / "timeout",
                        startup_timeout=45, run_timeout=.2, cleanup_grace=2)
    assert caught.value.report["cleanup_complete"]
    assert not any(row["alive"] for row in caught.value.report["processes"])


def test_forced_cleanup_requires_stopping_the_gpu_job(tmp_path):
    with pytest.raises(ParallelWorkerError, match="run timeout") as caught:
        run_spawn_group("test_boltz_parallel:_cpu_worker", [dict(value=1, seconds=20, ignore_abort=True)],
                        tmp_path / "forced", startup_timeout=45, run_timeout=.1, cleanup_grace=.1)
    assert caught.value.report["cleanup_complete"]
    assert caught.value.report["requires_job_stop"]
    assert any(row["forced_termination"] for row in caught.value.report["processes"])
