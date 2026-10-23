"""Serial, logged HPC subprocesses with explicit timeout and GPU-process cleanup.

Native preparation, latent optimization and native rescoring use separate child
processes. The orchestration process never loads a scorer on the GPU, so a later
native prediction cannot overlap resident affinity-model allocations.
"""

import json
import os
from pathlib import Path
import signal
import subprocess
import time


def _terminate_group(process, *, grace_seconds=10):
    """Terminate our child session even when its leader exits before workers.

    Waiting for Popen's direct child does not wait for grandchildren. Always
    send the final group kill after that wait, including its successful return,
    so a surviving native worker cannot retain the GPU during the next trial.
    """
    def stop(force):
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
            except ProcessLookupError:
                pass
        elif process.poll() is None:
            (process.kill if force else process.terminate)()

    stop(False)
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    finally:
        stop(True)
    process.wait()


def run_logged(command, log_path: Path, *, timeout_seconds: int) -> dict:
    """Run one child to termination; preserve failures and its ExperimentRun path.

    The Unix process group is also terminated on a timeout/interrupt, including
    native worker children. Logs use exclusive creation; callers must use fresh
    paths rather than appending a different attempt to an old result.
    """
    if type(timeout_seconds) is not int or timeout_seconds < 1:
        raise ValueError("Child timeout must be a positive integer number of seconds")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = list(map(str, command))
    started = time.monotonic()
    timed_out = False
    code = None
    failure = None
    interruption = None
    with log_path.open("x", encoding="utf-8") as log:
        log.write("COMMAND=" + json.dumps(command) + "\n")
        log.flush()
        try:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        except OSError as exc:
            # A launch failure is still a durable attempt; the batch controller
            # can retain this seed/count and continue without a replacement.
            failure = {"type": type(exc).__name__, "message": str(exc)}
        else:
            try:
                code = process.wait(timeout=timeout_seconds)
            except BaseException as exc:
                _terminate_group(process)
                code = process.returncode
                timed_out = isinstance(exc, subprocess.TimeoutExpired)
                failure = {"type": type(exc).__name__, "message": str(exc)}
                if not timed_out:
                    interruption = exc
    paths = [line.removeprefix("RUN_DIR=").strip()
             for line in log_path.read_text(errors="replace").splitlines()
             if line.startswith("RUN_DIR=")]
    result = {"command": command, "returncode": code, "timed_out": timed_out,
              "seconds": time.monotonic() - started, "log": str(log_path.resolve()),
              "run_paths": paths, "failure": failure,
              "interrupted": interruption is not None}
    # Save before propagating an interrupt: a cancelled child remains auditable
    # even when the parent has no opportunity to update its trial table.
    with log_path.open("a", encoding="utf-8") as log:
        log.write("CHILD_RESULT=" + json.dumps(result) + "\n")
    print("CHILD_RESULT=" + json.dumps(result), flush=True)
    if interruption is not None:
        raise interruption
    return result


def require_child(result: dict, *, experiment=False) -> Path | None:
    """A zero child exit alone does not establish a completed ExperimentRun."""
    if result["timed_out"] or result["returncode"] != 0:
        raise RuntimeError(f"Child failed or timed out; see {result['log']}")
    if not experiment:
        return None
    if len(result["run_paths"]) != 1:
        raise ValueError("Expected exactly one child ExperimentRun")
    path = Path(result["run_paths"][0])
    if json.loads((path / "run.json").read_text()).get("status") != "completed":
        raise ValueError("Child process exited without a completed experiment record")
    return path
