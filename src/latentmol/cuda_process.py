"""Process ownership checks for serial CUDA child workflows.

LSF's ``exclusive_process`` allocation permits one process to own the device's
CUDA context. Freeing model tensors or allocator cache does not end that process
context. A coordinator must therefore stay CUDA-free while each GPU child runs
and exits. This helper only inspects PyTorch's cached initialization flag; it
does not query devices or create a CUDA context itself.
"""

import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def require_cuda_free_coordinator():
    """Fail before spawning a CUDA child if this process already owns CUDA.

    Importing PyTorch on the CPU is permitted. ``is_available()``, device
    selection and GPU memory queries are deliberately absent: they are not a
    substitute for a process-lifetime boundary on an exclusive GPU.
    """
    torch = sys.modules.get("torch")
    initialized = bool(torch is not None and torch.cuda.is_initialized())
    if initialized:
        raise RuntimeError(
            "CUDA child requires a CUDA-free coordinator; the parent already "
            "initialized CUDA. End the GPU phase in a separate child process "
            "before launching native prediction on an exclusive-process GPU."
        )
    return {"pid": os.getpid(), "torch_imported": torch is not None,
            "cuda_initialized": False, "policy": "cuda-free-coordinator-v1"}


def _end_owned_group(process, *, grace_seconds=10):
    """Stop only the new session created for our direct child and its workers."""
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
        # A normal leader exit does not imply its native worker children exited.
        stop(True)
    process.wait()


def run_cuda_child(command, log_path, *, timeout_seconds):
    """Run one GPU phase to process exit while the coordinator stays CUDA-free.

    The log is immutable per attempt. A timeout/interrupt kills our entire child
    session, and successful leaders also have any surviving worker group ended.
    The caller must require success before launching the next phase.
    """
    if type(timeout_seconds) is not int or timeout_seconds < 1:
        raise ValueError("CUDA child timeout must be a positive integer")
    before = require_cuda_free_coordinator()
    command = list(map(str, command)); log_path = Path(log_path)
    if not command:
        raise ValueError("CUDA child command is empty")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic(); timed_out = False; interruption = None
    with log_path.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            process.wait(timeout=timeout_seconds)
        except BaseException as error:
            timed_out = isinstance(error, subprocess.TimeoutExpired)
            if not timed_out:
                interruption = error
        finally:
            _end_owned_group(process)
    result = dict(command=command, returncode=process.returncode, timed_out=timed_out,
        seconds=time.monotonic() - started, log=str(log_path.resolve()), child_pid=process.pid,
        coordinator_before=before, coordinator_after=require_cuda_free_coordinator())
    if interruption is not None:
        raise interruption
    return result


def require_cuda_child(result):
    """A timed-out/failed phase cannot release the next GPU process."""
    if result["timed_out"] or result["returncode"] != 0:
        raise RuntimeError("CUDA child phase failed; inspect " + result["log"])
