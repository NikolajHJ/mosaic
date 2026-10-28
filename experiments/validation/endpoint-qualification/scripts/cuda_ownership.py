"""Bounded two-client CUDA admission, before native/model initialization.

The coordinator is standard-library only.  The two isolated workers establish
real concurrent JAX CUDA ownership in the scheduler-provided MPS allocation;
they load no model/context and perform no optimizer updates.  This engineering
smoke does not replace any native precision, gradient, frozen-state or pilot
gate.  No MPS daemon or GPU compute mode is changed by this module.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import stat
import subprocess
import sys
import time
import uuid


VERSION = "two-client-jax-cuda-ownership-v1"
GPU_REQUEST = "num=1:mode=exclusive_process:mps=yes"
MAX_SECONDS = 240
MAX_LINE = 16384
MAX_STDERR = 128 * 1024
MAX_BYTES = 1024 * 1024
ROLES = ("production_fp32", "reference_fp64")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _write(path, value):
    with Path(path).open("x", encoding="utf8", newline="\n") as stream:
        json.dump(value, stream, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def binding(path):
    path = Path(path)
    info = path.lstat()
    require(path.is_absolute() and stat.S_ISREG(info.st_mode) and not path.is_symlink()
            and ".." not in path.parts and all(not p.is_symlink() for p in path.parents),
            "CUDA ownership identity must be a regular absolute unlinked file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return {"path": str(path), "bytes": info.st_size, "sha256": digest.hexdigest()}


def _bindings(items):
    require(isinstance(items, dict) and {"source_manifest", "plan", "context_identity"} <= set(items)
            and len(items) <= 12, "CUDA smoke requires source, plan and authenticated context identities")
    actual = {}
    for name, item in items.items():
        require(isinstance(name, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", name)
                and isinstance(item, dict) and set(item) == {"path", "bytes", "sha256"},
                "Malformed CUDA smoke input binding")
        require(isinstance(item["bytes"], int) and not isinstance(item["bytes"], bool)
                and item["bytes"] >= 0 and isinstance(item["sha256"], str)
                and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]), "Malformed CUDA smoke input digest")
        actual[name] = binding(item["path"])
        require(actual[name] == item, "CUDA smoke input identity changed: " + name)
    return actual


def allocation(environment, expected_job_id):
    require(str(expected_job_id).isdigit() and environment.get("LSB_JOBID") == str(expected_job_id),
            "CUDA smoke does not belong to this allocated job")
    require(environment.get("LSB_QUEUE") == "gpul40s" and environment.get("LSB_DJOB_NUMPROC") == "4",
            "CUDA smoke requires the declared gpul40s four-slot allocation")
    visible = environment.get("CUDA_VISIBLE_DEVICES", "")
    require(isinstance(visible, str) and re.fullmatch(r"(?:[0-9]+|GPU-[A-Za-z0-9-]+)", visible),
            "CUDA smoke requires exactly one explicitly visible GPU")
    return {"job_id": str(expected_job_id), "queue": "gpul40s", "cpu_slots": 4,
            "cuda_visible_devices": visible,
            "mps_environment": {key: environment.get(key) for key in
                                ("CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY", "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")}}


def _submission(path, own, identities):
    proof = binding(path)
    row = json.loads(Path(path).read_text(encoding="utf8"))
    require(row.get("status") == "submitted" and row.get("returncode") == 0
            and row.get("job_id") == own["job_id"]
            and row.get("source_manifest_sha256") == identities["source_manifest"]["sha256"],
            "CUDA smoke submission/source ownership mismatch")
    command = row.get("argv")
    require(isinstance(command, list) and command and command[0] == "bsub"
            and all(isinstance(part, str) for part in command), "Missing actual CUDA smoke submission command")
    for flag, value in (("-q", "gpul40s"), ("-n", "4"), ("-gpu", GPU_REQUEST)):
        require(command.count(flag) == 1 and command.index(flag) + 1 < len(command)
                and command[command.index(flag) + 1] == value, "CUDA smoke resource mismatch: " + flag)
    require(command.count("span[hosts=1]") == 1 and command.count("rusage[mem=8GB]") == 1,
            "CUDA smoke host or memory resources changed")
    return proof


def worker_command(role, nonce):
    require(role in ROLES and re.fullmatch(r"[0-9a-f]{32}", nonce), "Invalid CUDA worker identity")
    return [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--worker", role, "--nonce", nonce]


def worker_environment(environment, role):
    require(role in ROLES, "Unknown CUDA worker precision")
    result = dict(environment)
    result.update({"JAX_PLATFORMS": "cuda,cpu", "JAX_ENABLE_X64": "1" if role == "reference_fp64" else "0",
                   "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "JAX_ENABLE_COMPILATION_CACHE": "false",
                   "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "OPENBLAS_NUM_THREADS": "1"})
    return result


def _budget(directory):
    total = 0
    for path in directory.iterdir():
        info = path.lstat()
        require(stat.S_ISREG(info.st_mode) and not path.is_symlink(), "Unexpected CUDA smoke output type")
        total += info.st_size
    require(total <= MAX_BYTES, "CUDA smoke output budget exceeded")


def _receive(child, deadline, directory):
    with selectors.DefaultSelector() as selector:
        selector.register(child["process"].stdout, selectors.EVENT_READ)
        while True:
            _budget(directory)
            require(time.monotonic() < deadline, "CUDA smoke worker deadline exceeded")
            require(len(child["buffer"]) <= MAX_LINE, "Oversize CUDA smoke protocol line")
            if b"\n" in child["buffer"]:
                line, child["buffer"] = child["buffer"].split(b"\n", 1)
                require(line and len(line) <= MAX_LINE, "Invalid CUDA smoke protocol line")
                return json.loads(line.decode("utf8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            if not selector.select(min(.25, max(0, deadline - time.monotonic()))):
                require(child["process"].poll() is None, "CUDA smoke worker exited before response")
                continue
            block = os.read(child["process"].stdout.fileno(), MAX_LINE + 1)
            require(block, "CUDA smoke worker closed protocol")
            child["buffer"] += block


def _validate_message(row, own, role, nonce, sequence, previous=None):
    require(isinstance(row, dict) and set(row) == {"version", "status", "role", "nonce", "pid", "job_id", "queue",
            "cuda_visible_devices", "platform", "device_kind", "device_id", "device_count", "x64_enabled", "sequence", "measurements"},
            "CUDA smoke protocol schema changed")
    require(row["version"] == VERSION and row["status"] == "cuda_work_complete" and row["role"] == role
            and row["nonce"] == nonce and row["sequence"] == sequence
            and isinstance(row["pid"], int) and not isinstance(row["pid"], bool) and row["pid"] > 0,
            "CUDA smoke worker identity/sequence changed")
    require(row["job_id"] == own["job_id"] and row["queue"] == own["queue"]
            and row["cuda_visible_devices"] == own["cuda_visible_devices"]
            and row["platform"] in ("gpu", "cuda") and row["device_count"] == 1
            and row["device_id"] == 0 and isinstance(row["device_kind"], str) and "L40S" in row["device_kind"]
            and row["x64_enabled"] is (role == "reference_fp64"), "CUDA smoke allocation/device/precision mismatch")
    expected = {"float32": {"dtype": "float32", "sum_squares": sum((n + sequence)**2 for n in (1, 2, 4)),
                           "gradient": [2 * (n + sequence) for n in (1, 2, 4)]}}
    if role == "reference_fp64":
        expected["float64"] = {"dtype": "float64", "sum_squares": sum((n + sequence)**2 for n in (.5, 1.5, 2.5)),
                               "gradient": [2 * (n + sequence) for n in (.5, 1.5, 2.5)]}
    require(row["measurements"] == expected, "CUDA smoke actual dtype/primal/AD repeat mismatch")
    if previous is not None:
        require(all(row[key] == previous[key] for key in row if key not in ("sequence", "measurements")),
                "CUDA smoke process identity changed during co-residency")
    return row


def _send(child, value):
    require(child["process"].poll() is None, "CUDA smoke worker died before challenge")
    child["process"].stdin.write((json.dumps(value) + "\n").encode("utf8"))
    child["process"].stdin.flush()


def _cleanup(children):
    outcomes = []
    for child in children:
        process = child["process"]
        forced, error = False, None
        try:
            if process.poll() is None:
                forced = True
                process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        except BaseException as exc:
            error = type(exc).__name__ + ": " + str(exc)
        outcomes.append({"role": child["role"], "pid": process.pid, "returncode": process.returncode,
                         "forced_termination": forced, "reaped": process.poll() is not None, "error": error})
        for stream in (process.stdin, process.stdout, child["stderr"]):
            try:
                stream.close()
            except OSError as exc:
                outcomes[-1]["error"] = type(exc).__name__ + ": " + str(exc)
    return outcomes


def run_probe(directory, *, submission, expected_job_id, input_bindings, deadline):
    """Run once after storage admission, before importing/starting native models.

    ``deadline`` is the caller's existing monotonic phase deadline.  Bindings
    include the own source manifest, plan and an already authenticated original
    context/frozen/adoption proof document.  All are rehashed before and after.
    A failed or incomplete probe raises; existing output is never overwritten.
    """
    started = time.monotonic()
    require(isinstance(deadline, (int, float)) and deadline > started, "No CUDA smoke runtime remains")
    probe_deadline = min(deadline - 12, started + MAX_SECONDS)
    require(probe_deadline > started, "No bounded CUDA cleanup runtime remains")
    directory = Path(directory)
    require(directory.is_absolute() and not directory.exists() and not directory.is_symlink()
            and all(not p.is_symlink() for p in directory.parents), "CUDA smoke output already exists or has linked parent")
    directory.mkdir(exist_ok=False)
    children, records, error, cleanup = [], {}, None, []
    own, identities, identities_after, submission_proof = {}, {}, {}, None
    try:
        own = allocation(os.environ, expected_job_id)
        identities = _bindings(input_bindings)
        submission_proof = _submission(submission, own, identities)
        nonce = uuid.uuid4().hex
        _write(directory / "invocation.json", {"version": VERSION, "allocation": own, "input_bindings": identities,
            "submission": submission_proof, "gpu_request": GPU_REQUEST, "nonce": nonce,
            "max_seconds": MAX_SECONDS, "max_persistent_bytes": MAX_BYTES, "optimizer_updates": 0,
            "native_or_gradient_gate_replaced": False, "commands": {role: worker_command(role, nonce) for role in ROLES}})
        for role in ROLES:
            for child in children:
                require(child["process"].poll() is None, "First CUDA context ended before the second started")
            log = (directory / (role + ".stderr.log")).open("xb")
            try:
                process = subprocess.Popen(worker_command(role, nonce), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                           stderr=log, env=worker_environment(os.environ, role), bufsize=0)
            except BaseException:
                log.close()
                raise
            child = {"role": role, "process": process, "stderr": log, "buffer": b""}
            children.append(child)
            records[role] = [_validate_message(_receive(child, probe_deadline, directory), own, role, nonce, 0)]
            require(records[role][0]["pid"] == process.pid, "CUDA worker response does not belong to owned child")
        require(children[0]["process"].poll() is None and children[1]["process"].poll() is None
                and records[ROLES[0]][0]["pid"] != records[ROLES[1]][0]["pid"]
                and records[ROLES[0]][0]["device_kind"] == records[ROLES[1]][0]["device_kind"],
                "Two distinct CUDA clients are not simultaneously alive on the declared GPU")
        for sequence in (1, 2):
            for child in children:
                _send(child, {"command": "repeat", "sequence": sequence, "nonce": nonce})
            for child in children:
                role = child["role"]
                row = _validate_message(_receive(child, probe_deadline, directory), own, role, nonce, sequence, records[role][0])
                records[role].append(row)
            require(all(child["process"].poll() is None for child in children), "CUDA co-residency ended during repeats")
        for child in children:
            _send(child, {"command": "stop", "nonce": nonce})
        for child in children:
            child["process"].wait(timeout=max(.01, min(5, probe_deadline - time.monotonic())))
            require(child["process"].returncode == 0 and child["buffer"] == b""
                    and child["process"].stdout.read(MAX_LINE + 1) == b"", "CUDA smoke worker did not exit cleanly")
        require(_bindings(input_bindings) == identities and binding(submission) == submission_proof,
                "CUDA smoke source/context/submission changed")
    except BaseException as exc:
        error = type(exc).__name__ + ": " + str(exc)
    finally:
        try:
            cleanup = _cleanup(children)
            require(all(row["reaped"] and row["error"] is None for row in cleanup), "Owned child cleanup was incomplete")
        except BaseException as exc:
            error = (error + "; " if error else "") + "Owned CUDA cleanup failed: " + type(exc).__name__ + ": " + str(exc)
        if identities:
            try:
                identities_after = _bindings(input_bindings)
                require(identities_after == identities and binding(submission) == submission_proof,
                        "CUDA smoke identities changed during failure/cleanup")
            except BaseException as exc:
                error = (error + "; " if error else "") + "Input preservation failed: " + type(exc).__name__ + ": " + str(exc)
        receipt = {"version": VERSION, "status": "failed" if error else "passed_two_client_cuda_ownership",
                   "error": error, "job_id": str(expected_job_id), "allocation": own, "submission": submission_proof,
                   "input_bindings": identities, "input_bindings_after": identities_after, "records": records, "cleanup": cleanup,
                   "elapsed_seconds": time.monotonic() - started, "optimizer_updates": 0,
                   "native_or_gradient_gate_replaced": False, "scientific_results_changed": False,
                   "mps_daemon_or_compute_mode_changed": False}
        _write(directory / "assessment.json", receipt)
    _budget(directory)
    require(error is None and len(cleanup) == 2 and all(row["reaped"] and not row["forced_termination"] for row in cleanup),
            "Two-client CUDA ownership admission failed: " + str(error))
    return receipt


def _worker(role, nonce):
    require(role in ROLES and re.fullmatch(r"[0-9a-f]{32}", nonce), "Invalid CUDA smoke worker")
    allocation(os.environ, os.environ.get("LSB_JOBID", ""))
    require(os.environ.get("JAX_ENABLE_X64") == ("1" if role == "reference_fp64" else "0")
            and os.environ.get("JAX_PLATFORMS") == "cuda,cpu"
            and os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE") == "false", "CUDA worker runtime differs")
    import resource
    import signal
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_STDERR, MAX_STDERR))
    signal.alarm(MAX_SECONDS + 12)
    import jax
    import jax.numpy as jnp
    devices = jax.devices("gpu")
    require(len(devices) == 1 and "L40S" in devices[0].device_kind, "No unique allocated L40S CUDA device")
    device = devices[0]
    def measure(sequence):
        values = {}
        types = (("float32", jnp.float32, (1, 2, 4)),)
        if role == "reference_fp64":
            types += (("float64", jnp.float64, (.5, 1.5, 2.5)),)
        for name, dtype, starts in types:
            array = jax.device_put(jnp.asarray([number + sequence for number in starts], dtype=dtype), device)
            primal = jnp.sum(array * array).block_until_ready()
            gradient = jax.grad(lambda points: jnp.sum(points * points))(array).block_until_ready()
            require(array.device == device and primal.device == device and gradient.device == device,
                    "CUDA smoke computation fell back from the allocated GPU")
            values[name] = {"dtype": str(array.dtype), "sum_squares": float(primal), "gradient": gradient.tolist()}
        row = {"version": VERSION, "status": "cuda_work_complete", "role": role, "nonce": nonce, "pid": os.getpid(),
               "job_id": os.environ["LSB_JOBID"], "queue": os.environ["LSB_QUEUE"],
               "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "platform": device.platform,
               "device_kind": device.device_kind, "device_id": device.id, "device_count": len(devices),
               "x64_enabled": bool(jax.config.x64_enabled), "sequence": sequence, "measurements": values}
        print(json.dumps(row, allow_nan=False), flush=True)
    measure(0)
    for expected_sequence in (1, 2):
        line = sys.stdin.buffer.readline(MAX_LINE + 1)
        require(line.endswith(b"\n") and len(line) <= MAX_LINE, "Invalid CUDA worker challenge")
        require(json.loads(line) == {"command": "repeat", "sequence": expected_sequence, "nonce": nonce}, "CUDA challenge changed")
        measure(expected_sequence)
    stop = sys.stdin.buffer.readline(MAX_LINE + 1)
    require(stop.endswith(b"\n") and len(stop) <= MAX_LINE
            and json.loads(stop) == {"command": "stop", "nonce": nonce}, "CUDA stop changed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True, choices=ROLES)
    parser.add_argument("--nonce", required=True)
    arguments = parser.parse_args()
    _worker(arguments.worker, arguments.nonce)
