"""Portable, serial subprocess transport; native dependencies stay in the worker."""

import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import tempfile
from threading import Lock, Thread

import numpy as np

from .graph import ELEMENTS, Prediction, digest, validate_arrays

GUACAMOL_SOURCE_REVISION = "365bda9affadd5c2307014a0532ddaa244399441"
GUACAMOL_CHECKPOINT_SHA256 = (
    "bc97d52ccf33a20d991f98349a1d37662273ab1c3e3f9777b938f9f144bc61f5"
)
PROTOCOL = "mosaic-defog-teacher-v1"


class TeacherError(RuntimeError):
    """A worker, transport, model-identity or frozen-state failure."""


class DeFoGTeacher:
    """Frozen official GuacaMol teacher, running on CPU in its own interpreter.

    Source and weights are explicit local assets; nothing is downloaded.
    The defaults authenticate the checkpoint/source used in BENCHMARKS.md.
    Use a context manager to verify frozen state and stop the worker on exit.
    """

    def __init__(
        self,
        source,
        checkpoint,
        *,
        python=None,
        elements=ELEMENTS,
        feature_policy="hard_aux_soft_core",
        threads=2,
        timeout=300,
        source_revision=GUACAMOL_SOURCE_REVISION,
        checkpoint_sha256=GUACAMOL_CHECKPOINT_SHA256,
        stderr=None,
    ):
        source, checkpoint = Path(source).resolve(), Path(checkpoint).resolve()
        if not (source / "src/graph_discrete_flow_model.py").is_file():
            raise FileNotFoundError(f"Expected the official DeFoG source at {source}")
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if feature_policy not in ("hard_aux_soft_core", "native_soft"):
            raise ValueError("feature_policy must be hard_aux_soft_core or native_soft")
        if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
            raise ValueError("threads must be a positive integer")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        native_elements = (
            "C",
            "N",
            "O",
            "F",
            "B",
            "Br",
            "Cl",
            "I",
            "P",
            "S",
            "Se",
            "Si",
        )
        self.elements = tuple(elements)
        if (
            not self.elements
            or len(set(self.elements)) != len(self.elements)
            or any(e not in native_elements for e in self.elements)
        ):
            raise ValueError(
                "elements must be a distinct subset of the native GuacaMol vocabulary"
            )
        self.timeout, self.queries, self._sequence = timeout, 0, 0
        self._closed = False
        self._stderr_diagnostics = ""
        self._lock, self._responses = Lock(), Queue()
        self._stderr = (
            Path(stderr).open("x+", encoding="utf-8")
            if stderr is not None
            else tempfile.TemporaryFile(mode="w+", encoding="utf-8")
        )
        environment = dict(os.environ)
        environment.update(
            CUDA_VISIBLE_DEVICES="",
            OMP_NUM_THREADS=str(threads),
            MKL_NUM_THREADS=str(threads),
            PYTHONUNBUFFERED="1",
            PYTHONDONTWRITEBYTECODE="1",
        )
        # Never expose the coordinator's site-packages to another interpreter.
        # The entry script loads only this pure-Python package by absolute path.
        environment["PYTHONPATH"] = os.pathsep.join(
            map(str, (source, source / "src"))
        )
        environment.pop("PYTHONHOME", None)
        command = [
            str(python or sys.executable),
            "-u",
            str(Path(__file__).with_name("_worker_entry.py")),
            "--source",
            str(source),
            "--checkpoint",
            str(checkpoint),
            "--elements",
            ",".join(self.elements),
            "--feature-policy",
            feature_policy,
            "--threads",
            str(threads),
        ]
        if source_revision is not None:
            command += ["--source-revision", source_revision]
        if checkpoint_sha256 is not None:
            command += ["--checkpoint-sha256", checkpoint_sha256]
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._stderr,
                env=environment,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
            self._reader = Thread(target=self._read_output, daemon=True)
            self._reader.start()
            self.identity = self._receive()
            if (
                self.identity.get("status") != "ready"
                or self.identity.get("protocol") != PROTOCOL
                or self.identity.get("elements") != list(self.elements)
                or self.identity.get("feature_policy") != feature_policy
                or (
                    source_revision is not None
                    and self.identity.get("source_revision") != source_revision
                )
                or (
                    checkpoint_sha256 is not None
                    and self.identity.get("checkpoint_sha256") != checkpoint_sha256
                )
            ):
                raise TeacherError("Teacher asset/policy handshake mismatch")
        except BaseException:
            self.close()
            raise

    def _read_output(self):
        try:
            while True:
                line = self._process.stdout.readline(64 * 1024 * 1024)
                if not line:
                    self._responses.put(TeacherError("Teacher exited before replying"))
                    return
                if not line.endswith("\n"):
                    raise TeacherError("Teacher response exceeded the line budget")
                self._responses.put(json.loads(line))
        except Exception as error:
            self._responses.put(TeacherError(f"Invalid teacher response: {error}"))

    def _receive(self):
        try:
            reply = self._responses.get(timeout=self.timeout)
        except Empty as error:
            self.close()
            raise TeacherError(f"Teacher timed out after {self.timeout}s") from error
        if isinstance(reply, Exception):
            self.close()
            diagnostics = (
                f"\nWorker stderr:\n{self._stderr_diagnostics}"
                if self._stderr_diagnostics
                else ""
            )
            raise TeacherError(f"{reply}{diagnostics}") from reply
        if not isinstance(reply, dict):
            self.close()
            raise TeacherError("Teacher response must be an object")
        if reply.get("status") == "failed":
            self.close()
            raise TeacherError(reply.get("error", "Teacher failed"))
        return reply

    def _request(self, action, **values):
        if self._closed:
            raise TeacherError("Teacher is closed")
        self._sequence += 1
        try:
            self._process.stdin.write(
                json.dumps(
                    {"id": self._sequence, "action": action, **values}, allow_nan=False
                )
                + "\n"
            )
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            self.close()
            raise TeacherError("Could not send teacher request") from error
        reply = self._receive()
        if reply.get("id") != self._sequence or reply.get("status") != "passed":
            self.close()
            raise TeacherError("Teacher response ID/status mismatch")
        return reply

    def predict(self, graph, *, time=0.8):
        """Query exactly once, preserving slot/category order and student arrays."""
        if not math.isfinite(time) or not 0 <= time <= 1:
            raise ValueError("Teacher time must lie in [0,1]")
        x, e = validate_arrays(graph.X, graph.E_upper, n_elements=len(self.elements))
        payload = {
            "X": np.asarray(x, np.float32).tolist(),
            "E": np.asarray(e, np.float32).tolist(),
        }
        graph_sha = digest(payload)
        with self._lock:
            reply = self._request(
                "predict", **payload, time=float(time), graph_sha256=graph_sha
            )
            if (
                reply.get("graph_sha256") != graph_sha
                or reply.get("time") != time
                or reply.get("forward_calls") != 1
                or reply.get("target_sha256") != digest(reply.get("target"))
            ):
                self.close()
                raise TeacherError(
                    "Teacher graph, time, forward-count or target mismatch"
                )
            prediction = Prediction(
                np.asarray(reply["target"]["X"], np.float32),
                np.asarray(reply["target"]["E"], np.float32),
                float(time),
                reply.get("metrics", {}),
            )
            if prediction.X.shape != x.shape or prediction.E_upper.shape != e.shape:
                self.close()
                raise TeacherError("Teacher changed graph slots or categories")
            self.queries += 1
            return prediction

    def verify_frozen(self):
        """Re-hash the native state and reconcile all successful worker forwards."""
        with self._lock:
            result = self._request("frozen")
            if (
                result.get("state_digest") != self.identity.get("state_digest")
                or result.get("queries") != self.queries
                or result.get("forwards") != self.queries
            ):
                self.close()
                raise TeacherError("Frozen model state or query/forward count changed")
            return result

    def close(self):
        """Stop the worker, including after partial startup or failed queries."""
        if self._closed:
            return
        self._closed = True
        process = getattr(self, "_process", None)
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    stream.close()
            reader = getattr(self, "_reader", None)
            if reader is not None:
                reader.join(timeout=1)
        self._stderr.flush()
        buffer = self._stderr.buffer
        size = buffer.seek(0, os.SEEK_END)
        buffer.seek(max(0, size - 8192))
        self._stderr_diagnostics = (
            buffer.read().decode("utf-8", errors="replace").strip()
        )
        self._stderr.close()

    def __enter__(self):
        return self

    def __exit__(self, error_type, error, traceback):
        try:
            if error_type is None:
                self.verify_frozen()
        finally:
            self.close()
