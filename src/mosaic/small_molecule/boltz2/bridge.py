"""Explicit first-order JAX boundary to a persistent native PyTorch worker.

The worker lives in an isolated, pinned environment on the SAME allocation.
Mosaic keeps its own graph, optimizer and chemistry. Each AD call returns the
native scalar plus unmodified raw X/E derivatives; an independent primal call
does no backward. There is no gradient cache, learned bridge or VAE dependency.
"""

import hashlib
import json
import selectors
import subprocess
import threading
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


class AffinityWorker:
    """One serial model owner with a request identity and durable stderr."""

    def __init__(self, command, *, stderr, timeout=600):
        self.timeout, self.sequence = timeout, 0
        self.lock = threading.Lock()
        self.log = Path(stderr).open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
        )
        try:
            self.identity = self._read()
            if (
                self.identity.get("status") != "ready"
                or self.identity.get("protocol") != "mosaic-boltz-affinity-v1"
            ):
                raise ValueError(f"Native worker failed to initialize: {self.identity}")
        except BaseException:
            self.close()
            raise

    def _read(self):
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(self.timeout):
                raise TimeoutError("Native Boltz worker response timed out")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Native Boltz worker exited: {self.process.poll()}")
        return json.loads(line)

    def request(self, action, **values):
        with self.lock:
            self.sequence += 1
            request = {"id": self.sequence, "action": action, **values}
            self.process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
            self.process.stdin.flush()
            reply = self._read()
            if reply.get("id") != self.sequence or reply.get("status") != "passed":
                raise RuntimeError(f"Native Boltz request failed: {reply}")
            return reply

    def evaluate(self, x, edges, *, gradient):
        x, edges = np.asarray(x), np.asarray(edges)
        if (
            x.dtype != np.float32
            or edges.dtype != np.float32
            or x.shape != (6, 6)
            or edges.shape != (15, 5)
        ):
            raise ValueError("Worker bridge supports exact FP32 Mosaic six-slot graphs")
        graph_sha = hashlib.sha256(x.tobytes() + edges.tobytes()).hexdigest()
        reply = self.request(
            "evaluate",
            X=x.tolist(),
            E=edges.tolist(),
            gradient=gradient,
            graph_sha256=graph_sha,
        )
        if reply.get("graph_sha256") != graph_sha:
            raise ValueError("Native worker response graph identity differs")
        values = np.asarray(reply["values"], dtype=np.float32)
        gx = (
            np.asarray(reply["gradient_X"], dtype=np.float32)
            if gradient
            else np.zeros_like(x)
        )
        ge = (
            np.asarray(reply["gradient_E"], dtype=np.float32)
            if gradient
            else np.zeros_like(edges)
        )
        if (
            values.shape != (3,)
            or gx.shape != x.shape
            or ge.shape != edges.shape
            or any(not np.isfinite(v).all() for v in (values, gx, ge))
        ):
            raise FloatingPointError(
                "Native worker returned malformed/nonfinite values or derivatives"
            )
        return values, gx, ge

    def close(self):
        if hasattr(self, "process") and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=20)
        if hasattr(self, "log"):
            self.log.close()


def affinity_loss(worker):
    """Return a Mosaic graph loss with an explicit native first-order VJP.

    Quantitative affinity is the only differentiable output. Binary metrics are
    logging-only, explicitly stopped. Higher derivatives/JVP are unsupported;
    actual finite differences of the independent worker primal validate the VJP.
    """
    shapes = (
        jax.ShapeDtypeStruct((3,), jnp.float32),
        jax.ShapeDtypeStruct((6, 6), jnp.float32),
        jax.ShapeDtypeStruct((15, 5), jnp.float32),
    )

    def callback(x, edges, gradient):
        return jax.pure_callback(
            lambda a, b: worker.evaluate(a, b, gradient=gradient),
            shapes,
            x,
            edges,
            vmap_method="sequential",
        )

    @jax.custom_vjp
    def native(x, edges):
        values, _, _ = callback(x, edges, False)
        return values[0], jax.lax.stop_gradient(values[1:])

    def forward(x, edges):
        values, gx, ge = callback(x, edges, True)
        return (values[0], jax.lax.stop_gradient(values[1:])), (gx, ge)

    def backward(saved, cotangents):
        gx, ge = saved
        scalar, _ = cotangents
        return scalar * gx, scalar * ge

    native.defvjp(forward, backward)

    def loss(graph, *, key):
        del key  # Fixed prepared MSA/pose, eval native path has no sampling.
        value, metrics = native(graph.X, graph.E_upper)
        return value, {
            "affinity_pred_value": value,
            "affinity_probability_binary": jax.lax.stop_gradient(metrics[0]),
            "affinity_mean_binary_logit": jax.lax.stop_gradient(metrics[1]),
        }

    return loss
