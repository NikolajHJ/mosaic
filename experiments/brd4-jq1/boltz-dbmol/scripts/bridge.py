"""FP64 scalar/metrics and FP32 graph-gradient boundary to the DBMol worker."""

import hashlib
import subprocess
import threading
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mosaic.small_molecule.boltz2.bridge import AffinityWorker


PROTOCOL = "mosaic-boltz-dbmol-v1"
METRICS = (
    "dbmol_total_loss", "dbmol_affinity_loss", "dbmol_contact_loss",
    "dbmol_confidence_loss", "dbmol_weighted_affinity_loss",
    "dbmol_weighted_contact_loss", "dbmol_weighted_confidence_loss",
    "affinity_pred_value", "affinity_probability_binary", "affinity_logits_binary",
    "dbmol_contact_pocket_loss", "dbmol_contact_anchor_loss", "complex_plddt",
    "iptm", "complex_iplddt", "complex_ipde", "complex_pde", "confidence_score", "ptm",
)


class DBMolWorker(AffinityWorker):
    """Retain serial transport/IDs/timeouts; reject the old affinity protocol."""

    def __init__(self, command, *, stderr, timeout=1200):
        self.timeout, self.sequence = timeout, 0
        self.lock = threading.Lock()
        self.log = Path(stderr).open("w", encoding="utf-8")
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=self.log,
                                        text=True, bufsize=1)
        try:
            self.identity = self._read()
            if (self.identity.get("status") != "ready"
                    or self.identity.get("protocol") != PROTOCOL
                    or self.identity.get("optimized_output") != "dbmol_total_loss"
                    or self.identity.get("scalar_dtype") != "float64"
                    or self.identity.get("graph_dtype") != "float32"
                    or self.identity.get("gradient_dtype") != "float32"
                    or self.identity.get("metric_order") != list(METRICS)
                    or self.identity.get("n_atoms") != 25
                    or self.identity.get("recycling_steps") != 0
                    or self.identity.get("diffusion_gradient") is not False):
                raise ValueError(f"DBMol worker failed declared initialization: {self.identity}")
        except BaseException:
            self.close()
            raise

    def _evaluate_reply(self, x, edges, *, gradient, objective="dbmol_total_loss"):
        x, edges = np.asarray(x), np.asarray(edges)
        if (x.dtype != np.float32 or edges.dtype != np.float32
                or x.shape != (25, 6) or edges.shape != (300, 5)
                or not np.isfinite(x).all() or not np.isfinite(edges).all()):
            raise ValueError("DBMol worker requires finite exact FP32 X[25,6]/E[300,5]")
        if type(gradient) is not bool:
            raise ValueError("gradient must be an explicit boolean")
        graph_sha = hashlib.sha256(x.tobytes() + edges.tobytes()).hexdigest()
        reply = self.request("evaluate", X=x.tolist(), E=edges.tolist(),
                             gradient=gradient, graph_sha256=graph_sha,
                             objective=objective)
        if reply.get("graph_sha256") != graph_sha or reply.get("objective") != objective:
            raise ValueError("DBMol response graph/objective identity differs")
        metrics = reply.get("metrics")
        if not isinstance(metrics, dict) or set(metrics) != set(METRICS):
            raise ValueError("DBMol response lacks exact declared primitive schema")
        values = np.asarray(reply["values"], dtype=np.float64)
        all_metrics = np.asarray([metrics[key] for key in METRICS], dtype=np.float64)
        gx = (np.asarray(reply["gradient_X"], dtype=np.float32)
              if gradient else np.zeros_like(x))
        ge = (np.asarray(reply["gradient_E"], dtype=np.float32)
              if gradient else np.zeros_like(edges))
        if (values.shape != (3,) or all_metrics.shape != (len(METRICS),)
                or gx.shape != x.shape or ge.shape != edges.shape
                or any(not np.isfinite(value).all() for value in (values, all_metrics, gx, ge))
                or values[0] != metrics[objective]
                or values[1] != metrics["affinity_probability_binary"]
                or values[2] != metrics["affinity_logits_binary"]):
            raise FloatingPointError("Malformed/nonfinite DBMol values/derivatives")
        return values, all_metrics, gx, ge

    def evaluate(self, x, edges, *, gradient, objective="dbmol_total_loss"):
        """Independent FP64 primal plus FP32 raw selected-objective derivatives."""
        values, _, gx, ge = self._evaluate_reply(x, edges, gradient=gradient, objective=objective)
        return values, gx, ge

    def evaluate_metrics(self, x, edges, *, gradient, objective="dbmol_total_loss"):
        """All primitives/components, with the selected objective's gradients."""
        _, metrics, gx, ge = self._evaluate_reply(x, edges, gradient=gradient, objective=objective)
        return metrics, gx, ge


def dbmol_loss(worker, *, objective="dbmol_total_loss"):
    """Custom first-order VJP with no scalar narrowing or metric gradients.

    The surrounding driver must enable x64 BEFORE initializing graphs. Graph
    constructors must still explicitly request FP32. Reverse cotangents are
    cast back to the FP32 graph derivative type, as required by JAX's VJP shape
    contract. Higher derivatives and JVP through this callback are unsupported;
    allocated native raw FD and explicit bridge checks validate the boundary.
    """
    if not jax.config.jax_enable_x64:
        raise ValueError("Enable jax_enable_x64 before DBMol graph initialization")
    if worker.identity.get("protocol") != PROTOCOL:
        raise ValueError("DBMol loss requires the declared composite worker")
    if objective not in ("dbmol_affinity_loss", "dbmol_contact_loss",
                         "dbmol_confidence_loss", "dbmol_total_loss"):
        raise ValueError("Only declared DBMol components/total are loss objectives")
    objective_index = METRICS.index(objective)

    def callback(x, edges, gradient):
        if x.dtype != jnp.float32 or edges.dtype != jnp.float32:
            raise ValueError("DBMol JAX graph must remain FP32")
        shapes = (jax.ShapeDtypeStruct((len(METRICS),), jnp.float64),
                  jax.ShapeDtypeStruct(x.shape, jnp.float32),
                  jax.ShapeDtypeStruct(edges.shape, jnp.float32))
        return jax.pure_callback(
            lambda a, b: worker.evaluate_metrics(a, b, gradient=gradient, objective=objective),
            shapes, x, edges, vmap_method="sequential")

    @jax.custom_vjp
    def native(x, edges):
        metrics, _, _ = callback(x, edges, False)
        return metrics[objective_index], jax.lax.stop_gradient(metrics)

    def forward(x, edges):
        metrics, gx, ge = callback(x, edges, True)
        return (metrics[objective_index], jax.lax.stop_gradient(metrics)), (gx, ge)

    def backward(saved, cotangents):
        gx, ge = saved
        scalar, _ = cotangents
        return (scalar.astype(gx.dtype) * gx).astype(gx.dtype), (scalar.astype(ge.dtype) * ge).astype(ge.dtype)

    native.defvjp(forward, backward)

    def loss(graph, *, key):
        del key  # No sampling, pose update or MSA resampling occurs here.
        value, metrics = native(graph.X, graph.E_upper)
        auxiliary = {name: jax.lax.stop_gradient(metrics[index])
                     for index, name in enumerate(METRICS)}
        auxiliary[objective] = value
        return value, auxiliary

    return loss
