"""Bounded diagnostic capture of one wholly masked native attention row.

The context manager calls native attention unchanged and restores its callable
on exit. It captures detached inputs and, if backward reaches the chosen row,
its actual output cotangent. Isolated-row probes hold that cotangent fixed; they
do not establish how much this row changes the complete molecular objective.
This module is only for explicit HPC diagnostics, never production scoring.
"""

from contextlib import contextmanager
import inspect
import math
import os
from pathlib import Path

import torch

from latentmol.experiments import file_sha256


class AttentionMaskCapture:
    """Capture at most one native four-dimensional query/head row."""

    def __init__(self, attention, *, require_grad_enabled=True):
        self.attention = attention
        self.require_grad_enabled = require_grad_enabled
        self.row = None
        self.upstream = None
        self.calls = 0
        self.unsupported_shapes = 0
        path = Path(inspect.getfile(attention))
        self.native_identity = {"path": str(path), "sha256": file_sha256(path)}

    def _capture(self, query, key, value, biases, output):
        if (self.row is not None
                or (self.require_grad_enabled and not torch.is_grad_enabled())):
            return
        if (query.ndim != 4 or key.ndim != 4 or value.ndim != 4
                or len(biases) != 2 or key.shape[:2] != query.shape[:2]
                or value.shape[:3] != key.shape[:3] or key.shape[-2] > 4096
                or max(query.shape[-1], value.shape[-1]) > 256):
            self.unsupported_shapes += 1
            return
        mask_bias, triangle_bias = biases
        with torch.no_grad():
            # Reduce the original small mask, not a [batch,head,query,key]
            # expansion. Native chunked attention ordinarily uses [B,1,1,K].
            candidates = (mask_bias.detach().amax(dim=-1) < -1e5).nonzero()
            if not candidates.numel():
                return
            prefix = query.shape[:-1]
            coordinates = [0] * (len(prefix) - (mask_bias.ndim - 1)) + candidates[0].tolist()
            if len(coordinates) != 3:
                self.unsupported_shapes += 1
                return
            batch, head, slot = coordinates
            shape = (*prefix, key.shape[-2])
            try:
                masks = torch.broadcast_to(mask_bias, shape)[batch, head, slot]
                triangles = torch.broadcast_to(triangle_bias, shape)[batch, head, slot]
            except (RuntimeError, IndexError):
                self.unsupported_shapes += 1
                return
            self.row = {
                "query": query[batch:batch + 1, head:head + 1, slot:slot + 1].detach().cpu().clone(),
                "key": key[batch:batch + 1, head:head + 1].detach().cpu().clone(),
                "value": value[batch:batch + 1, head:head + 1].detach().cpu().clone(),
                "mask_bias": masks.detach().cpu().clone().reshape(1, 1, 1, -1),
                "triangle_bias": triangles.detach().cpu().clone().reshape(1, 1, 1, -1),
                "native_output": output[batch:batch + 1, head:head + 1, slot:slot + 1].detach().cpu().clone(),
                "source_coordinates": [batch, head, slot],
                "source_query_shape": list(query.shape),
                "source_output_requires_grad": output.requires_grad,
            }
        if output.requires_grad:
            def save_upstream(gradient):
                # A hook sees the actual objective cotangent, but a nonzero
                # cotangent alone is not proof of a numerical failure's cause.
                if self.upstream is None:
                    self.upstream = gradient[batch:batch + 1, head:head + 1,
                                             slot:slot + 1].detach().cpu().clone()
            output.register_hook(save_upstream)

    def __call__(self, query, key, value, biases):
        self.calls += 1
        output = self.attention(query, key, value, biases)
        self._capture(query, key, value, biases, output)
        return output

    def save(self, path: Path):
        """Persist only plain metadata and the bounded detached tensor slice."""
        torch.save({"format_version": 1, "native_attention": self.native_identity,
                    "row": self.row, "upstream_from_objective": self.upstream,
                    "calls_observed": self.calls,
                    "unsupported_shapes": self.unsupported_shapes}, path)

    def analyze(self, *, epsilons=(0.0001, 0.001, 0.01, 0.1)):
        """Measure local analytic/FD agreement using captured native arithmetic.

        This differentiates query, key and triangle bias while fixing value and
        mask. When available, the actual score cotangent supplies the readout;
        otherwise the synthetic mean-output readout is explicitly identified.
        A joint gradient direction supplies a measurable probe without selecting
        a random direction almost orthogonal to the local sensitivity.
        """
        if (not os.environ.get("LSB_JOBID")
                or not os.environ.get("LATENTMOL_SOURCE_SHA256")):
            raise ValueError("Native attention arithmetic probes require an immutable HPC allocation")
        if (not epsilons or len(epsilons) > 12
                or any(not math.isfinite(float(x)) or x <= 0 for x in epsilons)):
            raise ValueError("Declare between one and twelve positive finite probe sizes")
        result = {"captured": self.row is not None, "calls_observed": self.calls,
                  "unsupported_shapes": self.unsupported_shapes,
                  "claim_limit": "One isolated actual row; no end-to-end cause, patch or gate approval."}
        if self.row is None:
            return result
        row = self.row
        readout = (self.upstream if self.upstream is not None else
                   torch.ones_like(row["native_output"]) / row["native_output"].numel())
        result.update(actual_objective_upstream_available=self.upstream is not None,
                      readout="fixed_actual_objective_cotangent" if self.upstream is not None
                      else "synthetic_mean_output",
                      upstream_norm=float(readout.double().norm()),
                      source_coordinates=row["source_coordinates"],
                      source_query_shape=row["source_query_shape"],
                      mask_bias_min=float(row["mask_bias"].min()),
                      mask_bias_max=float(row["mask_bias"].max()))
        variables = [row[name].clone().requires_grad_() for name in ("query", "key", "triangle_bias")]
        q, k, tri = variables
        output = self.attention(q, k, row["value"], [row["mask_bias"], tri])
        loss = (output * readout).sum()
        gradients = torch.autograd.grad(loss, variables)
        norm = math.sqrt(sum(float(value.double().square().sum()) for value in gradients))
        result.update(local_gradient_norm=norm,
                      captured_output_max_difference=float((output.detach() - row["native_output"]).abs().max()),
                      component_gradient_norms={name: float(value.double().norm()) for name, value
                                               in zip(("query", "key", "triangle_bias"), gradients)},
                      finite_differences=[])
        if norm == 0:
            result["probe_status"] = "zero_local_readout_gradient"
            return result
        directions = [value.detach() / norm for value in gradients]
        analytic = sum(float((gradient.double() * direction.double()).sum())
                       for gradient, direction in zip(gradients, directions))
        for epsilon in epsilons:
            values = []
            for sign in (1, -1):
                with torch.no_grad():
                    perturbed = [base.detach() + sign * epsilon * direction
                                 for base, direction in zip(variables, directions)]
                    measured = self.attention(perturbed[0], perturbed[1], row["value"],
                                              [row["mask_bias"], perturbed[2]])
                    values.append(float((measured * readout).sum()))
            numerical = (values[0] - values[1]) / (2 * epsilon)
            result["finite_differences"].append({"epsilon": epsilon, "analytic": analytic,
                "f_plus": values[0], "f_minus": values[1], "central_difference": numerical,
                "absolute_error": abs(analytic - numerical)})
        result["probe_status"] = "measured"
        return result


@contextmanager
def capture_native_attention_mask(*, require_grad_enabled=True):
    """Observe unchanged native calls and always restore the module callable.

    Wrap one explicit analytic probe rather than model loading or an entire
    optimization. Nested capture is rejected. Captured slices are detached and
    contain no alternative molecule or learned representation.
    """
    import boltz.model.layers.triangular_attention.primitives as native
    original = native._attention
    if isinstance(original, AttentionMaskCapture):
        raise ValueError("Nested native attention diagnostic capture is unsupported")
    capture = AttentionMaskCapture(original, require_grad_enabled=require_grad_enabled)
    native._attention = capture
    try:
        yield capture
    finally:
        native._attention = original
