"""Torch VJP boundary for learned Protenix chemical input contributions."""
from __future__ import annotations

import json
import math
import queue

import torch

from latentmol.guidance.mosaic_protenix import MosaicProtenixScorer


PROTOCOL = "mosaic-protenix-chemical-embedding-vjp-v1"
HEADS = ("atom_element", "atom_charge", "pair_presence")


class _EmbeddingVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, element, charge, pair, owner):
        arrays = (element, charge, pair)
        result = owner._rpc(dict(op="evaluate", arrays={
            key: value.detach().cpu().tolist() for key, value in zip(HEADS, arrays)}))
        if not isinstance(result.get("loss"), (float, int)) or not math.isfinite(result["loss"]):
            raise FloatingPointError("Nonfinite native embedding objective")
        gradients = tuple(torch.as_tensor(result["gradients"][key], device=value.device,
                                         dtype=value.dtype) for key, value in zip(HEADS, arrays))
        if any(g.shape != v.shape or not bool(torch.isfinite(g).all())
               for g, v in zip(gradients, arrays)):
            raise ValueError("Nonfinite or incompatible native embedding VJP")
        ctx.save_for_backward(*gradients)
        owner.last_aux = result["aux"]
        return element.new_tensor(result["loss"])

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, upstream):
        return (*(upstream * g for g in ctx.saved_tensors), None)


class MosaicProtenixEmbeddingScorer(MosaicProtenixScorer):
    """Direct learned chemical contributions; no categorical probability law.

    ``projection_bank`` contains exact checkpoint-specific hard-label prototypes.
    Heads are atom_element, atom_charge (native arcsinh charge projection), and
    pair_presence (native token-bond projection). Inputs have a batch dimension
    and either all parent slots or selected occupied slots in ascending order.
    Pair contributions are symmetrized and their diagonal excluded by the child.
    Geometry, names, padding and protein context retain the prepared native view.
    """

    def _rpc(self, request, *, timeout=None):
        with self._lock:
            if self._closed:
                raise RuntimeError("Native scorer is closed")
            identifier = self._request_id
            self._request_id += 1
            request = dict(request, protocol=PROTOCOL, request_id=identifier)
            self._process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
            self._process.stdin.flush()
            try:
                result = self._responses.get(timeout=self.timeout if timeout is None else timeout)
            except queue.Empty as exc:
                self._terminate()
                raise TimeoutError("Native embedding response deadline reached") from exc
            if isinstance(result, BaseException):
                self._terminate()
                raise result
            if result.get("request_id") != identifier or result.get("protocol") != PROTOCOL:
                self._terminate()
                raise ValueError("Native embedding protocol/request identity differs")
            if result.get("status") != "ok":
                self._terminate()
                raise RuntimeError("Native embedding worker failed: " + result.get("error", "unknown"))
            if request["op"] == "init":
                self.projection_bank = result["projection_bank"]
                self.bank = self.projection_bank
            return result

    def __call__(self, embeddings):
        if set(embeddings) != set(HEADS):
            raise ValueError("Require exactly the three native chemical projection heads")
        arrays = []
        n = self.spec.max_nodes
        selected = int(self.active.sum())
        device = embeddings[HEADS[0]].device
        mask = self.active.to(device)
        mode = None
        for key in HEADS:
            value = embeddings[key]
            width = int(self.projection_bank[key]["width"])
            expected = (1, n, n, width) if key == "pair_presence" else (1, n, width)
            occupied = (1, selected, selected, width) if key == "pair_presence" else (1, selected, width)
            current = "full" if tuple(value.shape) == expected else "selected" if tuple(value.shape) == occupied else None
            if (current is None or (mode is not None and current != mode) or value.device != device or value.dtype != torch.float32
                    or not bool(torch.isfinite(value).all())):
                raise ValueError("Incompatible or nonfinite learned embedding: " + key)
            mode = current
            arrays.append(value[0] if current == "selected" else value[0, mask][:, mask] if key == "pair_presence" else value[0, mask])
        loss = _EmbeddingVJP.apply(*arrays, self)
        return loss, dict(self.last_aux)

    evaluate = __call__

    def _evaluate_arrays(self, atoms, charges, bonds):
        """Probability VJP for matched diagnostics using this resident model."""
        response = self._rpc(dict(op="evaluate_probabilities", arrays={
            key: value.detach().cpu().tolist() for key, value in
            zip(("atoms", "charges", "bonds"), (atoms, charges, bonds))}))
        if not isinstance(response.get("loss"), (float, int)) or not math.isfinite(response["loss"]):
            raise FloatingPointError("Nonfinite native probability objective")
        return response

    def score_graph(self, graph):
        """The original relaxed graph path, sharing model/context/key with the map."""
        return MosaicProtenixScorer.__call__(self, graph)

    def evaluate_graph(self, graph):
        """Detached exact-prototype native audit, outside the learned scorer path."""
        mask = self.active.to(graph.atoms.device)
        arrays = (graph.atoms[0, mask, 1:], graph.charges[0, mask], graph.bonds[0, mask][:, mask])
        response = self._rpc(dict(op="evaluate_graph", arrays={key: value.detach().cpu().tolist()
            for key, value in zip(("atoms", "charges", "bonds"), arrays)}))
        return response
