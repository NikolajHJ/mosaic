"""Stable -log q of a detached, slot-aligned target under the shared decoder law."""

from dataclasses import dataclass

import torch

from latentmol.graph_policy import graph_from_decoder
from latentmol.multimodel.study import hard_digest
from latentmol.optimization import NonfiniteObjective
from latentmol.representation import HardGraphBatch
from latentmol.sampling import hard_graph_log_prob_terms


@dataclass(frozen=True)
class FrozenRepairTarget:
    """Detached private copies; verify tensors haven't changed before using them.

    This is a reference graph for a loss, not the molecule scored by Boltz.
    Provider/runner acceptance checks chemical validity outside autograd.
    """

    graph: HardGraphBatch
    active: torch.Tensor
    identity: str
    refresh_step: int
    receipt_path: str

    @classmethod
    def create(cls, graph, active, *, refresh_step, receipt_path):
        copied = HardGraphBatch(*(v.detach().cpu().clone() for v in
                                 (graph.atoms, graph.charges, graph.bonds)))
        mask = active.detach().cpu().clone()
        if (mask.dtype != torch.bool or mask.ndim != 1 or not bool(mask.any())
                or copied.atoms.shape != (1, mask.numel())
                or copied.charges.shape != copied.atoms.shape
                or copied.bonds.shape != (1, mask.numel(), mask.numel())
                or any(v.dtype != torch.long for v in (copied.atoms, copied.charges, copied.bonds))
                or not torch.equal(copied.atoms[0] != 0, mask)):
            raise ValueError("Repair target must preserve the exact occupied slot IDs")
        return cls(copied, mask, hard_digest(copied), refresh_step, str(receipt_path))

    def verify(self, active):
        if (hard_digest(self.graph) != self.identity
                or not torch.equal(self.active, active.detach().cpu())
                or not torch.equal(self.graph.atoms[0] != 0, self.active)):
            raise ValueError("Frozen repair target identity/mask changed")


def fixed_target_nll(decoded, graph, target, spec, *, temperature=1., normalization="sum"):
    """Sum atoms, charges and every unordered occupied pair exactly once.

    Derive stable log probabilities from the SAME logits/temperature/mask that
    generated the supplied shared graph. Softmax probabilities may underflow;
    no clamp introduces zero gradient for very unlikely target categories.
    Parity guards reject another graph/temperature being substituted silently.
    """
    target.verify(target.active)
    expected = graph_from_decoder(decoded, spec, temperature=temperature,
                                  fixed_active=target.active.to(decoded.atoms.device))
    for name in ("atoms", "charges", "bonds"):
        torch.testing.assert_close(getattr(graph, name), getattr(expected, name), rtol=0, atol=0)
    terms = hard_graph_log_prob_terms(decoded, target.graph.to(decoded.atoms.device), spec,
                                     temperature=temperature, fixed_active=target.active.to(decoded.atoms.device))
    if not bool(terms["supported"].all()):
        raise ValueError("Repair target is outside the current fixed sampling law")
    nll = {name: -terms[name].reshape(()) for name in ("atoms", "charges", "bonds")}
    raw = nll["atoms"] + nll["charges"] + nll["bonds"]
    count = int(target.active.sum())
    decisions = 2 * count + count * (count - 1) // 2
    if normalization not in ("sum", "mean_per_decision"):
        raise ValueError("Unknown repair target normalization")
    loss = raw if normalization == "sum" else raw / decisions
    if not bool(loss.isfinite().all()):
        raise NonfiniteObjective("Nonfinite fixed repair-target NLL")
    hard = target.graph.to(graph.atoms.device)
    active = target.active.to(graph.atoms.device)
    left, right = torch.triu_indices(spec.max_nodes, spec.max_nodes, 1, device=graph.atoms.device)
    present_pairs = active[left] & active[right]
    underflows = {}
    for name, probabilities, labels, mask in (
            ("atoms", graph.atoms, hard.atoms, active[None]),
            ("charges", graph.charges, hard.charges, active[None]),
            ("bonds", graph.bonds[:, left, right], hard.bonds[:, left, right], present_pairs[None])):
        chosen = probabilities.detach().gather(-1, labels[..., None]).squeeze(-1)
        underflows[name] = int(((chosen == 0) & mask).sum())
    return loss, {"raw_nll": float(raw.detach()), "normalized_nll": float(loss.detach()),
                  "normalization": normalization, "terms": {k: float(v.detach()) for k, v in nll.items()},
                  "decisions": {"atoms": count, "charges": count, "bonds": count * (count - 1) // 2},
                  "probability_policy": "stable_log_softmax_no_floor",
                  "target_probability_underflows": underflows, "target_sha256": target.identity,
                  "refresh_step": target.refresh_step, "receipt_path": target.receipt_path,
                  "gradient_scope": "decoder log likelihood with repaired labels fixed; no DeFoG/draw/selection derivatives"}


def target_score_at(decoder, point, active, target, *, temperature=1., normalization="sum", gradients=False):
    """Decoder-only target probe; never calls Boltz or refreshes the oracle.

    The independent repair-term finite-difference gate prevents a much larger
    binding gradient from hiding a detached or incorrectly scaled repair term.
    """
    target.verify(active)
    probe = point.detach().clone().requires_grad_(gradients)
    with torch.set_grad_enabled(gradients):
        decoded = decoder.decode(probe)
        graph = graph_from_decoder(decoded, decoder.spec, temperature=temperature, fixed_active=active)
        loss, detail = fixed_target_nll(decoded, graph, target, decoder.spec,
                                       temperature=temperature, normalization=normalization)
        derivative = None
        if gradients:
            if not loss.requires_grad:
                raise ValueError("Repair objective detached from the molecular latent")
            derivative, = torch.autograd.grad(loss, probe, allow_unused=True)
            if derivative is None or not bool(derivative.isfinite().all()):
                raise ValueError("Missing/nonfinite repair latent derivative")
            derivative = derivative.detach()
        return {**detail, "loss": float(loss.detach()), "loss_dtype": str(loss.dtype)}, derivative
