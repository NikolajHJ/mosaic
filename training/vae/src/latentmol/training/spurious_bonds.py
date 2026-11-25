"""Focused reconstruction feedback for extra undirected bonds on matched parent graphs."""

import torch
from torch.nn import functional as F

from latentmol.representation import DecoderOutput, HardGraphBatch


def spurious_bond_loss(predicted: DecoderOutput, aligned_target: HardGraphBatch, *, max_pairs: int = 4):
    """Penalize the strongest invented bonds after the ordinary reconstruction alignment.

    The caller supplies target labels in prediction-slot order. On occupied atom
    pairs where the target has NO_BOND but a bond class wins, use binary
    cross-entropy between that class and NO_BOND. Average up to `max_pairs`
    such errors per graph, then average across graphs (zero for an error-free
    graph). The ordinary reconstruction term still supervises every target bond.
    """
    if type(max_pairs) is not int or max_pairs < 1:
        raise ValueError("max_pairs must be a positive integer")
    if predicted.atoms.ndim != 3 or predicted.bonds.ndim != 4:
        raise ValueError("Expected batched atom and bond logits")
    batch, nodes, _ = predicted.atoms.shape
    if (min(batch, nodes) < 1 or predicted.bonds.shape[:3] != (batch, nodes, nodes)
            or predicted.bonds.shape[-1] < 2 or aligned_target.atoms.shape != (batch, nodes)
            or aligned_target.bonds.shape != (batch, nodes, nodes)):
        raise ValueError("Incompatible predicted and aligned target shapes")

    logits = predicted.symmetric_bond_logits
    if not torch.isfinite(logits).all() or not torch.isfinite(predicted.atoms).all():
        raise FloatingPointError("Nonfinite inputs to spurious bond loss")
    occupied = aligned_target.atoms.ne(0) & predicted.atoms.argmax(-1).ne(0)
    active_pairs = (occupied[:, :, None] & occupied[:, None, :]).triu(1)
    margin = logits[..., 1:].max(-1).values - logits[..., 0]
    # A tie goes to class zero in the actual hard decode.
    false_positives = active_pairs & aligned_target.bonds.eq(0) & margin.gt(0)
    pair_budget = min(max_pairs, nodes * (nodes - 1) // 2)
    hardest = margin.masked_fill(~false_positives, -torch.inf).flatten(1).topk(pair_budget, dim=-1).values
    errors = false_positives.sum(dim=(1, 2))
    selected = errors.clamp(max=pair_budget)
    # softplus(-inf) is zero, including its derivative for masked positions.
    loss = (F.softplus(hardest).sum(-1) / selected.clamp_min(1)).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite spurious bond loss")
    affected, total, used = torch.stack([(errors > 0).sum(), errors.sum(), selected.sum()]).detach().tolist()
    return loss, dict(parents=batch, affected_parents=affected, false_positive_pairs=total, selected_pairs=used)
