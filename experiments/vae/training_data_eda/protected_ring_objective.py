"""Training-only aromatic-ring objective with same-latent distillation of off-site bond predictions."""

import math

import torch
from rdkit import rdBase
from torch.nn import functional as F

from experiments.vae.training_data_eda.aromatic_encodings_probe import patterns, select_sites


def _categorical_kl(logits, frozen, temperature):
    """KL(frozen || student), in logit units at the distillation temperature."""
    # The cost only covers selected edges, so float64 is cheap here and prevents
    # roundoff in small differences between almost-identical sharp distributions.
    reference = (frozen.double() / temperature).log_softmax(-1)
    student = (logits.double() / temperature).log_softmax(-1)
    return (reference.exp() * (reference - student)).sum(-1) * temperature**2


def _average_over_rows(values, row, selected, batch_size):
    """Count each represented graph once, regardless of its number of protected bonds."""
    if not selected.any():
        return None
    indices = row[selected]
    weights = torch.bincount(indices, minlength=batch_size)
    return (values[selected] / weights[indices]).sum() / weights.count_nonzero()


def protected_ring_loss(student, anchor, spec, *, retention_weight=1.0, temperature=2.0,
                        preserve_presence=True):
    """Score three equivalent ring assignments and optionally discourage off-site drift.

    Both decodes must use the same latent vectors and eval-mode dropout. Only the
    student supplies gradients. Eligibility comes from its detached argmax graph.
    The loss retains ordinary bond categories, atom representation, and sampler.

    Conditional on the six edges existing, the positive term is minus the log
    of the total probability of three disjoint verified six-edge assignments
    divided by six. The mean is over eligible graphs. Nearest substituent bonds
    and other bonds each get half the type-retention weight when both groups
    exist; inside each group, graphs receive equal weight. Only occupied pairs
    with an actual bond in the frozen starting graph incur retention. This keeps
    the many absent pairs from overwhelming the small number of relevant edges.
    Only molecules on which an aromatic edit was selected participate.
    """
    if not math.isfinite(retention_weight) or retention_weight < 0:
        raise ValueError("Retention weight must be finite and nonnegative")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Retention temperature must be finite and positive")
    if student.bonds.shape != anchor.bonds.shape or student.atoms.shape != anchor.atoms.shape:
        raise ValueError("Anchor and student must cover the same latent batch and graph schema")
    if student.bonds.device != anchor.bonds.device or spec.bond_names[0] != "NONE":
        raise ValueError("Bond distributions must share a device and the expected NONE category")
    logits = student.symmetric_bond_logits
    batch, slots, _, _ = logits.shape
    device = logits.device
    zero = logits.reshape(-1)[0] * 0
    with torch.no_grad(), rdBase.BlockLogs():
        hard = student.harden(spec).to("cpu")
        rings, _, _ = select_sites(hard, spec)
        ring_edges = torch.zeros((batch, slots, slots), dtype=torch.bool, device=device)
        ring_atoms = torch.zeros((batch, slots), dtype=torch.bool, device=device)
        if rings:
            r, i, j = torch.as_tensor(rings, device=device).unbind(-1)
            ring_edges[r, i, j] = ring_edges[r, j, i] = True
            ring_atoms[r, i] = ring_atoms[r, j] = True
        source = anchor.harden(spec)
        # The frozen model defines which bonds to preserve even if the student
        # deletes one. Both chemical validity and candidate generation remain
        # detached; ordinary real/prior losses handle the other graph components.
        present = source.bonds.ne(0).triu(1) & ring_atoms.any(dim=1)[:, None, None]
        row, left, right = present.nonzero(as_tuple=True)
        on_ring = ring_edges[row, left, right]
        incident = ring_atoms[row, left] | ring_atoms[row, right]
        near = incident & ~on_ring
        remote = ~incident
    if rings:
        per_edge = logits[r, i, j, 1:].double().log_softmax(-1)
        forms = patterns(spec).to(device) - 1
        alternatives = torch.stack([
            per_edge.gather(-1, form[None,:,None].expand(len(rings),-1,-1)).sum((1,2))
            for form in forms
        ], dim=-1)
        ring_nll = -alternatives.logsumexp(-1).mean()
    else:
        ring_nll = zero
    type_kl = presence_kl = zero
    if len(row):
        predicted = logits[row, left, right]
        with torch.no_grad():
            frozen = anchor.symmetric_bond_logits[row, left, right]
        types = _categorical_kl(predicted[:, 1:], frozen[:, 1:], temperature)
        groups = [_average_over_rows(types, row, mask, batch) for mask in (near, remote)]
        populated = [value for value in groups if value is not None]
        if populated:
            type_kl = torch.stack(populated).mean()
        if preserve_presence:
            # Distill the native binary marginal, applying temperature AFTER
            # combining present classes. Type targets on selected rings stay free.
            def binary(x):
                x = x.double()
                return torch.stack([x[:, 0], torch.logsumexp(x[:, 1:], dim=-1)], dim=-1)
            presence = _categorical_kl(binary(predicted), binary(frozen), temperature)
            presence_kl = _average_over_rows(presence, row, torch.ones_like(on_ring), batch)
    ring_per_edge = ring_nll / 6
    loss = ring_per_edge + retention_weight * (type_kl + presence_kl)
    if not bool(loss.detach().isfinite()):
        raise FloatingPointError("Nonfinite protected ring objective")
    return loss, dict(eligible_rows=len(rings), sites=len(rings), rows=batch,
                      protected_near_edges=int(near.sum()), protected_external_edges=int(remote.sum()),
                      presence_edges=len(row) if preserve_presence else 0, ring_nll_per_ring=float(ring_nll.detach()),
                      ring_nll_per_edge=float(ring_per_edge.detach()), offsite_type_kl=float(type_kl.detach()),
                      presence_kl=float(presence_kl.detach()), loss=float(loss.detach()))
