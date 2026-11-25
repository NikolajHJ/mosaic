"""Measure how well decoder probabilities distinguish real bonds from other slot pairs."""

import numpy as np
import torch

from latentmol.chemistry_early_health import preserve_observation_state
from latentmol.training.matching import align_reconstruction_targets


def _rank(scores, labels):
    """Pool all examples and include every score tied at the threshold."""
    if not np.isfinite(scores).all():
        raise FloatingPointError("Nonfinite bond probabilities in ranking")
    count = len(labels)
    positives = int(labels.sum())
    result = dict(pairs=count, target_bonds=positives, prevalence=positives / count if count else None)
    if not positives:
        return dict(result, threshold=None, selected_pairs=None, true_positives=None,
                    actual_recall=None, precision=None, average_precision=None)

    order = np.argsort(-scores, kind="stable")
    ordered = scores[order]
    # Each row represents a threshold that can actually be applied to all pairs.
    last_tied = np.flatnonzero(np.r_[ordered[1:] != ordered[:-1], True])
    selected = last_tied + 1
    tp = np.cumsum(labels[order], dtype=np.int64)[last_tied]
    precision = tp / selected
    recall = tp / positives
    first = np.flatnonzero(3 * tp >= 2 * positives)[0]
    average_precision = np.dot(np.diff(np.r_[0., recall]), precision)
    return dict(result, threshold=float(ordered[last_tied[first]]), selected_pairs=int(selected[first]),
                true_positives=int(tp[first]), actual_recall=float(recall[first]),
                precision=float(precision[first]), average_precision=float(average_precision))


def decoder_bond_ranking(model, batches, loss_config):
    """Rank held-out posterior-mean bond predictions at temperature one.

    Use the same batch boundaries and detached slot alignment as graph-matched
    reconstruction. The marginal score includes both endpoint occupancy
    probabilities and ranges over all unordered slot pairs. The conditional
    score compares only pairs of occupied *target* atoms after alignment,
    isolating bond discrimination when the endpoints are given. Both readouts
    marginalize over real bond types. Thresholds are descriptive operating
    points selected on this evaluation panel.
    """
    if loss_config.atom_order != "graph":
        raise ValueError("Bond ranking requires graph-matched reconstruction")
    device = next(model.parameters()).device
    i, j = torch.triu_indices(model.spec.max_nodes, model.spec.max_nodes, offset=1, device=device)
    marginal_scores, marginal_labels = [], []
    conditional_scores, conditional_labels = [], []
    samples = 0
    with preserve_observation_state(model, {}):
        for batch in batches:
            target = batch.to(device)
            decoded = model(target, sample_latent=False, from_canonical_cache=True).decoded
            if any(not field.isfinite().all() for field in (decoded.atoms, decoded.charges, decoded.bonds)):
                raise FloatingPointError("Nonfinite reconstruction logits in bond ranking")
            target = align_reconstruction_targets(decoded, target, loss_config)
            truth = target.bonds[:, i, j] != 0
            active = (target.atoms[:, i] != 0) & (target.atoms[:, j] != 0)
            marginal = decoded.probabilities().bonds[:, i, j, 1:].sum(-1)
            conditional = decoded.symmetric_bond_logits[:, i, j].softmax(-1)[..., 1:].sum(-1)
            marginal_scores.append(marginal.flatten().cpu().numpy())
            marginal_labels.append(truth.flatten().cpu().numpy())
            conditional_scores.append(conditional[active].cpu().numpy())
            conditional_labels.append(truth[active].cpu().numpy())
            samples += len(target.atoms)
    if not samples:
        raise ValueError("Bond ranking requires held-out molecules")
    return {
        "samples": samples,
        "threshold_policy": "Highest threshold reaching at least two-thirds pooled recall on this panel, including all ties",
        "all_pairs_marginal": _rank(np.concatenate(marginal_scores), np.concatenate(marginal_labels)),
        "target_occupied_pairs_conditional": _rank(np.concatenate(conditional_scores), np.concatenate(conditional_labels)),
    }
