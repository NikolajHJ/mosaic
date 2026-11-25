"""Ring and connectivity diagnostics for the matched graph-decoder comparison."""

import torch

from latentmol.chemistry_early_health import preserve_observation_state
from latentmol.guidance.connectivity import component_costs


def graph_topology(hard):
    """Return occupied atoms, components and independent cycles in each hard graph.

    E - V + C counts independent cycles, including in fused or disconnected graphs.
    Every undirected nonzero bond is one edge regardless of its order.
    """
    stats = component_costs(hard)
    edges = hard.bonds.ne(0).triu(diagonal=1).sum(dim=(-2, -1))
    return torch.stack((stats["active_count"], stats["components"],
                        edges - stats["active_count"] + stats["components"]), dim=-1)


def reconstruction_topology(model, batches):
    """Evaluate raw posterior-mean reconstructions on the experiment's fixed validation panel."""
    targets, predictions = [], []
    device = next(model.parameters()).device
    with preserve_observation_state(model, {}):
        for batch in batches:
            mu, _ = model.encoder(batch.to(device), from_canonical_cache=True)
            decoded = model.decode(mu)
            if any(not field.isfinite().all() for field in (decoded.atoms, decoded.charges, decoded.bonds)):
                raise FloatingPointError("Nonfinite reconstruction logits in topology measurement")
            targets.append(graph_topology(batch.to("cpu")))
            predictions.append(graph_topology(decoded.harden(model.spec).to("cpu")))
    if not targets:
        raise ValueError("Topology evaluation requires validation examples")
    truth, pred = torch.cat(targets), torch.cat(predictions)
    rank_equal = pred[:, 2] == truth[:, 2]
    # Treat an empty or fragmented 'tree' as a failed reconstruction of a connected tree.
    connected = pred[:, 1] == 1
    result = {}
    for group, mask in (("all", torch.ones_like(connected)), ("acyclic_targets", truth[:, 2] == 0),
                        ("cyclic_targets", truth[:, 2] > 0)):
        n = int(mask.sum())
        matched = int((mask & rank_equal & connected).sum())
        result[group] = {
            "targets": n, "connected_predictions": int((mask & connected).sum()),
            "empty_predictions": int((mask & (pred[:, 0] == 0)).sum()),
            "connected_and_correct_cycle_rank": matched,
            "connected_and_correct_cycle_rank_fraction": matched / n if n else None,
            "cycle_rank_mae": float((pred[mask, 2] - truth[mask, 2]).abs().float().mean()) if n else None,
        }
    return result
