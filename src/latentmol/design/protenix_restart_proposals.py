"""Score-free warm-start proposals for the fixed-slot Protenix comparison."""
from __future__ import annotations

import math

import torch

from latentmol.chemistry import inspect_graph
from latentmol.design.latent_protenix import hard_from_graph
from latentmol.design.protenix_restarts import RestartOffset
from latentmol.graph_policy import graph_from_decoder
from latentmol.representation import DecoderOutput


POLICIES = ("none", "all-row", "latent", "sparse")
LATENT_RADII = (0.03, 0.10, 0.25)
DIRECTIONS_PER_RADIUS = 4


def _graph(decoder, decoded, active, temperature):
    if not all(torch.isfinite(value).all() for value in (decoded.atoms, decoded.charges, decoded.bonds)):
        raise ValueError("Nonfinite decoder logits in restart proposal")
    graph = graph_from_decoder(decoded, decoder.spec, temperature=temperature, fixed_active=active)
    if not all(torch.isfinite(value).all() for value in (graph.atoms, graph.charges, graph.bonds)):
        raise ValueError("Nonfinite decoder probabilities in restart proposal")
    return graph


def _measure(graph, parent, pairs, spec):
    hard = hard_from_graph(graph)
    edit_count = (hard.atoms != parent.atoms).sum() + (hard.charges != parent.charges).sum()
    edit_count += (hard.bonds[:, pairs] != parent.bonds[:, pairs]).sum()
    checked = inspect_graph(hard.atoms[0], hard.charges[0], hard.bonds[0], spec)
    return dict(edit_count=int(edit_count), smiles=checked.smiles, failure_reason=checked.failure_reason,
                soft_bonds=float((1 - graph.bonds[0, :, :, 0])[pairs].sum()),
                hard_bonds=int((hard.bonds[0] != 0)[pairs].sum()))


def _latent_kick(decoder, parent, active, temperature, pairs, generator):
    trials = []
    directions = torch.randn((DIRECTIONS_PER_RADIUS, parent.latent.shape[-1]), generator=generator,
                             dtype=parent.latent.dtype)
    norms = torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
    if not (norms > 0).all():
        raise ValueError("Restart direction has zero latent norm")
    directions = directions / norms
    for radius in LATENT_RADII:
        for direction_index, direction in enumerate(directions):
            point = parent.latent + radius * direction
            graph = _graph(decoder, decoder.decode(point), active, temperature)
            measurement = _measure(graph, parent.hard, pairs, decoder.spec)
            valid = bool(measurement["smiles"] and "." not in measurement["smiles"])
            different = measurement["smiles"] != parent.smiles
            edits = measurement["edit_count"]
            if valid and different and 1 <= edits <= 3:
                rank = 0
            elif valid and different:
                rank = 1
            else:
                rank = 2 if valid else 3
            record = dict(measurement, radius=radius, direction=direction_index,
                          valid=valid, different=different)
            trials.append(((rank, radius, edits, direction_index), point, record))
    _, point, selected = min(trials, key=lambda trial: trial[0])
    return point, dict(selected=selected, attempted=len(trials),
        valid_trials=sum(row["valid"] for _, _, row in trials),
        changed_valid_trials=sum(row["valid"] and row["different"] for _, _, row in trials),
        radius_panel=[dict(radius=radius, valid=sum(row["valid"] for _, _, row in trials if row["radius"] == radius),
                           changed_valid=sum(row["valid"] and row["different"]
                                             for _, _, row in trials if row["radius"] == radius))
                      for radius in LATENT_RADII])


def _local_delta(logits, labels, temperature, generator):
    """Mix in local noise while retaining enough probability margin for the argmax."""
    log_prob = (logits / temperature).log_softmax(dim=-1)
    top = log_prob.exp().topk(2, dim=-1).values
    margin = top[..., :1] - top[..., 1:]
    # Even if all added noise lands on one rival, this coefficient is below
    # margin/(1+margin), the threshold at which it could overtake the parent.
    amount = (0.45 * margin).clamp_max(0.20)
    noise = torch.empty_like(log_prob).exponential_(generator=generator)
    noise.scatter_(-1, labels.unsqueeze(-1), 0)
    noise = noise / noise.sum(dim=-1, keepdim=True)
    mixed = torch.logaddexp(log_prob + torch.log1p(-amount), noise.log() + amount.log())
    return torch.where(amount > 0, temperature * mixed - logits, 0)


def _sparse_offset(base, hard, active, pairs, temperature, fade_updates, generator):
    """Soften a connected neighborhood of at most three atoms and existing bonds."""
    adjacency = hard.bonds[0] != 0
    candidates = active.nonzero().flatten()
    slots = [int(candidates[torch.randint(candidates.numel(), (), generator=generator)])]
    for _ in range(2):
        boundary = adjacency[slots].any(dim=0) & active
        boundary[slots] = False
        candidates = boundary.nonzero().flatten()
        if candidates.numel() == 0:
            break
        slots.append(int(candidates[torch.randint(candidates.numel(), (), generator=generator)]))
    chosen = torch.zeros_like(active)
    chosen[slots] = True
    left, right = (pairs & chosen[:, None] & chosen[None, :] & adjacency).nonzero(as_tuple=True)
    atoms = torch.zeros_like(base.atoms)
    charges = torch.zeros_like(base.charges)
    bonds = torch.zeros_like(base.bonds)
    atoms[:, chosen, 1:] = _local_delta(
        base.atoms[:, chosen, 1:], hard.atoms[:, chosen] - 1, temperature, generator)
    charges[:, chosen] = _local_delta(base.charges[:, chosen], hard.charges[:, chosen], temperature, generator)
    edge_delta = _local_delta(
        base.symmetric_bond_logits[:, left, right], hard.bonds[:, left, right], temperature, generator)
    bonds[:, left, right] = bonds[:, right, left] = edge_delta
    return RestartOffset(DecoderOutput(atoms, charges, bonds), fade_updates), dict(
        perturbed_atom_slots=slots, perturbed_bonds=list(map(list, zip(left.tolist(), right.tolist()))))


@torch.no_grad()
def propose_start(decoder, parent, active, *, policy, seed, temperature=1.0, fade_updates=25):
    """Return a latent, optional fading offset, and diagnostics before native scoring."""
    if policy not in POLICIES or type(fade_updates) is not int or fade_updates < 1:
        raise ValueError("Unknown restart policy or invalid fade budget")
    latent = parent.latent.detach().clone()
    if (latent.device.type != "cpu" or latent.ndim != 2 or latent.shape[0] != 1
            or not latent.is_floating_point() or not torch.isfinite(latent).all()
            or active.device.type != "cpu" or active.dtype != torch.bool
            or active.shape != (decoder.spec.max_nodes,) or not active.any()
            or not math.isfinite(temperature) or temperature <= 0):
        raise ValueError("Restart proposals require finite CPU latent, CPU mask and positive temperature")
    pairs = torch.triu(active[:, None] & active[None, :], diagonal=1)
    base = decoder.decode(latent)
    base_graph = _graph(decoder, base, active, temperature)
    base_hard = hard_from_graph(base_graph)
    if not all(torch.equal(getattr(base_hard, name), getattr(parent.hard, name))
               for name in ("atoms", "charges", "bonds")):
        raise ValueError("Bare parent latent does not reproduce saved graph labels")
    reference = _measure(base_graph, parent.hard, pairs, decoder.spec)
    if reference["smiles"] != parent.smiles or not parent.smiles or "." in parent.smiles:
        raise ValueError("Restart parent is not its recorded connected molecule")
    offset = None
    details = {}
    generator = torch.Generator().manual_seed(seed)
    if policy == "all-row":
        offset = RestartOffset.from_parent(decoder, latent, parent.hard, active, seed=seed,
                                            temperature=temperature, fade_updates=fade_updates)
    elif policy == "latent":
        latent, details = _latent_kick(decoder, parent, active, temperature, pairs, generator)
    elif policy == "sparse":
        offset, details = _sparse_offset(base, parent.hard, active, pairs, temperature, fade_updates, generator)
    if offset is not None:
        proposed = _measure(_graph(decoder, offset(base, 0), active, temperature), parent.hard, pairs, decoder.spec)
        if proposed["edit_count"] != 0:
            raise ValueError("Graph perturbation changed the parent hard labels at step zero")
    else:
        proposed = details.get("selected", reference)
    return latent, offset, dict(policy=policy, seed=seed, parent=reference, proposal=proposed,
        latent_displacement=float(torch.linalg.vector_norm(latent - parent.latent)),
        fade_updates=fade_updates if offset is not None else 0, **details)
