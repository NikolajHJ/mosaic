"""Matched, score-free restart doses for the trajectory sensitivity study."""
from __future__ import annotations

import math

import torch

from latentmol.design.protenix_restart_proposals import POLICIES, _graph, _measure, propose_start
from latentmol.design.protenix_restart_selection import max_pairwise_similarity
from latentmol.design.protenix_restarts import RestartOffset
from latentmol.representation import DecoderOutput


NOISE_POLICIES = (*POLICIES, "raw-latent", "prior", "sparse-mix", "dense-protected")
_GRAPH_MIX_POLICIES = ("sparse-mix", "dense-protected")


def validate_noise_policy(policy, strength):
    """A raw kick needs a radius; probability mixing needs a safe-budget fraction."""
    if policy not in NOISE_POLICIES:
        raise ValueError(f"Unknown restart noise policy: {policy}")
    if policy in _GRAPH_MIX_POLICIES or policy == "raw-latent":
        if (type(strength) not in (int, float) or not math.isfinite(strength) or strength <= 0
                or (policy in _GRAPH_MIX_POLICIES and strength > 1)):
            raise ValueError("Require a finite positive latent radius or graph budget fraction in (0, 1]")
    elif strength is not None:
        raise ValueError(f"Unexpected strength for restart noise policy {policy}")


def _mixed_delta(logits, labels, temperature, strength, generator, *, protect_none=False):
    """Use a fraction of the safe row-wise probability budget."""
    log_prob = (logits / temperature).log_softmax(dim=-1)
    if logits.shape[-1] == 1:
        return torch.zeros_like(logits), torch.zeros_like(logits[..., 0])
    top = log_prob.exp().topk(2, dim=-1).values
    margin = top[..., :1] - top[..., 1:]
    # Worst case: every unit of added mass lands on the nearest competitor.
    # Keep a margin even in that case and scale the resulting budget by dose.
    budget = (0.90 * margin / (1 + margin)).clamp_max(0.50)
    if protect_none:
        nonbond_budget = (0.20 * log_prob[..., 1:].exp().sum(dim=-1, keepdim=True)).clamp_max(0.05)
        budget = torch.where(labels.unsqueeze(-1) == 0, torch.minimum(budget, nonbond_budget), budget)
    amount = strength * budget
    noise = torch.empty_like(log_prob).exponential_(generator=generator)
    noise.scatter_(-1, labels.unsqueeze(-1), 0)
    noise = noise / noise.sum(dim=-1, keepdim=True)
    mixed = torch.logaddexp(log_prob + torch.log1p(-amount), noise.log() + amount.log())
    delta = torch.where(amount > 0, temperature * mixed - logits, 0)
    return delta, amount.squeeze(-1)


def _connected_slots(active, adjacency, generator):
    options = active.nonzero().flatten()
    slots = [int(options[torch.randint(options.numel(), (), generator=generator)])]
    for _ in range(2):
        boundary = adjacency[slots].any(dim=0) & active
        boundary[slots] = False
        options = boundary.nonzero().flatten()
        if options.numel() == 0:
            break
        slots.append(int(options[torch.randint(options.numel(), (), generator=generator)]))
    return slots


def _mix_readout(amount):
    return dict(rows=amount.numel(), changed_rows=int(amount.count_nonzero()),
                mean_fraction=float(amount.mean()) if amount.numel() else 0.0,
                max_fraction=float(amount.max()) if amount.numel() else 0.0)


def _graph_offset(base, hard, active, pairs, temperature, strength, fade_updates, generator, *, sparse):
    slots = active.nonzero().flatten()
    left, right = pairs.nonzero(as_tuple=True)
    bond_logits = base.symmetric_bond_logits[:, left, right]
    bond_labels = hard.bonds[:, left, right]
    # Draw all noise before selecting support. Matched seeds use the same
    # random category weights on rows that are shared across policies/doses.
    atom_delta, atom_mix = _mixed_delta(base.atoms[:, slots, 1:], hard.atoms[:, slots] - 1,
                                        temperature, strength, generator)
    charge_delta, charge_mix = _mixed_delta(base.charges[:, slots], hard.charges[:, slots],
                                            temperature, strength, generator)
    bond_delta, bond_mix = _mixed_delta(bond_logits, bond_labels, temperature, strength, generator,
                                        protect_none=True)
    chosen = active.clone()
    if sparse:
        chosen.fill_(False)
        chosen[_connected_slots(active, hard.bonds[0] != 0, generator)] = True
    node_support = chosen[slots]
    edge_support = (chosen[left] & chosen[right] & (bond_labels[0] != 0) if sparse
                    else torch.ones_like(left, dtype=torch.bool))
    atoms = torch.zeros_like(base.atoms)
    charges = torch.zeros_like(base.charges)
    bonds = torch.zeros_like(base.bonds)
    atoms[:, slots[node_support], 1:] = atom_delta[:, node_support]
    charges[:, slots[node_support]] = charge_delta[:, node_support]
    edge_left, edge_right = left[edge_support], right[edge_support]
    bonds[:, edge_left, edge_right] = bonds[:, edge_right, edge_left] = bond_delta[:, edge_support]
    existing = bond_labels != 0
    return RestartOffset(DecoderOutput(atoms, charges, bonds), fade_updates), dict(
        atom_slots=slots[node_support].tolist(),
        atoms=_mix_readout(atom_mix[:, node_support]),
        charges=_mix_readout(charge_mix[:, node_support]),
        existing_bonds=_mix_readout(bond_mix[edge_support.unsqueeze(0) & existing]),
        no_bonds=_mix_readout(bond_mix[edge_support.unsqueeze(0) & ~existing]),
    )


def _with_parent_similarity(proposal, parent_smiles):
    smiles = proposal["smiles"]
    valid = bool(smiles and "." not in smiles)
    return dict(proposal, valid=valid,
                parent_similarity=max_pairwise_similarity((parent_smiles, smiles)) if valid else None)


@torch.no_grad()
def propose_noise_start(decoder, parent, active, *, policy, seed, strength=None, temperature=1.0,
                        fade_updates=25):
    """Return a proposed latent, fading logit offset if any, and start diagnostics."""
    validate_noise_policy(policy, strength)
    if policy in POLICIES:
        latent, offset, receipt = propose_start(decoder, parent, active, policy=policy, seed=seed,
                                                temperature=temperature, fade_updates=fade_updates)
        receipt["proposal"] = _with_parent_similarity(receipt["proposal"], parent.smiles)
        return latent, offset, receipt

    # Reuse the exact parent, mask and decoder-replay validation for all new arms.
    latent, _, baseline = propose_start(decoder, parent, active, policy="none", seed=seed,
                                        temperature=temperature, fade_updates=fade_updates)
    pairs = torch.triu(active[:, None] & active[None, :], diagonal=1)
    generator = torch.Generator().manual_seed(seed)
    offset, details = None, {}
    if policy == "raw-latent":
        direction = torch.randn(latent.shape, generator=generator, dtype=latent.dtype)
        norm = torch.linalg.vector_norm(direction)
        if not torch.isfinite(norm) or norm <= 0:
            raise ValueError("Degenerate latent restart direction")
        latent = latent + strength * direction / norm
        decoded = decoder.decode(latent)
    elif policy == "prior":
        latent = torch.randn(latent.shape, generator=generator, dtype=latent.dtype)
        decoded = decoder.decode(latent)
    else:
        decoded = decoder.decode(latent)
        offset, mixing = _graph_offset(decoded, parent.hard, active, pairs, temperature, strength,
                                       fade_updates, generator, sparse=policy == "sparse-mix")
        no_bonds = parent.hard.bonds[:, pairs] == 0
        initial = (decoded.symmetric_bond_logits[:, pairs] / temperature).softmax(-1)
        mixing["no_bond_soft_mass_before"] = float((1 - initial[..., 0])[no_bonds].sum())
        details["mixing"] = mixing
        decoded = offset(decoded, 0)
    displacement = float(torch.linalg.vector_norm(latent - parent.latent))
    if not math.isfinite(displacement) or (policy == "raw-latent" and displacement == 0):
        raise ValueError("Restart latent displacement is nonfinite or unrepresentable")
    graph = _graph(decoder, decoded, active, temperature)
    proposed = _measure(graph, parent.hard, pairs, decoder.spec)
    if offset is not None:
        mixing["no_bond_soft_mass_after"] = float((1 - graph.bonds[..., 0][:, pairs])[no_bonds].sum())
        if proposed["edit_count"] != 0:
            raise ValueError("Graph mixing changed the parent hard labels at step zero")
    return latent, offset, dict(policy=policy, seed=seed, strength=strength, parent=baseline["parent"],
        proposal=_with_parent_similarity(proposed, parent.smiles), latent_displacement=displacement,
        fade_updates=fade_updates if offset is not None else 0, **details)
