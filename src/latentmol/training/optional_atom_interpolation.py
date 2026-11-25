"""Targets for a fixed shared core plus one optional terminal atom."""

import torch

from latentmol.evaluation.molecular_pairs import relabel_categories


def optional_atom_targets(smaller, larger, fractions, spec, loss_config):
    """Return weighted atom, conditional charge and conditional bond targets.

    Both endpoint graphs are already in the smaller atom order, with the extra
    atom mapped to a padding slot of the smaller endpoint. Charge and bond
    targets are weighted by target presence. Decoder probabilities apply the
    occupancy gate when those conditional predictions become marginals.
    """
    core = smaller.atoms != 0
    full = larger.atoms != 0
    optional = full & ~core
    core_pairs = core[:, :, None] & core[:, None, :]
    optional_pairs = (optional[:, :, None] & core[:, None, :]) | (core[:, :, None] & optional[:, None, :])
    if (bool((core & ~full).any()) or bool((optional.sum(-1) != 1).any())
            or not torch.equal(smaller.atoms[core], larger.atoms[core])
            or not torch.equal(smaller.charges[core], larger.charges[core])
            or not torch.equal(smaller.bonds[core_pairs], larger.bonds[core_pairs])
            or bool((smaller.bonds.masked_fill(core_pairs, 0) != 0).any())
            or bool((larger.bonds.masked_fill(core_pairs | optional_pairs, 0) != 0).any())
            or bool((((larger.bonds != 0) & optional_pairs).sum((1, 2)) != 2).any())
            or not torch.equal(larger.bonds, larger.bonds.transpose(1, 2))):
        raise ValueError("Require aligned endpoints with an identical core and one terminal atom")

    left, right = smaller.to_relaxed(spec), larger.to_relaxed(spec)
    alpha = torch.as_tensor(fractions, device=right.atoms.device, dtype=right.atoms.dtype)
    if (alpha.ndim != 1 or not alpha.numel() or not bool(alpha.isfinite().all())
            or bool(((alpha < 0) | (alpha > 1)).any())):
        raise ValueError("Require finite interpolation fractions in [0, 1]")
    alpha = alpha[None, :, None]
    occupancy = core[:, None, :] + alpha * optional[:, None, :]
    atom_weight = torch.where(full, 1., loss_config.padding_weight)
    atoms = torch.lerp(left.atoms[:, None], right.atoms[:, None], alpha[..., None])
    atoms = atoms * atom_weight[:, None, :, None]
    charges = right.charges[:, None] * occupancy[..., None]

    nodes = full.shape[-1]
    upper = torch.ones((nodes, nodes), device=full.device, dtype=torch.bool).triu(1)
    pair_weight = torch.where(larger.bonds != 0, 1., loss_config.no_bond_weight)
    pair_occupancy = occupancy[..., :, None] * occupancy[..., None, :]
    bonds = right.bonds[:, None] * (pair_occupancy * pair_weight[:, None] * upper)[..., None]
    return tuple(value.flatten(0, 1) for value in (atoms, charges, bonds))


def optional_atom_interpolation_loss(decoded, smaller, larger, fractions, spec, loss_config, *, orders=None):
    """Score aligned targets, optionally in either canonical endpoint frame."""
    targets = optional_atom_targets(smaller, larger, fractions, spec, loss_config)
    logits = (decoded.atoms, decoded.charges, decoded.symmetric_bond_logits)

    def score(values):
        components = []
        for value, target in zip(values, targets):
            if value.shape != target.shape:
                raise ValueError("Interpolation logits and optional-atom targets have different shapes")
            mass = target.flatten(1).sum(-1)
            components.append(-(target * value.log_softmax(-1)).flatten(1).sum(-1) / mass.clamp_min(1e-12))
        return torch.stack(components).mean(0)

    loss = score(logits)
    if orders is not None:
        loss = torch.minimum(loss, score(relabel_categories(
            *logits, orders.repeat_interleave(len(fractions), dim=0))))
    if not bool(loss.isfinite().all()):
        raise FloatingPointError("Nonfinite optional-atom interpolation loss")
    return loss.mean()
