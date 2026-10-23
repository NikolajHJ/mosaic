"""Versioned fixed-target chemistry CE with less weight on absent bonds.

Every target label remains fixed. All atom slots and occupied charges have
weight one; occupied unordered bond pairs have weight one for a present bond
and 0.1 for NONE. Each graph is divided by the sum of its decision weights.
This is a weighted categorical surrogate, not the graph's exact negative log
probability, a nearest-valid distance or a guarantee of chemical validity.

Module-level imports stay standard-library-only for scheduler policy readers.
Numerical entrypoints reuse the shared graph-law validation and symmetric raw
bond logits. Chemistry/target provenance belongs to the existing wrappers.
"""
from __future__ import annotations


VERSION = "bond-weighted-target-ce-v1"


def validate_chemistry_target_weighting(value):
    """Closed candidate policy; None means the exact historical objective."""
    if value is None:
        return None
    if (type(value) is not dict or set(value) != {"version", "none_bond_weight"}
            or value.get("version") != VERSION
            or type(value.get("none_bond_weight")) is not float or value["none_bond_weight"] != .1):
        raise ValueError("Require explicit bond-weighted target CE with NONE weight 0.1")
    return dict(value)


def fixed_hard_target_ce_terms(decoded, hard, spec, policy):
    """Return normalized per-graph terms and their exact hard-label denominator.

    ``atoms``, ``charges``, ``present_bonds`` and ``none_bonds`` sum to ``total``.
    NONE weighting is already included in ``none_bonds``. Counts/denominator
    depend only on detached target labels. Missing/chemically invalid targets
    must not be silently accepted by chemistry wrappers; this primitive checks
    the same structural support as the unconditional sampling law.
    """
    import torch
    from .representation import DecoderOutput
    from .sampling import hard_graph_log_prob_terms

    weighting = validate_chemistry_target_weighting(policy)
    if weighting is None:
        raise ValueError("The weighted CE primitive needs an explicit policy")
    # Match the existing loss wrappers' safe categorical arithmetic in reduced
    # precision; gradients still flow back through this ordinary cast.
    if decoded.atoms.dtype in (torch.float16, torch.bfloat16):
        decoded = DecoderOutput(*(value.float() for value in (decoded.atoms, decoded.charges, decoded.bonds)))
    original = hard_graph_log_prob_terms(decoded, hard, spec)
    if not bool(original["supported"].all()):
        raise ValueError("Weighted target lies outside the shared unconditional graph law")
    active = hard.atoms != 0
    left, right = torch.triu_indices(spec.max_nodes, spec.max_nodes, offset=1, device=decoded.atoms.device)
    occupied_pair = active[:, left] & active[:, right]
    labels = hard.bonds[:, left, right]
    is_none = occupied_pair & (labels == 0)
    is_present = occupied_pair & (labels != 0)
    # Reuse the validated atom/charge terms. Expose bond contributions without
    # subtracting two large aggregates, which can lose a small present-bond
    # signal. The second log-softmax is confined to the upper-triangle pairs.
    logits = decoded.symmetric_bond_logits[:, left, right]
    bond_nll = -logits.log_softmax(-1).gather(-1, labels[..., None]).squeeze(-1)
    occupied = active.sum(-1).to(decoded.atoms.dtype)
    present = is_present.sum(-1).to(decoded.atoms.dtype)
    absent = is_none.sum(-1).to(decoded.atoms.dtype)
    denominator = spec.max_nodes + occupied + present + weighting["none_bond_weight"] * absent
    atom = -original["atoms"] / denominator
    charge = -original["charges"] / denominator
    present_bond = torch.where(is_present, bond_nll, torch.zeros_like(bond_nll)).sum(-1) / denominator
    none_bond = (torch.where(is_none, bond_nll, torch.zeros_like(bond_nll)).sum(-1)
                 * weighting["none_bond_weight"] / denominator)
    total = atom + charge + present_bond + none_bond
    if not bool(torch.isfinite(total).all()):
        raise FloatingPointError("Weighted fixed-target chemistry CE is nonfinite")
    return dict(atoms=atom, charges=charge, present_bonds=present_bond, none_bonds=none_bond,
                total=total, denominator=denominator, occupied_atoms=occupied,
                present_pairs=present, none_pairs=absent)


def fixed_hard_target_ce(decoded, hard, spec, policy):
    """Return one weighted CE per graph; the caller owns the all-attempt mean."""
    return fixed_hard_target_ce_terms(decoded, hard, spec, policy)["total"]
