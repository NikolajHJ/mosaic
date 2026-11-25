"""Opt-in, bond-differentiable neighbor-capacity and sulfur environment risk.

No atom-type mixture selects an easier valence rule. Original hard identities,
charges, occupancy and outside bond orders are fixed during a loss evaluation
or line search. The shared decoder/latent may still change them via other
objectives. Always on: a valid hard graph can have unsafe probability tails.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

from latentmol.chemical_environment_policy import (BOND_ORDERS, SULFUR_NONFLUORINE_LIMIT,
    digest, prepare_environment_context, validate_environment_policy)


@dataclass(frozen=True)
class PreparedNeighborEnvironment:
    context: dict
    policy: dict
    identity: str

    def verify(self, spec):
        validate_environment_policy(self.policy)
        if (self.context.get("schema_sha256") != spec.fingerprint()
                or self.identity != digest(dict(context=self.context, policy=self.policy))):
            raise ValueError("Frozen neighbor context or policy changed")

    @property
    def receipt(self):
        return dict(version=self.policy["version"], context_sha256=digest(self.context),
            identity=self.identity, policy=deepcopy(self.policy),
            assessments=deepcopy(self.context["assessments"]),
            gradient_scope="bond logits only; fixed element/charge/occupancy/outside-bond context",
            normalization="mean over all active centers with supported valence caps",
            refresh="explicit between accepted optimizer states; fixed throughout trial evaluations")


def prepare_neighbor_environment(decoded, spec, policy):
    """Prepare from original decoder output, NOT accepted repair target labels."""
    policy = validate_environment_policy(policy)
    hard = decoded.harden(spec)
    fields = [getattr(hard, key).detach().cpu().tolist() for key in ("atoms", "charges", "bonds")]
    context = prepare_environment_context(*fields, spec)
    return PreparedNeighborEnvironment(context, policy, digest(dict(context=context, policy=policy)))


def _safe_lse(value):
    import torch
    supported = torch.isfinite(value).any(-1, keepdim=True)
    result = torch.logsumexp(torch.where(supported, value, torch.zeros_like(value)), -1)
    return torch.where(supported.squeeze(-1), result, torch.full_like(result, -torch.inf))


def neighbor_environment_loss(decoded, prepared, spec):
    """Exact log-DP mass for the declared conditional local event (no floor).

    State=(incident bond-order sum, nonfluorine sulfur coordination). Neighbor
    residual limits mask edge events without renormalizing away unsafe mass.
    The all-NONE incident assignment is always available, keeping finite-logit
    losses finite even for very broken graphs. Separate connectivity/repair
    guidance remains necessary; this event does not prevent disconnection.
    """
    import torch
    from latentmol.atom_preserving_chemistry import _validated_decoded
    prepared.verify(spec)
    batch = _validated_decoded(decoded, spec)
    context = prepared.context
    if batch != len(context["atoms"]):
        raise ValueError("Neighbor context batch changed")
    logits = decoded.symmetric_bond_logits
    # Normalization in FP64 avoids summing rounded categorical masses over50
    # slots; gradients return through the cast to the original logit dtype.
    logp = logits.double().log_softmax(-1)
    device, n = logits.device, spec.max_nodes
    active = torch.tensor(context["active"], device=device, dtype=torch.bool)
    caps = torch.tensor(context["caps"], device=device)
    elements = torch.tensor(context["atoms"], device=device)
    residual = torch.tensor(context["neighbor_remaining_caps"], device=device)
    maximum = max(max(row) for row in context["caps"])
    maximum = max(maximum, 0)
    limit = SULFUR_NONFLUORINE_LIMIT
    shape = (batch, n, maximum + 1, limit + 1)
    state = logp.new_full(shape, -torch.inf)
    state[:, :, 0, 0] = 0.
    v = torch.arange(maximum + 1, device=device)
    k = torch.arange(limit + 1, device=device)
    for j in range(n):
        included = active & active[:, j:j+1] & (torch.arange(n, device=device) != j)[None]
        increment = (elements == 16) & (elements[:, j:j+1] != 9)
        terms = []
        for category, name in enumerate(spec.bond_names):
            order = BOND_ORDERS[name]
            step = increment.long() if category else torch.zeros_like(elements)
            previous_v = v - order
            previous_k = k[None, None, :] - step[..., None]
            index_k = previous_k.clamp_min(0)[:, :, None, :].expand(batch, n, maximum+1, limit+1)
            shifted = state[:, :, previous_v.clamp_min(0), :].gather(-1, index_k)
            valid_state = (previous_v >= 0)[None, None, :, None] & (previous_k >= 0)[:, :, None, :]
            allowed = (residual[:, :, j] < 0) | (order <= residual[:, :, j]) | (category == 0)
            edge = torch.where(included, logp[:, :, j, category],
                               logp.new_zeros((batch, n)) if category == 0 else logp.new_full((batch, n), -torch.inf))
            event = shifted + edge[..., None, None]
            terms.append(torch.where(valid_state & allowed[..., None, None], event, -torch.inf))
        state = _safe_lse(torch.stack(terms, -1))
    supported = active & (caps >= 0)
    allowed_state = v[None, None, :, None] <= caps[..., None, None]
    mass = _safe_lse(torch.where(allowed_state, state, -torch.inf).flatten(-2))
    risk = torch.where(supported, -mass.clamp_max(0), torch.zeros_like(mass))
    # Explicit zero connections make expected zero atom/charge gradients easy
    # to audit without manufacturing gradients through detached rule choices.
    loss = risk.sum() / supported.sum().clamp_min(1)
    loss = loss + decoded.atoms.reshape(-1)[0] * 0 + decoded.charges.reshape(-1)[0] * 0
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite neighbor-environment event loss")
    return dict(loss=loss, weighted_loss=loss * prepared.policy["weight"],
        per_center_risk=risk, assessed_centers=int(supported.sum()),
        unassessed_centers=int((active & (caps < 0)).sum()), receipt=prepared.receipt)


def repair_with_neighbor_environment(decoded, repair_targets, spec, *, repair_config,
                                     environment_targets):
    """Opt-in composition; historical repair implementation stays unchanged."""
    from latentmol.repair_failure_guidance import failure_aware_repair_loss
    repair = failure_aware_repair_loss(decoded, repair_targets, spec, repair_config)
    environment = neighbor_environment_loss(decoded, environment_targets, spec)
    return dict(total=repair["total"] + environment["weighted_loss"], repair=repair,
                environment=environment)
