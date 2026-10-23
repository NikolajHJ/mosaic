"""Decoder-only objective components, independent of training orchestration.

The frozen reference anchors apply to *every* continuation arm. Local chemical
defects and sampled connectivity are probability-aware proxies; they are not
the distance to the nearest RDKit-valid graph. No chemistry conversion enters
this differentiable path, and no scorer or encoder is trained here.
"""

import math

import torch
from torch.nn import functional as F

from latentmol.chemistry_regularization import chemistry_regularization_losses
from latentmol.connectivity_regularization import connectivity_regularization_losses


def binary_log_probabilities(logits, positive_indices):
    """Stable Bernoulli log probabilities from a categorical partition.

    Return [negative, positive]; logsumexp retains confident tails that a
    probability clamp would erase. Both partitions must contain a category.
    """
    chosen = set(positive_indices)
    if not chosen or not chosen < set(range(logits.shape[-1])):
        raise ValueError("Expected a proper nonempty categorical partition")
    logp = F.log_softmax(logits, dim=-1)
    negative = [i for i in range(logits.shape[-1]) if i not in chosen]
    return torch.stack((logp[..., negative].logsumexp(-1),
                        logp[..., sorted(chosen)].logsumexp(-1)), -1)


def bernoulli_reference_kl(reference, current):
    """KL(frozen reference || current), without floors or moving targets."""
    reference = reference.detach()
    return (reference.exp() * (reference - current)).sum(-1)


def reference_anchors(output, reference, spec):
    """Preserve reference slot occupancy and raw aromatic-edge probability.

    Bond weights use the frozen reference occupancy, so deleting atoms cannot
    remove this anchor. Upper-triangle pairs are counted once. This controls
    size/aromatic shortcuts but does not fix every molecular identity.
    """
    current_occ = binary_log_probabilities(output.atoms, range(1, len(spec.atomic_numbers)))
    ref_occ = binary_log_probabilities(reference.atoms, range(1, len(spec.atomic_numbers))).detach()
    occupancy = bernoulli_reference_kl(ref_occ, current_occ).mean()
    aromatic_index = spec.bond_names.index("AROMATIC")
    cur_ar = binary_log_probabilities(output.symmetric_bond_logits, [aromatic_index])
    ref_ar = binary_log_probabilities(reference.symmetric_bond_logits, [aromatic_index]).detach()
    occ = ref_occ[..., 1].exp()
    upper = torch.ones((spec.max_nodes, spec.max_nodes), dtype=torch.bool,
                       device=occ.device).triu(1)
    weight = occ[:, :, None] * occ[:, None, :] * upper
    aromatic = (bernoulli_reference_kl(ref_ar, cur_ar) * weight).sum() / weight.sum().clamp_min(1)
    return {"occupancy_anchor": occupancy, "aromatic_anchor": aromatic}


def chemistry_objective(output, spec, chemistry_config, connectivity_config, *, generator):
    """Combine exact local expectation and first-order connectivity estimator.

    Call this even in coefficient-zero controls to match sampling budgets and
    RNG consumption. Outer coefficients, not helper disabling, define arms.
    """
    local = chemistry_regularization_losses(output, spec, chemistry_config)
    connection = connectivity_regularization_losses(output, spec, connectivity_config,
                                                    generator=generator)
    scalars = {"local_" + k: v for k, v in local.items()
               if isinstance(v, torch.Tensor) and v.ndim == 0}
    scalars.update({"connectivity_" + k: v for k, v in connection.items()
                    if isinstance(v, torch.Tensor) and v.ndim == 0})
    for name in ("expected_size", "expected_local_bond_order_excess", "unsupported_element_charge_mass",
                 "aromatic_affected_atom_mass", "expected_aromatic_edges"):
        scalars["local_diagnostic_" + name] = local["diagnostics"][name].mean()
    for name in ("costs", "advantages", "active_counts", "components",
                 "minimum_unrestricted_added_edges"):
        value = connection["diagnostics"].get(name)
        if isinstance(value, torch.Tensor):
            scalars["connectivity_diagnostic_" + name] = value.to(output.atoms.dtype).mean()
    advantage = connection["diagnostics"].get("advantages")
    if isinstance(advantage, torch.Tensor):
        scalars["connectivity_zero_advantage_fraction"] = (advantage == 0).to(output.atoms.dtype).mean()
        scalars["connectivity_advantage_variance"] = advantage.to(output.atoms.dtype).var(unbiased=False)
    scalars["chemistry"] = local["total"] + connection["total"]
    return scalars


def freeze_encoder(model):
    """Freeze parameters and mode; the caller must never call model.train()."""
    model.encoder.requires_grad_(False)
    model.encoder.eval()
    model.decoder.requires_grad_(True)
    model.decoder.train()
    return tuple(model.decoder.parameters())


def validate_coefficient(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("Chemistry coefficient must be finite and nonnegative")
    return float(value)
