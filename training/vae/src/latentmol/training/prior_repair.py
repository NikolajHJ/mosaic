"""Train on real reconstruction and detached repair targets for fresh priors."""

from contextlib import nullcontext

import torch
from torch.nn import functional as F

from latentmol.atom_preserving_chemistry import _validated_decoded, repair_loss
from latentmol.chemistry_classification import classify_repair_graph
from latentmol.chemistry_continuation import (
    chunks,
    hard_slice,
    reconstruction_chunk,
    reconstruction_denominators,
)
from latentmol.chemistry_optimizer_step import apply_optimizer_step
from latentmol.training.compact_prior import repaired_bond_loss
from latentmol.training.losses import latent_kl


GUIDANCE_VERSION = "frozen-parent-occupancy-chemistry-v1"


def occupancy_logits(atom_logits):
    """Log odds of any element versus NO_ATOM without subtracting rounded probabilities."""
    if atom_logits.ndim != 3 or atom_logits.shape[-1] < 2:
        raise ValueError("Occupancy requires [batch, slots, NO_ATOM + element categories]")
    return atom_logits[..., 1:].logsumexp(dim=-1) - atom_logits[..., 0]


@torch.no_grad()
def parent_prior_targets(reference, prior_z):
    """Decode the same fresh prior once and verify the parent's unmodified argmax graph.

    Occupancy supervision is available on every row. Conditional labels are
    eligible only after the detached graph passes the existing strict native
    chemistry and schema checks. Preserve the original slot order for training.
    """
    if reference.training or any(p.requires_grad for p in reference.parameters()):
        raise ValueError("Prior guidance requires a frozen reference in eval mode")
    decoded = reference.decode(prior_z)
    if _validated_decoded(decoded, reference.spec) != len(prior_z):
        raise ValueError("Reference decoder returned a different number of prior graphs")
    hard = decoded.harden(reference.spec)
    cpu = hard.to("cpu")
    valid = [classify_repair_graph(a, c, b, reference.spec)["strict_teacher_valid"]
             for a, c, b in zip(cpu.atoms, cpu.charges, cpu.bonds, strict=True)]
    return (occupancy_logits(decoded.atoms).sigmoid(), hard,
            torch.tensor(valid, dtype=torch.bool, device=prior_z.device))


def conditional_parent_loss(decoded, hard, valid):
    """Parent element/charge and undirected bond CE conditional on occupied slots.

    Match the recovery pilot: element and charge each carry weight one, bonds
    carry weight two with NONE weighted 0.1. Scale by the eligible fraction so
    a single eligible graph does not carry a full batch of chemistry weight.
    """
    batch, slots = decoded.atoms.shape[:2]
    if (batch < 1 or hard.atoms.shape != (batch, slots) or hard.charges.shape != (batch, slots)
            or decoded.charges.shape[:2] != (batch, slots) or hard.bonds.shape != (batch, slots, slots)
            or decoded.bonds.shape[:3] != (batch, slots, slots) or valid.shape != (batch,)
            or valid.dtype != torch.bool):
        raise ValueError("Parent labels and eligibility must align with the live graph batch")
    active = valid[:, None] & hard.atoms.ne(0)
    zero = decoded.bonds.reshape(-1)[0] * 0
    if not active.any():
        return zero
    # Removing NO_ATOM from the softmax removes its direct label gradient.
    atom = F.cross_entropy(decoded.atoms[..., 1:][active], hard.atoms[active] - 1)
    charge = F.cross_entropy(decoded.charges[active], hard.charges[active])
    pairs = (active[:, :, None] & active[:, None, :]).triu(1)
    bond = zero
    if pairs.any():
        weights = decoded.bonds.new_ones(decoded.bonds.shape[-1])
        weights[0] = .1
        bond = F.cross_entropy(decoded.symmetric_bond_logits[pairs], hard.bonds[pairs], weight=weights)
    return (atom + charge + 2 * bond) * valid.to(dtype=atom.dtype).mean()


def prior_repair_update(model, optimizer, hard, training, teacher, prior_rng, *,
                        real_per_prior, prior_weight, update, epoch, microbatch, step_control,
                        timings=None, reference=None, guidance=None):
    """Search once on the prior decode while the real VAE branch backpropagates."""
    if (reference is None) != (guidance is None):
        raise ValueError("Frozen-parent prior guidance needs both a reference and its policy")
    count = len(hard.atoms)
    prior_count, remainder = divmod(count, real_per_prior)
    if remainder or not prior_count:
        raise ValueError("Each batch must contain a whole number of real-to-prior groups")
    measure = timings.measure if timings is not None else _untimed
    with measure("prepare_inputs", cuda=True):
        parameter = next(model.parameters())
        hard = hard.to(parameter.device)
        epsilon = torch.randn(count, model.config.latent, device=parameter.device)
        prior = torch.randn(prior_count, model.config.latent, generator=prior_rng).to(parameter.device)
    with measure("prior_decode", cuda=True):
        prior_output = model.decode(prior)
    first_id = update * training.batch_size
    with measure("repair_submit"):
        pending = teacher.build_targets_async(
            prior_output, candidate_ids=range(first_id, first_id + prior_count), update=update
        )
    if reference is not None:
        with measure("parent_guidance_targets", cuda=True):
            presence, parent_hard, valid = parent_prior_targets(reference, prior)
    with measure("real_forward_backward", cuda=True):
        optimizer.zero_grad(set_to_none=True)
        denominators = reconstruction_denominators(hard, training.loss, dtype=parameter.dtype)
        beta = training.beta * min(epoch / max(training.kl_warmup_epochs, 1), 1.)
        reconstruction = kl_total = 0.
        for start, stop in chunks(count, microbatch):
            target = hard_slice(hard, start, stop)
            mu, logvar = model.encoder(target, from_canonical_cache=True)
            z = mu + epsilon[start:stop] * (.5 * logvar).exp()
            rec = reconstruction_chunk(model.decode(z), target, training.loss, denominators)
            kl = latent_kl(mu, logvar) * ((stop - start) / count)
            loss = rec + beta * kl
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite real VAE objective")
            loss.backward()
            reconstruction += float(rec.detach())
            kl_total += float(kl.detach())

    with measure("repair_wait"):
        targets = pending.result()
    receipt = targets.receipt
    statuses = receipt["status_counts"]
    if receipt["attempted_candidates"] != prior_count or statuses.get("rejected_or_error", 0):
        raise ValueError("Prior repair targets did not match their scheduled graph inputs")
    if guidance is not None and (sum(statuses.values()) != prior_count
            or targets.candidate_ids != tuple(range(first_id, first_id + prior_count))):
        raise ValueError("Guided prior repair did not account for every scheduled latent")
    details = {}
    with measure("prior_loss_backward", cuda=True):
        if guidance is None:
            prior_loss = repair_loss(prior_output, targets, model.spec)
            prior_objective = prior_weight * prior_loss
        else:
            prior_loss = repaired_bond_loss(prior_output, targets, model.spec)
            occupancy = F.binary_cross_entropy_with_logits(occupancy_logits(prior_output.atoms), presence)
            conditional = conditional_parent_loss(prior_output, parent_hard, valid)
            if not all(loss.requires_grad for loss in (prior_loss, occupancy, conditional)):
                raise ValueError("Detached bond, occupancy or parent chemistry loss")
            prior_objective = (prior_weight * prior_loss + guidance["occupancy_weight"] * occupancy
                               + guidance["conditional_weight"] * conditional)
            details = dict(prior_guidance=guidance["version"], parent_valid_priors=int(valid.sum()),
                           prior_occupancy_bce=float(occupancy.detach()), prior_conditional_ce=float(conditional.detach()),
                           prior_occupancy_weight=guidance["occupancy_weight"],
                           prior_conditional_weight=guidance["conditional_weight"])
        if not prior_objective.requires_grad or not bool(torch.isfinite(prior_objective)):
            raise ValueError("Detached or nonfinite prior repair objective")
        prior_objective.backward()
        if any(p.grad is None for p in (model.encoder.mu.weight, model.decoder.atom_head.weight)):
            raise RuntimeError("Required encoder or decoder path is detached")
    with measure("adamw_and_checks", cuda=True):
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), training.clip_grad, error_if_nonfinite=True))
        apply_optimizer_step(optimizer, model.parameters(), step_control)
        if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise FloatingPointError("Nonfinite VAE parameters after AdamW")
    repair = float(prior_loss.detach())
    weighted_prior = prior_weight * repair if guidance is None else float(prior_objective.detach())
    return dict(real_points=count, fresh_prior_points=prior_count, reconstruction=reconstruction,
                kl=kl_total, beta=beta, prior_repair=repair, prior_weight=prior_weight,
                total=reconstruction + beta * kl_total + weighted_prior,
                repair_targets={"prior": receipt["accepted_targets"]},
                prior_already_valid=statuses.get("skipped_already_valid", 0), repair_statuses=statuses,
                search_seconds=receipt["elapsed_seconds"], pre_clip_gradient_norm=norm, **details)


def _untimed(_name, *, cuda=False):
    return nullcontext()
