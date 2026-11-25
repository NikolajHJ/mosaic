"""Teach the canonical encoder to reconstruct valid argmax graphs from the current prior."""

from collections import Counter
from dataclasses import replace
import math

import torch
from torch.func import functional_call

from latentmol.chemistry import MoleculeRejected, canonical_slot_order, decode_graph
from latentmol.representation import DecoderOutput, HardGraphBatch
from latentmol.sampling import _validate_graph_fields
from latentmol.training.confidence_margins import matched_confidence_loss, validate_excess_confidence
from latentmol.training.losses import reconstruction_losses


@torch.no_grad()
def canonical_generated_parents(hard: HardGraphBatch, spec, *, max_parents=None, include_smiles=False):
    """Select exact, losslessly canonicalizable graphs; report selected source rows in draw order.

    The scan stops at the cap. Only examined rows enter the rejection counts.
    Set include_smiles to also return the accepted canonical identities for evaluation.
    """
    draws = len(hard.atoms)
    device, n = hard.atoms.device, spec.max_nodes
    if (hard.atoms.shape != (draws, n) or hard.charges.shape != (draws, n)
            or hard.bonds.shape != (draws, n, n)):
        raise ValueError("Incompatible generated parent graph shapes")
    for field in (hard.atoms, hard.charges, hard.bonds):
        if field.device != device or field.dtype != torch.long:
            raise ValueError("Generated parent labels must be long tensors on one device")
    if max_parents is not None and (type(max_parents) is not int or not 1 <= max_parents <= draws):
        raise ValueError("max_parents must be a positive integer within the candidate batch")
    hard = hard.to("cpu")
    rows, orders, smiles = [], [], []
    rejections = Counter()
    for row in range(draws):
        try:
            order = canonical_slot_order(hard.atoms[row], hard.charges[row], hard.bonds[row], spec)
        except MoleculeRejected as error:
            rejections[error.reason] += 1
            continue
        rows.append(row)
        orders.append(order)
        if include_smiles:
            smiles.append(decode_graph(hard.atoms[row], hard.charges[row], hard.bonds[row], spec))
        if len(rows) == max_parents:
            break
    stats = dict(draws=draws, examined=len(rows) + sum(rejections.values()),
                 selected=len(rows), rejections=dict(rejections), rows=rows)
    if include_smiles:
        stats["smiles"] = smiles
    if not rows:
        return None, stats

    # canonical_slot_order verifies a bijection to the training order. Gather
    # the source labels themselves, with the same permutation on both bond axes.
    row = torch.tensor(rows)
    order = torch.tensor(orders)
    targets = HardGraphBatch(hard.atoms[row[:, None], order], hard.charges[row[:, None], order],
                             hard.bonds[row[:, None, None], order[:, :, None], order[:, None, :]])
    return targets.to(device), stats


def _decode_posterior_mean(decoder, mu):
    # Keep the input Jacobian while excluding every decoder parameter from this
    # branch. Functional tensors preserve existing gradients from ordinary training.
    params = {name: value.detach() for name, value in decoder.named_parameters()}
    buffers = {name: value.detach().clone() for name, value in decoder.named_buffers()}
    modes = [(module, module.training) for module in decoder.modules()]
    try:
        decoder.eval()
        return functional_call(decoder, (params, buffers), (mu,), strict=True)
    finally:
        for module, training in modes:
            module.training = training


def _reencode(model, targets):
    mu, _ = model.encoder(targets, from_canonical_cache=True)
    if not mu.requires_grad:
        raise RuntimeError("Detached posterior mean in generated round-trip training")
    if not torch.isfinite(mu).all():
        raise FloatingPointError("Nonfinite posterior mean in generated round-trip training")
    return _decode_posterior_mean(model.decoder, mu)


def validate_parent_options(parent_source, false_positive_bonds):
    if parent_source not in ("train", "eval"):
        raise ValueError("generated_roundtrip.parent_source must be train or eval")
    if false_positive_bonds is not None:
        if (parent_source != "eval" or not isinstance(false_positive_bonds, dict)
                or set(false_positive_bonds) != {"weight", "max_pairs"}
                or type(false_positive_bonds["weight"]) not in (int, float)
                or not math.isfinite(false_positive_bonds["weight"]) or false_positive_bonds["weight"] < 0
                or type(false_positive_bonds["max_pairs"]) is not int or false_positive_bonds["max_pairs"] < 1):
            raise ValueError("false_positive_bonds requires eval parents, finite nonnegative weight and a positive pair cap")


def _reconstruction(predicted, targets, loss_config, focus, stats):
    # Each parent has its own atom and bond CE denominators and graph matching.
    # When focused supervision is enabled, both objectives share the very same
    # detached assignment. An arbitrary reordering must not create negative edges.
    per_parent = []
    mapping = []
    ordered = replace(loss_config, atom_order="ordered")
    for i in range(len(targets.atoms)):
        row = slice(i, i + 1)
        output = DecoderOutput(predicted.atoms[row], predicted.charges[row], predicted.bonds[row])
        target = HardGraphBatch(targets.atoms[row], targets.charges[row], targets.bonds[row])
        if focus:
            from latentmol.training.matching import align_reconstruction_targets

            if loss_config.atom_order != "graph":
                raise ValueError("Focused bond supervision requires graph-matched reconstruction")
            target = align_reconstruction_targets(output, target, loss_config)
            mapping.append(target)
        per_parent.append(reconstruction_losses(output, target, ordered if focus else loss_config))
    means = {name: torch.stack([terms[name] for terms in per_parent]).mean() for name in per_parent[0]}
    loss = means["reconstruction"]
    if not loss.requires_grad:
        raise RuntimeError("Detached generated round-trip reconstruction")
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite generated round-trip reconstruction")
    stats.update(zip(means, torch.stack(list(means.values())).detach().cpu().tolist()))
    if focus:
        from latentmol.training.spurious_bonds import spurious_bond_loss

        aligned = HardGraphBatch(*(torch.cat([getattr(parent, field) for parent in mapping])
                                   for field in ("atoms", "charges", "bonds")))
        term, counts = spurious_bond_loss(predicted, aligned, max_pairs=focus["max_pairs"])
        if counts["selected_pairs"] and not term.requires_grad:
            raise RuntimeError("Detached spurious-bond objective")
        weighted = focus["weight"] * term
        loss = loss + weighted
        stats["false_positive_bonds"] = dict(counts, loss=float(term.detach()), weighted_loss=float(weighted.detach()))
    return loss


def generated_roundtrip_loss(model, decoded: DecoderOutput, loss_config, *, max_parents: int,
                             prior_latents=None, excess_confidence=None, parent_source="train",
                             false_positive_bonds=None):
    """Reconstruct generated graphs with encoder-only gradients through the decoder.

    Historical recipes take reconstruction targets from the training decode and
    confidence targets from an independent evaluation-mode decode. With eval
    parents both losses and the optional false-positive term share one cohort,
    one evaluation-mode source decode and one differentiable posterior decode.
    """
    validate_parent_options(parent_source, false_positive_bonds)
    if excess_confidence is not None:
        validate_excess_confidence(excess_confidence)
    confidence = excess_confidence if excess_confidence and excess_confidence["weight"] else None
    focus = false_positive_bonds if false_positive_bonds and false_positive_bonds["weight"] else None
    _validate_graph_fields(decoded, model.spec)

    # Retain historical source/re-encoding call order when processing the two cohorts.
    if parent_source == "train":
        targets, stats = canonical_generated_parents(decoded.harden(model.spec), model.spec, max_parents=max_parents)
        del stats["rows"]
        loss = decoded.atoms.new_zeros(())
        if targets is not None:
            loss = _reconstruction(_reencode(model, targets), targets, loss_config, None, stats)
        if confidence is None:
            return loss, stats

    if (prior_latents is None or prior_latents.shape != (len(decoded.atoms), model.config.latent)
            or prior_latents.device != decoded.atoms.device or not prior_latents.isfinite().all()):
        raise ValueError("Evaluation parents require matching finite original prior latents")
    with torch.no_grad():
        reference = _decode_posterior_mean(model.decoder, prior_latents)
        _validate_graph_fields(reference, model.spec)
        parents, detail = canonical_generated_parents(reference.harden(model.spec), model.spec, max_parents=max_parents)
    rows = detail.pop("rows")
    if parent_source == "eval":
        stats = dict(detail, parent_source="eval")
        loss = reference.atoms.new_zeros(())
    if parents is None:
        if focus:
            stats["false_positive_bonds"] = dict(parents=0, affected_parents=0, false_positive_pairs=0,
                                                 selected_pairs=0, loss=0., weighted_loss=0.)
        if confidence:
            stats["excess_confidence"] = dict(detail, reference_matches=0, posterior_matches=0,
                                              exact_pairs=0, loss=0., weighted_loss=0.)
        return loss, stats

    predicted = _reencode(model, parents)
    _validate_graph_fields(predicted, model.spec)
    if parent_source == "eval":
        loss = _reconstruction(predicted, parents, loss_config, focus, stats)
    if confidence:
        chosen = DecoderOutput(reference.atoms[rows], reference.charges[rows], reference.bonds[rows])
        margin, counts = matched_confidence_loss(predicted, chosen, parents, model.spec, confidence["tolerance"])
        if counts["exact_pairs"] and not margin.requires_grad:
            raise RuntimeError("Detached generated excess-confidence objective")
        loss = loss + confidence["weight"] * margin
        stats["excess_confidence"] = dict(detail, **counts, weighted_loss=confidence["weight"] * counts["loss"])
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite generated round-trip objective")
    return loss, stats
