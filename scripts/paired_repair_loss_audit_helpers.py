"""Reusable matched teacher loss/logit-gradient audits, allocated DTU HPC only.

The objective follows the chemistry continuation's reconstruction, categorical
repair NLL, KL schedule, global composition cotangent and occupancy anchor.
Teacher search remains detached. ``objective_terms`` retains the caller's graph
for a decoder-parameter audit; ``loss_audit`` explicitly creates independent
FP32 logit leaves and therefore certifies logits, not encoder/latent gradients.
Nothing in this module trains a model or updates optimizer state.
"""
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math

import torch

from latentmol.atom_preserving_chemistry import (
    RepairTargetBatch, composition_statistics, population_js_cotangent, repair_loss,
)
from latentmol.chemistry_continuation import reconstruction_chunk, reconstruction_denominators
from latentmol.losses import latent_kl
from latentmol.representation import DecoderOutput


FIELDS = ("atoms", "charges", "bonds")
VERSION = "paired-fixed-logit-chemistry-loss-v1"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def semantic_receipt(receipt):
    """Use the production receipt identity, excluding only execution counters."""
    result = {key: value for key, value in receipt.items()
              if key not in ("elapsed_seconds", "cache_hits", "search_checks")}
    result["attempts"] = [{key: value for key, value in attempt.items() if key != "cache_hit"}
                          for attempt in receipt.get("attempts", [])]
    return result


def _target_bindings(real, prior):
    unique, bindings, checks = {}, {}, 0
    for name, batch in (("real", real), ("prior", prior)):
        if not isinstance(batch, RepairTargetBatch):
            raise ValueError("Require actual stopped RepairTargetBatch objects")
        receipt = semantic_receipt(batch.receipt)
        identity = _digest(receipt)
        bindings[name] = {"selected_candidate_ids": list(batch.candidate_ids), "semantic_receipt_sha256": identity}
        if identity not in unique:
            unique[identity] = receipt
            checks += batch.receipt.get("search_checks", 0)
    # A shared concatenated teacher batch is counted once, even when .subset()
    # supplies separate real/prior views of its unchanged full receipt.
    chain = "0" * 64
    for receipt in unique.values():
        chain = _digest({"previous": chain, "receipt": receipt})
    return {"branches": bindings, "unique_teacher_receipts": len(unique),
            "ordered_teacher_receipt_chain_sha256": chain, "logical_search_checks": checks}


def _validate(decoded_real, decoded_prior, hard, real_targets, prior_targets, spec, protocol,
              mu, logvar, reference_prior_occupancy, epoch, phase_start_epoch):
    count = len(decoded_real.atoms)
    if count < 1 or len(decoded_prior.atoms) != 3 * count or protocol["prior_ratio"] != 3:
        raise ValueError("Matched audit requires exactly one real to three fresh-prior points")
    if protocol["real_repair_weight"] != .25 or protocol["prior_repair_weight"] != .75:
        raise ValueError("Preserve the declared one-quarter/three-quarter repair mixture")
    values = tuple(getattr(decoded, key) for decoded in (decoded_real, decoded_prior) for key in FIELDS)
    if any(value.dtype != torch.float32 or value.device != values[0].device or not bool(value.isfinite().all()) for value in values):
        raise ValueError("Matched audit requires common-device finite FP32 decoder logits")
    if (len(hard.atoms) != count or len(real_targets.targets) != count or len(prior_targets.targets) != 3 * count
            or mu.shape != logvar.shape or mu.ndim != 2 or mu.shape[0] != count
            or mu.device != values[0].device or logvar.device != values[0].device
            or mu.dtype != torch.float32 or logvar.dtype != torch.float32
            or not bool(mu.isfinite().all()) or not bool(logvar.isfinite().all())):
        raise ValueError("Real labels, posterior and stopped target populations disagree")
    ids = (*real_targets.candidate_ids, *prior_targets.candidate_ids)
    if len(ids) != 4 * count or len(set(ids)) != len(ids) or any(type(value) is not int or value < 0 for value in ids):
        raise ValueError("Real/prior candidate IDs must be jointly unique nonnegative integers")
    if (reference_prior_occupancy.shape != (3 * count, spec.max_nodes)
            or reference_prior_occupancy.device != values[0].device or reference_prior_occupancy.requires_grad
            or reference_prior_occupancy.dtype != torch.float32
            or not bool(reference_prior_occupancy.isfinite().all())):
        raise ValueError("Require a fixed finite same-coordinate prior occupancy reference")
    if type(epoch) is not int or type(phase_start_epoch) is not int or not 0 <= phase_start_epoch < epoch:
        raise ValueError("Require an absolute epoch after the declared chemistry phase start")
    for key in ("repair_weight", "element_js_weight", "charge_js_weight", "occupancy_anchor_weight"):
        value = protocol[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid chemistry objective coefficient: " + key)
    if type(protocol["repair_ramp_epochs"]) is not int or protocol["repair_ramp_epochs"] < 1:
        raise ValueError("Repair ramp must contain a positive number of epochs")


def objective_terms(decoded_real, decoded_prior, hard, real_targets, prior_targets, spec, config, protocol,
                    *, mu, logvar, reference_prior_occupancy, epoch=36, phase_start_epoch=35,
                    composition=None):
    """Return JSON component values and a live objective for caller-owned AD.

    ``composition`` is the authenticated training-population reference. It is
    required explicitly: no distribution is fitted to the evaluated candidates.
    The true JS scalar is paired with the production detached-cotangent VJP.
    Inputs are neither detached nor cloned, so decoder parameter paths survive.
    """
    if composition is None:
        raise ValueError("Require the authenticated empirical training composition")
    _validate(decoded_real, decoded_prior, hard, real_targets, prior_targets, spec, protocol,
              mu, logvar, reference_prior_occupancy, epoch, phase_start_epoch)
    hard = hard.to(decoded_real.atoms.device)
    denominators = reconstruction_denominators(hard, config.loss, dtype=decoded_real.atoms.dtype)
    reconstruction = reconstruction_chunk(decoded_real, hard, config.loss, denominators)
    kl = latent_kl(mu, logvar)
    real_nll = repair_loss(decoded_real, real_targets, spec)
    prior_nll = repair_loss(decoded_prior, prior_targets, spec)
    statistics = [composition_statistics(decoded, spec) for decoded in (decoded_real, decoded_prior)]
    js = {}
    for key in ("elements", "charges"):
        counts = getattr(statistics[0], key).double() + getattr(statistics[1], key).double()
        value, cotangent = population_js_cotangent(counts.detach(), composition[key + "_distribution"])
        surrogate = (counts * cotangent).sum()
        # The zero-valued parenthesized term retains the exact production VJP
        # while logging the true nonnegative JS value, never the linear score.
        js[key] = value + (surrogate - surrogate.detach())
    anchor = (decoded_prior.probabilities().occupancy - reference_prior_occupancy).square().mean()
    coefficient = protocol["repair_weight"] * min((epoch - phase_start_epoch) / protocol["repair_ramp_epochs"], 1.)
    beta = config.beta * min(epoch / max(config.kl_warmup_epochs, 1), 1.)
    weighted_repair = coefficient * (.25 * real_nll + .75 * prior_nll)
    total = (reconstruction + beta * kl + weighted_repair
             + protocol["element_js_weight"] * js["elements"]
             + protocol["charge_js_weight"] * js["charges"]
             + protocol["occupancy_anchor_weight"] * anchor)
    scalars = {"reconstruction": reconstruction, "kl": kl, "repair_real": real_nll, "repair_prior": prior_nll,
               "element_js": js["elements"], "charge_js": js["charges"], "occupancy_anchor": anchor,
               "weighted_repair": weighted_repair, "total": total}
    if any(value.numel() != 1 or not bool(value.isfinite().all()) for value in scalars.values()):
        raise ValueError("Nonfinite or nonscalar matched chemistry objective")
    report = {key: float(value.detach()) for key, value in scalars.items()}
    report.update(real_points=len(decoded_real.atoms), fresh_prior_points=len(decoded_prior.atoms),
        repair_targets={"real": sum(value is not None for value in real_targets.targets),
                        "prior": sum(value is not None for value in prior_targets.targets)},
        coefficients={"repair": coefficient, "real_repair": .25, "prior_repair": .75, "beta": beta,
            "element_js": protocol["element_js_weight"], "charge_js": protocol["charge_js_weight"],
            "occupancy_anchor": protocol["occupancy_anchor_weight"]},
        epoch=epoch, phase_start_epoch=phase_start_epoch, teacher_bindings=_target_bindings(real_targets, prior_targets))
    return report, total


def gradient_fingerprint(value):
    if value is None:
        return {"attached": False}
    raw = value.detach().cpu().contiguous()
    if not bool(raw.isfinite().all()):
        raise ValueError("Nonfinite audit gradient")
    return {"attached": True, "shape": list(raw.shape), "dtype": str(raw.dtype),
            "sha256": hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
            "max_abs": float(raw.double().abs().max()), "l2": float(raw.double().norm())}


@dataclass
class LossAudit:
    metrics: dict
    gradients: dict
    input_sha256: str
    repair_gradients: dict = field(default_factory=dict)

    def record(self):
        return {"version": VERSION, "scope": "fixed_FP32_decoder_logits_no_model_or_latent_updates",
                "input_sha256": self.input_sha256, "metrics": self.metrics,
                "gradients": {name: gradient_fingerprint(value) for name, value in self.gradients.items()},
                "repair_gradients": {term: {name: gradient_fingerprint(value) for name, value in values.items()}
                                     for term, values in self.repair_gradients.items()}}


def _input_identity(decoded_real, decoded_prior, hard, mu, logvar, reference_prior_occupancy, spec, config, protocol, composition):
    digest = hashlib.sha256()
    values = {**{branch + "." + key: getattr(decoded, key) for branch, decoded in (("real", decoded_real), ("prior", decoded_prior)) for key in FIELDS},
              **{"hard." + key: getattr(hard, key) for key in FIELDS}, "mu": mu, "logvar": logvar,
              "reference_prior_occupancy": reference_prior_occupancy}
    for name, value in sorted(values.items()):
        raw = value.detach().cpu().contiguous()
        digest.update(f"{name}|{raw.dtype}|{tuple(raw.shape)}|".encode()); digest.update(raw.numpy().tobytes())
    digest.update(_digest({"schema": spec.to_dict(), "loss": asdict(config.loss), "beta": config.beta,
        "kl_warmup_epochs": config.kl_warmup_epochs, "protocol": protocol, "composition": composition}).encode())
    return digest.hexdigest()


def loss_audit(decoded_real, decoded_prior, hard, real_targets, prior_targets, spec, config, protocol,
               *, mu, logvar, reference_prior_occupancy, epoch=36, phase_start_epoch=35, composition=None):
    """Differentiate the full objective and isolated repair terms on six leaves."""
    identity = _input_identity(decoded_real, decoded_prior, hard, mu, logvar, reference_prior_occupancy,
                               spec, config, protocol, composition)
    cloned = [DecoderOutput(*(getattr(decoded, key).detach().clone().requires_grad_(True) for key in FIELDS))
              for decoded in (decoded_real, decoded_prior)]
    values = {branch + "." + key: getattr(decoded, key)
              for branch, decoded in zip(("real", "prior"), cloned) for key in FIELDS}
    with torch.enable_grad():
        metrics, total = objective_terms(*cloned, hard, real_targets, prior_targets, spec, config, protocol,
            mu=mu.detach(), logvar=logvar.detach(), reference_prior_occupancy=reference_prior_occupancy,
            epoch=epoch, phase_start_epoch=phase_start_epoch, composition=composition)
        gradients = torch.autograd.grad(total, tuple(values.values()))
        isolated = {}
        for name, decoded, targets in (("repair_real", cloned[0], real_targets), ("repair_prior", cloned[1], prior_targets)):
            nll = repair_loss(decoded, targets, spec)
            derivatives = torch.autograd.grad(nll, tuple(values.values()), allow_unused=True)
            isolated[name] = {key: None if value is None else value.detach().clone()
                              for key, value in zip(values, derivatives)}
    return LossAudit(metrics, {key: value.detach().clone() for key, value in zip(values, gradients)}, identity, isolated)


def compare_gradients(left, right):
    """Exact differences; relative errors are observations with no tolerance gate."""
    rows = {}
    for name in sorted(set(left) | set(right)):
        a, b = left.get(name), right.get(name)
        if a is None or b is None:
            rows[name] = {"exact": name in left and name in right and a is None and b is None,
                          "reference": gradient_fingerprint(a), "candidate": gradient_fingerprint(b)}
            continue
        if a.shape != b.shape or a.dtype != b.dtype:
            rows[name] = {"exact": False, "reference": gradient_fingerprint(a), "candidate": gradient_fingerprint(b)}
            continue
        a, b = a.detach().cpu(), b.detach().cpu()
        difference = (a.double() - b.double()).abs()
        denominator = torch.maximum(a.double().abs(), b.double().abs()).clamp_min(torch.finfo(torch.float64).tiny)
        rows[name] = {"exact": torch.equal(a, b), "reference": gradient_fingerprint(a), "candidate": gradient_fingerprint(b),
                      "max_abs_difference": float(difference.max()), "max_relative_difference": float((difference / denominator).max()),
                      "different_entries": int(torch.count_nonzero(a != b))}
    return {"passed": bool(rows) and all(row["exact"] for row in rows.values()), "rows": rows}


def compare_loss_audits(reference, candidate):
    total = compare_gradients(reference.gradients, candidate.gradients)
    repairs = {name: compare_gradients(reference.repair_gradients[name], candidate.repair_gradients[name])
               for name in ("repair_real", "repair_prior")}
    differing = sorted(key for key in set(reference.metrics) | set(candidate.metrics)
                       if key not in reference.metrics or key not in candidate.metrics or reference.metrics[key] != candidate.metrics[key])
    return {"passed": reference.input_sha256 == candidate.input_sha256 and not differing and total["passed"]
                       and all(value["passed"] for value in repairs.values()),
            "inputs_exact": reference.input_sha256 == candidate.input_sha256,
            "components_and_weighted_total_exact": not differing, "differing_metric_fields": differing,
            "weighted_objective_logit_gradients": total, "isolated_repair_logit_gradients": repairs}
