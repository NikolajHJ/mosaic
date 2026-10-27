"""Frozen supplied-pose DBMol objective and mandatory fresh qualification.

This wrapper reuses the unchanged ordinary relaxed-graph derivative boundary.
Every context is created for its own complete graph by scorer_context; every
hard endpoint is scored with ordinary native chemical features at that pose.
No learned affinity/embedding proxy, reference-pose replacement or old campaign
certificate is accepted by this module. Numerical execution is HPC-only.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from latentmol.graph_policy import fixed_occupancy_graph
from latentmol.boltz.adapter import adapt_graph
from latentmol.boltz.contracts import AFFINITY_SHA256
from latentmol.boltz.dbmol_model import FrozenBoltzDBMol, load_confidence_model
from latentmol.boltz.dbmol_objective import DBMolObjectiveConfig, CONTACT_HEAD64_ARITHMETIC, COMPONENT_OUTPUTS
from latentmol.boltz.embedding_bridge import NativeProjectionBank
from latentmol.boltz.embedding_comparison_validation import learned_start_gate
from latentmol.boltz.embedding_validation import native_gate
from latentmol.boltz.model import load_affinity_model
from latentmol.boltz.provenance import context_digest
from latentmol.train import state_digest
from .paired_data import check_deadline, deterministic_seed, vae_identity, write_json


SCORER_VERSION = "diffsbdd-supplied-pose-relaxed-dbmol-v1"
FD_EPSILONS = (0.01, 0.003, 0.001)
COMPONENTS = (*COMPONENT_OUTPUTS, "dbmol_total_loss")


def load_scorer(affinity_checkpoint, confidence_checkpoint, pocket_residues, device="cuda"):
    """Strictly load both pretrained trunks with the existing full objective."""
    config = DBMolObjectiveConfig.from_mapping(dict(confidence_checkpoint=confidence_checkpoint,
        pocket_residues=list(pocket_residues), pocket_origin="explicit_native_residues",
        affinity_regression_weight=1.0, affinity_logit_weight=1.0,
        contact_weight=2.5, confidence_weight=2.5,
        objective_arithmetic=CONTACT_HEAD64_ARITHMETIC), Path.cwd())
    affinity = load_affinity_model(Path(affinity_checkpoint))
    structure = load_confidence_model(config.confidence_checkpoint)
    return FrozenBoltzDBMol(affinity, structure, config, recycling_steps=0).to(device).eval().requires_grad_(False)


def fixed_context_scores(scorer, prepared, vae, z):
    """Attached z->ordinary decoder->relaxed graph->frozen native score path."""
    prepared.validate()
    if vae.spec != prepared.spec or z.shape != (1, vae.config.latent):
        raise ValueError("Supplied-pose objective VAE/context/latent mismatch")
    if (vae.training or scorer.training
            or any(p.requires_grad for p in (*vae.parameters(), *scorer.parameters()))):
        raise ValueError("Supplied-pose objective requires frozen eval models")
    before = context_digest(prepared)
    graph = fixed_occupancy_graph(vae.decode(z), prepared.active, prepared.spec)
    features, bonds = adapt_graph(graph, prepared)
    scores = scorer(features, bonds, prepared.coordinates)
    if any(key not in scores or scores[key].numel() != 1 or not bool(scores[key].isfinite().all()) for key in COMPONENTS):
        raise FloatingPointError("Supplied-pose DBMol components are absent or nonfinite")
    if context_digest(prepared) != before:
        raise ValueError("Fixed native context changed within an inner latent update")
    return scores


@torch.no_grad()
def score_hard_native(scorer, context):
    """Independent ordinary integer evaluator at the newly prepared supplied pose."""
    prepared = context.prepared if hasattr(context, "prepared") else context
    before = context_digest(prepared)
    prepared.validate()
    values = scorer.discrete_reference(prepared.features, prepared.coordinates)
    if context_digest(prepared) != before:
        raise ValueError("Native hard-endpoint evaluator mutated the fixed context")
    result = {key: float(value.reshape(())) for key, value in values.items()}
    if any(not math.isfinite(value) for value in result.values()):
        raise FloatingPointError("Nonfinite independently rebuilt hard native endpoint")
    return result


def finite_difference_certificate(primal, point, component_gradients, directions, *,
                                  epsilons=FD_EPSILONS, deadline=None):
    """Additional strict engineering FD contract, preserving inherited AD gates.

    ``primal`` executes a fresh native forward at each point and returns finite
    scalar tensors. Two repeats bound observed forward jitter; differences must
    exceed 10x that range (and 1e-12). At least two adjacent declared scales must
    be resolved and agree within 1e-6 + .02*max(|AD|,|FD|), for every component
    and direction. An unresolved result is not a pass. This is a finite probe,
    not a derivative through bond perception, canonicalization or force fields.
    """
    if len(epsilons) < 2 or any(not math.isfinite(x) or x <= 0 for x in epsilons):
        raise ValueError("Finite-difference scales require at least two positive finite values")
    baseline, repeated = primal(point), primal(point)
    keys = tuple(component_gradients)
    if set(keys) - set(baseline) or set(keys) - set(repeated) or not directions:
        raise ValueError("Finite-difference primal/gradient components or directions differ")
    if any(not math.isfinite(float(values[key])) for values in (baseline, repeated) for key in keys):
        raise FloatingPointError("Nonfinite repeated native finite-difference baseline")
    repeat_delta = {key: abs(float(baseline[key]) - float(repeated[key])) for key in keys}
    rows, component_checks = [], {}
    for name, direction in directions.items():
        norm = float(direction.double().norm())
        if direction.shape != point.shape or not math.isfinite(norm):
            raise ValueError("Malformed finite-difference direction")
        results = {key: [] for key in keys}
        for epsilon in epsilons:
            check_deadline(deadline)
            plus, minus = primal(point + epsilon * direction), primal(point - epsilon * direction)
            for key in keys:
                analytic = float((component_gradients[key].double() * direction.double()).sum())
                positive, negative = float(plus[key]), float(minus[key])
                central = (positive - negative) / (2 * epsilon)
                allowed = 1e-6 + 0.02 * max(abs(analytic), abs(central))
                finite = all(math.isfinite(x) for x in (analytic, positive, negative, central))
                resolved = finite and abs(positive - negative) > max(10 * repeat_delta[key], 1e-12)
                agreement = finite and abs(analytic - central) <= allowed
                wrong = analytic + (1.0 + 10 * abs(analytic))
                wrong_allowed = 1e-6 + .02 * max(abs(wrong), abs(central))
                row = dict(direction=name, component=key, epsilon=epsilon, analytic=analytic,
                    plus=positive, minus=negative, central=central, allowed_error=allowed,
                    repeat_delta=repeat_delta[key], finite=finite, resolved=resolved, agrees=agreement,
                    passed=resolved and agreement,
                    wrong_gradient_control_rejected=finite and abs(wrong - central) > wrong_allowed,
                    null_direction_cannot_establish_nonzero_gradient=norm == 0)
                rows.append(row)
                results[key].append(row["passed"] and row["wrong_gradient_control_rejected"])
        for key, values in results.items():
            component_checks[name + "/" + key] = any(a and b for a, b in zip(values, values[1:]))
    return dict(version="diffsbdd-prospective-resolved-fd-v1", passed=all(component_checks.values()),
                checks=component_checks, rows=rows, epsilons=list(epsilons), repeat_delta=repeat_delta,
                null_directions_are_uninformative=True,
                policy="two adjacent feasible resolved scales; atol1e-6+rtol.02; inherited AD gates remain mandatory")


def qualify_context(scorer, context, vae, z, output, *, deadline=None):
    """Fresh native oracle, unchanged independent AD gate, then strict FD probes.

    A context's feature preparation receipt alone never releases optimization.
    Original checks/tolerances are called unchanged, including all elements and
    bonds and exact objective recomposition. Failure writes a terminal receipt
    and prevents alternating steps. No earlier source137 certificate is reused.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    prepared = context.prepared if hasattr(context, "prepared") else context
    before, model_before, vae_before = context_digest(prepared), state_digest(scorer.state_dict()), state_digest(vae.state_dict())
    report = dict(version=SCORER_VERSION, status="qualifying_not_passed", passed=False,
                  context_sha256=before, optimization="not_run")
    try:
        vae_identity(vae)
        if tuple(scorer.dbmol_config.pocket_residues) != tuple(prepared.metadata["native_pocket"]):
            raise ValueError("Scorer biological pocket differs from fresh supplied-pose context")
        bank = NativeProjectionBank.from_models({"affinity": scorer.affinity.model, "structure": scorer.structure},
            checkpoint_sha256={"affinity": AFFINITY_SHA256,
                "structure": scorer.dbmol_config.confidence_checkpoint["sha256"]}).to(z.device).eval().requires_grad_(False)
        oracle = native_gate(scorer, prepared, bank, vae, output / "native-oracle.json", deadline=deadline)
        report["native_oracle_status"] = oracle["status"]
        placeholder = nn.Identity().eval().requires_grad_(False)
        derivative = learned_start_gate(scorer, prepared, placeholder, vae, z,
            output / "relaxed-start-ad.json", deadline=deadline, scoring_path="relaxed")
        report["independent_ad_status"] = derivative["status"]
        with torch.enable_grad():
            point = z.detach().clone().requires_grad_(True)
            scores = fixed_context_scores(scorer, prepared, vae, point)
            gradients = {key: torch.autograd.grad(scores[key].reshape(()), point, retain_graph=True)[0].detach()
                         for key in COMPONENTS}
        gradient = gradients["dbmol_total_loss"]
        norm = float(gradient.double().norm())
        if not math.isfinite(norm) or norm <= 1e-12:
            raise ValueError("Supplied-pose latent gradient is nonfinite or uninformative")
        directions = {"gradient": gradient / norm}
        generator = torch.Generator(device="cpu").manual_seed(deterministic_seed(17, "supplied_pose_fd"))
        for index in range(3):
            direction = torch.randn(z.shape, generator=generator).to(z)
            directions[f"random_{index + 1}"] = direction / direction.double().norm().to(direction)

        @torch.no_grad()
        def primal(value):
            check_deadline(deadline)
            return fixed_context_scores(scorer, prepared, vae, value)

        finite = finite_difference_certificate(primal, z.detach(), gradients, directions, deadline=deadline)
        write_json(output / "resolved-fd.json", finite)
        report["resolved_fd_passed"] = finite["passed"]
        if not finite["passed"]:
            raise ValueError("Additional planned finite-difference gate failed or remained unresolved")
        report.update(status="passed_supplied_pose_scorer_and_latent_gates", passed=True)
    except Exception as error:
        report.update(status="failed_not_qualified", passed=False, error_type=type(error).__name__, error=str(error))
        raise
    finally:
        unchanged = (context_digest(prepared) == before and state_digest(scorer.state_dict()) == model_before
                     and state_digest(vae.state_dict()) == vae_before)
        report["frozen_context_and_models_unchanged"] = unchanged
        if not unchanged:
            report.update(status="failed_frozen_context_or_model_changed", passed=False)
        write_json(output / "qualification.json", report)
    if not unchanged:
        raise RuntimeError("Supplied-pose qualification changed frozen context or models")
    return report
