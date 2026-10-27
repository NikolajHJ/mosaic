"""Fresh pointwise derivative checks for a frozen learned embedding start.

This is a separate implementation of the learned score's local derivative.  The
detached hard placeholder is frozen at the start, while decoder hidden states,
the mapping and every native downstream operation remain attached.  Forward AD
is independent of reverse AD.  Finite chords are recorded at predeclared scales,
with the same nonsmooth native semantics as ``pointwise_native_ad_v4``; their
agreement is not substituted for the pointwise derivative contract.

Passing does not establish chemical fidelity or a useful optimization direction.
All numerical execution, including the fixtures, belongs on allocated DTU HPC.
"""

from contextlib import contextmanager
import json
import math
from pathlib import Path

import torch

from latentmol.chemistry_classification import classify_repair_graph
from latentmol.multimodel.study import tensor_digest
from .adapter import adapt_graph
from . import additive_fd
from .dbmol_objective import COMPONENT_OUTPUTS, CONTACT_HEAD64_ARITHMETIC
from .embedding_comparison_policy import (DEFAULT_POLICY, comparison_graph,
                                          policy_binding, policy_record, validate_policy)
from .embedding_injection import inject_native_embeddings
from .embedding_training import conditioned_hard
from .embedding_validation import (_comparison, _deadline, _frozen_versions,
                                   _scalars, score_learned_bridge)
from .provenance import context_digest


VERSION = "embedding-comparison-start-pointwise-ad-v1"
JVP_RTOL, JVP_ATOL = 1e-3, 1e-6
SCALAR_RTOL, SCALAR_ATOL = 1e-4, 1e-5
FD_EPSILONS = (1e-4, 1e-3, 1e-2)
DIRECTION_SEED = 138036
MIN_GRADIENT_NORM = 1e-12
_COMPONENT_NAMES = (*COMPONENT_OUTPUTS, "dbmol_total_loss")


def _grouped(scores, config, key):
    component = "combined" if key == "dbmol_total_loss" else key
    weights = additive_fd.primitive_weights(config, component)
    return additive_fd.grouped_native_expression(
        {name: scores[name].reshape(()) for name in weights}, config, component).reshape(())


@contextmanager
def _local_scores(scorer, prepared, bridge, vae, point, features, bonds, scoring_path="learned",
                  comparison_graph_policy=DEFAULT_POLICY):
    """Keep hooks alive through reverse AD and activation recomputation."""
    if scoring_path == "relaxed":
        graph = comparison_graph(vae.decode(point), prepared.active, prepared.spec,
                                 policy=comparison_graph_policy)
        native_features, native_bonds = adapt_graph(graph, prepared)
        yield scorer(native_features, native_bonds, prepared.coordinates), None
        return
    _, hidden = vae.decode_with_features(point)
    predictions = bridge(hidden.atom_hidden, hidden.bond_input, prepared.active,
                         hidden.charge_input)
    with inject_native_embeddings(scorer, prepared, predictions) as receipt:
        yield scorer(features, bonds, prepared.coordinates), receipt


def _receipt(receipt):
    return receipt.record() if receipt is not None else {"scoring_path": "relaxed", "injection_used": False}


def score_relaxed_path(scorer, prepared, vae, z, *, require_gradient=False, deadline=None,
                       comparison_graph_policy=DEFAULT_POLICY):
    """Declared fixed-occupancy final arithmetic with detached score records."""
    comparison_graph_policy = validate_policy(comparison_graph_policy)
    _deadline(deadline)
    prepared.validate()
    if vae.spec != prepared.spec or z.shape != (1, vae.config.latent):
        raise ValueError("Relaxed path latent/prepared VAE schema mismatch")
    versions = _frozen_versions(scorer=scorer, vae=vae)
    before = context_digest(prepared)
    with torch.enable_grad() if require_gradient else torch.no_grad():
        point = z.detach().clone().requires_grad_(require_gradient)
        decoded = vae.decode(point)
        graph = comparison_graph(decoded, prepared.active, prepared.spec, policy=comparison_graph_policy)
        features, bonds = adapt_graph(graph, prepared)
        _deadline(deadline)
        scores = scorer(features, bonds, prepared.coordinates)
        values = _scalars(scores)
        gradient = torch.autograd.grad(scores["dbmol_total_loss"].reshape(()), point)[0] if require_gradient else None
        if gradient is not None and not bool(gradient.isfinite().all()):
            raise ValueError("Relaxed latent derivative is nonfinite")
        with torch.no_grad():
            hard = conditioned_hard(decoded, prepared.active[None], prepared.spec)
    assessment = classify_repair_graph(hard.atoms[0], hard.charges[0], hard.bonds[0], prepared.spec)
    unchanged = _frozen_versions(scorer=scorer, vae=vae) == versions
    context_unchanged = context_digest(prepared) == before
    if not unchanged or not context_unchanged:
        raise ValueError("Frozen model/context changed during relaxed score")
    return dict(scores=values, gradient=None if gradient is None else gradient.detach(), hard=hard,
        injection_receipt=_receipt(None), context_sha256=before,
        hard_classification=assessment, canonical_smiles=assessment["smiles"],
        gradient_norm=None if gradient is None else float(gradient.double().norm()),
        frozen_modules_unchanged=unchanged, context_unchanged=context_unchanged,
        comparison_graph_policy=policy_record(comparison_graph_policy),
        comparison_graph_policy_sha256=policy_binding(comparison_graph_policy),
        scope="same-hard-identity-fixed-context-relaxed-path")


def _directions(point, gradient):
    # Selection is fixed before any tangent or chord is observed; no retry pool.
    generator = torch.Generator(device="cpu").manual_seed(DIRECTION_SEED)
    result = {"gradient": gradient.detach().clone()}
    for index in range(1, 4):
        random = torch.randn(point.shape, generator=generator, dtype=torch.float64)
        result["random_" + str(index)] = random.to(device=point.device, dtype=point.dtype)
    for name, value in result.items():
        norm = float(value.double().norm())
        if not math.isfinite(norm) or norm <= MIN_GRADIENT_NORM:
            raise ValueError("Uninformative predeclared learned direction: " + name)
        result[name] = value / norm
    return result


def learned_start_gate(scorer, prepared, bridge, vae, z, output, deadline=None, *, scoring_path="learned",
                        comparison_graph_policy=DEFAULT_POLICY):
    """Write a fresh start certificate; return it only after all mandatory checks.

    Exact scalar/reverse/JVP recomposition applies to the inherited contact-head
    FP64 DBMol arithmetic.  Native reverse versus independent native forward AD
    uses the original 1e-3/1e-6 tolerance.  The complete finite-difference table
    must be finite, but chord agreement remains diagnostic, as predeclared here.
    The production score is also checked against the fixed-placeholder local
    function, including at every FD endpoint, so hard-placeholder switching
    cannot silently become a different objective.
    """
    if scoring_path not in ("learned", "relaxed"):
        raise ValueError("Comparison gate scoring_path must be learned or relaxed")
    comparison_graph_policy = validate_policy(comparison_graph_policy)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("Learned start gate requires a fresh result path")
    report = dict(version=VERSION, scoring_path=scoring_path, passed=False, status="running_not_passed", stage="initializing",
        comparison_graph_policy=policy_record(comparison_graph_policy),
        comparison_graph_policy_sha256=policy_binding(comparison_graph_policy),
        graph_policy_binding_required=True,
        point_sha256=tensor_digest({"z": z}), direction_seed=DIRECTION_SEED,
        contract="pointwise_native_ad_v4_extension_for_frozen_learned_embedding",
        scope="pointwise implementation consistency at this start only",
        requires_fresh_native_oracle_gate=True, useful_gradient_validated=False,
        molecular_improvement_validated=False, thresholds_tuned_from_audit=False,
        jvp_tolerance=dict(rtol=JVP_RTOL, atol=JVP_ATOL),
        scalar_tolerance=dict(rtol=SCALAR_RTOL, atol=SCALAR_ATOL),
        minimum_gradient_norm=MIN_GRADIENT_NORM,
        finite_difference_policy=dict(epsilons=list(FD_EPSILONS),
            finite_measurements_required=True, secant_agreement_required=False,
            reason="native nonsmooth pointwise AD contract; finite chords are separate diagnostics"),
        calls=dict(forward_attempted=0, reverse_attempted=0, independent_forward_ad_attempted=0))

    def save():
        path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    def forward():
        _deadline(deadline)
        report["calls"]["forward_attempted"] += 1
        save()

    versions = before = None

    def production_score(candidate, require_gradient=False):
        if scoring_path == "learned":
            result = score_learned_bridge(scorer, prepared, bridge, vae, candidate,
                require_gradient=require_gradient, deadline=deadline, include_hard_score=False)
            # The selected final categorical policy belongs to the paired
            # comparison; this injected boundary does not execute that policy.
            result.update(comparison_graph_policy=policy_record(comparison_graph_policy),
                          comparison_graph_policy_sha256=policy_binding(comparison_graph_policy))
            return result
        return score_relaxed_path(scorer, prepared, vae, candidate,
            require_gradient=require_gradient, deadline=deadline,
            comparison_graph_policy=comparison_graph_policy)

    def local_scores(candidate):
        return _local_scores(scorer, prepared, bridge, vae, candidate, features, bonds, scoring_path,
                             comparison_graph_policy=comparison_graph_policy)

    save()
    try:
        prepared.validate()
        if (vae.spec != prepared.spec or z.shape != (1, vae.config.latent)
                or not z.is_floating_point() or not bool(z.isfinite().all())):
            raise ValueError("Learned-start VAE/prepared/latent shape or finite-value mismatch")
        config = scorer.dbmol_config
        if config.objective_arithmetic != CONTACT_HEAD64_ARITHMETIC:
            raise ValueError("Learned-start exact recomposition requires declared contact-head FP64 arithmetic")
        if getattr(scorer, "recycling_steps", 0) != 0:
            raise ValueError("Learned-start contract requires zero recycling")
        versions = _frozen_versions(scorer=scorer, bridge=bridge, vae=vae)
        before = context_digest(prepared)
        report["context_sha256"] = before
        point = z.detach().clone()
        with torch.no_grad():
            hard = conditioned_hard(vae.decode(point), prepared.active[None], prepared.spec)
            features, bonds = adapt_graph(hard.to_relaxed(prepared.spec), prepared)
        report["fixed_placeholder_sha256"] = tensor_digest(
            {name: getattr(hard, name) for name in ("atoms", "charges", "bonds")})
        report["fixed_occupancy_sha256"] = tensor_digest({"active": prepared.active})

        report["stage"] = "production_reverse_and_independent_local_reverse"
        forward()
        report["calls"]["reverse_attempted"] += 1
        production = production_score(point, require_gradient=True)
        if (production.get("comparison_graph_policy_sha256") != report["comparison_graph_policy_sha256"]
                or production.get("comparison_graph_policy") != report["comparison_graph_policy"]):
            raise ValueError("Production start score changed the declared graph arithmetic policy")
        supplied_gradient = production["gradient"]
        report["production_gradient_sha256"] = tensor_digest({"gradient": supplied_gradient})
        norm = float(supplied_gradient.double().norm())
        report["gradient_norm"] = norm
        if not math.isfinite(norm) or norm <= MIN_GRADIENT_NORM:
            raise ValueError("Learned-start total gradient is zero or nonfinite")
        report["production_injection_receipt"] = production["injection_receipt"]
        report["hard_classification"] = production["hard_classification"]
        report["canonical_smiles"] = production["canonical_smiles"]
        with torch.enable_grad():
            probe = point.clone().requires_grad_(True)
            forward()
            with local_scores(probe) as (scores, receipt):
                baseline = _scalars(scores)
                exact_production = baseline == production["scores"]
                report["production_local_scalar_exact"] = exact_production
                if not exact_production:
                    raise ValueError("Production learned score differs from independent fixed-placeholder baseline")
                native_gradients, grouped_gradients, grouped_scalars = {}, {}, {}
                report["component_gradient_norms"] = {}
                for key in _COMPONENT_NAMES:
                    value = scores[key].reshape(())
                    expression = _grouped(scores, config, key)
                    grouped_scalars[key] = torch.equal(value, expression)
                    for destination, scalar in ((native_gradients, value), (grouped_gradients, expression)):
                        _deadline(deadline)
                        report["calls"]["reverse_attempted"] += 1
                        destination[key] = torch.autograd.grad(scalar, probe, retain_graph=True)[0].detach()
                    if not bool(native_gradients[key].isfinite().all()):
                        raise ValueError("Nonfinite learned component derivative: " + key)
                    component_norm = float(native_gradients[key].double().norm())
                    report["component_gradient_norms"][key] = component_norm
                    if component_norm <= MIN_GRADIENT_NORM:
                        raise ValueError("Zero/uninformative comparison component derivative: " + key)
                supplied_exact = torch.equal(supplied_gradient, native_gradients["dbmol_total_loss"])
                grouped_exact = {key: torch.equal(native_gradients[key], grouped_gradients[key])
                                 for key in _COMPONENT_NAMES}
            report["local_reverse_injection_receipt"] = _receipt(receipt)
            del scores, probe, value, expression, scalar
        report["native_grouped_parity"] = dict(scalar_exact=grouped_scalars,
            reverse_exact=grouped_exact, production_total_reverse_exact=supplied_exact)
        if not (all(grouped_scalars.values()) and all(grouped_exact.values()) and supplied_exact):
            raise ValueError("Learned-start scalar/reverse native grouped recomposition failed")
        report["baseline_scores"] = baseline

        report["stage"] = "grad_no_grad_and_repeated_primal"
        with torch.no_grad():
            forward()
            with local_scores(point) as (scores, _):
                report["grad_no_grad_primal_exact"] = _scalars(scores) == baseline
            forward()
            repeated = production_score(point)
        report["production_repeated_primal_exact"] = repeated["scores"] == baseline
        if not (report["grad_no_grad_primal_exact"] and report["production_repeated_primal_exact"]):
            raise ValueError("Learned grad/no-grad or repeated production primal differs")

        directions = _directions(point, supplied_gradient)
        report["direction_bindings"] = {name: tensor_digest({"direction": value})
                                        for name, value in directions.items()}
        report["stage"] = "independent_native_forward_ad"
        report["independent_forward_ad"] = []
        for name, direction in directions.items():
            forward()
            report["calls"]["independent_forward_ad_attempted"] += 1
            with torch.no_grad(), torch.autograd.forward_ad.dual_level():
                dual = torch.autograd.forward_ad.make_dual(point, direction)
                with local_scores(dual) as (scores, receipt):
                    _scalars(scores)
                    rows = []
                    for key in _COMPONENT_NAMES:
                        primal, tangent = torch.autograd.forward_ad.unpack_dual(scores[key].reshape(()))
                        grouped_primal, grouped_tangent = torch.autograd.forward_ad.unpack_dual(_grouped(scores, config, key))
                        if tangent is None or grouped_tangent is None or not bool(tangent.isfinite().all()):
                            raise ValueError("Missing or nonfinite learned native forward tangent: " + key)
                        analytic = float((native_gradients[key].double() * direction.double()).sum())
                        actual = float(tangent)
                        tolerance = JVP_ATOL + JVP_RTOL * max(abs(actual), abs(analytic))
                        exact = (float(primal) == baseline[key] and torch.equal(primal, grouped_primal)
                                 and torch.equal(tangent, grouped_tangent))
                        rows.append(dict(component=key, primal=float(primal), tangent=actual,
                            reverse_projection=analytic, absolute_error=abs(actual - analytic),
                            allowed_error=tolerance, native_grouped_primal_jvp_exact=exact,
                            passed=exact and abs(actual - analytic) <= tolerance))
            report["independent_forward_ad"].append(dict(direction=name, rows=rows,
                injection_receipt=_receipt(receipt), passed=all(row["passed"] for row in rows)))
            save()
        if not all(item["passed"] for item in report["independent_forward_ad"]):
            raise ValueError(f"Comparison {scoring_path} start independent forward/reverse AD agreement failed")

        report["stage"] = "mandatory_finite_chord_diagnostics"
        report["finite_difference_observations"] = []
        parity_limits = dict(scalar_rtol=SCALAR_RTOL, scalar_atol=SCALAR_ATOL)
        for name, direction in directions.items():
            for epsilon in FD_EPSILONS:
                endpoints, production_endpoints, cancellation = [], [], []
                with torch.no_grad():
                    for sign in (1, -1):
                        candidate = point + sign * epsilon * direction
                        forward()
                        with local_scores(candidate) as (scores, _):
                            local = _scalars(scores)
                        forward()
                        natural = production_score(candidate)
                        endpoints.append(local)
                        production_endpoints.append(natural["scores"])
                        cancellation.append(_comparison(natural["scores"], local, parity_limits))
                central = {key: (endpoints[0][key] - endpoints[1][key]) / (2 * epsilon)
                           for key in baseline}
                if not all(math.isfinite(value) for value in central.values()):
                    raise ValueError("Nonfinite mandatory finite-chord derivative observation")
                analytic = {key: float((native_gradients[key].double() * direction.double()).sum())
                            for key in _COMPONENT_NAMES}
                report["finite_difference_observations"].append(dict(direction=name,
                    epsilon=epsilon, fixed_placeholder_endpoints=endpoints,
                    production_endpoints=production_endpoints, central=central, analytic=analytic,
                    absolute_ad_secant_gap={key: abs(central[key] - analytic[key]) for key in analytic},
                    production_placeholder_cancellation=cancellation,
                    secant_agreement_required=False, finite_measurement_required=True))
                save()
                if not all(row["passed"] for row in cancellation):
                    raise ValueError("Learned production objective changes with its detached hard placeholder")
        report.update(stage="complete", passed=True, status="passed_pointwise_start_only")
    except Exception as error:
        report.update(passed=False, status="failed_not_passed", error_type=type(error).__name__, error=str(error))
        save()
        raise
    finally:
        if versions is not None:
            try:
                report["frozen_modules_unchanged"] = _frozen_versions(scorer=scorer, bridge=bridge, vae=vae) == versions
            except Exception as error:
                report.update(frozen_modules_unchanged=False, frozen_identity_detail=str(error))
        if before is not None:
            try:
                report["context_unchanged"] = context_digest(prepared) == before
            except Exception as error:
                report.update(context_unchanged=False, context_identity_detail=str(error))
        if report.get("frozen_modules_unchanged") is False or report.get("context_unchanged") is False:
            report.update(passed=False, status="failed_frozen_identity")
        save()
    if not report["passed"]:
        raise ValueError("Learned-start frozen model/context identity check failed")
    return report
