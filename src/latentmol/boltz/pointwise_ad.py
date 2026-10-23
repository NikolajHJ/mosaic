"""Pointwise native AD consistency, separate from finite-chord accuracy.

Independent native forward AD and reverse AD follow the unchanged shared graph,
frozen models and stopped repair target. Exact scalar/both-AD recomposition and
an independent graph-to-native affine bridge check are mandatory. Native ReLU,
abs and minimum subgradient conventions remain native; no branch is held.
Natural finite differences are retained as non-gating finite-step observations.
The optimizer must separately accept bounded steps against its actual natural
objective. This is not a smoothness, chemistry or predicted-binding guarantee.
All numerical execution and fixtures require allocated DTU HPC.
"""
from copy import deepcopy
import math

import torch

from latentmol.representation import RelaxedGraph
from latentmol.multimodel.study import tensor_digest
from . import additive_fd
from .additive_branch_fd import _native_scalars, _scalar, _reverse
from .additive_contract import profile_record
from .dbmol_objective import COMPONENT_OUTPUTS, CONTACT_HEAD64_ARITHMETIC
from .local_fd import direction_plan
from .pointwise_cuts import validate_bridge

POLICY = "pointwise_native_ad_v4"
JVP_RTOL, JVP_ATOL = 1e-3, 1e-6
DIAGNOSTIC_EPSILONS = (1e-4, .001)
MAXIMUM_FULL_FORWARDS = 32
DIRECTIONS = ("gradient", "random_1", "random_2", "random_3")


def _graph_copy(graph):
    return RelaxedGraph(*(torch.autograd.forward_ad.unpack_dual(getattr(graph, key)).primal.detach().clone()
                          for key in ("atoms", "charges", "bonds")))


def _graph_digest(graph):
    return tensor_digest({key: getattr(graph, key) for key in ("atoms", "charges", "bonds")})


def _frozen_versions(decoder, scorer):
    versions = []
    for name, module in (("decoder", decoder), ("scorer", scorer)):
        if module.training or any(value.requires_grad for value in module.parameters()):
            raise ValueError("Pointwise validation requires frozen eval models: " + name)
        versions.extend((name, key, value._version) for key, value in [*module.named_parameters(), *module.named_buffers()])
    return tuple(versions)


def validate_branch_derivatives(decoder, point, branch, config, counters, report, persist,
                                measured, gradient, components, *, target=None, weight=0.):
    """Same production signature as earlier policies, with a new explicit claim."""
    if (config.fd_policy != POLICY or config.dbmol is None
            or config.dbmol.objective_arithmetic != CONTACT_HEAD64_ARITHMETIC
            or config.recycling_steps != 0 or set(components) != set(COMPONENT_OUTPUTS)
            or (target is None and weight != 0)):
        raise ValueError("Unsupported pointwise native AD contract or missing components/target")
    versions_before = _frozen_versions(decoder, branch.scorer)
    context_before = branch.verify_context_unchanged()
    target_identity = target.identity if target is not None else None
    if target is not None: target.verify(branch.prepared.active)
    profile = profile_record(config.dbmol, config.objective, repair_weight=weight if target is not None else None)
    contract = {"version": POLICY, "passed": False, "profile": profile, "latent_shape": list(point.shape),
        "point_sha256": tensor_digest({"z": point}), "context_sha256": context_before,
        "target_sha256": target_identity, "graph_sha256": None,
        "scope": "pointwise_native_AD_consistency", "full_network_fd_required": False,
        "historical_gate_approval": False, "branch_holding_used_in_optimization": False,
        "finite_step_acceptance": "separate unchanged bounded natural-objective step control",
        "native_subgradient_conventions_retained": True}
    report.update(derivative_contract=contract, component_gradients={}, actual_pointwise_full_forwards=0)
    persist()
    used, dual_cache = 0, {}

    def forward(candidate, purpose):
        nonlocal used
        if used >= MAXIMUM_FULL_FORWARDS:
            raise ValueError("Pointwise native forward reservation exhausted")
        used += 1
        report["actual_pointwise_full_forwards"] = used
        captured = []
        def observe(_module, arguments):
            if len(arguments) != 1:
                raise ValueError("Shared graph branch invocation changed")
            captured.append(_graph_copy(arguments[0]))
        handle = branch.register_forward_pre_hook(observe)
        try:
            outputs = counters.call("boltz", purpose, "forward",
                lambda: _native_scalars(decoder, candidate, branch, config, target=target, weight=weight), point.device)
        finally:
            handle.remove()
        if len(captured) != 1:
            raise ValueError("Require exactly one observed shared-graph branch call")
        return outputs, captured[0]

    def reverse(value, probe, purpose):
        return counters.call("repair" if purpose.endswith("repair_only") else "boltz", purpose, "backward",
                             lambda: _reverse(value, probe), point.device)

    keys = (*COMPONENT_OUTPUTS, "combined", *(["repair_only"] if target is not None else []))
    names = {key: "repair_nll" if key == "repair_only" else key for key in keys}
    with torch.enable_grad():
        probe = point.detach().clone().requires_grad_(True)
        native, current_graph = forward(probe, "pointwise_native_baseline")
        graph_sha = _graph_digest(current_graph)
        contract["graph_sha256"] = graph_sha
        baseline = {key: float(_scalar(value, key).detach()) for key, value in native.items()}
        if (baseline["combined"] != measured["loss"] or set(measured["components"]) != set(native) - {"combined", "repair_nll"}
                or any(baseline[key] != value for key, value in measured["components"].items())):
            raise ValueError("Pointwise native baseline differs from the supplied production measurement")
        grouped, native_gradients, grouped_gradients = {}, {}, {}
        for key in keys:
            if key == "repair_only":
                grouped[key] = _scalar(native["repair_nll"], key)
            else:
                kwargs = {"objective": config.objective, "repair_weight": weight if target is not None else None} if key == "combined" else {}
                weights = additive_fd.primitive_weights(config.dbmol, key, **kwargs)
                grouped[key] = _scalar(additive_fd.grouped_native_expression(
                    {name: _scalar(native[name], name) for name in weights}, config.dbmol, key, **kwargs), key)
            native_gradients[key] = reverse(_scalar(native[names[key]], key), probe, "pointwise_native:" + key)
            grouped_gradients[key] = reverse(grouped[key], probe, "pointwise_grouped:" + key)
        supplied = {**components, "combined": gradient}
        supplied_exact = {key: torch.equal(native_gradients[key], value) for key, value in supplied.items()}
        grouped_gradient_exact = {key: torch.equal(native_gradients[key], grouped_gradients[key]) for key in keys}
        grouped_scalar_exact = {key: torch.equal(_scalar(native[names[key]], key), grouped[key]) for key in keys}
        values = {name: _scalar(value, name).detach().clone() for name, value in native.items()}
        del native, grouped, probe
    with torch.no_grad():
        repeated, repeated_graph = forward(point, "pointwise_repeat_native_primal")
        repeated_exact = set(repeated) == set(values) and all(torch.equal(_scalar(repeated[key], key), value) for key, value in values.items())
        repeated_graph_exact = _graph_digest(repeated_graph) == graph_sha
    del repeated, repeated_graph
    parity = {"scalar_exact": all(grouped_scalar_exact.values()), "supplied_gradient_exact": supplied_exact,
        "native_grouped_gradient_exact": grouped_gradient_exact, "native_grouped_scalar_exact": grouped_scalar_exact,
        "repeated_native_scalar_exact": repeated_exact, "repeated_shared_graph_exact": repeated_graph_exact,
        "target_unchanged": target is None or target.identity == target_identity}
    parity["passed"] = (parity["scalar_exact"] and all(supplied_exact.values()) and all(grouped_gradient_exact.values())
        and repeated_exact and repeated_graph_exact and parity["target_unchanged"])
    report["native_grouped_baseline_parity"] = parity
    report["native_input_bridge_audit"] = validate_bridge(current_graph, branch.prepared,
        context_sha256=context_before, parity_rtol=config.diagnostics.parity_rtol, parity_atol=config.diagnostics.parity_atol)
    persist()
    if not parity["passed"] or not report["native_input_bridge_audit"]["passed"]:
        raise ValueError("Pointwise native/grouped parity or independent bridge audit failed")

    plans, errors = {}, []
    # Every direction is chosen from the original plan before any new native
    # tangent or finite-step endpoint is observed. No direction is replaced.
    for key in keys:
        derivative = native_gradients[key]
        norm = float(derivative.double().norm())
        item = {"validation_contract": POLICY, "passed": False, "status": "validating", "attached": True,
            "fixed_pose_and_mask": True, "gradient_norm": norm, "baseline_loss": baseline[names[key]],
            "independent_forward_ad": {"rtol": JVP_RTOL, "atol": JVP_ATOL, "passed": False, "rows": []},
            "recomposition_proofs": {}}
        if key == "combined": report["local_fd_validation"] = item
        elif key == "repair_only": report["repair_only_validation"] = item
        else: report["component_gradients"][key] = item
        if not math.isfinite(norm) or norm <= 1e-12:
            item.update(status="failed", error="Zero/nonfinite component derivative is uninformative")
            errors.append(key)
            continue
        directions, _, selection = direction_plan(point, derivative, seed=config.diagnostics.seed)
        item.update(direction_selection=selection, direction_bindings={label: tensor_digest({"direction": direction})
                                                                      for label, direction in directions.items()})
        plans[key] = (directions, item)
        if selection["pool_exhausted"]:
            item.update(status="failed", error="Original predeclared direction pool exhausted")
            errors.append(key)
    persist()

    def dual_probe(direction):
        identity = tensor_digest({"direction": direction})
        if identity not in dual_cache:
            with torch.no_grad(), torch.autograd.forward_ad.dual_level():
                dual = torch.autograd.forward_ad.make_dual(point, direction)
                outputs, graph = forward(dual, "pointwise_independent_native_forward_ad")
                if _graph_digest(graph) != graph_sha:
                    raise ValueError("Native dual changed the shared primal graph")
                tangents, proofs = {}, {}
                for name, output in outputs.items():
                    primal, tangent = torch.autograd.forward_ad.unpack_dual(_scalar(output, name))
                    if tangent is None or not bool(tangent.isfinite().all()) or not torch.equal(primal, values[name]):
                        raise ValueError("Missing native tangent or changed native primal: " + name)
                    tangents[name] = {"primal": float(primal), "tangent": float(tangent)}
                for key in keys:
                    if key == "repair_only": expression = _scalar(outputs["repair_nll"], key)
                    else:
                        kwargs = {"objective": config.objective, "repair_weight": weight if target is not None else None} if key == "combined" else {}
                        weights = additive_fd.primitive_weights(config.dbmol, key, **kwargs)
                        expression = _scalar(additive_fd.grouped_native_expression(
                            {name: _scalar(outputs[name], name) for name in weights}, config.dbmol, key, **kwargs), key)
                    primal, tangent = torch.autograd.forward_ad.unpack_dual(_scalar(outputs[names[key]], key))
                    grouped_primal, grouped_tangent = torch.autograd.forward_ad.unpack_dual(expression)
                    proofs[key] = additive_fd.exact_recomposition_proof(
                        {"scalar": primal, "reverse": native_gradients[key], "jvp": tangent},
                        {"scalar": grouped_primal, "reverse": grouped_gradients[key], "jvp": grouped_tangent})
                    if not proofs[key]["passed"]:
                        raise ValueError("Exact native/grouped recomposition failed: " + key)
            dual_cache[identity] = {"outputs": tangents, "proofs": proofs}
        return dual_cache[identity]

    for key, (directions, item) in plans.items():
        try:
            if item["direction_selection"]["pool_exhausted"]: continue
            for label, direction in directions.items():
                dual = dual_probe(direction)
                observation = dual["outputs"][names[key]]
                analytic = float((native_gradients[key].double() * direction.double()).sum())
                error = abs(observation["tangent"] - analytic)
                tolerance = JVP_ATOL + JVP_RTOL * max(abs(analytic), abs(observation["tangent"]))
                item["independent_forward_ad"]["rows"].append({"direction": label, "direction_sha256": item["direction_bindings"][label],
                    **observation, "reverse_projection": analytic, "absolute_error": error, "tolerance": tolerance,
                    "primal_exact": observation["primal"] == item["baseline_loss"],
                    "passed": observation["primal"] == item["baseline_loss"] and error <= tolerance})
                item["recomposition_proofs"][label] = deepcopy(dual["proofs"][key])
                persist()
            item["independent_forward_ad"]["passed"] = len(item["independent_forward_ad"]["rows"]) == 4 and all(
                row["passed"] for row in item["independent_forward_ad"]["rows"])
            item["passed"] = item["independent_forward_ad"]["passed"] and len(item["recomposition_proofs"]) == 4
            item["status"] = "validated_nonzero" if item["passed"] else "failed"
            if not item["passed"]: errors.append(key)
        except Exception as error:
            item.update(passed=False, status="failed", error=str(error), error_type=type(error).__name__)
            errors.append(key)
        persist()

    if errors:
        # Every requested component has its own partial receipt. Missing native
        # tangents cannot be converted into optional endpoint observations.
        raise ValueError("Pointwise native AD gate failed: " + "; ".join(errors))

    diagnostics = {"required_for_pass": False, "epsilons": list(DIAGNOSTIC_EPSILONS), "directions": {}, "rows": []}
    report["natural_secant_diagnostics"] = diagnostics
    diagnostic_directions = {}
    if "combined" in plans and "gradient" in plans["combined"][0]:
        diagnostic_directions["combined_gradient"] = plans["combined"][0]["gradient"]
    if "dbmol_contact_loss" in plans and "random_2" in plans["dbmol_contact_loss"][0]:
        diagnostic_directions["contact_random_2"] = plans["dbmol_contact_loss"][0]["random_2"]
    for label, direction in diagnostic_directions.items():
        identity = tensor_digest({"direction": direction})
        diagnostics["directions"][label] = identity
        for epsilon in DIAGNOSTIC_EPSILONS:
            row = {"direction": label, "direction_sha256": identity, "epsilon": epsilon, "required_for_pass": False}
            # Completing the declared measurements is mandatory. Their secant
            # agreement is not a pointwise derivative condition. Unsupported
            # shapes, nonfinite endpoints and execution/budget errors propagate
            # instead of being disguised as harmless numerical disagreement.
            with torch.no_grad():
                plus, gp = forward(point + epsilon * direction, "pointwise_natural_secant_diagnostic")
                minus, gm = forward(point - epsilon * direction, "pointwise_natural_secant_diagnostic")
                fp, fm = ({key: float(_scalar(value, key)) for key, value in outputs.items()} for outputs in (plus, minus))
            tangents = dual_probe(direction)["outputs"]
            row.update(status="completed", f_plus=fp, f_minus=fm,
                tangents={key: value["tangent"] for key, value in tangents.items()},
                secants={key: (fp[key] - fm[key]) / (2 * epsilon) for key in fp},
                plus_graph_sha256=_graph_digest(gp), minus_graph_sha256=_graph_digest(gm))
            diagnostics["rows"].append(row)
            persist()
    if target is not None: target.verify(branch.prepared.active)
    frozen = versions_before == _frozen_versions(decoder, branch.scorer)
    context = branch.verify_context_unchanged() == context_before
    report["pointwise_state_unchanged"] = {"frozen_versions": frozen, "context": context,
                                           "target": target is None or target.identity == target_identity}
    if not all(report["pointwise_state_unchanged"].values()): errors.append("scientific_state_changed")
    report["independent_forward_ad_direction_count"] = len(dual_cache)
    contract["passed"] = not errors
    persist()
    if errors:
        raise ValueError("Pointwise native AD gate failed: " + "; ".join(errors))
