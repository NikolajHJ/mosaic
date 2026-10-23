"""Opt-in candidate additive derivative certificate for the frozen DBMol path.

This module changes validation arithmetic, never the model or objective. It is
UNVALIDATED on native trajectories until allocated replay checks pass. Original
net finite differences retain their own outcomes. New approval requires every
declared primitive to pass at shared neighboring scales on the original four
directions, exact native/grouped scalar and both-AD recomposition, and native
forward/reverse agreement. Contact primitives must also pass on natural heads.
Only diagnostic readout probes hold baseline ReLU branches; optimization never
uses those hooks. No fallback, replacement direction, surrogate or new model is
implemented here. The caller owns immutable source/context/target/state checks.
"""

from copy import deepcopy
import math

import torch

from . import additive_fd
from .branch_local_fd import AffinityReadoutBranches, JVP_ATOL, JVP_RTOL, validate_derivative
from .dbmol_objective import COMPONENT_OUTPUTS, CONTACT_HEAD64_ARITHMETIC
from .local_fd import direction_plan, validate_local_fd

POLICY = "additive_branch_v3"


def _native_scalars(decoder, point, branch, config, *, target=None, weight=0.):
    """Return actual live outputs without detaching a forward-AD latent dual."""
    from latentmol.graph_policy import graph_from_decoder
    from latentmol.multimodel.engine import scalar_loss
    decoded = decoder.decode(point)
    graph = graph_from_decoder(decoded, decoder.spec, temperature=config.temperature,
                               fixed_active=branch.prepared.active)
    outputs = dict(branch(graph))
    combined, _ = scalar_loss(outputs, config.objective)
    if target is not None:
        if config.repair_guidance is None or not math.isfinite(weight) or weight <= 0:
            raise ValueError("A fixed repair target requires declared positive guidance")
        from latentmol.repair_guidance.loss import fixed_target_nll
        target.verify(branch.prepared.active)
        repair, _ = fixed_target_nll(decoded, graph, target, decoder.spec,
            temperature=config.temperature, normalization=config.repair_guidance.normalization)
        outputs[additive_fd.REPAIR_PRIMITIVE] = repair
        combined = combined + weight * repair
    outputs["combined"] = combined
    return outputs


def _scalar(value, key):
    if (not isinstance(value, torch.Tensor) or value.numel() != 1
            or value.dtype not in (torch.float32, torch.float64) or not bool(value.isfinite().all())):
        raise ValueError("Require a finite native FP32/FP64 scalar: " + key)
    return value.reshape(())


def _reverse(value, probe):
    if not value.requires_grad:
        raise ValueError("Native/grouped scalar is detached")
    result, = torch.autograd.grad(value, probe, retain_graph=True, allow_unused=True)
    if result is None or result.shape != probe.shape or not bool(result.isfinite().all()):
        raise ValueError("Missing/nonfinite native/grouped reverse derivative")
    return result.detach()


def validate_branch_derivatives(decoder, point, branch, config, counters, report, persist,
                                measured, gradient, components, *, target=None, weight=0.):
    """Candidate v3 gate; production signature matches branch_local_v2 exactly.

    All components are attempted after a component failure. Unknown operations,
    absent tangents, nonfinite values or changed baseline identities fail closed.
    Cached endpoints contain actual scalar values/dtypes, never cached graphs.
    No numerical readiness is established by this implementation alone.
    """
    from latentmol.multimodel.study import tensor_digest
    if (config.dbmol is None or config.dbmol.objective_arithmetic != CONTACT_HEAD64_ARITHMETIC
            or config.fd_policy != POLICY or config.recycling_steps != 0
            or set(components) != set(COMPONENT_OUTPUTS)):
        raise ValueError("Additive branch gate requires the explicit contact-head DBMol profile and all components")
    if target is None and weight != 0:
        raise ValueError("A nonzero supplied repair weight without a target is unsupported")
    branches = AffinityReadoutBranches(branch.scorer)
    from .additive_contract import profile_record
    report["derivative_contract"] = {"version": POLICY, "passed": False,
        "scope": "exact native AD/recomposition and shared-scale additive primitive finite differences",
        "branch_holding_used_in_optimization": False, "historical_gate_approval": False,
        "natural_contact_primitive_certificate_required": True,
        "original_net_fd_results_required_to_be_retained": True,
        "latent_shape": list(point.shape),
        "profile": profile_record(config.dbmol, config.objective,
                                  repair_weight=weight if target is not None else None)}
    report["component_gradients"] = {}
    errors, jvp_cache = [], {}
    endpoint_cache = {"native": {}, "held": {}}

    def forward(candidate, purpose):
        return counters.call("boltz", purpose, "forward",
            lambda: _native_scalars(decoder, candidate, branch, config, target=target, weight=weight), point.device)

    def reverse(value, probe, purpose):
        model = "repair" if purpose.endswith(":repair_only") else "boltz"
        return counters.call(model, purpose, "backward", lambda: _reverse(value, probe), point.device)

    # Direct live probes retain the actual native scalar graph. Recomposition
    # starts from its primitive tensors; it does not sum detached VJP vectors.
    with torch.enable_grad():
        probe = point.detach().clone().requires_grad_(True)
        with branches.capture():
            native = forward(probe, "additive_native_baseline")
        baseline = {key: float(_scalar(value, key).detach()) for key, value in native.items()}
        if baseline["combined"] != measured["loss"] or any(
                baseline[key] != measured["components"][key] for key in measured["components"]):
            raise ValueError("Direct native baseline differs from supplied production measurement")
        keys = (*COMPONENT_OUTPUTS, "combined")
        weights, grouped, native_gradients, grouped_gradients = {}, {}, {}, {}
        for key in keys:
            kwargs = ({"objective": config.objective, "repair_weight": weight if target is not None else None}
                      if key == "combined" else {})
            weights[key] = additive_fd.primitive_weights(config.dbmol, key, **kwargs)
            primitives = {name: _scalar(native[name], name) for name in weights[key]}
            grouped[key] = _scalar(additive_fd.grouped_native_expression(primitives, config.dbmol, key, **kwargs), key)
            native_gradients[key] = reverse(_scalar(native[key], key), probe, "additive_native_baseline:" + key)
            grouped_gradients[key] = reverse(grouped[key], probe, "additive_grouped_baseline:" + key)
        if target is not None:
            native_gradients["repair_only"] = reverse(_scalar(native[additive_fd.REPAIR_PRIMITIVE], "repair_nll"),
                                                      probe, "additive_native_baseline:repair_only")
        supplied = {**components, "combined": gradient}
        native_exact = {key: torch.equal(native_gradients[key], value) for key, value in supplied.items()}
        grouped_exact = {key: torch.equal(native_gradients[key], grouped_gradients[key]) for key in keys}
        grouped_scalar_exact = {key: torch.equal(_scalar(native[key], key), grouped[key]) for key in keys}
        native_values = {key: value.detach().clone() for key, value in native.items()}
        # Release the first two-trunk tape before building the held baseline.
        # Only actual detached scalar values and reverse vectors are needed for
        # parity; retaining two Boltz activation graphs would waste GPU memory.
        del native, grouped, primitives
        with branches.hold():
            held = forward(probe, "additive_held_baseline")
            held_gradients = {key: reverse(_scalar(held[additive_fd.REPAIR_PRIMITIVE if key == "repair_only" else key], key),
                                           probe, "additive_held_baseline:" + key)
                              for key in native_gradients}
        held_scalar_exact = set(held) == set(native_values) and all(torch.equal(native_values[key], held[key]) for key in native_values)
        held_gradient_exact = {key: torch.equal(value, held_gradients[key]) for key, value in native_gradients.items()}
    parity = all(native_exact.values()) and all(grouped_exact.values()) and all(grouped_scalar_exact.values()) \
        and held_scalar_exact and all(held_gradient_exact.values())
    report["local_branch_baseline_parity"] = {"passed": parity,
        "scalar_exact": held_scalar_exact, "total_gradient_exact": held_gradient_exact["combined"],
        "component_gradient_exact": {key: held_gradient_exact[key] for key in COMPONENT_OUTPUTS},
        "supplied_native_gradient_exact": native_exact, "native_grouped_gradient_exact": grouped_exact,
        "native_grouped_scalar_exact": grouped_scalar_exact,
        "held_gradient_exact": held_gradient_exact, "captured_readout_relus": branches.baseline}
    persist()
    if not parity:
        raise ValueError("Exact supplied/native/grouped/held baseline parity failed")
    # Do not keep a native Boltz reverse tape resident while evaluating the
    # endpoint grid. Derivatives above are detached; dual proofs below use a
    # fresh native forward for each direction and the saved reverse tensors.
    del held, probe, native_values

    def dual_probe(direction):
        identity = tensor_digest({"direction": direction})
        if identity not in jvp_cache:
            with torch.no_grad(), torch.autograd.forward_ad.dual_level():
                dual = torch.autograd.forward_ad.make_dual(point, direction)
                outputs = forward(dual, "additive_native_forward_ad")
                observed, proofs = {}, {}
                for name, output in outputs.items():
                    primal, tangent = torch.autograd.forward_ad.unpack_dual(_scalar(output, name))
                    if tangent is None or not bool(tangent.isfinite().all()) or float(primal) != baseline[name]:
                        raise ValueError("Missing/nonfinite tangent or changed native primal: " + name)
                    observed[name] = {"primal": float(primal), "tangent": float(tangent)}
                for key in keys:
                    kwargs = ({"objective": config.objective, "repair_weight": weight if target is not None else None}
                              if key == "combined" else {})
                    expression = additive_fd.grouped_native_expression(
                        {name: _scalar(outputs[name], name) for name in weights[key]}, config.dbmol, key, **kwargs)
                    original, original_jvp = torch.autograd.forward_ad.unpack_dual(_scalar(outputs[key], key))
                    recomposed, recomposed_jvp = torch.autograd.forward_ad.unpack_dual(_scalar(expression, key))
                    proofs[key] = additive_fd.exact_recomposition_proof(
                        {"scalar": original, "reverse": native_gradients[key], "jvp": original_jvp},
                        {"scalar": recomposed, "reverse": grouped_gradients[key], "jvp": recomposed_jvp})
                    if not proofs[key]["passed"]:
                        raise ValueError("Native/grouped scalar/reverse/JVP recomposition failed: " + key)
            jvp_cache[identity] = {"outputs": observed, "proofs": proofs}
        return jvp_cache[identity]

    def endpoint(candidate, *, held=False):
        mode = "held" if held else "native"
        identity = tensor_digest({"point": candidate})
        if identity not in endpoint_cache[mode]:
            with torch.no_grad():
                if held:
                    with branches.hold(): outputs = forward(candidate, "additive_secant:" + mode)
                else: outputs = forward(candidate, "additive_secant:" + mode)
                if set(outputs) != set(baseline):
                    raise ValueError("Endpoint native scalar membership changed")
                endpoint_cache[mode][identity] = {"values": {name: {"loss": float(_scalar(value, name)),
                    "loss_dtype": str(value.dtype)} for name, value in outputs.items()},
                    "contact_selection": deepcopy(getattr(branch.scorer, "last_contact_selection", {}))}
        return endpoint_cache[mode][identity]

    def scalar_endpoint(candidate, key, *, held=False):
        row = endpoint(candidate, held=held)
        name = additive_fd.REPAIR_PRIMITIVE if key == "repair_only" else key
        return {**row["values"][name], "contact_selection": row["contact_selection"]}

    def forward_for_repair(direction):
        values = dual_probe(direction)["outputs"]
        return {"repair_only": values[additive_fd.REPAIR_PRIMITIVE]}

    # Order matches the old full gate: three components, optional repair, total.
    ordered = [*COMPONENT_OUTPUTS, *(["repair_only"] if target is not None else []), "combined"]
    for key in ordered:
        derivative = native_gradients[key]
        base_key = additive_fd.REPAIR_PRIMITIVE if key == "repair_only" else key
        item = {"validation_contract": POLICY, "baseline_loss": baseline[base_key],
                "gradient_norm": float(derivative.double().norm()), "attached": True,
                "fixed_pose_and_mask": True, "passed": False}
        if key == "combined": report["local_fd_validation"] = item
        elif key == "repair_only": report["repair_only_validation"] = item
        else: report["component_gradients"][key] = item
        try:
            if not math.isfinite(item["gradient_norm"]) or item["gradient_norm"] <= 1e-12:
                raise ValueError("Zero/nonfinite derivative is uninformative")
            natural = lambda candidate, key=key: scalar_endpoint(candidate, key)
            held_score = lambda candidate, key=key: scalar_endpoint(candidate, key, held=True)
            if key == "repair_only":
                validate_derivative(point, derivative, natural, held_score, forward_for_repair,
                    baseline=baseline[base_key], key=key, seed=config.diagnostics.seed,
                    rtol=config.diagnostics.fd_rtol, atol=config.diagnostics.fd_atol,
                    minimum_passes=config.diagnostics.minimum_fd_passes, report=item, persist=persist)
                # Preserve v2 repair semantics, while identifying the enclosing
                # v3 policy. The repair target and categorical NLL are unchanged.
                item["validation_contract"] = POLICY
                item["repair_derivative_contract"] = "unchanged_branch_local_v2"
                item["original_repair_policy"] = item["policy"]
                item["policy"] = {**item["policy"], "version": POLICY,
                                  "repair_derivative_contract": "unchanged_branch_local_v2"}
            else:
                directions, _, selection = direction_plan(point, derivative, seed=config.diagnostics.seed)
                item.update(direction_selection=selection, original_net_fd={}, additive_inputs={}, additive_certificates={},
                    independent_forward_ad={"rtol": JVP_RTOL, "atol": JVP_ATOL, "rows": [], "passed": False})
                persist()
                if selection["pool_exhausted"]:
                    raise ValueError("Original direction pool exhausted")
                bindings = {label: tensor_digest({"direction": direction}) for label, direction in directions.items()}
                # Keep the old scalar tolerances and full grid, with both
                # original outcomes preserved separately from the new contract.
                for mode, score in (("natural", natural), ("held", held_score)):
                    old = {}
                    item["original_net_fd"][mode] = old
                    validate_local_fd(point, derivative, score, baseline=baseline[key], seed=config.diagnostics.seed,
                        rtol=config.diagnostics.fd_rtol, atol=config.diagnostics.fd_atol,
                        minimum_passes=config.diagnostics.minimum_fd_passes, report=old, persist=persist)
                    if old["direction_selection"] != selection:
                        raise ValueError("Original net and additive direction selections differ")
                item["natural_secants"] = item["original_net_fd"]["natural"]
                item["local_branch_fd"] = item["original_net_fd"]["held"]
                for label, direction in directions.items():
                    observed = dual_probe(direction)["outputs"][key]
                    analytic = float((derivative.double() * direction.double()).sum())
                    error = abs(observed["tangent"] - analytic)
                    tolerance = JVP_ATOL + JVP_RTOL * max(abs(analytic), abs(observed["tangent"]))
                    item["independent_forward_ad"]["rows"].append({"direction": label, **observed,
                        "reverse_projection": analytic, "absolute_error": error, "tolerance": tolerance,
                        "primal_exact": observed["primal"] == baseline[key],
                        "passed": observed["primal"] == baseline[key] and error <= tolerance})
                item["independent_forward_ad"]["passed"] = all(
                    row["passed"] for row in item["independent_forward_ad"]["rows"])
                offset = additive_fd.objective_offset(config.objective) if key == "combined" else 0.
                for mode in ("held", "natural"):
                    rows = []
                    item["additive_inputs"][mode] = rows
                    for label, direction in directions.items():
                        dual = dual_probe(direction)
                        for epsilon in additive_fd.EPSILONS:
                            plus = endpoint(point + epsilon * direction, held=mode == "held")["values"]
                            minus = endpoint(point - epsilon * direction, held=mode == "held")["values"]
                            rows.append({"direction": label, "direction_sha256": bindings[label], "epsilon": epsilon,
                                "analytic_kind": additive_fd.ANALYTIC_KIND,
                                "primitives": {name: {"analytic": dual["outputs"][name]["tangent"],
                                    "f_plus": plus[name]["loss"], "f_minus": minus[name]["loss"],
                                    "loss_dtype_plus": plus[name]["loss_dtype"], "loss_dtype_minus": minus[name]["loss_dtype"]}
                                    for name in weights[key]},
                                "aggregate": {"analytic": dual["outputs"][key]["tangent"],
                                    "reverse_projection": float((derivative.double() * direction.double()).sum()),
                                    "baseline_loss": baseline[key], "f_plus": plus[key]["loss"], "f_minus": minus[key]["loss"],
                                    "loss_dtype_plus": plus[key]["loss_dtype"], "loss_dtype_minus": minus[key]["loss_dtype"]},
                                "recomposition": dual["proofs"][key]})
                            persist()
                    item["additive_certificates"][mode] = additive_fd.certify(weights[key], rows, directions=bindings,
                        rtol=config.diagnostics.fd_rtol, atol=config.diagnostics.fd_atol,
                        minimum_passes=config.diagnostics.minimum_fd_passes, offset=offset)
                    persist()
                item["passed"] = (item["independent_forward_ad"]["passed"]
                    and item["additive_certificates"]["held"]["passed"]
                    and (key != "dbmol_contact_loss" or item["additive_certificates"]["natural"]["passed"]))
                item["policy"] = {"version": POLICY, "original_tolerances_and_grid_retained": True,
                    "native_model_during_optimization": "unmodified ReLU branches",
                    "primitive_approval_requires_shared_neighboring_scales": True,
                    "natural_contact_primitive_certificate_required": key == "dbmol_contact_loss"}
            item["status"] = "validated_nonzero" if item["passed"] else "failed"
            if not item["passed"]: errors.append(key + ": derivative checks failed")
        except Exception as error:
            item.update(passed=False, status="failed", error=str(error), error_type=type(error).__name__)
            errors.append(key + ": " + str(error))
        persist()
    report["independent_forward_ad_direction_count"] = len(jvp_cache)
    report["endpoint_cache_counts"] = {mode: len(rows) for mode, rows in endpoint_cache.items()}
    report["finite_differences"] = report["local_fd_validation"].get("local_branch_fd", {}).get("finite_differences", [])
    report["derivative_contract"]["passed"] = not errors
    persist()
    if errors:
        raise ValueError("Additive branch derivative gate failed: " + "; ".join(errors))
