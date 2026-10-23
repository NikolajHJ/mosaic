"""Allocated exact-state diagnostic of the small frozen Boltz affinity heads.

Native FP32 pooled endpoints, weights, mask, pose and latent directions are
unchanged. Cached head arithmetic is then explicitly varied: FP32/FP64 and
baseline-ReLU-sign-held counterfactuals. The latter extends a local affine
branch; it is NOT a replacement model, production fix or likelihood. Tiny
head-only probes use a declared cached pooled chord, not an exact native JVP.
All evidence refers to the two source091 step15 failures, authenticated against
source093 and completed source097. No DeFoG inference or latent updates occur.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import math
import os
from pathlib import Path
import sys


CASE_TIMEOUT_SECONDS = 3450
SOURCE097_ARCHIVE = "672f7098bf7f5248752f56310e78f1d74d222c7ad9376011721d4c4b16de4e69"
SOURCE097_MANIFEST = "925d5ed1ac132566a54500df8279867ae4870b1ed16bcb8f70648abf5a9fea96"
SOURCE097_OBSERVER = "201e3e03382efa3f2a9eeccb7b1421daf54794479f5e87c842185a87eedac7d8"
SOURCE097_HELPER = "286e3683d32bc377c1542e51eefba7f7c2fb9938f107d29d8419a5912716ab9a"
VARIANTS = (("fp32_native", "float32", "none"), ("fp64_native", "float64", "none"),
            ("fp32_shared_sign_held", "float32", "shared"),
            ("fp64_shared_sign_held", "float64", "shared"),
            ("fp32_all_sign_held", "float32", "all"),
            ("fp64_all_sign_held", "float64", "all"))
TANGENT_FRACTIONS = (1 / 16, 1 / 32, 1 / 64)
MAXIMUM_CACHED_HEAD_CALLS = 502
MAXIMUM_SHARED_SWITCH_CALLS = 72
CLAIM_LIMIT = __doc__
_HELPER = None
_BOUNDARY = None
_CASE = None
_ENFORCE_HEAD_BUDGET = False
_HEAD_CALLS = 0
_SWITCH_CALLS = 0


def helper():
    """Load instrumentation before original-source bootstrap imports latentmol."""
    global _HELPER
    if _HELPER is None:
        spec = importlib.util.spec_from_file_location("_cached_head_replay_helper",
            Path(__file__).with_name("failed_gate_attribution.py"))
        _HELPER = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_HELPER)
    return _HELPER


def boundary(case):
    """Use the sealed positive-depth source097 observer, never the checkout draft."""
    global _BOUNDARY
    paths = [Path(path) for path in case["parent_artifacts_sha256"]
             if "source-097-operation-boundary-recovery-attempt001/src/latentmol/boltz/" in path
             and Path(path).name == "operation_boundary_diagnostic.py"]
    if len(paths) != 1 or helper()._sha(helper().authenticated_artifact(case, paths[0])) != SOURCE097_OBSERVER:
        raise ValueError("Require the exact sealed source097 BoundaryCapture")
    if helper()._sha(helper().authenticated_artifact(case, paths[0].with_name("failed_gate_attribution.py"))) != SOURCE097_HELPER:
        raise ValueError("Require source097's authenticated adjacent replay helper")
    if _BOUNDARY is None:
        spec = importlib.util.spec_from_file_location("_sealed097_boundary_observer", paths[0])
        _BOUNDARY = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_BOUNDARY)
    if Path(_BOUNDARY.__file__).resolve() != paths[0].resolve():
        raise ValueError("Cases resolved different boundary instrumentation")
    return _BOUNDARY


def source097_reference(case, report):
    paths = [Path(path) for path in case["parent_artifacts_sha256"]
             if "source-097-operation-boundary-recovery-attempt001-replay" in path
             and Path(path).name == "attribution.json"]
    if len(paths) != 1:
        raise ValueError("Require one source097 completed case reference")
    path = helper().authenticated_artifact(case, paths[0])
    record = helper()._read(helper().authenticated_artifact(case, path.with_name("run.json")))
    previous = helper()._read(path)
    if (record.get("status") != "completed" or record.get("source_archive_sha256") != SOURCE097_ARCHIVE
            or record.get("source", {}).get("sha256") != SOURCE097_MANIFEST
            or any(previous.get(key) is not True for key in
                   ("diagnostic_completed", "frozen_state_unchanged", "parent_unchanged"))
            or previous.get("identities_before") != report["identities_before"]
            or previous.get("identities_before") != previous.get("identities_after")
            or previous.get("target") != report["target"]
            or previous.get("original_item") != report["original_item"]):
        raise ValueError("Source097 evidence does not authenticate this completed scientific case")
    return previous, {"path": str(path), "sha256": helper()._sha(path),
        "run_sha256": helper()._sha(path.with_name("run.json")), "archive_sha256": SOURCE097_ARCHIVE,
        "manifest_sha256": SOURCE097_MANIFEST, "observer_sha256": SOURCE097_OBSERVER}


def reference_baselines(reference):
    """Pair source093's different forward/analytic schemas without merging them.

    The no-grad forward records contain the canonical graph digest and output
    dtypes. Analytic records contain the full backward digest, but deliberately
    do not contain ``graph_sha256``. Source099 incorrectly read that key from
    an analytic record. Require a unique same-repeat forward record and exact
    scalar agreement; never invent a digest or weaken a native parity check.
    """
    rows = reference.get("baselines", [])
    analytic = next((row for row in rows if row.get("grad_enabled") is True), None)
    if (analytic is None or type(analytic.get("repeat")) is not int or analytic["repeat"] < 0
            or not isinstance(analytic.get("values"), dict) or not analytic["values"]
            or not helper()._hash(analytic.get("sum_gradient_identity", {}).get("combined_gradient_sha256"))):
        raise ValueError("Require the source093 analytic baseline and original backward digest")
    forward = [row for row in rows if row.get("grad_enabled") is False
               and row.get("repeat") == analytic["repeat"]]
    if (len(forward) != 1 or not helper()._hash(forward[0].get("graph_sha256"))
            or forward[0].get("values") != analytic["values"]
            or ("graph_sha256" in analytic and analytic["graph_sha256"] != forward[0]["graph_sha256"])):
        raise ValueError("Require a unique same-repeat source093 forward graph digest and exact analytic scalar parity")
    return forward[0], analytic


def heads_forward(heads, pooled, *, held_scope="none", masks=None):
    """Reproduce both native readouts and ensemble reductions from tiny inputs.

    Each Sequential contains only Linear/ReLU. No norm or epsilon lies here.
    Sign-held multiplication has the native derivative at the baseline (zero
    inputs retain ReLU's zero derivative), but changes the off-baseline model.
    """
    global _HEAD_CALLS
    if _ENFORCE_HEAD_BUDGET:
        if _HEAD_CALLS >= MAXIMUM_CACHED_HEAD_CALLS:
            raise ValueError("Cached ensemble-head call budget exceeded")
        _HEAD_CALLS += 1
    import torch
    if held_scope not in ("none", "shared", "all") or len(heads) != 2 or len(pooled) != 2:
        raise ValueError("Require the declared two-member head variant")
    if held_scope != "none" and masks is None:
        raise ValueError("Sign-held variants require baseline masks")
    pre, shared, values, logits = {}, {}, [], []

    def sequence(module, x, prefix, hold):
        if not isinstance(module, torch.nn.Sequential):
            raise ValueError("Unexpected native head topology")
        for index, layer in enumerate(module):
            name = prefix + "." + str(index)
            if isinstance(layer, torch.nn.ReLU):
                if layer.inplace:
                    raise ValueError("Observer requires non-inplace native ReLU")
                pre[name] = x
                if hold:
                    if name not in masks or masks[name].shape != x.shape:
                        raise ValueError("Missing or differently shaped baseline ReLU mask")
                    x = x * masks[name].to(x)
                else:
                    x = layer(x)
            elif isinstance(layer, torch.nn.Linear):
                x = layer(x)
            else:
                raise ValueError("Unsupported cached head operation: " + type(layer).__name__)
        return x

    for index, (head, x) in enumerate(zip(heads, pooled)):
        prefix = "head." + str(index)
        g = sequence(head.affinity_out_mlp, x, prefix + ".affinity_out_mlp", held_scope != "none")
        shared[str(index)] = g
        value = sequence(head.to_affinity_pred_value, g, prefix + ".to_affinity_pred_value", held_scope == "all")
        score = sequence(head.to_affinity_pred_score, g, prefix + ".to_affinity_pred_score", held_scope == "all")
        if not isinstance(head.to_affinity_logits_binary, torch.nn.Linear):
            raise ValueError("Expected native linear binary-logit projection")
        values.append(value.reshape(-1, 1))
        logits.append(head.to_affinity_logits_binary(score).reshape(-1, 1))
    outputs = {"affinity_pred_value": torch.stack(values).mean(0),
               "affinity_probability_binary": torch.stack([x.sigmoid() for x in logits]).mean(0),
               "affinity_logits_binary": torch.stack(logits).mean(0)}
    return outputs, shared, pre


def crossings(current, baseline):
    """Save exact switched unit indices and their values, not just counts."""
    import torch
    if set(current) != set(baseline):
        raise ValueError("Head ReLU membership changed")
    result = {}
    for name, value in current.items():
        base = baseline[name].to(value)
        indices = torch.nonzero((value > 0) != (base > 0), as_tuple=False)
        result[name] = {"positive_branch_changes": len(indices),
            "baseline_min_abs": float(base.abs().min()), "baseline_exact_zeros": int((base == 0).sum()),
            "changed_units": [{"index": row.tolist(), "baseline": float(base[tuple(row.tolist())]),
                               "endpoint": float(value[tuple(row.tolist())])} for row in indices]}
    return result


def head_scalar(outputs, regression_weight, logit_weight):
    """Source091 DBMol affinity component, without repair/structural terms."""
    return (regression_weight * outputs["affinity_pred_value"].double().reshape(())
            - logit_weight * outputs["affinity_logits_binary"].double().reshape(()))


def projection(values, cotangents):
    return sum((value.double() * cotangent.double()).sum() for value, cotangent in zip(values, cotangents))


def centered_projection(values, baseline, cotangents):
    """Source097 readout order: widen each endpoint BEFORE subtraction."""
    return projection(tuple(value.double() - base.double() for value, base in zip(values, baseline)), cotangents)


def centered_secant(plus, minus, baseline, cotangents, epsilon):
    """Source097 forms two centered readout scalars before differencing them."""
    p = float(centered_projection(plus, baseline, cotangents))
    m = float(centered_projection(minus, baseline, cotangents))
    return (p - m) / (2 * epsilon)


def shared_switch_contributions(heads, pooled, masks, cotangents):
    """Exact two-ReLU switch identity, with individual unit contributions.

    For y=ReLU(W2 ReLU(u)+b2), baseline masks m1,m2 give
    y-y_held = m2*W2*(ReLU(u)-m1*u) + (ReLU(v)-m2*v).
    Contract with the original fixed shared-output cotangent. This predicts
    the sign-held counterfactual difference without differentiating a switch.
    It is a declared algebraic attribution, not an alternative autograd rule.
    """
    global _SWITCH_CALLS
    if _ENFORCE_HEAD_BUDGET:
        if _SWITCH_CALLS >= MAXIMUM_SHARED_SWITCH_CALLS:
            raise ValueError("Shared-switch attribution call budget exceeded")
        _SWITCH_CALLS += 1
    import torch
    result, total = {}, 0.
    for index, (head, point, cotangent) in enumerate(zip(heads, pooled, cotangents)):
        layers = head.affinity_out_mlp
        if (len(layers) != 4 or not isinstance(layers[0], torch.nn.Linear)
                or not isinstance(layers[1], torch.nn.ReLU) or not isinstance(layers[2], torch.nn.Linear)
                or not isinstance(layers[3], torch.nn.ReLU)):
            raise ValueError("Switch identity requires native Linear/ReLU/Linear/ReLU shared MLP")
        name1, name2 = f"head.{index}.affinity_out_mlp.1", f"head.{index}.affinity_out_mlp.3"
        u = layers[0](point)
        v = layers[2](layers[1](u))
        first = layers[1](u) - masks[name1].to(u) * u
        second = layers[3](v) - masks[name2].to(v) * v
        effective = (cotangent.to(v) * masks[name2].to(v)) @ layers[2].weight
        for name, residual, coefficient, pre in ((name1, first, effective, u),
                                                  (name2, second, cotangent.to(v), v)):
            contributions = residual * coefficient
            indices = torch.nonzero(residual != 0, as_tuple=False)
            value = float(contributions.sum())
            total += value
            result[name] = {"contracted_switch_effect": value,
                "units": [{"index": row.tolist(), "preactivation": float(pre[tuple(row.tolist())]),
                           "relu_minus_baseline_linear_branch": float(residual[tuple(row.tolist())]),
                           "fixed_effective_cotangent": float(coefficient[tuple(row.tolist())]),
                           "predicted_contribution": float(contributions[tuple(row.tolist())])} for row in indices]}
    return {"total_predicted_shared_counterfactual_difference": total, "layers": result,
            "scope": "FP64 cached head with original fixed shared cotangent and native baseline sign masks"}


def scalar_comparison(analytic, fd, *, rtol, atol):
    """Report the existing tolerance separately from diagnostic signal quality."""
    if not all(math.isfinite(value) for value in (analytic, fd)):
        raise ValueError("Nonfinite isolated-head slope")
    bound = atol + rtol * max(abs(analytic), abs(fd))
    return {"analytic": analytic, "central_difference": fd, "absolute_error": abs(analytic - fd),
            "original_tolerance": bound, "within_original_tolerance": abs(analytic - fd) <= bound,
            "historical_gate_approved": False}


def weighted_output_spacing(outputs, regression_weight, logit_weight):
    """Primitive spacing is a signal diagnostic, not a network error bound."""
    import torch
    total = 0.
    for name, weight in (("affinity_pred_value", regression_weight), ("affinity_logits_binary", logit_weight)):
        value = outputs[name].detach()
        up = torch.nextafter(value, torch.full_like(value, math.inf))
        down = torch.nextafter(value, torch.full_like(value, -math.inf))
        total += abs(weight) * max(float((up.double() - value.double()).abs().max()),
                                  float((value.double() - down.double()).abs().max()))
    return total


def branch_radius(preactivations, derivatives):
    """Exact symmetric open-cell radius of a sign-held piecewise-linear head.

    Layer preactivations are affine on the baseline cell. The first possible
    crossing bounds the entire local cell in either direction; a baseline zero
    with nonzero slope has no symmetric differentiable neighborhood.
    """
    import torch
    if set(preactivations) != set(derivatives):
        raise ValueError("Head tangent telemetry differs")
    limiting, radius = None, math.inf
    for name, value in preactivations.items():
        slope = derivatives[name]
        if value.shape != slope.shape or not bool(torch.isfinite(value).all() and torch.isfinite(slope).all()):
            raise ValueError("Malformed head branch tangent")
        active = slope != 0
        if bool(active.any()):
            candidates = torch.where(active, (value / torch.where(active, slope, torch.ones_like(slope))).abs(),
                                     torch.full_like(value, math.inf))
            candidate, flat = candidates.flatten().min(dim=0)
            if float(candidate) < radius:
                radius = float(candidate)
                limiting = {"layer": name, "flat_index": int(flat), "preactivation": float(value.flatten()[flat]),
                            "preactivation_slope": float(slope.flatten()[flat])}
    return {"symmetric_open_cell_radius": None if math.isinf(radius) else radius,
            "limiting_unit": limiting, "baseline_zero_with_nonzero_tangent": radius == 0,
            "radius_is_head_only_not_latent_space": True}


def tangent_probes(heads, pooled, tangent, masks, *, regression_weight, logit_weight, rtol, atol):
    """Autograd/FD within the same head ReLU cell, using identical head weights.

    Input tangent is a declared cached pooled chord. FP64 and FP32 head probes
    are separately labelled; realized representable chords and branch changes
    are retained. A crossing or no representable motion cannot certify a probe.
    """
    import torch
    base64 = tuple(x.detach().double() for x in pooled)
    tangent64 = tuple(x.detach().double() for x in tangent)
    native64_pre = heads_forward(heads["float64"], base64)[2]
    masks64 = {name: value > 0 for name, value in native64_pre.items()}
    held_pre_names = list(native64_pre)

    def pre_tuple(*inputs):
        pre = heads_forward(heads["float64"], inputs, held_scope="all", masks=masks64)[2]
        return tuple(pre[name] for name in held_pre_names)

    pre, derivatives = torch.autograd.functional.jvp(pre_tuple, base64, tangent64, create_graph=False)
    radius = branch_radius(dict(zip(held_pre_names, pre)), dict(zip(held_pre_names, derivatives)))
    result = {"scope": "Isolated cached heads along original pooled central chord; not an exact native latent JVP",
        "cell": radius, "probes": [], "fractions_of_open_cell_or_original_scale": list(TANGENT_FRACTIONS)}
    if radius["baseline_zero_with_nonzero_tangent"]:
        result["status"] = "no_symmetric_same_branch_neighborhood"
        return result
    # Never enlarge the original smallest latent scale. This is a tiny head
    # test, not an outcome-selected replacement finite-difference grid.
    cap = min(radius["symmetric_open_cell_radius"] or 1e-5, 1e-5)
    for dtype_name in ("float32", "float64"):
        dtype = getattr(torch, dtype_name)
        base = tuple(x.to(dtype).requires_grad_() for x in pooled)
        direction = tuple(x.to(dtype) for x in tangent)
        outputs, _, base_pre = heads_forward(heads[dtype_name], base)
        value = head_scalar(outputs, regression_weight, logit_weight)
        gradient = torch.autograd.grad(value, base)
        analytic = float(projection(gradient, direction))
        for fraction in TANGENT_FRACTIONS:
            epsilon = cap * fraction
            plus_inputs = tuple(x.detach() + epsilon * d for x, d in zip(base, direction))
            minus_inputs = tuple(x.detach() - epsilon * d for x, d in zip(base, direction))
            with torch.no_grad():
                p, _, pp = heads_forward(heads[dtype_name], plus_inputs)
                m, _, mp = heads_forward(heads[dtype_name], minus_inputs)
                plus, minus = float(head_scalar(p, regression_weight, logit_weight)), float(head_scalar(m, regression_weight, logit_weight))
                switches = {"plus": crossings(pp, base_pre), "minus": crossings(mp, base_pre)}
                spacing = max(weighted_output_spacing(p, regression_weight, logit_weight),
                              weighted_output_spacing(m, regression_weight, logit_weight))
            same_branch = not any(row["positive_branch_changes"] for side in switches.values() for row in side.values())
            chord = tuple(p.double() - m.double() for p, m in zip(plus_inputs, minus_inputs))
            motion = float(sum(x.square().sum() for x in chord).sqrt())
            realized = float(projection(chord, gradient) / (2 * epsilon))
            comparison = scalar_comparison(analytic, (plus - minus) / (2 * epsilon), rtol=rtol, atol=atol)
            informative = abs(analytic) > atol and abs(2 * epsilon * analytic) > spacing and motion > 0 and plus != minus
            result["probes"].append({"dtype": dtype_name, "epsilon": epsilon, "fraction": fraction,
                "same_relu_branches": same_branch, "realized_chord_norm": motion,
                "realized_chord_linear_prediction": realized, "f_plus": plus, "f_minus": minus,
                "endpoint_scalar_difference": plus - minus, "branch_crossings": switches,
                "weighted_native_head_output_spacing": spacing, "informative": informative,
                "signal_disclaimer": "Output rounding/quantization can dominate FP32 tiny probes; no tolerance retuning",
                "comparison": comparison, "head_only_same_branch_agreement": same_branch and informative and
                    comparison["within_original_tolerance"]})
    result["status"] = "completed"
    return result


def forward_budget(direction_counts):
    if len(direction_counts) != 2 or any(type(count) is not int or not 2 <= count <= 4 for count in direction_counts):
        raise ValueError("Require two original bounded direction sets")
    explicit = sum(1 + 2 * 9 * count for count in direction_counts)
    if explicit > 146:
        raise ValueError("Endpoint plan exceeds head-observer budget")
    return {"explicit_native_affinity_forwards": explicit, "maximum_explicit_native_affinity_forwards": 146,
        "auxiliary_full_dbmol_or_native_parity_affinity_traversals": 6,
        "total_native_affinity_traversals_upper_bound": explicit + 6,
        "maximum_cached_ensemble_head_calls_per_case": MAXIMUM_CACHED_HEAD_CALLS,
        "maximum_shared_switch_identity_calls_per_case": MAXIMUM_SHARED_SWITCH_CALLS,
        "maximum_child_seconds": 2 * CASE_TIMEOUT_SECONDS, "gpu_wall_seconds": 7200,
        "controller_headroom_seconds": 7200 - 2 * CASE_TIMEOUT_SECONDS,
        "new_defog_calls": 0, "optimizer_updates": 0}


def probe_case(decoder, point, branch, config, counters, target, weight, item_name, item,
               *, persist=lambda: None, report=None):
    global _ENFORCE_HEAD_BUDGET, _HEAD_CALLS, _SWITCH_CALLS
    _ENFORCE_HEAD_BUDGET, _HEAD_CALLS, _SWITCH_CALLS = True, 0, 0
    import torch
    from latentmol.boltz.active_set import score_at
    from latentmol.boltz.local_fd import EPSILONS, endpoint_row
    from latentmol.multimodel.study import tensor_digest
    report = report if report is not None else {}
    observer = boundary(_CASE)
    reference, receipt = observer.authenticated_reference(_CASE, report)
    previous, receipt097 = source097_reference(_CASE, report)
    forward_base, analytic_base = reference_baselines(reference)
    if config.recycling_steps != 0:
        raise ValueError("Original zero recycling required")
    report.update(status="probing", diagnostic_completed=False, reference093=receipt, reference097=receipt097,
        observer_module_sha256=helper()._sha(__file__), original_gate_not_reclassified=True,
        finite_differences=[], head_only_tangents={}, scope=CLAIM_LIMIT)
    report["native_provenance"] = observer.native_provenance(_CASE, branch.scorer.affinity.model)
    components = {}
    measured, total = score_at(decoder, point, branch, config, counters, "head_original_full_analytic",
        gradients=True, component_gradients=components, repair_target=target, repair_weight=weight)
    if (measured["components"] != {key: analytic_base["values"][key] for key in measured["components"]}
            or measured["loss"] != analytic_base["values"]["combined"]
            or tensor_digest({"gradient": total}) != analytic_base["sum_gradient_identity"]["combined_gradient_sha256"]):
        raise ValueError("Original source093 full baseline/backward differs")
    directions = helper().reconstruct_directions(point, total if item_name == "combined" else components[item_name], item)
    if any(metadata["replayed_sha256"] != previous["direction_replay"][name]["replayed_sha256"]
           for name, (_, metadata) in directions.items()):
        raise ValueError("Original source097 direction identity differs")
    direct = components["dbmol_affinity_loss"].detach()
    report["direction_replay"] = {name: metadata for name, (_, metadata) in directions.items()}
    report["source093_baseline_pair"] = {"repeat": analytic_base["repeat"],
        "forward_graph_sha256": forward_base["graph_sha256"],
        "analytic_combined_gradient_sha256": analytic_base["sum_gradient_identity"]["combined_gradient_sha256"],
        "forward_and_analytic_scalars_exactly_equal": True,
        "graph_identity_source": "Authenticated same-repeat no-grad forward record",
        "backward_identity_source": "Authenticated grad-enabled analytic record"}
    del total, components

    def capture_baseline():
        probe = point.detach().clone().requires_grad_()
        with torch.enable_grad(), observer.BoundaryCapture(branch.scorer.affinity.model) as capture:
            outputs, loss, graph_sha = observer.native_forward(decoder, probe, branch, config, counters, capture, "head_native_baseline")
            if (float(loss) != forward_base["values"]["dbmol_affinity_loss"] or graph_sha != forward_base["graph_sha256"]
                    or any(float(value) != forward_base["values"][name] for name, value in outputs.items())):
                raise ValueError("Native head baseline output differs")
            gradient, = torch.autograd.grad(loss, probe, retain_graph=True)
            clone_parity = observer.chain_completeness(direct, gradient)
            pooled = tuple(capture.values[f"member.{i}.pooled"] for i in range(2))
            shared = tuple(capture.values[f"member.{i}.shared"] for i in range(2))
            pool_cotangent = torch.autograd.grad(loss, pooled, retain_graph=True)
            shared_cotangent = torch.autograd.grad(loss, shared, retain_graph=True)
            chain, = torch.autograd.grad(pooled, probe, grad_outputs=pool_cotangent)
            chain_parity = observer.chain_completeness(direct, chain)
            if not clone_parity["engineering_chain_complete"] or not chain_parity["engineering_chain_complete"]:
                raise ValueError("Original native pooled frontier VJP differs")
            metadata = observer.tensor_metadata(capture.values)
            if metadata != previous["boundary_tensors"]:
                raise ValueError("Exact source097 native baseline boundary bytes differ")
            return {"pooled": tuple(x.detach().clone() for x in pooled),
                "shared": tuple(x.detach().clone() for x in shared),
                "pool_cotangent": tuple(x.detach().clone() for x in pool_cotangent),
                "shared_cotangent": tuple(x.detach().clone() for x in shared_cotangent),
                "head_pre": {key: value.detach().clone() for key, value in capture.relus.items() if key.startswith("head.")},
                "outputs": {key: value.detach().clone() for key, value in outputs.items()},
                "baseline": float(loss), "graph_sha256": graph_sha,
                "clone_backward_parity": clone_parity, "pooled_complete_vjp": chain_parity,
                "topology": deepcopy(capture.topology)}
    base = observer.phase("capture_exact_native_pooled_baseline", capture_baseline, point.device, report, persist)
    native_heads = tuple(member.affinity_heads for member in
                         (branch.scorer.affinity.model.affinity_module1, branch.scorer.affinity.model.affinity_module2))
    heads = {name: tuple(deepcopy(head).to(dtype=getattr(torch, name)).eval().requires_grad_(False)
                        for head in native_heads) for name in ("float32", "float64")}
    masks = {key: value > 0 for key, value in base["head_pre"].items()}
    regression, logit = config.dbmol.affinity_regression_weight, config.dbmol.affinity_logit_weight
    baseline_variants = {}
    for variant, dtype_name, scope in VARIANTS:
        inputs = tuple(x.to(getattr(torch, dtype_name)).requires_grad_() for x in base["pooled"])
        outputs, shared, pre = heads_forward(heads[dtype_name], inputs, held_scope=scope, masks=masks)
        scalar = head_scalar(outputs, regression, logit)
        pool_grad = torch.autograd.grad(scalar, inputs)
        if dtype_name == "float32":
            if any(not torch.equal(outputs[key], base["outputs"][key]) for key in outputs):
                raise ValueError("Cached FP32 baseline output parity failed: " + variant)
            if any(not torch.equal(pre[key], base["head_pre"][key]) for key in pre):
                raise ValueError("Cached FP32 baseline preactivation parity failed: " + variant)
            vjp = observer.chain_completeness(torch.cat([x.flatten() for x in base["pool_cotangent"]]),
                                             torch.cat([x.flatten() for x in pool_grad]))
            if not vjp["engineering_chain_complete"]:
                raise ValueError("Cached FP32 head VJP parity failed: " + variant)
        else:
            vjp = {"scope": "Declared widened head arithmetic; original parity not claimed",
                "difference_norm_float64": float(sum((expected.double() - current.double()).square().sum()
                                          for expected, current in zip(base["pool_cotangent"], pool_grad)).sqrt())}
        baseline_variants[variant] = {"loss": float(scalar), "outputs": {key: float(value) for key, value in outputs.items()},
            "native_pool_vjp_parity": vjp, "pooled_gradient": observer.tensor_metadata(dict(enumerate(pool_grad))),
            "shared": tuple(shared[str(i)].detach().clone() for i in range(2)),
            "pool_gradient": tuple(x.detach().clone() for x in pool_grad)}
    report["baseline"] = {key: value for key, value in base.items() if not isinstance(value, (dict, tuple))}
    report["native_baseline_parity"] = {"clone_backward": base["clone_backward_parity"],
        "pooled_complete_vjp": base["pooled_complete_vjp"], "source097_all_boundary_bytes_equal": True}
    report["variants"] = {name: {key: value for key, value in data.items() if key not in ("shared", "pool_gradient")}
                           for name, data in baseline_variants.items()}
    persist()

    cache = {"baseline": base, "baseline_variants": baseline_variants,
        "weights": [{key: value.detach().cpu() for key, value in head.state_dict().items()}
                                             for head in native_heads], "endpoints": {}}

    def capture_endpoint(value, label, original):
        with torch.no_grad(), observer.BoundaryCapture(branch.scorer.affinity.model) as capture:
            outputs, loss, graph_sha = observer.native_forward(decoder, value, branch, config, counters, capture, label)
            if (float(loss) != original["values"]["dbmol_affinity_loss"] or graph_sha != original["graph_sha256"]
                    or any(float(output) != original["values"][key] for key, output in outputs.items())):
                raise ValueError("Native endpoint differs from exact source093 endpoint")
            return {"pooled": tuple(capture.values[f"member.{i}.pooled"].detach().clone() for i in range(2)),
                "shared": tuple(capture.values[f"member.{i}.shared"].detach().clone() for i in range(2)),
                "head_pre": {key: x.detach().clone() for key, x in capture.relus.items() if key.startswith("head.")},
                "outputs": {key: x.detach().clone() for key, x in outputs.items()},
                "loss": float(loss), "graph_sha256": graph_sha}

    for direction_name, (direction, _) in directions.items():
        tangent = None
        for epsilon in EPSILONS:
            oldrow = [row for row in reference["finite_differences"] if row["direction"] == direction_name
                      and row["epsilon"] == epsilon and row["grad_enabled"] is False]
            old097 = [row for row in previous["finite_differences"] if row["direction"] == direction_name and row["epsilon"] == epsilon]
            if len(oldrow) != 1 or len(old097) != 1:
                raise ValueError("Missing unique original endpoint")
            plus = capture_endpoint(point + epsilon * direction, "head_endpoint_plus", oldrow[0]["plus"])
            minus = capture_endpoint(point - epsilon * direction, "head_endpoint_minus", oldrow[0]["minus"])
            if tangent is None:
                tangent = tuple((p.double() - m.double()) / (2 * epsilon) for p, m in zip(plus["pooled"], minus["pooled"]))
            native_row = endpoint_row(point, direction, direct, epsilon,
                {"loss": plus["loss"], "loss_dtype": "torch.float64"},
                {"loss": minus["loss"], "loss_dtype": "torch.float64"}, base["baseline"],
                rtol=config.diagnostics.fd_rtol, atol=config.diagnostics.fd_atol)
            if native_row != old097[0]["native_affinity"]:
                raise ValueError("Original source097 derivative row differs")
            variants = {}
            for variant, dtype_name, scope in VARIANTS:
                endpoint_values = []
                for endpoint in (plus, minus):
                    inputs = tuple(x.to(getattr(torch, dtype_name)) for x in endpoint["pooled"])
                    with torch.no_grad():
                        outputs, shared, pre = heads_forward(heads[dtype_name], inputs, held_scope=scope, masks=masks)
                    if variant == "fp32_native" and (any(not torch.equal(outputs[key], endpoint["outputs"][key]) for key in outputs)
                            or any(not torch.equal(pre[key], endpoint["head_pre"][key]) for key in pre)):
                        raise ValueError("Cached unchanged FP32 endpoint parity failed")
                    endpoint_values.append({"loss": float(head_scalar(outputs, regression, logit)),
                        "shared_readout": float(centered_projection(tuple(shared[str(i)] for i in range(2)),
                                                    baseline_variants[variant]["shared"], base["shared_cotangent"])),
                        "branch_crossings": crossings(pre, base["head_pre"])})
                p, m = endpoint_values
                variants[variant] = {"plus": p, "minus": m,
                    "full_affinity_secant": (p["loss"] - m["loss"]) / (2 * epsilon),
                    "shared_fixed_cotangent_secant": (p["shared_readout"] - m["shared_readout"]) / (2 * epsilon),
                    "same_cached_native_pooled_endpoint_tensors": True,
                    "diagnostic_counterfactual": scope != "none" or dtype_name != "float32"}
            pooled_secant = centered_secant(plus["pooled"], minus["pooled"], base["pooled"],
                                             base["pool_cotangent"], epsilon)
            shared_original = old097[0]["cuts"]["shared_head_mlp"]["original_native_reference"]["central_difference"]
            pooled_original = old097[0]["cuts"]["pooled"]["original_native_reference"]["central_difference"]
            if (pooled_secant != pooled_original or variants["fp32_native"]["shared_fixed_cotangent_secant"] != shared_original):
                raise ValueError("Cached pooled/shared source097 fixed-cotangent readout parity failed")
            # This is an explicit additive comparison, not a claim that the
            # production derivative or head precision has been corrected.
            v32, v64, held = variants["fp32_native"], variants["fp64_native"], variants["fp64_shared_sign_held"]
            with torch.no_grad():
                switch_predictions = [shared_switch_contributions(heads["float64"],
                    tuple(x.double() for x in endpoint["pooled"]), masks, base["shared_cotangent"])
                    for endpoint in (plus, minus)]
            predicted_switch_secant = (switch_predictions[0]["total_predicted_shared_counterfactual_difference"]
                                      - switch_predictions[1]["total_predicted_shared_counterfactual_difference"]) / (2 * epsilon)
            decomposition = {"shared_precision_secant_difference": v32["shared_fixed_cotangent_secant"] - v64["shared_fixed_cotangent_secant"],
                "shared_relu_switch_secant_effect": v64["shared_fixed_cotangent_secant"] - held["shared_fixed_cotangent_secant"],
                "predicted_shared_relu_switch_secant": predicted_switch_secant,
                "switch_identity_secant_residual": predicted_switch_secant -
                    (v64["shared_fixed_cotangent_secant"] - held["shared_fixed_cotangent_secant"]),
                "switch_contributions_plus": switch_predictions[0], "switch_contributions_minus": switch_predictions[1],
                "shared_sign_held_minus_native_pooled_secant": held["shared_fixed_cotangent_secant"] - pooled_secant,
                "native_pooled_secant_minus_latent_analytic": pooled_secant - native_row["analytic"],
                "predicted_shared_total_residual": v32["shared_fixed_cotangent_secant"] - native_row["analytic"],
                "full_head_precision_secant_difference": v32["full_affinity_secant"] - v64["full_affinity_secant"],
                "full_head_all_relu_switch_secant_effect": v64["full_affinity_secant"] - variants["fp64_all_sign_held"]["full_affinity_secant"]}
            report["finite_differences"].append({"direction": direction_name, "epsilon": epsilon,
                "original_native_affinity": native_row, "pooled_fixed_cotangent_secant": pooled_secant,
                "variants": variants, "residual_decomposition": decomposition,
                "source093_endpoint_and_source097_readout_parity": True})
            cache["endpoints"][direction_name + ":" + str(epsilon)] = {"plus": plus, "minus": minus}
            persist()
        report["head_only_tangents"][direction_name] = tangent_probes(heads, base["pooled"], tangent, masks,
            regression_weight=regression, logit_weight=logit,
            rtol=config.diagnostics.fd_rtol, atol=config.diagnostics.fd_atol)
        persist()
    # Tiny tensors only: save exact pooled endpoints, masks, preactivations and
    # weights for inspectable replay. No full model graph or full pair state.
    def cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: cpu(child) for key, child in value.items()}
        if isinstance(value, tuple):
            return tuple(cpu(child) for child in value)
        if isinstance(value, list):
            return [cpu(child) for child in value]
        return value
    cache_path = _OUTPUT / "cached_heads.pt"
    torch.save(cpu(cache), cache_path)
    report.update(status="completed", diagnostic_completed=True, old_gate_certified_by_this_diagnostic=False,
        cache={"path": str(cache_path), "sha256": helper()._sha(cache_path)},
        actual_explicit_native_affinity_forwards=1 + 2 * len(directions) * len(EPSILONS),
        actual_cached_ensemble_head_calls=_HEAD_CALLS, actual_shared_switch_identity_calls=_SWITCH_CALLS,
        auxiliary_full_dbmol_or_native_parity_affinity_traversals=3, new_defog_calls=0, optimizer_updates=0,
        interpretation="Cached head comparisons isolate precision and ReLU-switch effects; held-sign controls change off-baseline math and cannot certify historical gates. Control full combined contact/confidence failure is outside this scope.")
    persist()
    return report


_OUTPUT = None


def run_case(plan, case, output_root):
    global _CASE, _OUTPUT, _ENFORCE_HEAD_BUDGET
    _CASE = case
    replay = helper()
    boundary(case)  # Authenticate instrumentation before old source imports.
    original = replay.probe_case

    def probe_with_path(*args, **kwargs):
        global _OUTPUT
        # ExperimentRun creates its unique child directory before probing.
        roots = list(Path(output_root).glob("*/run.json"))
        if len(roots) != 1:
            raise ValueError("Require one unique new child directory")
        _OUTPUT = roots[0].parent
        return probe_case(*args, **kwargs)

    replay.probe_case = probe_with_path
    replay.CASE_TIMEOUT_SECONDS, replay.REPEATS, replay.CLAIM_LIMIT = CASE_TIMEOUT_SECONDS, 1, CLAIM_LIMIT
    try:
        return replay.run_case(plan, case, output_root)
    finally:
        replay.probe_case = original
        _ENFORCE_HEAD_BUDGET = False


def validate_cases(plan):
    if ([case["id"] for case in plan["cases"]] != ["boltz_only", "boltz_defog"]
            or any(case["step"] != 15 or case["arm"] != "fixed_initial_mask" for case in plan["cases"])):
        raise ValueError("Require the original two source091 step15 failures")
    for case in plan["cases"]:
        boundary(case)
        paths = [path for path in case["parent_artifacts_sha256"]
                 if "source-097-operation-boundary-recovery-attempt001-replay" in path and Path(path).name == "attribution.json"]
        if len(paths) != 1:
            raise ValueError("Missing unique source097 reference")
    return plan


def run_study(path, expected):
    plan = validate_cases(helper().load_plan(path, expected))
    from latentmol.experiments import ExperimentRun, write_json
    from latentmol.boltz.workflow_process import run_logged
    counts = []
    for case in plan["cases"]:
        gate = Path(case["parent_run"]) / case["arm"] / "gate-0015.json"
        _, item = helper().failed_item(helper()._read(helper().authenticated_artifact(case, gate)))
        counts.append(1 + sum(name != "gradient" and row.get("passed") is False for name, row in item["direction_results"].items()))
    budget = forward_budget(counts)
    with ExperimentRun(Path(plan["output_root"]), plan["name"], {"plan": plan, "plan_sha256": expected,
            "diagnostic_module_sha256": helper()._sha(__file__), "budget": budget, "claim_limit": CLAIM_LIMIT}, Path.cwd()) as run:
        rows = []
        for case in plan["cases"]:
            execution = run_logged([sys.executable, str(Path(__file__).resolve()), "--plan", str(Path(path).resolve()),
                "--sha256", expected, "--child", case["id"], "--child-output", str(run.path / case["id"])],
                run.path / (case["id"] + ".log"), timeout_seconds=CASE_TIMEOUT_SECONDS)
            completed, reason = boundary(case).execution_completion(execution)
            rows.append({"case_id": case["id"], "execution": execution, "diagnostic_completed": completed,
                         "incomplete_reason": reason, "old_gate_certified": False})
            write_json(run.path / "cases.json", rows)
        run.annotate(diagnostic_cases_completed=sum(row["diagnostic_completed"] for row in rows),
                     diagnostic_cases_requested=2, old_gates_certified=False)
        print(run.path, flush=True)
        boundary(plan["cases"][0]).require_completed_cases(rows)
        return run.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--child", help=argparse.SUPPRESS)
    parser.add_argument("--child-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID") or not helper()._hash(os.environ.get("LATENTMOL_SOURCE_SHA256")):
        raise ValueError("Observer requires an authenticated allocated DTU HPC job")
    plan = validate_cases(helper().load_plan(args.plan, args.sha256))
    if args.child:
        matches = [case for case in plan["cases"] if case["id"] == args.child]
        if len(matches) != 1 or args.child_output is None:
            raise ValueError("Unknown child or missing unique output root")
        print(run_case(plan, matches[0], args.child_output), flush=True)
    else:
        if args.child_output is not None:
            raise ValueError("Child output requires --child")
        print(run_study(args.plan, args.sha256), flush=True)


if __name__ == "__main__":
    main()
