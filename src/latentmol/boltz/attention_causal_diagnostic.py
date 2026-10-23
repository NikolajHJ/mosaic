"""HPC-only causal controls for native finite-mask attention arithmetic.

This observer authenticates the saved source079 experiment using source080's
loader. It neither edits that release nor runs an optimizer/DeFoG. Three
prespecified controls use exactly the same saved point, graph, target, objective
and directions: native, forward-identical masked-logit stop-gradient, and FP64
masked-row logits/softmax with probabilities cast back before the value product.
The last two deliberately change derivatives; FP64 can also change forward
values. Unchanged weights do NOT establish unchanged scorer mathematics.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys


NATIVE_ATTENTION_SHA256 = "d7245ffe8eefa580c693967e047da9950ed2b366c3595804e19a4477d789a412"
MODES = ("native", "stop_masked_logits", "masked_logits_float64")
EPSILONS = (3e-5, 1e-4, 3e-4, 0.001, 0.003, 0.01)
CASE_TIMEOUT_SECONDS = 3600
CLAIM_LIMIT = ("Causal diagnostic only: no optimizer/DeFoG, production patch, tolerance change or historical gate approval. "
               "Stop-logit derivatives and FP64 masked attention are explicit counterfactual scorer mathematics.")


def _helper():
    """Load instrumentation by file, before original scientific-package imports."""
    path = Path(__file__).with_name("failed_gate_attribution.py")
    spec = importlib.util.spec_from_file_location("_authenticated_failed_gate_loader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def whole_mask_rows(mask_bias, query, key):
    """Recognize the pinned native finite mask, preserving arbitrary batch axes."""
    import torch
    if mask_bias.requires_grad or not mask_bias.is_floating_point():
        raise ValueError("Native mask must be a fixed floating tensor")
    if not bool(((mask_bias == 0) | (mask_bias == -1e9)).all()):
        raise ValueError("Unsupported native mask: require exact 0/-1e9 entries")
    if query.ndim < 3 or key.ndim != query.ndim or query.shape[-1] != key.shape[-1]:
        raise ValueError("Unsupported native attention query/key shapes")
    shape = (*query.shape[:-1], key.shape[-2])
    if torch.broadcast_shapes(tuple(mask_bias.shape), shape) != shape:
        raise ValueError("Native mask does not broadcast to attention scores")
    # Reducing the small mask first avoids materializing its Q/head broadcast.
    return torch.broadcast_to((mask_bias == -1e9).all(-1, keepdim=True), (*shape[:-1], 1))


def controlled_attention(query, key, value, biases, *, mode, softmax):
    """Copy pinned arithmetic, intervening only in wholly masked logit rows.

    Stop-gradient keeps q/k/triangle sensitivity for other rows and retains the
    value path everywhere. FP64 recomputes QK in double (not just the addition),
    adds both biases and evaluates softmax in double only for the selected rows;
    probabilities return to the original dtype before the native value matmul.
    """
    import torch
    if mode not in MODES or len(biases) != 2:
        raise ValueError("Require a declared attention control and two native biases")
    rows = whole_mask_rows(biases[0], query, key)
    logits = torch.matmul(query, key.transpose(-1, -2))
    for bias in biases:
        logits += bias
    if mode == "stop_masked_logits":
        logits = torch.where(rows, logits.detach(), logits)
    probabilities = softmax(logits, -1)
    if mode == "masked_logits_float64" and bool(rows.any()):
        # This is a deliberately changed forward, confined to all-masked rows.
        wide = torch.matmul(query.double(), key.double().transpose(-1, -2))
        for bias in biases:
            wide += bias.double()
        probabilities = torch.where(rows, wide.softmax(-1).to(probabilities), probabilities)
    return torch.matmul(probabilities, value), rows


class AttentionControl:
    """Bounded exposure/cotangent observer with an explicit attention intervention."""

    def __init__(self, native, mode):
        self.original = native._attention
        self.softmax = native.softmax_no_cast
        self.mode = mode
        self.phase = "forward"
        self.capture_phase = None
        self.statistics = {"mode": mode, "calls": 0, "calls_with_wholly_masked_rows": 0,
                           "wholly_masked_rows_over_calls": 0, "all_rows_over_calls": 0,
                           "masked_output_cotangents": {},
                           "scope": "Counts include rematerialization and repeated backwards; not unique biological rows.",
                           "row_capture_budget": "First four wholly masked rows per attention call until one has nonzero cotangent in the failed objective's backward; aggregate cotangent norms cover all wholly masked rows."}
        self.capture = None

    def __call__(self, query, key, value, biases):
        import torch
        if len(biases) != 2:
            raise ValueError("Pinned native triangle attention requires two biases")
        rows = whole_mask_rows(biases[0], query, key)
        if self.mode == "native":
            output = self.original(query, key, value, biases)
        else:
            output, _ = controlled_attention(query, key, value, biases, mode=self.mode, softmax=self.softmax)
        count = int(rows.sum())
        self.statistics["calls"] += 1
        self.statistics["all_rows_over_calls"] += rows.numel()
        self.statistics["wholly_masked_rows_over_calls"] += count
        self.statistics["calls_with_wholly_masked_rows"] += int(count > 0)
        if count and output.requires_grad:
            candidates = []
            if self.capture is None and self.mode == "native":
                # Keep bounded detached CPU slices, never q/k/v/output tensors
                # or their graphs in a hook closure. Retaining full activations
                # here would defeat native activation rematerialization.
                full = (*query.shape[:-1], key.shape[-2])
                for coords in rows[..., 0].nonzero()[:4].tolist():
                    coords = tuple(coords)
                    prefix, qindex = coords[:-1], coords[-1]
                    candidates.append({
                        "source_query_shape": list(query.shape), "coordinates": list(coords),
                        "query": query.detach()[prefix + (slice(qindex, qindex + 1),)].cpu().clone(),
                        "key": key.detach()[prefix].cpu().clone(), "value": value.detach()[prefix].cpu().clone(),
                        "mask_bias": torch.broadcast_to(biases[0].detach(), full)[coords].cpu().clone().unsqueeze(0),
                        "triangle_bias": torch.broadcast_to(biases[1].detach(), full)[coords].cpu().clone().unsqueeze(0),
                        "output": output.detach()[coords].cpu().clone().unsqueeze(0)})
            # Save at most one actual row receiving nonzero objective cotangent.
            # The hook follows each named production backward, so disconnected
            # padding is distinguished from rows with genuine downstream use.
            def observe(gradient):
                selected = gradient.detach().masked_fill(~rows, 0)
                norm = float(selected.double().norm())
                item = self.statistics["masked_output_cotangents"].setdefault(self.phase,
                    {"backward_hook_calls": 0, "nonzero_calls": 0, "maximum_norm": 0.0})
                item["backward_hook_calls"] += 1
                item["nonzero_calls"] += int(norm > 0)
                item["maximum_norm"] = max(item["maximum_norm"], norm)
                if self.capture is None and norm > 0 and self.phase == self.capture_phase:
                    for candidate in candidates:
                        upstream = gradient.detach()[tuple(candidate["coordinates"])]
                        if bool(upstream.any()):
                            self.capture = {**candidate, "phase": self.phase,
                                            "upstream": upstream.cpu().clone().unsqueeze(0)}
                            break
            output.register_hook(observe)
        return output


@contextmanager
def attention_control(mode, counters=None):
    """Restore native callable/cost observer after ordinary or exceptional exit."""
    import boltz.model.layers.triangular_attention.primitives as native
    helper = _helper()
    native_path = Path(inspect.getfile(native._attention))
    if helper._sha(native_path) != NATIVE_ATTENTION_SHA256:
        raise ValueError("Native attention source differs from the authenticated pinned bytes")
    observer = AttentionControl(native, mode)
    old_call = counters.call if counters is not None else None
    if counters is not None:
        def phased(model, purpose, action, function, device):
            old_phase = observer.phase
            observer.phase = f"{model}:{purpose}:{action}"
            try:
                return old_call(model, purpose, action, function, device)
            finally:
                observer.phase = old_phase
        counters.call = phased
    native._attention = observer
    try:
        yield observer
    finally:
        native._attention = observer.original
        if counters is not None:
            counters.call = old_call


def isolated_row_report(capture, *, device):
    """Test the actual captured row with its fixed, nonzero native cotangent."""
    import torch
    if capture is None:
        return {"captured": False, "reason": "No nonzero failed-objective cotangent among the bounded captured row candidates; full-row aggregate exposure remains separately reported."}
    def softmax(x, dim):
        return x.softmax(dim)
    # A singleton leading head axis gives the same general broadcast contract.
    bases = [capture[key].to(device).unsqueeze(0) for key in ("query", "key", "triangle_bias")]
    value, mask, cotangent = (capture[key].to(device).unsqueeze(0) for key in ("value", "mask_bias", "upstream"))
    result = {"captured": True, "phase": capture["phase"], "source_query_shape": capture["source_query_shape"],
              "coordinates": capture["coordinates"], "variants": {},
              "device": str(device), "dtype": str(bases[0].dtype),
              "scope": "Isolated actual inputs on original device with fixed objective cotangent; reduced matrix shape can select another kernel, so captured-output equality is checked; separate from end-to-end attribution."}
    # The same native-gradient direction is shared by all variants, including
    # a possible zero stop-logit derivative; no direction is chosen after FD.
    variables = [x.clone().requires_grad_() for x in bases]
    out, _ = controlled_attention(variables[0], variables[1], value, [mask, variables[2]], mode="native", softmax=softmax)
    result["captured_native_output_max_absolute_difference"] = float((out.detach() - capture["output"].to(device).unsqueeze(0)).abs().max())
    result["captured_native_output_exact"] = result["captured_native_output_max_absolute_difference"] == 0
    grads = torch.autograd.grad((out * cotangent).sum(), variables)
    norm = sum(float(x.double().square().sum()) for x in grads) ** .5
    if norm <= 1e-20:
        result.update(native_logit_gradient_norm=norm, status="zero_native_logit_sensitivity")
        return result
    directions = [g / norm for g in grads]
    for mode in MODES:
        variables = [x.clone().requires_grad_() for x in bases]
        out, _ = controlled_attention(variables[0], variables[1], value, [mask, variables[2]], mode=mode, softmax=softmax)
        loss = (out * cotangent).sum()
        if loss.requires_grad:
            gs = torch.autograd.grad(loss, variables, allow_unused=True)
            gs = [torch.zeros_like(x) if g is None else g for x, g in zip(variables, gs)]
        else:
            gs = [torch.zeros_like(x) for x in variables]
        analytic = sum(float((g.double() * d.double()).sum()) for g, d in zip(gs, directions))
        rows = []
        for epsilon in (1e-4, 0.001, 0.01, 0.1):
            values = []
            for sign in (1, -1):
                xs = [x + sign * epsilon * d for x, d in zip(bases, directions)]
                with torch.no_grad():
                    y, _ = controlled_attention(xs[0], xs[1], value, [mask, xs[2]], mode=mode, softmax=softmax)
                    values.append(float((y * cotangent).sum()))
            rows.append({"epsilon": epsilon, "analytic": analytic, "central_difference": (values[0] - values[1]) / (2 * epsilon),
                         "f_plus": values[0], "f_minus": values[1]})
        result["variants"][mode] = {"baseline": float(loss.detach()), "analytic": analytic, "rows": rows}
    return result


def decoder_readout_probe(decoder, point, branch, config, counters, directions, *, output_name, expected_gradient):
    """Freeze native graph cotangents to separate decoder from scorer curvature.

    Only one native forward/backward obtains this local linear readout. All
    endpoint calls below are decoder-only. The readout is a diagnostic tangent,
    not a learned reward surrogate or a replacement optimization objective.
    """
    import torch
    from latentmol.graph_policy import graph_from_decoder
    from latentmol.multimodel.study import tensor_digest
    probe = point.detach().clone().requires_grad_()
    graph = graph_from_decoder(decoder.decode(probe), decoder.spec, temperature=config.temperature,
                               fixed_active=branch.prepared.active)
    tensors = tuple(getattr(graph, key) for key in ("atoms", "charges", "bonds"))
    outputs = counters.call("boltz", "decoder_readout", "forward", lambda: branch(graph), point.device)
    value = outputs[output_name]
    cotangents = counters.call("boltz", "decoder_readout", "backward", lambda:
        torch.autograd.grad(value, tensors, retain_graph=True), point.device)
    gradient, = torch.autograd.grad(tensors, probe, grad_outputs=cotangents)
    cotangents = tuple(x.detach() for x in cotangents)
    base = tuple(x.detach() for x in tensors)
    result = {"native_output": output_name, "gradient_norm": float(gradient.double().norm()),
              "gradient_vs_original_native_component_max_absolute_difference": float((gradient.double() - expected_gradient.double()).abs().max()),
              "cotangent_sha256": tensor_digest(dict(zip(("atoms", "charges", "bonds"), cotangents))), "rows": [],
              "claim_limit": "Fixed native graph cotangent readout isolates decoder/graph policy; binding-only, excludes fixed-target NLL and its logits path; not a new scorer."}
    for label, (direction, _) in directions.items():
        analytic = float((gradient.double() * direction.double()).sum())
        for epsilon in EPSILONS:
            values = []
            for sign in (1, -1):
                with torch.no_grad():
                    endpoint = graph_from_decoder(decoder.decode(point + sign * epsilon * direction), decoder.spec,
                        temperature=config.temperature, fixed_active=branch.prepared.active)
                    # Subtract baseline channel tensors before widened reduction;
                    # float64 arithmetic cannot restore their rounded FP32 values.
                    values.append(sum(float(((getattr(endpoint, key).double() - original.double()) * cot.double()).sum())
                        for key, original, cot in zip(("atoms", "charges", "bonds"), base, cotangents)))
            numerical = (values[0] - values[1]) / (2 * epsilon)
            result["rows"].append({"direction": label, "epsilon": epsilon, "analytic": analytic,
                "central_difference": numerical, "absolute_error": abs(analytic - numerical),
                "passed_original_tolerance": abs(analytic - numerical) <= config.diagnostics.fd_atol +
                    config.diagnostics.fd_rtol * max(abs(analytic), abs(numerical))})
    return result


def causal_probe(decoder, point, branch, config, counters, target, weight, item_name, item, *, persist=lambda: None, report=None):
    """Hold original directions fixed across all interventions and all endpoints."""
    import torch
    from latentmol.boltz.active_set import score_at
    from latentmol.boltz.local_fd import endpoint_row
    from latentmol.multimodel.study import tensor_digest
    helper = _helper()
    report = report if report is not None else {}
    report.update(status="probing", claim_limit=CLAIM_LIMIT, causal_module_sha256=helper._sha(__file__),
                  diagnostic_epsilons=list(EPSILONS), native_attention_sha256=NATIVE_ATTENTION_SHA256,
                  controls={}, original_gate_not_reclassified=True)
    original_endpoints, directions, original_gradients = {}, None, None
    for mode in MODES:
        current = {"mode": mode, "status": "running", "forward_expected_unchanged": mode != "masked_logits_float64",
                   "gradient_mathematics_changed": mode != "native", "finite_differences": []}
        report["controls"][mode] = current
        persist()
        with attention_control(mode, counters) as observer:
            observer.capture_phase = (f"boltz:causal_{mode}:backward" if item_name == "combined" else
                                      f"boltz:causal_{mode}:{item_name}:backward")
            components = {}
            measured, gradient = score_at(decoder, point, branch, config, counters, f"causal_{mode}", gradients=True,
                component_gradients=components, repair_target=target, repair_weight=weight)
            terms = {**components, "combined": gradient}
            baseline = {**measured["components"], "combined": measured["loss"]}
            current.update(baseline=baseline, gradients={key: {"norm": float(value.double().norm()),
                "sha256": tensor_digest({"gradient": value})} for key, value in terms.items()})
            if mode == "native":
                relevant = gradient if item_name == "combined" else components[item_name]
                directions = helper.reconstruct_directions(point, relevant, item)
                report["direction_replay"] = {key: meta for key, (_, meta) in directions.items()}
                if any(meta["hash_matches_original"] is not True for key, (_, meta) in directions.items() if key != "gradient"):
                    raise ValueError("Original random direction identity did not reproduce")
                original_gradients = {key: value.clone() for key, value in terms.items()}
            else:
                current["gradient_change_from_native"] = {key: {
                    "norm": float((value.double() - original_gradients[key].double()).norm()),
                    "directional_difference": {name: float(((value.double() - original_gradients[key].double()) * d.double()).sum())
                                               for name, (d, _) in directions.items()}}
                    for key, value in terms.items()}
                current["baseline_exact_native_match"] = baseline == report["controls"]["native"]["baseline"]
                if mode == "stop_masked_logits" and not current["baseline_exact_native_match"]:
                    raise ValueError("Stop-logit diagnostic changed forward values at baseline")
            persist()
            for label, (direction, _) in directions.items():
                for epsilon in EPSILONS:
                    plus = helper.forward_values(decoder, point + epsilon * direction, branch, config, counters, target, weight, grad_enabled=False)
                    minus = helper.forward_values(decoder, point - epsilon * direction, branch, config, counters, target, weight, grad_enabled=False)
                    key = (label, epsilon)
                    if mode == "native":
                        original_endpoints[key] = (deepcopy(plus), deepcopy(minus))
                    old_plus, old_minus = original_endpoints[key]
                    exact = (plus["values"] == old_plus["values"] and minus["values"] == old_minus["values"] and
                             plus["graph_sha256"] == old_plus["graph_sha256"] and minus["graph_sha256"] == old_minus["graph_sha256"])
                    if mode == "stop_masked_logits" and not exact:
                        raise ValueError("Stop-logit diagnostic changed forward values at a fixed endpoint")
                    pairs = {key: endpoint_row(point, direction, derivative, epsilon,
                        {"loss": plus["values"][key], "loss_dtype": plus["dtypes"][key]},
                        {"loss": minus["values"][key], "loss_dtype": minus["dtypes"][key]}, baseline[key],
                        rtol=config.diagnostics.fd_rtol, atol=config.diagnostics.fd_atol) for key, derivative in terms.items()}
                    current["finite_differences"].append({"direction": label, "epsilon": epsilon,
                        "forward_exact_native_match": exact, "terms": pairs, "plus": plus, "minus": minus})
                    persist()
            current["attention_exposure"] = deepcopy(observer.statistics)
            if mode == "native":
                current["isolated_actual_row"] = isolated_row_report(observer.capture, device=point.device)
            current["status"] = "completed"
        persist()
    # Run after the attention context has restored native functions. This extra
    # test attributes decoder vs scorer behavior without touching old math.
    expected = original_gradients[item_name] if item_name != "combined" else original_gradients["combined"]
    if item_name == "combined":
        from latentmol.repair_guidance.loss import target_score_at
        _, repair_gradient = target_score_at(decoder, point, branch.prepared.active, target,
            temperature=config.temperature, normalization=config.repair_guidance.normalization, gradients=True)
        # Remove the separately differentiated repair term, then undo the
        # declared scalar objective sign/scale to match raw dbmol_total_loss.
        sign = -1 if config.objective.direction == "maximize" else 1
        expected = (expected - weight * repair_gradient) * config.objective.scale / sign
    report["decoder_fixed_graph_readout"] = decoder_readout_probe(decoder, point, branch, config, counters, directions,
        output_name="dbmol_contact_loss" if item_name != "combined" else "dbmol_total_loss", expected_gradient=expected)
    report.update(status="completed", diagnostic_completed=True, old_gate_certified_by_this_diagnostic=False,
                  scored_directions=list(directions), no_new_production_gradient_policy=True)
    persist()
    return report


def run_case(plan, case, output_root):
    helper = _helper()
    helper.probe_case = causal_probe
    helper.REPEATS = 1
    helper.CASE_TIMEOUT_SECONDS = CASE_TIMEOUT_SECONDS
    helper.CLAIM_LIMIT = CLAIM_LIMIT
    # The existing loader authenticates all source079 bytes before importing the
    # old package and checks frozen weights/context/parent hashes again on exit.
    return helper.run_case(plan, case, output_root)


def run_study(plan_path, expected):
    """Run bounded fresh children; completion does not reclassify old failures."""
    helper = _helper()
    plan = helper.load_plan(plan_path, expected)
    from latentmol.experiments import ExperimentRun, write_json
    from latentmol.boltz.workflow_process import run_logged
    maximum_calls = 0
    for case in plan["cases"]:
        gate = helper._read(helper.authenticated_artifact(case, Path(case["parent_run"]) / case["arm"] / f"gate-{case['step']:04d}.json"))
        _, item = helper.failed_item(gate)
        failed_random = sum(name != "gradient" and value.get("passed") is False for name, value in item["direction_results"].items())
        maximum_calls += 3 + len(MODES) * (1 + (1 + failed_random) * len(EPSILONS) * 2)
    if maximum_calls > 450:
        raise ValueError("Causal probe plan exceeds 450 native scorer forwards")
    config = {"plan": plan, "plan_sha256": expected, "instrumentation_sha256": helper._sha(__file__),
              "loader_sha256": helper._sha(Path(__file__).with_name("failed_gate_attribution.py")),
              "native_attention_sha256": NATIVE_ATTENTION_SHA256, "modes": list(MODES),
              "epsilons": list(EPSILONS), "maximum_scorer_forwards": maximum_calls,
              "maximum_child_seconds": CASE_TIMEOUT_SECONDS, "claim_limit": CLAIM_LIMIT}
    with ExperimentRun(Path(plan["output_root"]), plan["name"], config, Path.cwd()) as experiment:
        rows = []
        for case in plan["cases"]:
            execution = run_logged([sys.executable, str(Path(__file__).resolve()), "--plan", str(Path(plan_path).resolve()),
                "--sha256", expected, "--child", case["id"], "--child-output", str(experiment.path / case["id"])],
                experiment.path / f"{case['id']}.log", timeout_seconds=CASE_TIMEOUT_SECONDS)
            completed = False
            if not execution["timed_out"] and execution["returncode"] == 0 and len(execution["run_paths"]) == 1:
                child = Path(execution["run_paths"][0])
                evidence = helper._read(child / "attribution.json")
                completed = (helper._read(child / "run.json")["status"] == "completed" and evidence.get("status") == "completed"
                    and evidence.get("frozen_state_unchanged") is True and evidence.get("parent_unchanged") is True
                    and all(evidence.get("controls", {}).get(mode, {}).get("status") == "completed" for mode in MODES))
            rows.append({"case_id": case["id"], "diagnostic_completed": completed, "execution": execution,
                         "old_gate_certified": False})
            write_json(experiment.path / "cases.json", rows)
        experiment.annotate(diagnostic_cases_completed=sum(row["diagnostic_completed"] for row in rows),
                            diagnostic_cases_requested=len(rows), old_gates_certified=False)
        if not all(row["diagnostic_completed"] for row in rows):
            raise ValueError("One or more causal diagnostic children failed; retained in cases.json")
        return experiment.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--child")
    parser.add_argument("--child-output", type=Path)
    args = parser.parse_args()
    helper = _helper()
    if not os.environ.get("LSB_JOBID") or not helper._hash(os.environ.get("LATENTMOL_SOURCE_SHA256")):
        raise ValueError("Causal attention probes require an immutable DTU HPC allocation")
    plan = helper.load_plan(args.plan, args.sha256)
    if args.child is None:
        if args.child_output is not None:
            raise ValueError("Child output requires a child case")
        print(run_study(args.plan, args.sha256), flush=True)
    else:
        cases = [case for case in plan["cases"] if case["id"] == args.child]
        if len(cases) != 1 or args.child_output is None:
            raise ValueError("Unknown authenticated saved case or missing output root")
        print(run_case(plan, cases[0], args.child_output), flush=True)


if __name__ == "__main__":
    main()
