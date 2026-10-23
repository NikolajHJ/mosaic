"""HPC-only decoder attribution after source081's frozen-cotangent replay.

One authentic native contact forward/backward supplies a fixed graph cotangent.
All subsequent calls use the frozen decoder alone. Prespecified controls are
original FP32, FP32 stable internal occupancy, a FP64 decoder copy, and a FP32
absolute-value branch extension. These are declared changes of diagnostic
mathematics, not production fixes, learned surrogates or historical FD approval.
The same historical prior-3 point, fixed mask and directions are used throughout.
Engineering fixtures and actual-state evidence must run on DTU HPC; no local
numerical validation is claimed.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import math
import os
from pathlib import Path
import sys


MODES = ("original_fp32", "stable_occupancy_fp32", "original_fp64", "frozen_abs_fp32")
EPSILONS = (3e-5, 1e-4, 3e-4, 0.001, 0.003, 0.01)
CASE_ID, CASE_STEP = "prior-3", 50
CASE_TIMEOUT_SECONDS = 1800
MAXIMUM_SCORER_FORWARDS = 3  # Authenticated loader's two parity calls plus one cotangent forward.
MAXIMUM_DECODER_FORWARDS = 126  # Includes FP64 endpoints at original realized FP32 coordinates.
MAXIMUM_SIGN_ELEMENTS = 4 * 1024 * 1024
CLAIM_LIMIT = (
    "Fixed-native-cotangent decoder diagnostic only; no optimization, new DeFoG calls, "
    "tolerance changes, production patch or historical gate approval. FP64 and stable "
    "occupancy change forward arithmetic; frozen absolute signs change endpoint mathematics."
)


def _helper():
    path = Path(__file__).with_name("failed_gate_attribution.py")
    spec = importlib.util.spec_from_file_location("_authenticated_decoder_gate_loader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selected_case(plan):
    cases = [case for case in plan["cases"] if case["id"] == CASE_ID]
    if len(cases) != 1 or cases[0]["step"] != CASE_STEP or cases[0]["arm"] != "fixed_initial_mask":
        raise ValueError("Decoder probe requires the prespecified prior-3 fixed-mask step-50 case")
    return cases[0]


def reference_from_plan(plan, case):
    """Authenticate completed source081 proof before accepting its gradient hash."""
    helper = _helper()
    paths = [Path(path) for path in case["parent_artifacts_sha256"]
             if "source-081-attention-causal-attempt001-replay" in path and Path(path).name == "attribution.json"]
    if len(paths) != 1:
        raise ValueError("Require exactly one pinned source081 prior-3 attribution proof")
    attribution_path = helper.authenticated_artifact(case, paths[0])
    run_path = helper.authenticated_artifact(case, paths[0].with_name("run.json"))
    reference, record = helper._read(attribution_path), helper._read(run_path)
    before = reference.get("identities_before")
    production = record.get("config", {})
    old_case = production.get("case", {})
    source = record.get("source", {})
    if (record.get("status") != "completed" or reference.get("status") != "completed"
            or reference.get("diagnostic_completed") is not True or reference.get("native_seed_parity") is not True
            or reference.get("frozen_state_unchanged") is not True or reference.get("parent_unchanged") is not True
            or not isinstance(before, dict) or set(before) != {"point_sha256", "context_sha256", "decoder_state_sha256", "scorer_state_sha256"}
            or any(not helper._hash(value) for value in before.values()) or before != reference.get("identities_after")
            or old_case.get("id") != CASE_ID or old_case.get("step") != CASE_STEP
            or old_case.get("parent_run") != case["parent_run"]
            or production.get("production_release") != plan["production_release"]
            or production.get("production_manifest_sha256") != plan["production_manifest_sha256"]
            or production.get("production_archive_sha256") != plan["production_archive_sha256"]
            or not source.get("files") or helper._digest(source["files"]) != source.get("sha256")
            or not helper._hash(record.get("source_archive_sha256"))):
        raise ValueError("Source081 proof is incomplete or has different scientific/source identities")
    if any(case["parent_artifacts_sha256"].get(path) != expected
           for path, expected in old_case.get("parent_artifacts_sha256", {}).items()):
        raise ValueError("Source081 proof parent artifacts differ from the current pinned plan")
    native = reference.get("controls", {}).get("native", {})
    gradient = native.get("gradients", {}).get("dbmol_contact_loss", {})
    norm = gradient.get("norm")
    if (native.get("status") != "completed" or not helper._hash(gradient.get("sha256"))
            or type(norm) not in (int, float) or not math.isfinite(norm) or norm <= 0):
        raise ValueError("Source081 native contact gradient identity is unavailable")
    return {"identities": before, "target_sha256": reference.get("target", {}).get("target_sha256"),
            "native_contact_gradient": gradient, "direction_replay": reference.get("direction_replay", {}),
            "evidence": {"attribution_path": str(attribution_path), "run_path": str(run_path),
                         "attribution_sha256": helper._sha(attribution_path), "run_sha256": helper._sha(run_path),
                         "source_manifest_sha256": source["sha256"], "source_archive_sha256": record["source_archive_sha256"]}}


class DecoderObserver:
    """Observe occupancy cancellation and freeze only declared abs branches.

    Detached int8 signs/boolean cotangent support are bounded. Floating decoder
    activations are never retained across evaluations. Counts are per helper
    invocation, not unique atoms; frozen-abs endpoint signs are evaluated on
    that counterfactual decoder's intermediate features.
    """

    def __init__(self, *, maximum_sign_elements=MAXIMUM_SIGN_ELEMENTS):
        if type(maximum_sign_elements) is not int or maximum_sign_elements < 1:
            raise ValueError("Require a positive sign capture budget")
        self.maximum_sign_elements = maximum_sign_elements
        self.mode = "original_fp32"
        self.baselines, self.sign_elements = [], 0
        self.active, self.capture, self.backward = False, False, False
        self.abs_calls, self.occupancy_calls = [], []
        self.current_atoms = None
        self.expected_rounds = None

    def begin(self, mode, *, capture=False):
        if mode not in MODES or self.active or (capture and self.baselines) or (not capture and not self.baselines):
            raise ValueError("Unsupported decoder observer mode/capture sequence")
        self.mode, self.capture, self.active = mode, capture, True
        self.abs_calls, self.occupancy_calls, self.current_atoms = [], [], None

    def absolute_features(self, original, h):
        import torch
        if not self.active:
            raise ValueError("Absolute helper called outside the scoped decoder evaluation")
        index = len(self.abs_calls)
        # Bookkeeping must not create an earlier differentiable subtraction or
        # views that are reused by the counterfactual graph below.
        observed = h.detach()
        left, right = observed[:, :, None, :], observed[:, None, :, :]
        difference = left - right
        signs = difference.detach().sign().to(torch.int8)
        if self.capture:
            self.sign_elements += signs.numel()
            if self.sign_elements > self.maximum_sign_elements:
                raise ValueError("Decoder absolute signs exceed the declared bounded memory budget")
            entry = {"shape": list(signs.shape), "signs": signs.clone(), "cotangent_support": None}
            self.baselines.append(entry)
        else:
            if index >= len(self.baselines) or list(signs.shape) != self.baselines[index]["shape"]:
                raise ValueError("Decoder absolute helper invocation/shape alignment changed")
            entry = self.baselines[index]
        crossed = signs != entry["signs"]
        support = entry["cotangent_support"]
        off_diagonal = ~torch.eye(h.shape[1], device=h.device, dtype=torch.bool)
        self.abs_calls.append({"invocation": index, "shape": list(signs.shape),
            "sign_crossings": int(crossed.sum()), "zeros": int((signs == 0).sum()),
            "minimum_absolute_difference": float(difference.detach().abs().min()),
            "off_diagonal_zeros": int(((signs == 0) & off_diagonal[None, :, :, None]).sum()),
            "minimum_off_diagonal_absolute_difference": float(difference.detach()[:, off_diagonal].abs().min()) if h.shape[1] > 1 else None,
            "baseline_cotangent_support_observed": support is not None,
            "crossings_with_nonzero_baseline_output_cotangent": int((crossed & support).sum()) if support is not None else 0})
        if self.mode == "frozen_abs_fp32":
            # Preserve native branch construction order: sum, fresh difference
            # and its absolute branch, product, then cat. Reusing the earlier
            # observation difference reordered autograd's accumulation into h
            # and failed the strict FP32 baseline-gradient engineering guard.
            live_left, live_right = h[:, :, None, :], h[:, None, :, :]
            summed = live_left + live_right
            signed_difference = live_left - live_right
            absolute_extension = signed_difference * entry["signs"].to(signed_difference)
            product = live_left * live_right
            output = torch.cat((summed, absolute_extension, product), -1)
        else:
            output = original(h)
        if self.capture and output.requires_grad:
            width = h.shape[-1]
            def observe(gradient, entry=entry, width=width):
                if not self.backward:
                    return
                if entry["cotangent_support"] is not None:
                    raise ValueError("Absolute-helper cotangent observed more than once")
                entry["cotangent_support"] = (gradient.detach()[..., width:2 * width] != 0).clone()
            output.register_hook(observe)
        return output

    def predictions(self, original, h):
        if not self.active:
            raise ValueError("Node prediction called outside a decoder evaluation")
        result = original(h)
        self.current_atoms = result[0]
        return result

    def refinement(self, index, original, h, edge, node_context, occupancy):
        import torch
        if not self.active or self.current_atoms is None or index != len(self.occupancy_calls):
            raise ValueError("Decoder refinement/atom-logit invocation alignment changed")
        probabilities = self.current_atoms.softmax(-1)
        subtraction = 1.0 - probabilities[..., 0]
        stable = probabilities[..., 1:].sum(-1)
        if not torch.equal(occupancy, subtraction):
            raise ValueError("Native decoder no longer uses the authenticated subtraction occupancy")
        effective = stable if self.mode == "stable_occupancy_fp32" else occupancy
        self.occupancy_calls.append({"round": index, "dtype": str(occupancy.dtype),
            "no_atom_probability": probabilities[..., 0].detach().cpu().tolist(),
            "subtraction_occupancy": subtraction.detach().cpu().tolist(),
            "stable_sum_occupancy": stable.detach().cpu().tolist(),
            "effective_occupancy": effective.detach().cpu().tolist(),
            "maximum_absolute_difference": float((subtraction.detach() - stable.detach()).abs().max()),
            "subtraction_clamped_count": int((subtraction.detach() < 1e-6).sum()),
            "stable_clamped_count": int((stable.detach() < 1e-6).sum()),
            "clamp_branch_disagreements": int(((subtraction.detach() < 1e-6) != (stable.detach() < 1e-6)).sum()),
            "subtraction_at_clamp_boundary": int((subtraction.detach() == 1e-6).sum()),
            "stable_at_clamp_boundary": int((stable.detach() == 1e-6).sum())})
        return original(h, edge, node_context, effective)

    def finish(self):
        if not self.active or not self.abs_calls or len(self.abs_calls) != len(self.baselines):
            raise ValueError("Decoder absolute helper call count differs from the saved point")
        if self.expected_rounds is not None and len(self.occupancy_calls) != self.expected_rounds:
            raise ValueError("Decoder refinement round count differs from the authenticated decoder")
        for call, entry in zip(self.abs_calls, self.baselines):
            # Initial capture hooks run after forward bookkeeping, during the
            # VJP. Report their completed support at finish rather than the
            # earlier, not-yet-observed state.
            support = entry["cotangent_support"]
            call["baseline_cotangent_support_observed"] = support is not None
            call["nonzero_baseline_output_cotangent_elements"] = int(support.sum()) if support is not None else None
        row = {"mode": self.mode, "absolute_calls": deepcopy(self.abs_calls),
               "occupancy_rounds": deepcopy(self.occupancy_calls),
               "sign_crossings": sum(call["sign_crossings"] for call in self.abs_calls),
               "crossings_with_nonzero_baseline_output_cotangent": sum(
                   call["crossings_with_nonzero_baseline_output_cotangent"] for call in self.abs_calls)}
        self.abort()
        return row

    def abort(self):
        self.active, self.capture, self.backward = False, False, False
        self.current_atoms = None


@contextmanager
def decoder_control(model, observer):
    """Instrument this decoder copy; restore helper/methods even after failure.

    The abs helper is a production-module global shared by decoder classes.
    Its replacement exists only in this synchronous decoder-only scope. No
    native scorer call or original-model evaluation occurs while it is patched.
    """
    decoder = model.decoder
    namespace = decoder.forward.__func__.__globals__
    original_features = namespace.get("symmetric_pair_features")
    if original_features is None or not callable(original_features) or not hasattr(decoder, "rounds") or not decoder.rounds:
        raise ValueError("Require the authenticated refined decoder with pair rounds")
    saved = []
    old_expected_rounds = observer.expected_rounds
    observer.expected_rounds = len(decoder.rounds)
    try:
        def features(h):
            return observer.absolute_features(original_features, h)
        namespace["symmetric_pair_features"] = features
        for owner, attribute, callback in [(decoder, "_node_predictions", observer.predictions)]:
            present, previous = attribute in owner.__dict__, owner.__dict__.get(attribute)
            original = getattr(owner, attribute)
            saved.append((owner, attribute, present, previous))
            def wrapped(h, original=original, callback=callback):
                return callback(original, h)
            setattr(owner, attribute, wrapped)
        for index, layer in enumerate(decoder.rounds):
            present, previous = "forward" in layer.__dict__, layer.__dict__.get("forward")
            original = layer.forward
            saved.append((layer, "forward", present, previous))
            def refinement(h, edge, node_context, occupancy, index=index, original=original):
                return observer.refinement(index, original, h, edge, node_context, occupancy)
            layer.forward = refinement
        yield observer
    finally:
        observer.abort()
        observer.expected_rounds = old_expected_rounds
        namespace["symmetric_pair_features"] = original_features
        for owner, attribute, present, previous in reversed(saved):
            if present:
                setattr(owner, attribute, previous)
            else:
                owner.__dict__.pop(attribute, None)


def fixed_readout(graph, base, cotangents):
    """Widen only the fixed linear reduction; decoder tensors retain their dtype."""
    return sum(((getattr(graph, key).double() - reference.double()) * cotangent.double()).sum()
               for key, reference, cotangent in zip(("atoms", "charges", "bonds"), base, cotangents))


def evaluate_decoder(model, point, active, temperature, base, cotangents, observer, mode,
                     *, gradients=False, capture=False):
    import torch
    from latentmol.graph_policy import graph_from_decoder
    from latentmol.multimodel.study import tensor_digest
    probe = point.detach().clone().requires_grad_(gradients)
    observer.begin(mode, capture=capture)
    try:
        with torch.set_grad_enabled(gradients):
            graph = graph_from_decoder(model.decode(probe), model.spec, temperature=temperature, fixed_active=active)
            value = fixed_readout(graph, base, cotangents)
            gradient = None
            if gradients:
                observer.backward = True
                # Match source081's graph-tensor VJP exactly. A fresh widened
                # scalar reduction can choose another FP32 accumulation order;
                # widening endpoint secants does not require changing this path.
                fields = tuple(getattr(graph, key) for key in ("atoms", "charges", "bonds"))
                gradient, = torch.autograd.grad(fields, probe,
                    grad_outputs=tuple(cotangent.to(field) for cotangent, field in zip(cotangents, fields)))
                observer.backward = False
                if not bool(gradient.isfinite().all()):
                    raise ValueError("Nonfinite fixed-cotangent decoder gradient")
        if not bool(value.isfinite().all()):
            raise ValueError("Nonfinite fixed-cotangent decoder value")
        metadata = {"value": float(value.detach()), "decoder_dtype": str(probe.dtype),
            "gradient_evaluation": "Source081 graph-tensor VJP with fixed cotangents cast to graph dtype; widened scalar reduction is for endpoint secants only.",
            "graph_sha256": tensor_digest({key: getattr(graph, key) for key in ("atoms", "charges", "bonds")}),
            "trace": observer.finish()}
        return metadata, gradient.detach() if gradient is not None else None
    except Exception:
        observer.abort()
        raise


def decoder_probe(decoder, point, branch, config, counters, target, weight, item_name, item,
                  *, reference=None, persist=lambda: None, report=None):
    import torch
    from latentmol.graph_policy import graph_from_decoder
    from latentmol.multimodel.study import module_digest, tensor_digest
    if item_name != "dbmol_contact_loss" or any(parameter.dtype != torch.float32 for parameter in decoder.parameters()):
        raise ValueError("Require the historical FP32 contact-component decoder")
    helper = _helper()
    report = report if report is not None else {}
    if reference is None or report.get("identities_before") != reference["identities"]:
        raise ValueError("Current scientific identities differ from completed source081 evidence")
    if target.identity != reference["target_sha256"]:
        raise ValueError("Repaired target differs from the source081 saved-state proof")
    report.update(status="probing", claim_limit=CLAIM_LIMIT, decoder_module_sha256=helper._sha(__file__),
                  diagnostic_epsilons=list(EPSILONS), controls={}, original_gate_not_reclassified=True)
    # Copy before installing any observer. Numerical FP32 parameter values are
    # represented exactly in FP64; the new dtype/forward math remains explicit.
    original_digest = module_digest(decoder)
    copy32 = deepcopy(decoder).eval().requires_grad_(False)
    copy64 = deepcopy(decoder).double().eval().requires_grad_(False)
    if module_digest(copy32) != original_digest or any(
            not torch.equal(copy64.state_dict()[name].to(value), value) for name, value in decoder.state_dict().items()):
        raise ValueError("Decoder copies do not preserve the original numerical state values")
    probe = point.detach().clone().requires_grad_()
    graph = graph_from_decoder(decoder.decode(probe), decoder.spec, temperature=config.temperature,
                               fixed_active=branch.prepared.active)
    tensors = tuple(getattr(graph, key) for key in ("atoms", "charges", "bonds"))
    outputs = counters.call("boltz", "decoder_causal_cotangent", "forward", lambda: branch(graph), point.device)
    contact = outputs[item_name]
    cotangents = counters.call("boltz", "decoder_causal_graph", "backward", lambda:
        torch.autograd.grad(contact, tensors, retain_graph=True), point.device)
    native_gradient, = counters.call("boltz", "decoder_causal_direct", "backward", lambda:
        torch.autograd.grad(contact, probe, retain_graph=True), point.device)
    if (tensor_digest({"gradient": native_gradient}) != reference["native_contact_gradient"]["sha256"]
            or float(native_gradient.double().norm()) != reference["native_contact_gradient"]["norm"]):
        raise ValueError("Captured native contact gradient differs from authenticated source081 gradient")
    chain_gradient, = torch.autograd.grad(tensors, probe, grad_outputs=cotangents)
    if not torch.equal(native_gradient, chain_gradient):
        raise ValueError("Native contact and fixed-graph-cotangent latent gradients do not match exactly")
    cotangents, base = tuple(value.detach() for value in cotangents), tuple(value.detach() for value in tensors)
    directions = helper.reconstruct_directions(point, native_gradient, item)
    if set(directions) != {"gradient", "random_1"} or directions["random_1"][1]["hash_matches_original"] is not True:
        raise ValueError("Historical prior-3 direction identities did not reproduce")
    if any(meta["replayed_sha256"] != reference["direction_replay"].get(name, {}).get("replayed_sha256")
           for name, (_, meta) in directions.items()):
        raise ValueError("Prespecified direction differs from completed source081 evidence")
    report.update(native_contact_value=float(contact.detach()), native_gradient_norm=float(native_gradient.double().norm()),
        native_gradient_sha256=tensor_digest({"gradient": native_gradient}),
        native_vs_decoder_chain_gradient_max_absolute_difference=0.0,
        direction_replay={name: meta for name, (_, meta) in directions.items()},
        cotangent_sha256=tensor_digest(dict(zip(("atoms", "charges", "bonds"), cotangents))),
        original_decoder_state_sha256=original_digest, fp64_decoder_state_sha256=module_digest(copy64),
        fp64_parameters_preserve_original_numerical_values=True,
        source081_reference=reference["evidence"], source081_native_gradient_exact_match=True,
        gradient_scope="Binding/contact-only fixed native graph cotangent; repair target is authenticated but its logits NLL is excluded.")
    # Release native float activations before cheap decoder-only counterfactuals.
    del outputs, contact, graph, tensors, probe, chain_gradient
    observer = DecoderObserver()
    for mode in MODES:
        model = copy64 if mode == "original_fp64" else copy32
        current = {"status": "running", "mode": mode, "rows": [],
            "forward_arithmetic_changed": mode in ("stable_occupancy_fp32", "original_fp64"),
            "endpoint_piecewise_extension": mode == "frozen_abs_fp32"}
        report["controls"][mode] = current
        with decoder_control(model, observer):
            origin = point.double() if mode == "original_fp64" else point
            baseline, gradient = evaluate_decoder(model, origin, branch.prepared.active, config.temperature,
                base, cotangents, observer, mode, gradients=True, capture=mode == "original_fp32")
            current.update(baseline=baseline, gradient_sha256=tensor_digest({"gradient": gradient}),
                gradient_norm=float(gradient.double().norm()),
                gradient_change_from_original_norm=float((gradient.double() - native_gradient.double()).norm()))
            if mode in ("original_fp32", "frozen_abs_fp32"):
                exact = baseline["graph_sha256"] == tensor_digest(dict(zip(("atoms", "charges", "bonds"), base)))
                exact = exact and baseline["value"] == 0 and torch.equal(gradient, native_gradient)
                current["baseline_graph_readout_and_gradient_exact_native_match"] = exact
                if not exact:
                    raise ValueError("Original/frozen-abs decoder baseline differs from original native gradient/graph")
            persist()
            for label, (original_direction, _) in directions.items():
                direction = original_direction.double() if mode == "original_fp64" else original_direction
                analytic = float((gradient.double() * direction.double()).sum())
                for epsilon in EPSILONS:
                    endpoints = [evaluate_decoder(model, origin + sign * epsilon * direction,
                        branch.prepared.active, config.temperature, base, cotangents, observer, mode)[0]
                        for sign in (1, -1)]
                    numerical = (endpoints[0]["value"] - endpoints[1]["value"]) / (2 * epsilon)
                    current["rows"].append({"direction": label, "epsilon": epsilon, "analytic": analytic,
                        "central_difference": numerical, "absolute_error": abs(analytic - numerical),
                        "passed_original_tolerance": abs(analytic - numerical) <= config.diagnostics.fd_atol +
                            config.diagnostics.fd_rtol * max(abs(analytic), abs(numerical)),
                        "plus": endpoints[0], "minus": endpoints[1]})
                    if mode == "original_fp64":
                        rounded = [evaluate_decoder(model, (point + sign * epsilon * original_direction).double(),
                            branch.prepared.active, config.temperature, base, cotangents, observer, mode)[0]
                            for sign in (1, -1)]
                        current["rows"][-1]["original_realized_fp32_latent_endpoints"] = {
                            "central_difference": (rounded[0]["value"] - rounded[1]["value"]) / (2 * epsilon),
                            "plus": rounded[0], "minus": rounded[1],
                            "scope": "Same FP64 decoder, original rounded FP32 latent endpoints; distinguishes latent from internal arithmetic."}
                    persist()
            current["status"] = "completed"
        persist()
    if module_digest(decoder) != original_digest:
        raise ValueError("Decoder diagnostic mutated original frozen state")
    report.update(status="completed", diagnostic_completed=True, all_decoder_methods_restored=True,
                  old_gate_certified_by_this_diagnostic=False, maximum_scorer_forwards=MAXIMUM_SCORER_FORWARDS,
                  maximum_decoder_forwards=MAXIMUM_DECODER_FORWARDS, no_new_production_gradient_policy=True)
    persist()
    return report


def run_case(plan, case, output_root):
    if case != selected_case(plan):
        raise ValueError("Only the prespecified prior-3 decoder case is supported")
    helper = _helper()
    reference = reference_from_plan(plan, case)
    def probe_with_reference(*args, **kwargs):
        return decoder_probe(*args, **kwargs, reference=reference)
    helper.probe_case, helper.REPEATS = probe_with_reference, 1
    helper.CASE_TIMEOUT_SECONDS, helper.CLAIM_LIMIT = CASE_TIMEOUT_SECONDS, CLAIM_LIMIT
    return helper.run_case(plan, case, output_root)


def run_study(plan_path, expected):
    helper = _helper()
    plan = helper.load_plan(plan_path, expected)
    case = selected_case(plan)
    from latentmol.experiments import ExperimentRun, write_json
    from latentmol.boltz.workflow_process import run_logged
    inputs = {"plan": plan, "plan_sha256": expected, "selected_case": CASE_ID,
        "instrumentation_sha256": helper._sha(__file__), "modes": list(MODES), "epsilons": list(EPSILONS),
        "maximum_scorer_forwards": MAXIMUM_SCORER_FORWARDS, "maximum_decoder_forwards": MAXIMUM_DECODER_FORWARDS,
        "maximum_child_seconds": CASE_TIMEOUT_SECONDS, "claim_limit": CLAIM_LIMIT}
    with ExperimentRun(Path(plan["output_root"]), plan["name"], inputs, Path.cwd()) as experiment:
        execution = run_logged([sys.executable, str(Path(__file__).resolve()), "--plan", str(Path(plan_path).resolve()),
            "--sha256", expected, "--child", CASE_ID, "--child-output", str(experiment.path / CASE_ID)],
            experiment.path / f"{CASE_ID}.log", timeout_seconds=CASE_TIMEOUT_SECONDS)
        completed = False
        if not execution["timed_out"] and execution["returncode"] == 0 and len(execution["run_paths"]) == 1:
            child = Path(execution["run_paths"][0])
            evidence = helper._read(child / "attribution.json")
            completed = (helper._read(child / "run.json")["status"] == "completed" and evidence.get("status") == "completed"
                and evidence.get("diagnostic_completed") is True and evidence.get("frozen_state_unchanged") is True
                and evidence.get("parent_unchanged") is True and evidence.get("all_decoder_methods_restored") is True
                and all(evidence.get("controls", {}).get(mode, {}).get("status") == "completed" for mode in MODES))
        write_json(experiment.path / "cases.json", [{"case_id": case["id"], "diagnostic_completed": completed,
            "execution": execution, "old_gate_certified": False}])
        experiment.annotate(diagnostic_cases_completed=int(completed), diagnostic_cases_requested=1, old_gates_certified=False)
        if not completed:
            raise ValueError("Decoder causal diagnostic child failed; retained in cases.json")
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
        raise ValueError("Decoder causal probes require an immutable DTU HPC allocation")
    plan = helper.load_plan(args.plan, args.sha256)
    if args.child is None:
        if args.child_output is not None:
            raise ValueError("Child output requires a child case")
        print(run_study(args.plan, args.sha256), flush=True)
    else:
        if args.child != CASE_ID or args.child_output is None:
            raise ValueError("Unknown authenticated saved case or missing output root")
        print(run_case(plan, selected_case(plan), args.child_output), flush=True)


if __name__ == "__main__":
    main()
