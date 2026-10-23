"""HPC-only regression-specific canonical-graph/decoder attribution.

This conditional follow-up isolates the native continuous-affinity contribution,
not the full binding gradient previously examined in source081. Its fixed graph
cotangents come from the unchanged source079 scorer and exact saved prior-5 point.
Held decoder absolute-value signs change diagnostic endpoints only. Neither a
linear readout nor its branch extension can certify an original production gate.

The simultaneous graph cut may change FP32 reverse-accumulation order. Exactness
is recorded; the existing 64-epsilon maximum/L2 engineering bound must hold.
Decoder baselines must match that measured graph-chain VJP exactly. Both native
and chain gradients retain separate FD rows at the original tolerances. Failed
rows support localization only when both fail informatively and their projected
chain discrepancy is below one percent of BOTH FD residuals. Other disagreements
remain ambiguous. This protocol changes neither a production gate nor its math.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import sys


CASE_ID = "prior-5"
CASE_STEP = 20
CASE_TIMEOUT_SECONDS = 900
FIELDS = ("atoms", "charges", "bonds")
MODES = ("original_fp32", "frozen_abs_fp32")
MAXIMUM_SCORER_FORWARDS = 3
CONTROLLED_DECODER_FORWARDS = 146  # Two baseline calls plus 2*4*9*2 endpoints.
SOURCE087_REPORT = ("/work3/s234818/latentmol/runs/source-087-affinity-causal-attempt002-replay/"
    "20261010T102212Z_source087-affinity-causal-attempt002_474638f7/prior-5/"
    "20261010T102334Z_prior-5_f2a301b1/attribution.json")
SOURCE087_REPORT_SHA256 = "45a092a8786143124c723a5e9c682c0c00f0f0efc9f7788eeb14d83057486073"
SOURCE087_RUN_SHA256 = "6d9b227ebe0a2b82b479718a429e1aab47bcd6f3e804af80aa48121d78d64444"
SOURCE087_ARCHIVE = "b15a1b032ef4e36f10fa6500b9bcd23c22952a60061da910549b5fb0f746245e"
CHAIN_RESIDUAL_FRACTION = .01  # Prespecified attribution guard, never an FD tolerance.
CHAIN_PROTOCOL = {
    "version": "regression_graph_cut_dual_fd_v1",
    "structural_scope": "Simultaneous exact live canonical atoms, charges and bonds consumed by native branch",
    "completeness": "Existing 64 native-epsilon maximum and L2 chain bounds; exactness separately recorded",
    "decoder_baseline": "Exact graph, zero readout and exact measured simultaneous graph-chain VJP",
    "fd_references": ["source086_native_regression", "measured_graph_chain"],
    "failure_attribution_maximum_projected_chain_fraction": CHAIN_RESIDUAL_FRACTION,
    "original_fd_policy_unchanged": True,
    "classification_disagreement_or_low_signal": "ambiguous",
}
CLAIM_LIMIT = __doc__
_AFFINITY = None
_DECODER = None


def observers():
    """Load only observer files before the authenticated source079 bootstrap."""
    global _AFFINITY, _DECODER
    for attribute, filename, alias in (("affinity", "affinity_causal_diagnostic.py", "_regression_affinity_reference"),
                                       ("decoder", "decoder_causal_diagnostic.py", "_regression_decoder_observer")):
        if (_AFFINITY if attribute == "affinity" else _DECODER) is None:
            spec = importlib.util.spec_from_file_location(alias, Path(__file__).with_name(filename))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if attribute == "affinity":
                _AFFINITY = module
            else:
                _DECODER = module
    return _AFFINITY, _DECODER


def regression_graph_cut(output, output_cotangent, graph_fields, point, *, call=lambda label, function: function()):
    """Use the exact graph tensors consumed by this native forward.

    All three channels form one simultaneous cut. An unused channel receives
    an explicitly reported zero; omitting its tensor would conceal cut scope.
    The widened endpoint readout is never used to define this analytic VJP.
    """
    import torch
    if (len(graph_fields) != len(FIELDS) or any(not value.requires_grad for value in graph_fields)
            or output_cotangent.requires_grad or output.shape != output_cotangent.shape):
        raise ValueError("Require three live canonical fields and a detached same-shape output cotangent")
    direct, = call("regression_direct", lambda: torch.autograd.grad(
        output, point, grad_outputs=output_cotangent, retain_graph=True))
    raw = call("regression_graph_cotangent", lambda: torch.autograd.grad(
        output, graph_fields, grad_outputs=output_cotangent, retain_graph=True, allow_unused=True))
    cotangents = tuple(torch.zeros_like(field) if value is None else value.detach()
                      for field, value in zip(graph_fields, raw))
    chain, = call("regression_graph_chain", lambda: torch.autograd.grad(
        graph_fields, point, grad_outputs=cotangents, retain_graph=False))
    if any(not bool(value.isfinite().all()) for value in (direct, chain, *cotangents)):
        raise ValueError("Nonfinite regression graph cut")
    return direct.detach(), chain.detach(), cotangents, [name for name, value in zip(FIELDS, raw) if value is None]


def require_complete_chain(record):
    """Engineering completeness is separate from unchanged scientific FD gates."""
    if record.get("engineering_chain_complete") is not True:
        raise ValueError("Regression graph cut is incomplete; no decoder attribution")


def dual_fd_attribution(native, chain):
    """Qualify localization without replacing either independently computed row.

    Nominal FP32 projections define the original FD rows. Wider dot products and
    actual representable chords also bound the observed projected discrepancy;
    none replace those rows or correct a failing numerical derivative.
    """
    import math
    if any(native[key] != chain[key] for key in ("epsilon", "f_plus", "f_minus", "central_difference")):
        raise ValueError("Dual FD comparisons must share identical endpoints and nominal epsilon")
    projections = {name: abs(native[name] - chain[name]) for name in (
        "analytic", "analytic_dot_float64", "actual_secant_linear_prediction")}
    residuals = {"native": native["absolute_error"], "chain": chain["absolute_error"]}
    if not all(math.isfinite(value) for value in (*projections.values(), *residuals.values())):
        raise ValueError("Nonfinite dual FD attribution")
    discrepancy = max(projections.values())
    same_classification = native["passed"] == chain["passed"]
    both_fail = (native["informative"] and chain["informative"]
                 and not native["passed_tolerance"] and not chain["passed_tolerance"])
    separated = all(discrepancy < CHAIN_RESIDUAL_FRACTION * value for value in residuals.values())
    status = ("consistent_pass" if native["passed"] and chain["passed"] else
              "consistent_failure_separated_from_chain_roundoff" if both_fail and separated else "ambiguous")
    return {"status": status, "classifications_agree": same_classification,
        "attribution_eligible": status != "ambiguous", "failure_localization_eligible": both_fail and separated,
        "projected_chain_differences": projections, "maximum_projected_chain_discrepancy": discrepancy,
        "native_and_chain_fd_residuals": residuals,
        "maximum_residual_fraction": CHAIN_RESIDUAL_FRACTION,
        "both_failures_separated_from_chain_roundoff": both_fail and separated,
        "original_fd_rows_unmodified": True}


def source086_reference():
    affinity, _ = observers()
    helper = affinity.head_helpers()._helper()
    if (helper._sha(affinity.SOURCE086_REPORT) != affinity.SOURCE086_REPORT_SHA256
            or helper._sha(Path(affinity.SOURCE086_REPORT).parent / "run.json") != affinity.SOURCE086_RUN_SHA256):
        raise ValueError("Pinned source086 reference changed")
    return helper._read(affinity.SOURCE086_REPORT)


def source087_reference(report, capture):
    """Bind the completed upstream-MLP investigation without changing plan assets."""
    affinity, _ = observers()
    helper = affinity.head_helpers()._helper()
    if (helper._sha(SOURCE087_REPORT) != SOURCE087_REPORT_SHA256
            or helper._sha(Path(SOURCE087_REPORT).parent / "run.json") != SOURCE087_RUN_SHA256):
        raise ValueError("Pinned source087 follow-up evidence changed")
    previous = helper._read(SOURCE087_REPORT)
    run = helper._read(Path(SOURCE087_REPORT).parent / "run.json")
    if (run.get("status") != "completed" or run.get("source_archive_sha256") != SOURCE087_ARCHIVE
            or not all(previous.get(name) is True for name in (
                "diagnostic_completed", "affinity_baseline_validated", "frozen_state_unchanged", "parent_unchanged"))
            or len(previous.get("finite_differences", [])) != 36
            or previous.get("target") != report["target"]
            or previous.get("identities_before") != report["identities_before"]
            or previous.get("identities_before") != previous.get("identities_after")
            or previous["baseline"]["gradients"]["affinity_regression"] != capture["record"]["gradients"]["regression"]):
        raise ValueError("Source087 is incomplete or differs from the same regression state")
    return previous, {"path": SOURCE087_REPORT, "sha256": SOURCE087_REPORT_SHA256,
        "run_sha256": SOURCE087_RUN_SHA256, "source_archive_sha256": SOURCE087_ARCHIVE,
        "completed_supporting_diagnostic": True, "production_gate_approval": False}


def native_capture(decoder, point, branch, config, counters, target, weight):
    """One native graph; return only detached observations for decoder-only work."""
    import torch
    from latentmol.multimodel.study import tensor_digest
    affinity, _ = observers()
    head = affinity.head_helpers()
    _, architecture = affinity.native_heads(branch.scorer)
    probe = point.detach().clone().requires_grad_()
    with torch.enable_grad(), head.capture_heads(branch.scorer) as heads:
        _, graph, outputs, terms, detail = head._forward(
            decoder, probe, branch, config, counters, target, weight, "regression_decoder_native_capture")
        graph_fields = tuple(getattr(graph, name) for name in FIELDS)
        gradients = {}
        for name in ("combined", "binding"):
            gradients[name], = counters.call("boltz", "regression_decoder:" + name, "backward", lambda name=name:
                torch.autograd.grad(terms[name], probe, retain_graph=True), point.device)
            gradients[name] = gradients[name].detach()
        mean = heads["affinity_pred_value"]
        coefficient, = torch.autograd.grad(terms["binding"], mean, retain_graph=True)
        coefficient = coefficient.detach()
        direct, chain, cotangents, unused = regression_graph_cut(mean, coefficient, graph_fields, probe,
            call=lambda label, function: counters.call("boltz", label, "backward", function, point.device))
        base = tuple(value.detach().clone() for value in graph_fields)
        gradients.update(regression=direct, graph_chain=chain)
        record = {"values": {name: float(value.detach()) for name, value in terms.items()},
            "dtypes": {name: str(value.dtype) for name, value in terms.items()},
            "native_outputs": {name: float(value.detach()) for name, value in outputs.items()},
            "gradients": {name: head._gradient_record(value) for name, value in gradients.items()},
            "direct_vs_graph_chain": head.chain_completeness(direct, chain),
            "graph_sha256": tensor_digest(dict(zip(FIELDS, base))),
            "cotangent_sha256": tensor_digest(dict(zip(FIELDS, cotangents))),
            "graph_tensors": head._metadata(dict(zip(FIELDS, base))),
            "graph_cotangents": head._metadata(dict(zip(FIELDS, cotangents))),
            "unused_graph_channels_explicit_zero": unused,
            "native_heads": head._metadata(heads), "native_affinity_architecture": architecture,
            "native_affinity_output_cotangent": coefficient.cpu().tolist(),
            "regression_component_scalar": float((mean.detach() * coefficient).sum()),
            "target_sha256": detail["target_sha256"],
            "scope": "Actual binding cotangent on mean affinity_pred_value only; excludes binary affinity, structural heads and direct repair-logit NLL."}
    return {"record": record, "gradients": gradients, "base": base, "cotangents": cotangents}, record


def verify_native_reference(capture, report, reference):
    """Authenticate total direction and isolated component independently."""
    affinity, _ = observers()
    current = capture["record"]
    prior = source086_reference()
    proof = affinity.verify_source086(current["gradients"]["regression"], report)
    native = reference["controls"]["native"]
    old_combined = native["gradients"]["combined"]
    if (current["gradients"]["combined"]["sha256"] != old_combined["sha256"]
            or current["gradients"]["combined"]["norm_float64"] != old_combined["norm"]
            or current["gradients"]["binding"] != prior["baseline"]["gradients"]["binding"]
            or current["values"]["combined"] != native["baseline"]["combined"]
            or current["native_outputs"] != prior["baseline"]["native_outputs"]
            or any(value != prior["baseline"]["values"][name] for name, value in current["values"].items())
            or any(value != native["baseline"][name] for name, value in current["native_outputs"].items())
            or report["identities_before"] != reference["identities_before"]
            or current["target_sha256"] != report["target"]["target_sha256"]):
        raise ValueError("Regression decoder capture changed original binding/combined baseline or identities")
    require_complete_chain(current["direct_vs_graph_chain"])
    return proof, prior


def verify_directions(point, combined_gradient, item, reference):
    affinity, _ = observers()
    directions = affinity.head_helpers()._helper().reconstruct_directions(point, combined_gradient, item)
    if set(directions) != {"gradient", "random_1", "random_2", "random_3"}:
        raise ValueError("Require all original combined-selected directions")
    for name, (_, metadata) in directions.items():
        if (metadata["replayed_sha256"] != reference["direction_replay"][name]["replayed_sha256"]
                or name != "gradient" and metadata["hash_matches_original"] is not True):
            raise ValueError("Changed original combined-selected direction: " + name)
    return directions


def require_decoder_baseline(row, gradient, expected_graph_sha256, expected_gradient):
    import torch
    exact = (row["graph_sha256"] == expected_graph_sha256 and row["value"] == 0.
             and torch.equal(gradient, expected_gradient))
    if not exact:
        raise ValueError("Original/held-abs decoder baseline graph or gradient differs from measured regression graph chain")
    return exact


def probe_case(decoder, point, branch, config, counters, target, weight, item_name, item, *,
               reference, reference_asset, persist=lambda: None, report=None):
    import torch
    from latentmol.boltz.local_fd import endpoint_row, neighboring_passes
    from latentmol.multimodel.study import module_digest, tensor_digest
    affinity, decoder_observer = observers()
    head = affinity.head_helpers()
    helper = head._helper()
    if item_name != "combined" or any(value.dtype != torch.float32 for value in decoder.parameters()):
        raise ValueError("Require original prior-5 combined gate and FP32 decoder")
    report = report if report is not None else {}
    report.update(status="probing", claim_limit=CLAIM_LIMIT, regression_decoder_module_sha256=helper._sha(__file__),
        decoder_observer_sha256=helper._sha(Path(__file__).with_name("decoder_causal_diagnostic.py")),
        affinity_observer_sha256=helper._sha(Path(__file__).with_name("affinity_causal_diagnostic.py")),
        source081_reference=reference_asset, modes=list(MODES), epsilons=list(head.EPSILONS), controls={},
        regression_graph_cut_protocol=CHAIN_PROTOCOL,
        readout_scope="FP64 reduction of fixed cotangents times FP32 graph changes; output ULP is not a decoder-error bound",
        original_gate_not_reclassified=True, no_new_production_gradient_policy=True)
    def phase_progress(label, event):
        report["native_capture_phase"] = event
        persist()
    persist()
    capture = affinity.detached_phase("regression_graph_capture", lambda:
        native_capture(decoder, point, branch, config, counters, target, weight), point.device, phase_progress)
    report["native_capture"] = capture["record"]
    persist()  # Preserve direct/chain discrepancy before any completeness rejection.
    proof, source086 = verify_native_reference(capture, report, reference)
    source087, supporting_proof = source087_reference(report, capture)
    report.update(source086_regression_reference=proof, original_native_baseline_exact=True,
        original_binding_gradient_exact=True, original_combined_gradient_exact=True,
        direct_regression_graph_chain_exact=capture["record"]["direct_vs_graph_chain"]["exact_equal"],
        engineering_regression_graph_chain_complete=True, source087_reference=supporting_proof)
    directions = verify_directions(point, capture["gradients"]["combined"], item, reference)
    report["direction_replay"] = {name: metadata for name, (_, metadata) in directions.items()}
    if report["direction_replay"] != source087["direction_replay"]:
        raise ValueError("Source087 original combined-selected direction identity changed")
    native_endpoints = {(row["direction"], row["epsilon"]): row for row in source086["finite_differences"]}
    if set(native_endpoints) != {(name, epsilon) for name in directions for epsilon in head.EPSILONS}:
        raise ValueError("Source086 endpoint grid is incomplete")
    original_state = module_digest(decoder)
    model = deepcopy(decoder).eval().requires_grad_(False)
    if module_digest(model) != original_state:
        raise ValueError("Decoder copy changed frozen parameters")
    observer = decoder_observer.DecoderObserver()
    base, cotangents = capture["base"], capture["cotangents"]
    reference_gradient = capture["gradients"]["regression"]
    chain_gradient = capture["gradients"]["graph_chain"]
    # These measured projections are recorded before any FD comparison. They
    # disclose accumulation effects independently of endpoint classification.
    report["native_and_chain_direction_projections"] = {name: {
        "native_dot": float((reference_gradient * direction).sum()),
        "chain_dot": float((chain_gradient * direction).sum()),
        "native_dot_float64": float((reference_gradient.double() * direction.double()).sum()),
        "chain_dot_float64": float((chain_gradient.double() * direction.double()).sum()),
        "difference_dot_float64": float(((reference_gradient.double() - chain_gradient.double()) * direction.double()).sum()),
        "direction_sha256": metadata["replayed_sha256"],
    } for name, (direction, metadata) in directions.items()}
    graph_sha = capture["record"]["graph_sha256"]
    completed_decoder_calls = 0
    for mode in MODES:
        current = {"status": "running", "mode": mode, "rows": [], "production_scorer_forward_changed": False,
            "endpoint_counterfactual": mode == "frozen_abs_fp32", "baseline_exact_graph_chain_match": False}
        report["controls"][mode] = current
        persist()
        with decoder_observer.decoder_control(model, observer):
            baseline, gradient = decoder_observer.evaluate_decoder(model, point, branch.prepared.active, config.temperature,
                base, cotangents, observer, mode, gradients=True, capture=mode == "original_fp32")
            completed_decoder_calls += 1
            current.update(baseline=baseline, gradient=head._gradient_record(gradient),
                gradient_difference_from_native=head._difference(reference_gradient, gradient))
            persist()
            current["baseline_exact_graph_chain_match"] = require_decoder_baseline(
                baseline, gradient, graph_sha, chain_gradient)
            for label, (direction, _) in directions.items():
                for epsilon in head.EPSILONS:
                    endpoints = {}
                    for sign, name in ((1, "plus"), (-1, "minus")):
                        endpoint_point = point + epsilon * direction if sign == 1 else point - epsilon * direction
                        endpoint, _ = decoder_observer.evaluate_decoder(model, endpoint_point,
                            branch.prepared.active, config.temperature, base, cotangents, observer, mode)
                        endpoint["latent_point_sha256"] = tensor_digest({"z": endpoint_point})
                        completed_decoder_calls += 1
                        endpoints[name] = endpoint
                    native_row = native_endpoints[label, epsilon]
                    graph_parity = {name: endpoints[name]["graph_sha256"] == native_row[name]["graph_sha256"] for name in endpoints}
                    plus_summary = {
                        "loss": endpoints["plus"]["value"], "loss_dtype": "torch.float64",
                        "natural_selector_crossed": native_row["plus"]["natural_selector_crossed"]}
                    minus_summary = {
                        "loss": endpoints["minus"]["value"], "loss_dtype": "torch.float64",
                        "natural_selector_crossed": native_row["minus"]["natural_selector_crossed"]}
                    comparisons = {name: endpoint_row(point, direction, grad, epsilon,
                        plus_summary, minus_summary, 0.,
                        rtol=config.diagnostics.fd_rtol, atol=config.diagnostics.fd_atol)
                        for name, grad in (("source086_native_regression", reference_gradient),
                                           ("measured_graph_chain", gradient))}
                    row = {"direction": label, "epsilon": epsilon, "finite_differences": comparisons,
                        "attribution": dual_fd_attribution(*comparisons.values()),
                        "original_native_endpoint_graph_parity": graph_parity,
                        "selector_warning_scope": "Source086 original decoder endpoint; both diagnostic modes use the same fixed mask.",
                        **endpoints}
                    current["rows"].append(row)
                    report["controlled_decoder_forward_calls_completed"] = completed_decoder_calls
                    persist()
                    if mode == "original_fp32" and not all(graph_parity.values()):
                        raise ValueError("Original decoder endpoint graph differs from source086 native endpoint")
            current["direction_diagnostics"] = {comparison: {label: neighboring_passes(
                [row["finite_differences"][comparison] for row in current["rows"] if row["direction"] == label],
                minimum=item["policy"]["minimum_neighboring_passes"]) for label in directions}
                for comparison in CHAIN_PROTOCOL["fd_references"]}
            current["all_original_directions_have_neighboring_window"] = {comparison: all(
                result["passed"] for result in diagnostics.values())
                for comparison, diagnostics in current["direction_diagnostics"].items()}
            current["attribution_counts"] = {status: sum(row["attribution"]["status"] == status for row in current["rows"])
                for status in ("consistent_pass", "consistent_failure_separated_from_chain_roundoff", "ambiguous")}
            current["status"] = "completed"
        persist()
    if completed_decoder_calls != CONTROLLED_DECODER_FORWARDS or module_digest(decoder) != original_state or module_digest(model) != original_state:
        raise ValueError("Decoder call budget or frozen state changed")
    report.update(status="completed", diagnostic_completed=True, all_decoder_methods_restored=True,
        old_gate_certified_by_this_diagnostic=False, maximum_scorer_forwards=MAXIMUM_SCORER_FORWARDS,
        controlled_decoder_forward_calls_completed=completed_decoder_calls,
        decoder_calls_in_native_capture=1, loader_mask_initialization_excluded_from_controlled_decoder_count=True,
        no_optimization_or_new_defog_calls=True)
    persist()
    return report


def selected_case(plan):
    cases = [case for case in plan["cases"] if case["id"] == CASE_ID]
    if len(cases) != 1 or cases[0]["step"] != CASE_STEP or cases[0]["arm"] != "fixed_initial_mask":
        raise ValueError("Require original prior-5 fixed-mask step20 case")
    return cases[0]


def run_case(plan, case, output_root):
    affinity, _ = observers()
    head = affinity.head_helpers()
    helper = head._helper()
    reference, asset = head.reference_replay(case, plan)
    helper.probe_case = lambda *args, **kwargs: probe_case(*args, reference=reference, reference_asset=asset, **kwargs)
    helper.REPEATS, helper.CASE_TIMEOUT_SECONDS, helper.CLAIM_LIMIT = 1, CASE_TIMEOUT_SECONDS, CLAIM_LIMIT
    return helper.run_case(plan, case, output_root)


def run_study(path, sha):
    affinity, _ = observers()
    head = affinity.head_helpers()
    helper = head._helper()
    plan = helper.load_plan(path, sha)
    case = selected_case(plan)
    head.reference_replay(case, plan)
    from latentmol.experiments import ExperimentRun, write_json
    from latentmol.boltz.workflow_process import run_logged
    configuration = {"plan": plan, "plan_sha256": sha, "module_sha256": helper._sha(__file__),
        "selected_cases": [CASE_ID], "modes": list(MODES), "epsilons": list(head.EPSILONS),
        "maximum_scorer_forwards": MAXIMUM_SCORER_FORWARDS,
        "controlled_decoder_forwards": CONTROLLED_DECODER_FORWARDS,
        "regression_graph_cut_protocol": CHAIN_PROTOCOL,
        "source087_supporting_report_sha256": SOURCE087_REPORT_SHA256,
        "case_timeout_seconds": CASE_TIMEOUT_SECONDS, "claim_limit": CLAIM_LIMIT}
    with ExperimentRun(Path(plan["output_root"]), plan["name"], configuration, Path.cwd()) as run:
        execution = run_logged([sys.executable, str(Path(__file__).resolve()), "--plan", str(Path(path).resolve()),
            "--sha256", sha, "--child", CASE_ID, "--child-output", str(run.path / CASE_ID)],
            run.path / (CASE_ID + ".log"), timeout_seconds=CASE_TIMEOUT_SECONDS)
        complete = False
        if not execution["timed_out"] and execution["returncode"] == 0 and len(execution["run_paths"]) == 1:
            child = Path(execution["run_paths"][0])
            evidence = helper._read(child / "attribution.json")
            complete = (helper._read(child / "run.json")["status"] == "completed"
                and evidence.get("controlled_decoder_forward_calls_completed") == CONTROLLED_DECODER_FORWARDS
                and all(evidence.get(name) is True for name in ("diagnostic_completed", "frozen_state_unchanged", "parent_unchanged",
                    "original_native_baseline_exact", "original_binding_gradient_exact", "original_combined_gradient_exact",
                    "engineering_regression_graph_chain_complete", "all_decoder_methods_restored"))
                and evidence.get("regression_graph_cut_protocol") == CHAIN_PROTOCOL
                and evidence.get("source087_reference", {}).get("sha256") == SOURCE087_REPORT_SHA256
                and set(evidence.get("controls", {})) == set(MODES)
                and all(row.get("status") == "completed" and row.get("baseline_exact_graph_chain_match") is True
                        and len(row.get("rows", [])) == 36 for row in evidence.get("controls", {}).values()))
        write_json(run.path / "cases.json", [{"case_id": CASE_ID, "diagnostic_completed": complete,
            "execution": execution, "old_gate_certified": False}])
        run.annotate(diagnostic_cases_completed=int(complete), diagnostic_cases_requested=1, old_gates_certified=False)
        if not complete:
            raise ValueError("Regression decoder diagnostic incomplete; receipts retained")
        return run.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--child")
    parser.add_argument("--child-output", type=Path)
    args = parser.parse_args()
    affinity, _ = observers()
    helper = affinity.head_helpers()._helper()
    if not os.environ.get("LSB_JOBID") or not helper._hash(os.environ.get("LATENTMOL_SOURCE_SHA256")):
        raise ValueError("Regression decoder probes require an immutable DTU HPC allocation")
    plan = helper.load_plan(args.plan, args.sha256)
    case = selected_case(plan)
    if args.child is None:
        if args.child_output is not None:
            raise ValueError("Child output requires selected child")
        print(run_study(args.plan, args.sha256), flush=True)
    else:
        if args.child != CASE_ID or args.child_output is None:
            raise ValueError("Require original prior-5 child/output")
        print(run_case(plan, case, args.child_output), flush=True)


if __name__ == "__main__":
    main()
