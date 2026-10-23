"""HPC-only isolated native-head derivatives at the original prior-5 failure.

The six frozen cotangents come from the complete original binding objective.
Every endpoint is an unchanged production forward. Widened linear readouts
expose native-head changes; they do not restore bits lost inside the model.
Reported ULP budgets cover only final native-head tensor rounding, not internal
matrix operations, nonlinear curvature, activation crossings or backward error.
No historical gate is certified and no production arithmetic is replaced.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import sys


CASE_ID = "prior-5"
CASE_TIMEOUT_SECONDS = 1800
MAXIMUM_SCORER_FORWARDS = 75
CLAIM_LIMIT = __doc__
_HEAD_MODULE = None


def head_helpers():
    """Load observer helpers without importing latentmol before old-source boot."""
    global _HEAD_MODULE
    if _HEAD_MODULE is None:
        path = Path(__file__).with_name("head_causal_diagnostic.py")
        spec = importlib.util.spec_from_file_location("_isolated_head_observer", path)
        _HEAD_MODULE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_HEAD_MODULE)
    return _HEAD_MODULE


def channel_name(key):
    return "isolated_head_" + key


def isolated_head_vjps(heads, cotangents, point, call=lambda label, function: function()):
    """Separate paths through one non-ancestor cut; weights remain detached."""
    import torch
    result = {}
    for key in head_helpers().HEADS:
        if heads[key].shape != cotangents[key].shape or cotangents[key].requires_grad:
            raise ValueError("Require a detached same-shape head cotangent")
        gradient, = call(key, lambda key=key: torch.autograd.grad(
            heads[key], point, grad_outputs=cotangents[key], retain_graph=True, allow_unused=True))
        result[key] = torch.zeros_like(point) if gradient is None else gradient.detach()
        if not bool(result[key].isfinite().all()):
            raise ValueError("Nonfinite isolated head derivative: " + key)
    return result


def require_complete_head_sum(direct, gradients):
    """A missing/corrupted channel cannot authorize localization."""
    head = head_helpers()
    if set(gradients) != set(head.HEADS):
        raise ValueError("Incomplete isolated head membership")
    total = sum(gradients[key] for key in head.HEADS)
    return head.chain_completeness(direct, total)


def native_rounding_budget(value, cotangent):
    """Conservative final-rounding scale using actual native endpoint spacing.

    This assumes each native tensor entry is rounded once from an exact real
    value. It is NOT an error bound for the neural network that produced it.
    Both neighboring spacings are inspected at powers of two and near zero.
    The frozen baseline subtraction cancels in the central endpoint difference.
    """
    import torch
    if value.dtype not in (torch.float32, torch.float64) or value.shape != cotangent.shape:
        raise ValueError("Require same-shape FP32/FP64 native tensor and cotangent")
    if not bool(value.isfinite().all()) or not bool(cotangent.isfinite().all()):
        raise ValueError("Nonfinite tensor cannot receive a finite rounding budget")
    detached = value.detach()
    upper = torch.nextafter(detached, torch.full_like(detached, float("inf")))
    lower = torch.nextafter(detached, torch.full_like(detached, -float("inf")))
    spacing = torch.maximum(upper.double() - detached.double(), detached.double() - lower.double())
    if not bool(spacing.isfinite().all()):
        raise ValueError("Native tensor touches floating-point overflow")
    return {"native_dtype": str(value.dtype), "elements": value.numel(),
            "maximum_native_spacing": float(spacing.max()),
            "weighted_half_ulp_sum": float((cotangent.detach().double().abs() * spacing).sum() * .5),
            "scope": "Final native-head rounding only; excludes accumulated upstream/model error."}


def baseline_probe(decoder, point, branch, config, counters, target, weight):
    """Original baseline plus six separately measured reverse-mode paths."""
    import torch
    from latentmol.boltz.dbmol_objective import pocket_token_mask
    head = head_helpers()
    probe = point.detach().clone().requires_grad_()
    pocket, ligand = pocket_token_mask(branch.prepared.features, config.dbmol)
    with torch.enable_grad(), head.capture_heads(branch.scorer) as heads:
        _, _, outputs, terms, detail = head._forward(
            decoder, probe, branch, config, counters, target, weight, "isolated_head_baseline")
        centers = head.native_bin_centers(heads["distogram"])
        controls = head.cached_controls(outputs, heads, config, pocket, ligand, centers, terms["repair_nll"], weight)
        gradients = {}
        for key, value in {**terms, **controls}.items():
            gradient, = counters.call("boltz", "head_cut_direct:" + key, "backward", lambda value=value:
                torch.autograd.grad(value, probe, retain_graph=True), point.device)
            if not bool(gradient.isfinite().all()):
                raise ValueError("Nonfinite direct derivative: " + key)
            gradients[key] = gradient.detach()
        cotangents, chain, metadata = {}, {}, {}
        for key in ("contact", "binding"):
            chain[key], cotangents[key], unused = counters.call("boltz", "head_cut_chain:" + key, "backward", lambda key=key:
                head.cut_vjp(terms[key], heads, probe), point.device)
            metadata[key] = {"direct_vs_cut_chain": head.chain_completeness(gradients[key], chain[key]),
                "unused_cut_channels": unused, "cotangents": head._metadata(cotangents[key])}
        isolated = isolated_head_vjps(heads, cotangents["binding"], probe,
            call=lambda label, function: counters.call("boltz", "isolated_head:" + label, "backward", function, point.device))
        completeness = require_complete_head_sum(gradients["binding"], isolated)
        for key, gradient in isolated.items():
            gradients[channel_name(key)] = gradient
        base = {key: value.detach().clone() for key, value in heads.items()}
        report = {"head_tensors": head._metadata(base), "cut_chain": metadata,
            "isolated_head_sum": completeness,
            "isolated_head_cotangent_scope": "Entire binding objective, including distogram-dependent confidence PDE weights; excludes repair logits path.",
            "direct_binding_vs_total_minus_weighted_repair": head._difference(
                gradients["binding"], gradients["combined"] - weight * gradients["repair_nll"]),
            "values": {**{key: float(value.detach()) for key, value in {**terms, **controls}.items()},
                       **{channel_name(key): 0. for key in head.HEADS}},
            "dtypes": {**{key: str(value.dtype) for key, value in {**terms, **controls}.items()},
                       **{channel_name(key): "torch.float64" for key in head.HEADS}},
            "native_outputs": {key: float(value.detach()) for key, value in outputs.items()},
            "gradients": {key: head._gradient_record(value) for key, value in gradients.items()},
            "contact_selection": deepcopy(branch.scorer.last_contact_selection), "target_sha256": detail["target_sha256"],
            "all_hooks_observational": True, "head_checkpoint_replays": 0,
            "relu_observation": "Not captured: this diagnostic localizes channels, not activation branches."}
    return report, gradients, base, cotangents, chain, (pocket, ligand, centers.detach())


def endpoint_probe(decoder, point, branch, config, counters, target, weight, base, cotangents, geometry):
    """The same native endpoint supplies all six readouts and rounding scales."""
    import torch
    from latentmol.boltz.active_set import select_slots
    from latentmol.multimodel.study import tensor_digest
    head = head_helpers()
    with torch.no_grad(), head.capture_heads(branch.scorer) as heads:
        decoded, graph, outputs, terms, detail = head._forward(
            decoder, point, branch, config, counters, target, weight, "isolated_head_endpoint")
        controls = head.cached_controls(outputs, heads, config, *geometry, terms["repair_nll"], weight)
        readouts, contributions = {}, {}
        for key, cotangent in cotangents.items():
            readouts[key], contributions[key] = head.fixed_readout(heads, base, cotangent)
        channels = {channel_name(key): contributions["binding"][key] for key in head.HEADS}
        values = {**terms, **controls, **{"head_readout_" + key: value for key, value in readouts.items()}}
        natural = select_slots(decoded, temperature=config.temperature, threshold=config.threshold, require_nonempty=False)
        result = {"values": {**{key: float(value) for key, value in values.items()}, **channels},
            "dtypes": {**{key: str(value.dtype) for key, value in values.items()}, **{key: "torch.float64" for key in channels}},
            "head_contributions": contributions,
            "native_head_rounding": {key: native_rounding_budget(heads[key], cotangents["binding"][key]) for key in head.HEADS},
            "native_outputs": {key: float(value) for key, value in outputs.items()},
            "graph_sha256": tensor_digest({key: getattr(graph, key) for key in ("atoms", "charges", "bonds")}),
            "target_sha256": detail["target_sha256"], "contact_selection": deepcopy(branch.scorer.last_contact_selection),
            "natural_selector_crossed": not torch.equal(natural.active, branch.prepared.active)}
    return result


def annotate_rows(report):
    """Metadata arithmetic only; never treat a rounding budget as a gate pass."""
    head = head_helpers()
    for row in report.get("finite_differences", []):
        if "isolated_head_accounting" in row:
            continue
        budgets = {}
        for key in head.HEADS:
            plus = row["plus"]["native_head_rounding"][key]["weighted_half_ulp_sum"]
            minus = row["minus"]["native_head_rounding"][key]["weighted_half_ulp_sum"]
            bound = (plus + minus) / (2 * row["epsilon"])
            observed = row["terms"][channel_name(key)]["absolute_error"]
            budgets[key] = {"central_slope_final_rounding_budget": bound,
                "observed_error_exceeds_final_rounding_budget": observed > bound,
                "not_a_model_error_bound_or_pass_criterion": True}
        analytic = [row["terms"][channel_name(key)]["analytic_dot_float64"] for key in head.HEADS]
        numerical = [row["terms"][channel_name(key)]["central_difference"] for key in head.HEADS]
        row["isolated_head_accounting"] = {"rounding_only": budgets,
            "sum_channel_analytic_dot_float64": sum(analytic), "sum_channel_central_difference": sum(numerical),
            "sum_absolute_channel_analytic": sum(abs(value) for value in analytic),
            "sum_absolute_channel_central_difference": sum(abs(value) for value in numerical),
            "scope": "Individual fixed-cotangent channels may cancel; readout output FP64 ULPs do not describe native model precision."}


@contextmanager
def diagnostic_helpers():
    """Scope replacement of observer functions only, never production modules."""
    head = head_helpers()
    original = head.baseline_probe, head.endpoint_probe
    head.baseline_probe, head.endpoint_probe = baseline_probe, endpoint_probe
    try:
        yield head
    finally:
        head.baseline_probe, head.endpoint_probe = original


def probe_case(*args, persist=lambda: None, report=None, **kwargs):
    """Reuse original source/reference/direction checks and incremental storage."""
    report = report if report is not None else {}
    report.update(isolated_head_module_sha256=head_helpers()._helper()._sha(__file__),
        isolated_head_claim_limit=CLAIM_LIMIT)
    def checked_persist():
        baseline = report.get("baseline")
        if baseline is not None:
            complete = baseline["isolated_head_sum"]["engineering_chain_complete"]
            report["engineering_isolated_head_sum_complete"] = complete
            if not complete:
                report["isolated_head_status"] = "incomplete"
                persist()
                raise ValueError("Isolated head sum is incomplete; no endpoint localization authorized")
        annotate_rows(report)
        persist()
    with diagnostic_helpers() as head:
        result = head.probe_case(*args, persist=checked_persist, report=report, **kwargs)
    result["isolated_head_status"] = "completed"
    checked_persist()
    return result


def selected_case(plan):
    matches = [case for case in plan["cases"] if case["id"] == CASE_ID]
    if len(matches) != 1 or matches[0]["step"] != 20:
        raise ValueError("Require the single original prior-5 step20 failure")
    return matches[0]


def run_case(plan, case, output_root):
    head = head_helpers()
    helper = head._helper()
    reference, reference_asset = head.reference_replay(case, plan)
    helper.probe_case = lambda *args, **kwargs: probe_case(
        *args, reference=reference, reference_asset=reference_asset, **kwargs)
    helper.REPEATS, helper.CASE_TIMEOUT_SECONDS, helper.CLAIM_LIMIT = 1, CASE_TIMEOUT_SECONDS, CLAIM_LIMIT
    return helper.run_case(plan, case, output_root)


def run_study(path, sha):
    head = head_helpers()
    helper = head._helper()
    plan = helper.load_plan(path, sha)
    case = selected_case(plan)
    _, item = helper.failed_item(helper._read(helper.authenticated_artifact(case,
        Path(case["parent_run"]) / case["arm"] / "gate-0020.json")))
    randoms = [key for key, row in item["direction_results"].items() if key != "gradient" and row.get("passed") is False]
    calls = 3 + (1 + len(randoms)) * len(head.EPSILONS) * 2
    if set(randoms) != {"random_1", "random_2", "random_3"} or calls != MAXIMUM_SCORER_FORWARDS:
        raise ValueError("Original prior-5 directions or native call budget changed")
    head.reference_replay(case, plan)
    from latentmol.experiments import ExperimentRun, write_json
    from latentmol.boltz.workflow_process import run_logged
    config = {"plan": plan, "plan_sha256": sha, "module_sha256": helper._sha(__file__),
        "head_helper_sha256": helper._sha(Path(__file__).with_name("head_causal_diagnostic.py")),
        "loader_sha256": helper._sha(Path(__file__).with_name("failed_gate_attribution.py")),
        "selected_cases": [CASE_ID], "maximum_scorer_forwards": calls,
        "case_timeout_seconds": CASE_TIMEOUT_SECONDS, "claim_limit": CLAIM_LIMIT}
    with ExperimentRun(Path(plan["output_root"]), plan["name"], config, Path.cwd()) as run:
        execution = run_logged([sys.executable, str(Path(__file__).resolve()), "--plan", str(Path(path).resolve()),
            "--sha256", sha, "--child", CASE_ID, "--child-output", str(run.path / CASE_ID)],
            run.path / (CASE_ID + ".log"), timeout_seconds=CASE_TIMEOUT_SECONDS)
        complete = False
        if not execution["timed_out"] and execution["returncode"] == 0 and len(execution["run_paths"]) == 1:
            child = Path(execution["run_paths"][0])
            evidence = helper._read(child / "attribution.json")
            complete = (helper._read(child / "run.json")["status"] == "completed"
                and evidence.get("isolated_head_status") == "completed"
                and all(evidence.get(key) is True for key in ("diagnostic_completed", "frozen_state_unchanged", "parent_unchanged",
                    "baseline_exact_source081_scalar_parity", "engineering_head_cut_complete",
                    "engineering_isolated_head_sum_complete", "original_failed_gradient_exact_source081_match")))
        write_json(run.path / "cases.json", [{"case_id": CASE_ID, "diagnostic_completed": complete,
            "execution": execution, "old_gate_certified": False}])
        run.annotate(diagnostic_cases_completed=int(complete), diagnostic_cases_requested=1, old_gates_certified=False)
        if not complete:
            raise ValueError("Isolated native-head diagnostic incomplete; receipts retained")
        return run.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--child")
    parser.add_argument("--child-output", type=Path)
    args = parser.parse_args()
    helper = head_helpers()._helper()
    if not os.environ.get("LSB_JOBID") or not helper._hash(os.environ.get("LATENTMOL_SOURCE_SHA256")):
        raise ValueError("Isolated native-head probes require an immutable DTU HPC allocation")
    plan = helper.load_plan(args.plan, args.sha256)
    case = selected_case(plan)
    if args.child is None:
        if args.child_output is not None:
            raise ValueError("Child output requires a selected child")
        print(run_study(args.plan, args.sha256), flush=True)
    else:
        if args.child != CASE_ID or args.child_output is None:
            raise ValueError("Require the original prior-5 child and output directory")
        print(run_case(plan, case, args.child_output), flush=True)


if __name__ == "__main__":
    main()
