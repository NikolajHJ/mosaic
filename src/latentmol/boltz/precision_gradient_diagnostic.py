"""Bounded HPC validation of a declared DBMol precision candidate.

Prior 0 compares the actual new contact/assembly reducer with cached FP64
controls, then runs unchanged local_v1 derivative gates. Prior 5 retains the
original scorer and isolates its native-head derivatives. Completion and a
passed candidate gate are distinct: scientific failures remain in the report.
No optimizer, DeFoG call, tolerance change or historical gate approval occurs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import sys
from types import MethodType


CASE_IDS = ("prior-0", "prior-5")
PRIOR0_TIMEOUT_SECONDS = 2700
PRIOR0_MAXIMUM_FORWARDS = 200
PRIOR0_FIXED_FORWARDS = 6  # Original and candidate parity pairs and baselines.
ARITHMETIC = "contact_and_assembly_float64_v1"
CLAIM_LIMIT = __doc__


def direct_module(filename, name, *, register=False):
    """Direct loading preserves old production imports in a fresh child.

    The new objective dataclass requires its alias in sys.modules. It never
    replaces latentmol.boltz.dbmol_objective or another historical module.
    """
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    if register:
        sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if register:
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
        raise
    return module


def head_helpers():
    return direct_module("head_causal_diagnostic.py", "_precision_original_head_observer")


@contextmanager
def candidate_reducer(branch, config, objective_module):
    """Temporarily bind the actual new reducer with a truthful profile manifest.

    Native trunks, heads, prepared features and weights are never replaced.
    Only the declared objective configuration/reducer/branch manifest change;
    all are restored even after a failed scientific or operational check.
    """
    from latentmol.boltz.scoring import boltz_branch_manifest
    scorer = branch.scorer
    original_config, original_manifest = scorer.dbmol_config, branch.manifest
    had_instance_method = "_reduce" in scorer.__dict__
    original_method = scorer.__dict__.get("_reduce")
    original_selection = scorer.last_contact_selection
    raw = {**config.dbmol.record(), "objective_arithmetic": ARITHMETIC}
    candidate = objective_module.DBMolObjectiveConfig.from_mapping(raw, Path.cwd())
    active_config = replace(config, dbmol=candidate)
    def reduce(this, affinity, structure, distogram, features):
        this.last_contact_selection = {}
        return objective_module.reduce_profile(affinity, structure, distogram, features,
            this.dbmol_config, diagnostics=this.last_contact_selection)
    try:
        scorer.dbmol_config = candidate
        scorer._reduce = MethodType(reduce, scorer)
        branch.manifest = boltz_branch_manifest(branch.prepared,
            adapter_source_sha256=original_manifest.adapter_source_sha256,
            recycling_steps=scorer.recycling_steps,
            validation_report_sha256=original_manifest.validation_report_sha256,
            dbmol_profile=candidate.record())
        branch.verify_context_unchanged()
        yield active_config
        branch.verify_context_unchanged()
    finally:
        scorer.dbmol_config = original_config
        if had_instance_method:
            scorer._reduce = original_method
        else:
            del scorer._reduce
        scorer.last_contact_selection = original_selection
        branch.manifest = original_manifest


def authenticate_baseline(head, helper, baseline, gradients, point, item, report, reference):
    """Authenticate the exact old failure before installing new arithmetic."""
    if not all(row["direct_vs_cut_chain"]["engineering_chain_complete"] for row in baseline["cut_chain"].values()):
        raise ValueError("Original native head cut is incomplete")
    old = reference["controls"]["native"]
    expected = old["gradients"]["dbmol_contact_loss"]
    observed = baseline["gradients"]["contact"]
    if observed["sha256"] != expected["sha256"] or observed["norm_float64"] != expected["norm"]:
        raise ValueError("Original contact gradient differs from authenticated source081")
    if (baseline["values"]["combined"] != old["baseline"]["combined"]
            or any(value != old["baseline"][key] for key, value in baseline["native_outputs"].items())
            or report["identities_before"] != reference["identities_before"]):
        raise ValueError("Original baseline/frozen identities differ from source081")
    directions = helper.reconstruct_directions(point, gradients["contact"], item)
    if set(directions) != set(reference["direction_replay"]):
        raise ValueError("Original failed direction membership changed")
    for key, (_, metadata) in directions.items():
        if (metadata["replayed_sha256"] != reference["direction_replay"][key]["replayed_sha256"]
                or key != "gradient" and metadata["hash_matches_original"] is not True):
            raise ValueError("Original failed direction identity changed: " + key)
    report.update(baseline_exact_source081_scalar_parity=True, original_failed_gradient_exact_source081_match=True,
        original_direction_replay={key: metadata for key, (_, metadata) in directions.items()},
        engineering_head_cut_complete=True)
    return directions


def candidate_baseline(head, decoder, point, branch, config, counters, target, weight):
    import torch
    probe = point.detach().clone().requires_grad_()
    with torch.enable_grad(), head.capture_heads(branch.scorer) as heads:
        _, _, outputs, terms, detail = head._forward(
            decoder, probe, branch, config, counters, target, weight, "precision_candidate_baseline")
        gradients = {}
        for key, value in terms.items():
            gradient, = counters.call("boltz", "precision_candidate:" + key, "backward", lambda value=value:
                torch.autograd.grad(value, probe, retain_graph=True), point.device)
            if not bool(gradient.isfinite().all()):
                raise ValueError("Nonfinite candidate gradient: " + key)
            gradients[key] = gradient.detach()
        record = {"values": {key: float(value.detach()) for key, value in terms.items()},
            "dtypes": {key: str(value.dtype) for key, value in terms.items()},
            "gradients": {key: head._gradient_record(value) for key, value in gradients.items()},
            "raw_heads": head._metadata(heads), "target_sha256": detail["target_sha256"],
            "native_outputs": {key: float(value.detach()) for key, value in outputs.items()},
            "contact_selection": deepcopy(branch.scorer.last_contact_selection),
            "repair_weighting": "Original weight * FP32 repair_loss, then addition to FP64 binding; not a widened repair reduction."}
    return record, gradients


def require_cached_parity(candidate, candidate_gradients, original, original_gradients):
    """Strict equality to controls that used the identical native logits."""
    import torch
    for key, cached in (("contact", "contact_float64"), ("binding", "binding_contact_float64")):
        if (candidate["values"][key] != original["values"][cached]
                or not torch.equal(candidate_gradients[key], original_gradients[cached])):
            raise ValueError("Actual candidate differs from cached FP64 control: " + key)
    if candidate["raw_heads"] != original["head_tensors"]:
        raise ValueError("Candidate reduction changed native raw heads")
    if any(row["dtype"] != "torch.float32" for row in candidate["raw_heads"].values()):
        raise ValueError("Native heads unexpectedly changed precision")
    if (candidate["dtypes"]["contact"] != "torch.float64" or candidate["dtypes"]["binding"] != "torch.float64"
            or candidate["dtypes"]["combined"] != "torch.float64" or candidate["dtypes"]["repair_nll"] != "torch.float32"):
        raise ValueError("Candidate dtype contract differs from declared arithmetic")
    if (candidate["values"]["repair_nll"] != original["values"]["repair_nll"]
            or not torch.equal(candidate_gradients["repair_nll"], original_gradients["repair_nll"])
            or candidate["target_sha256"] != original["target_sha256"]):
        raise ValueError("Candidate changed held-target repair mathematics or identity")


def planned_endpoints(point, gradients, original_directions, *, seed, maximum):
    """Declare both native local_v1 plans and a hard budget before any endpoint."""
    from latentmol.boltz.local_fd import direction_plan, EPSILONS
    from latentmol.multimodel.study import tensor_digest
    plans, unique = {}, set()
    def include(direction):
        for epsilon in EPSILONS:
            for sign in (1, -1):
                unique.add(tensor_digest({"z": point + sign * epsilon * direction}))
    for key in ("contact", "combined"):
        directions, legacy, metadata = direction_plan(point, gradients[key], seed=seed)
        plans[key] = metadata
        for direction in (*directions.values(), legacy):
            include(direction)
    # Keep the original failing random projections even if the new analytic
    # gradient's prespecified proposal filter selects a different set.
    for key, (direction, _) in original_directions.items():
        if key != "gradient":
            include(direction)
    if len(unique) > maximum:
        raise ValueError("Prespecified candidate endpoint union exceeds bounded budget; no direction was dropped")
    return plans, unique


def candidate_endpoint(head, decoder, point, branch, config, counters, target, weight):
    import torch
    from latentmol.boltz.active_set import select_slots
    with torch.no_grad(), head.capture_heads(branch.scorer) as heads:
        decoded, _, outputs, terms, detail = head._forward(
            decoder, point, branch, config, counters, target, weight, "precision_candidate_endpoint")
        if any(value.dtype != torch.float32 for value in heads.values()):
            raise ValueError("An endpoint changed native head precision")
        if terms["contact"].dtype != torch.float64 or terms["combined"].dtype != torch.float64:
            raise ValueError("An endpoint lost candidate scalar precision")
        natural = select_slots(decoded, temperature=config.temperature, threshold=config.threshold, require_nonempty=False)
        return {"values": {key: float(value) for key, value in terms.items()},
            "dtypes": {key: str(value.dtype) for key, value in terms.items()},
            "target_sha256": detail["target_sha256"], "contact_selection": deepcopy(branch.scorer.last_contact_selection),
            "natural_selector_crossed": not torch.equal(natural.active, branch.prepared.active)}


def probe_precision(decoder, point, branch, config, counters, target, weight, item_name, item, *,
                    reference, reference_asset, persist=lambda: None, report=None):
    import torch
    from latentmol.boltz.local_fd import validate_local_fd, endpoint_row, neighboring_passes, EPSILONS
    from latentmol.multimodel.study import tensor_digest
    head = head_helpers()
    helper = head._helper()
    report = report if report is not None else {}
    if item_name != "dbmol_contact_loss":
        raise ValueError("Precision candidate requires the original prior-0 contact failure")
    report.update(status="probing", claim_limit=CLAIM_LIMIT, reference_replay=reference_asset,
                  precision_module_sha256=helper._sha(__file__),
                  head_helper_sha256=helper._sha(Path(__file__).with_name("head_causal_diagnostic.py")),
                  candidate_gate_passed=False, candidate_gates={}, original_direction_candidate_rows=[])
    persist()
    original, old_gradients, _, _, _, _ = head.baseline_probe(decoder, point, branch, config, counters, target, weight)
    report["original_baseline"] = original
    directions = authenticate_baseline(head, helper, original, old_gradients, point, item, report, reference)
    persist()
    objective_path = Path(__file__).with_name("dbmol_objective.py")
    objective_sha = helper._sha(objective_path)
    objective = direct_module("dbmol_objective.py", "_declared_precision_objective", register=True)
    original_profile = deepcopy(config.dbmol.record())
    original_manifest = branch.manifest
    report["candidate_objective"] = {"path": str(objective_path), "sha256": objective_sha,
        "objective_arithmetic": ARITHMETIC, "historical_objective_module_replaced": False,
        "native_model_arithmetic_changed": False, "new_defog_calls": 0, "optimizer_updates": 0}
    with candidate_reducer(branch, config, objective) as candidate_config:
        report["candidate_config"] = candidate_config.record()
        report["candidate_config_sha256"] = helper._digest(candidate_config.record())
        report["candidate_branch_manifest"] = deepcopy(branch.manifest.to_dict())
        # This parity uses the new reducer on both original hard-input paths.
        with torch.no_grad():
            native = counters.call("boltz", "precision_seed_parity", "native_forward", lambda:
                branch.scorer.discrete_reference(branch.prepared.features, branch.prepared.coordinates), point.device)
            adapted = counters.call("boltz", "precision_seed_parity", "forward", lambda:
                branch(branch.prepared.seed.to_relaxed(decoder.spec)), point.device)
        if set(native) != set(adapted):
            raise ValueError("Candidate native/adapted output membership differs")
        for key in native:
            torch.testing.assert_close(native[key], adapted[key], rtol=config.diagnostics.parity_rtol, atol=config.diagnostics.parity_atol)
        report["candidate_native_seed_parity"] = True
        baseline, gradients = candidate_baseline(head, decoder, point, branch, candidate_config, counters, target, weight)
        report["candidate_baseline"] = baseline
        persist()
        require_cached_parity(baseline, gradients, original, old_gradients)
        report["candidate_cached_contact_binding_exact_parity"] = True
        plans, planned = planned_endpoints(point, gradients, directions, seed=config.diagnostics.seed,
            maximum=PRIOR0_MAXIMUM_FORWARDS - PRIOR0_FIXED_FORWARDS)
        report["endpoint_plan"] = {"direction_plans": plans, "unique_native_endpoints": len(planned),
            "maximum_scorer_forwards": len(planned) + PRIOR0_FIXED_FORWARDS,
            "endpoint_cache": "Exact production-dtype point SHA256; no approximate point reuse",
            "original_failed_random_directions_retained": [key for key in directions if key != "gradient"]}
        cache = {}
        def endpoint(candidate):
            digest = tensor_digest({"z": candidate})
            if digest not in planned:
                raise ValueError("Unplanned native endpoint; refuse implicit budget expansion")
            if digest not in cache:
                cache[digest] = candidate_endpoint(head, decoder, candidate, branch, candidate_config, counters, target, weight)
                if cache[digest]["target_sha256"] != baseline["target_sha256"]:
                    raise ValueError("Candidate endpoint changed held repair target")
                report["unique_endpoints_scored"] = len(cache)
            return cache[digest]
        def score(candidate, key):
            row = endpoint(candidate)
            return {"loss": row["values"][key], "loss_dtype": row["dtypes"][key],
                "contact_selection": row["contact_selection"], "natural_selector_crossed": row["natural_selector_crossed"]}
        persist()
        for key in ("contact", "combined"):
            gate = {}
            report["candidate_gates"][key] = gate
            validate_local_fd(point, gradients[key], lambda p, key=key: score(p, key),
                baseline=baseline["values"][key], seed=config.diagnostics.seed,
                rtol=config.diagnostics.fd_rtol, atol=config.diagnostics.fd_atol,
                minimum_passes=config.diagnostics.minimum_fd_passes, report=gate, persist=persist)
            if gate["direction_selection"] != plans[key]:
                raise ValueError("Actual local_v1 selection differs from the pre-endpoint plan")
        for label, (direction, _) in directions.items():
            if label == "gradient":
                continue
            for epsilon in EPSILONS:
                plus, minus = endpoint(point + epsilon * direction), endpoint(point - epsilon * direction)
                terms = {key: endpoint_row(point, direction, gradients[key], epsilon,
                    {"loss": plus["values"][key], "loss_dtype": plus["dtypes"][key]},
                    {"loss": minus["values"][key], "loss_dtype": minus["dtypes"][key]},
                    baseline["values"][key], rtol=config.diagnostics.fd_rtol, atol=config.diagnostics.fd_atol)
                    for key in ("contact", "binding", "combined", "repair_nll")}
                report["original_direction_candidate_rows"].append({"direction": label, "epsilon": epsilon,
                    "terms": terms, "plus": plus, "minus": minus})
                persist()
        report["original_direction_candidate_neighboring"] = {
            key: {label: neighboring_passes([row["terms"][key] for row in report["original_direction_candidate_rows"]
                if row["direction"] == label], minimum=config.diagnostics.minimum_fd_passes)
                for label in directions if label != "gradient"} for key in ("contact", "combined")}
        report["candidate_gate_passed"] = all(gate["passed"] for gate in report["candidate_gates"].values())
        report["candidate_original_failed_projection_passed"] = all(row["passed"] for row in
            report["original_direction_candidate_neighboring"]["contact"].values())
        report["candidate_validation_scope"] = "New-arithmetic contact and combined local_v1 only; no full component/repair/start/trajectory readiness certification."
        report["actual_scorer_forwards"] = len(cache) + PRIOR0_FIXED_FORWARDS
    if (config.dbmol.record() != original_profile or branch.manifest != original_manifest
            or helper._sha(objective_path) != objective_sha):
        raise ValueError("Candidate failed to restore original profile or objective source changed")
    branch.verify_context_unchanged()
    report.update(status="completed", diagnostic_completed=True, objective_context_restored=True,
        old_gate_certified_by_this_diagnostic=False, no_new_production_gradient_policy=True)
    persist()
    return report


def selected_cases(plan):
    cases = [case for case in plan["cases"] if case["id"] in CASE_IDS]
    if tuple(case["id"] for case in cases) != CASE_IDS or any(case["step"] != 20 for case in cases):
        raise ValueError("Require original prior-0 and prior-5 step20 cases")
    return cases


def run_case(plan, case, output_root):
    if case["id"] == "prior-5":
        isolated = direct_module("native_head_gradient_diagnostic.py", "_precision_isolated_head_child")
        return isolated.run_case(plan, case, output_root)
    head = head_helpers()
    helper = head._helper()
    reference, reference_asset = head.reference_replay(case, plan)
    helper.probe_case = lambda *args, **kwargs: probe_precision(
        *args, reference=reference, reference_asset=reference_asset, **kwargs)
    helper.REPEATS, helper.CASE_TIMEOUT_SECONDS, helper.CLAIM_LIMIT = 1, PRIOR0_TIMEOUT_SECONDS, CLAIM_LIMIT
    return helper.run_case(plan, case, output_root)


def child_complete(case_id, run, evidence):
    """Operational success never infers a passed scientific candidate gate."""
    common = (run.get("status") == "completed" and all(evidence.get(key) is True for key in
        ("diagnostic_completed", "frozen_state_unchanged", "parent_unchanged", "baseline_exact_source081_scalar_parity",
         "original_failed_gradient_exact_source081_match", "engineering_head_cut_complete")))
    if case_id == "prior-0":
        gates = evidence.get("candidate_gates", {})
        return (common and set(gates) == {"contact", "combined"}
            and all(gate.get("status") in {"passed", "failed", "failed_direction_pool"}
                    and gate.get("policy", {}).get("version") == "local_v1"
                    and type(gate.get("passed")) is bool
                    and gate["passed"] == (gate["status"] == "passed") for gate in gates.values())
            and type(evidence.get("candidate_gate_passed")) is bool
            and evidence["candidate_gate_passed"] == all(gate["passed"] for gate in gates.values())
            and all(evidence.get(key) is True for key in
                ("objective_context_restored", "candidate_native_seed_parity", "candidate_cached_contact_binding_exact_parity")))
    return common and evidence.get("isolated_head_status") == "completed" and evidence.get("engineering_isolated_head_sum_complete") is True


def run_study(path, sha):
    head = head_helpers()
    helper = head._helper()
    plan = helper.load_plan(path, sha)
    cases = selected_cases(plan)
    for case in cases:
        head.reference_replay(case, plan)
    from latentmol.experiments import ExperimentRun, write_json
    from latentmol.boltz.workflow_process import run_logged
    config = {"plan": plan, "plan_sha256": sha, "module_sha256": helper._sha(__file__),
        "candidate_objective_sha256": helper._sha(Path(__file__).with_name("dbmol_objective.py")),
        "isolated_head_module_sha256": helper._sha(Path(__file__).with_name("native_head_gradient_diagnostic.py")),
        "selected_cases": list(CASE_IDS), "maximum_scorer_forwards": 275,
        "excluded_cases": [case["id"] for case in plan["cases"] if case not in cases],
        "exclusion_reason": "Prior-3 original decoder kink is a separate source082 mechanism; no cure claimed for that case.",
        "case_timeout_seconds": {"prior-0": PRIOR0_TIMEOUT_SECONDS, "prior-5": 1800}, "claim_limit": CLAIM_LIMIT}
    with ExperimentRun(Path(plan["output_root"]), plan["name"], config, Path.cwd()) as run:
        rows = []
        for case in cases:
            execution = run_logged([sys.executable, str(Path(__file__).resolve()), "--plan", str(Path(path).resolve()),
                "--sha256", sha, "--child", case["id"], "--child-output", str(run.path / case["id"])],
                run.path / (case["id"] + ".log"), timeout_seconds=config["case_timeout_seconds"][case["id"]])
            evidence, complete = {}, False
            if not execution["timed_out"] and execution["returncode"] == 0 and len(execution["run_paths"]) == 1:
                child = Path(execution["run_paths"][0])
                evidence = helper._read(child / "attribution.json")
                complete = child_complete(case["id"], helper._read(child / "run.json"), evidence)
            rows.append({"case_id": case["id"], "diagnostic_completed": complete, "execution": execution,
                "candidate_gate_passed": evidence.get("candidate_gate_passed") if case["id"] == "prior-0" else None,
                "candidate_original_failed_projection_passed": evidence.get("candidate_original_failed_projection_passed"),
                "old_gate_certified": False})
            write_json(run.path / "cases.json", rows)
        run.annotate(diagnostic_cases_completed=sum(row["diagnostic_completed"] for row in rows),
            diagnostic_cases_requested=len(rows), old_gates_certified=False,
            candidate_gate_passed=rows[0]["candidate_gate_passed"])
        if not all(row["diagnostic_completed"] for row in rows):
            raise ValueError("Precision/head diagnostic operationally incomplete; all outcomes retained")
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
        raise ValueError("Precision/head validation requires an immutable DTU HPC allocation")
    plan = helper.load_plan(args.plan, args.sha256)
    cases = selected_cases(plan)
    if args.child is None:
        if args.child_output is not None:
            raise ValueError("Child output requires selected child")
        print(run_study(args.plan, args.sha256), flush=True)
    else:
        matches = [case for case in cases if case["id"] == args.child]
        if len(matches) != 1 or args.child_output is None:
            raise ValueError("Unknown precision/head child or missing output")
        print(run_case(plan, matches[0], args.child_output), flush=True)


if __name__ == "__main__":
    main()
