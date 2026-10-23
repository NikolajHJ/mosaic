"""HPC-only saved-state head-cut diagnostic; no production math is replaced.

Passive hooks observe the complete distogram/affinity/confidence-logit cut.
Original endpoints feed both the actual objective and a fixed-cotangent linear
readout. Cached FP64 scalar/contact controls explicitly change arithmetic; they
cannot restore precision lost upstream or certify historical failed gates.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import sys


HEADS = ("distogram", "affinity_pred_value", "affinity_logits_binary", "pae_logits", "pde_logits", "plddt_logits")
CASE_IDS = ("prior-0", "prior-5")
EPSILONS = (1e-5, 3e-5, 1e-4, 3e-4, .001, .003, .01, .03, .1)
CASE_TIMEOUT_SECONDS = 1800
SOURCE081_ARCHIVE = "c50513aaaa3d9a2a16bcf6baa3a751751c551f679242af9bb0a7aa16d0e9e477"
CLAIM_LIMIT = ("Passive native head-cut attribution, not optimization, a learned surrogate, a production fix, "
               "a tolerance change or historical gate certification. FP64 cached controls change downstream arithmetic only.")


def _helper():
    path = Path(__file__).with_name("failed_gate_attribution.py")
    spec = importlib.util.spec_from_file_location("_head_authenticated_loader", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def capture_heads(scorer):
    """Capture live outputs exactly once; hooks neither return nor replace data.

    Hooks remain installed through backward to detect unexpected head replay.
    Only Pairformer blocks are checkpointed in the authenticated implementation.
    The raw distogram also influences confidence PDE weighting, so contact-only
    cotangents cannot stand in for cotangents of the entire binding objective.
    """
    captured, counts, handles = {}, {}, []
    def hook(label, keys):
        def observe(module, args, output):
            counts[label] = counts.get(label, 0) + 1
            if counts[label] != 1:
                raise ValueError("Native head unexpectedly recomputed: " + label)
            values = {"distogram": output} if label == "distogram" else {key: output[key] for key in keys}
            captured.update(values)
        return observe
    modules = (("affinity", scorer.affinity, HEADS[1:3]),
               ("distogram", scorer.structure.distogram_module, ("distogram",)),
               ("confidence", scorer.structure.confidence_module.confidence_heads, HEADS[3:]))
    try:
        for label, module, keys in modules:
            handles.append(module.register_forward_hook(hook(label, keys)))
        yield captured
        if set(captured) != set(HEADS) or counts != {"affinity": 1, "distogram": 1, "confidence": 1}:
            raise ValueError("Incomplete native head capture")
    finally:
        for handle in handles:
            handle.remove()


def cut_vjp(value, heads, point):
    """Measure a complete non-ancestor cut without assuming bitwise chain parity."""
    import torch
    tensors = tuple(heads[key] for key in HEADS)
    if any(not tensor.requires_grad for tensor in tensors):
        raise ValueError("Detached tensor in native head cut")
    raw = torch.autograd.grad(value, tensors, retain_graph=True, allow_unused=True)
    cotangents = tuple(torch.zeros_like(tensor) if cotangent is None else cotangent.detach()
                       for tensor, cotangent in zip(tensors, raw))
    if any(not bool(value.isfinite().all()) for value in cotangents):
        raise ValueError("Nonfinite native head cotangent")
    gradient, = torch.autograd.grad(tensors, point, grad_outputs=cotangents, retain_graph=True)
    return gradient.detach(), dict(zip(HEADS, cotangents)), [key for key, value in zip(HEADS, raw) if value is None]


def fixed_readout(heads, base, cotangents):
    """Subtract observed FP32 tensors before widened reduction; no invented bits."""
    terms = {key: ((heads[key].double() - base[key].double()) * cotangents[key].double()).sum() for key in HEADS}
    return sum(terms.values()), {key: float(value.detach()) for key, value in terms.items()}


def native_bin_centers(logits):
    """Freeze the original native-dtype representatives before any widening."""
    import torch
    if logits.shape[-1] != 64:
        raise ValueError("Require the original 64-bin native distogram")
    boundaries = torch.linspace(2., 22., 63, device=logits.device, dtype=logits.dtype)
    half = (boundaries[1] - boundaries[0]) / 2
    return torch.cat((boundaries[:1] - half, (boundaries[:-1] + boundaries[1:]) / 2, boundaries[-1:] + half))


def cached_controls(outputs, heads, config, pocket, ligand, centers, repair, weight):
    """Declared changed arithmetic, sharing every native head/endpoint tensor.

    Assembly-only controls widen already-rounded primitive scalar outputs.
    Contact controls additionally reevaluate the contact reduction in FP64 using
    the original bin representatives/masks/cutoff/epsilon. Native confidence
    reductions remain unchanged: this is not an FP64 confidence/model test.
    """
    from latentmol.boltz.dbmol_objective import contact_loss, reduce_objective
    from latentmol.multimodel.engine import scalar_loss
    native_contact = contact_loss(heads["distogram"], pocket, ligand,
        cutoff=config.dbmol.distance_cutoff, epsilon=config.dbmol.epsilon, bin_centers=centers)["dbmol_contact_loss"]
    if float(native_contact.detach()) != float(outputs["dbmol_contact_loss"].detach()):
        raise ValueError("Cached native contact reconstruction changed the scalar")
    wide = {key: value.double() for key, value in outputs.items()}
    binding, _ = scalar_loss(reduce_objective(wide, config.dbmol), config.objective)
    contact = contact_loss(heads["distogram"].double(), pocket, ligand,
        cutoff=config.dbmol.distance_cutoff, epsilon=config.dbmol.epsilon, bin_centers=centers.double())["dbmol_contact_loss"]
    with_contact, _ = scalar_loss(reduce_objective({**wide, "dbmol_contact_loss": contact}, config.dbmol), config.objective)
    return {"binding_assembly_float64": binding, "combined_assembly_float64": binding + weight * repair.double(),
            "contact_float64": contact, "binding_contact_float64": with_contact,
            "combined_contact_float64": with_contact + weight * repair.double()}


def _forward(decoder, point, branch, config, counters, target, weight, purpose):
    from latentmol.graph_policy import graph_from_decoder
    from latentmol.multimodel.engine import scalar_loss
    from latentmol.repair_guidance.loss import fixed_target_nll
    decoded = decoder.decode(point)
    graph = graph_from_decoder(decoded, decoder.spec, temperature=config.temperature, fixed_active=branch.prepared.active)
    outputs = counters.call("boltz", purpose, "forward", lambda: branch(graph), point.device)
    binding, _ = scalar_loss(outputs, config.objective)
    repair, detail = fixed_target_nll(decoded, graph, target, decoder.spec,
        temperature=config.temperature, normalization=config.repair_guidance.normalization)
    terms = {"contact": outputs["dbmol_contact_loss"], "binding": binding, "repair_nll": repair,
             "combined": binding + weight * repair}
    return decoded, graph, outputs, terms, detail


def _gradient_record(gradient):
    from latentmol.multimodel.study import tensor_digest
    return {"sha256": tensor_digest({"gradient": gradient}), "dtype": str(gradient.dtype),
            "norm_float64": float(gradient.double().norm())}


def _difference(left, right):
    import torch
    delta = left.double() - right.double()
    return {"exact_equal": torch.equal(left, right), "max_absolute_difference": float(delta.abs().max()),
            "difference_norm_float64": float(delta.norm()), "left_norm_float64": float(left.double().norm()),
            "right_norm_float64": float(right.double().norm())}


def chain_completeness(direct, chained):
    """Tight disclosed engineering criterion; unrelated to historical FD gates."""
    import torch
    result = _difference(direct, chained)
    relative_bound = 64 * torch.finfo(direct.dtype).eps
    maximum_bound = relative_bound * max(float(direct.detach().double().abs().max()), 1e-12)
    norm_bound = relative_bound * max(result["left_norm_float64"], 1e-12)
    result.update(engineering_criterion="Both maximum and L2 chain error <=64 native-dtype eps times direct scale",
        maximum_absolute_bound=maximum_bound, norm_error_bound=norm_bound,
        engineering_chain_complete=result["max_absolute_difference"] <= maximum_bound and result["difference_norm_float64"] <= norm_bound,
        historical_fd_tolerance_changed=False)
    return result


def _metadata(heads):
    from latentmol.multimodel.study import tensor_digest
    return {key: {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device),
                  "sha256": tensor_digest({key: value}), "norm_float64": float(value.detach().double().norm())}
            for key, value in heads.items()}


def baseline_probe(decoder, point, branch, config, counters, target, weight):
    """One production forward; independently differentiate binding and total."""
    import torch
    from latentmol.boltz.dbmol_objective import pocket_token_mask
    probe = point.detach().clone().requires_grad_()
    pocket, ligand = pocket_token_mask(branch.prepared.features, config.dbmol)
    with torch.enable_grad(), capture_heads(branch.scorer) as heads:
        _, _, outputs, terms, detail = _forward(decoder, probe, branch, config, counters, target, weight, "head_cut_baseline")
        centers = native_bin_centers(heads["distogram"])
        controls = cached_controls(outputs, heads, config, pocket, ligand, centers, terms["repair_nll"], weight)
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
                cut_vjp(terms[key], heads, probe), point.device)
            metadata[key] = {"direct_vs_cut_chain": chain_completeness(gradients[key], chain[key]),
                "unused_cut_channels": unused, "cotangents": _metadata(cotangents[key]),
                "claim_limit": "Measured finite-precision chain comparison; cut membership alone is not proof of exact derivative equality."}
        base = {key: value.detach().clone() for key, value in heads.items()}
        combined_minus_repair = gradients["combined"] - weight * gradients["repair_nll"]
        report = {"head_tensors": _metadata(base), "cut_chain": metadata,
            "direct_binding_vs_total_minus_weighted_repair": _difference(gradients["binding"], combined_minus_repair),
            "values": {key: float(value.detach()) for key, value in {**terms, **controls}.items()},
            "dtypes": {key: str(value.dtype) for key, value in {**terms, **controls}.items()},
            "native_outputs": {key: float(value.detach()) for key, value in outputs.items()},
            "gradients": {key: _gradient_record(value) for key, value in gradients.items()},
            "contact_selection": deepcopy(branch.scorer.last_contact_selection), "target_sha256": detail["target_sha256"],
            "all_hooks_observational": True, "head_checkpoint_replays": 0}
    return report, gradients, base, cotangents, chain, (pocket, ligand, centers.detach())


def endpoint_probe(decoder, point, branch, config, counters, target, weight, base, cotangents, geometry):
    """One no-grad native endpoint supplies every comparison; release all heads."""
    import torch
    from latentmol.boltz.active_set import select_slots
    from latentmol.multimodel.study import tensor_digest
    with torch.no_grad(), capture_heads(branch.scorer) as heads:
        decoded, graph, outputs, terms, detail = _forward(decoder, point, branch, config, counters, target, weight, "head_cut_endpoint")
        controls = cached_controls(outputs, heads, config, *geometry, terms["repair_nll"], weight)
        readouts, head_contributions = {}, {}
        for key, cotangent in cotangents.items():
            readouts[key], head_contributions[key] = fixed_readout(heads, base, cotangent)
        values = {**terms, **controls, **{"head_readout_" + key: value for key, value in readouts.items()}}
        natural = select_slots(decoded, temperature=config.temperature, threshold=config.threshold, require_nonempty=False)
        result = {"values": {key: float(value) for key, value in values.items()},
            "dtypes": {key: str(value.dtype) for key, value in values.items()}, "head_contributions": head_contributions,
            "native_outputs": {key: float(value) for key, value in outputs.items()},
            "graph_sha256": tensor_digest({key: getattr(graph, key) for key in ("atoms", "charges", "bonds")}),
            "target_sha256": detail["target_sha256"], "contact_selection": deepcopy(branch.scorer.last_contact_selection),
            "natural_selector_crossed": not torch.equal(natural.active, branch.prepared.active)}
    return result


def reference_replay(case, plan):
    """Use the completed pinned source081 replay, never a newly chosen direction."""
    helper = _helper()
    paths = [Path(path) for path in case["parent_artifacts_sha256"] if Path(path).name == "attribution.json"]
    if len(paths) != 1:
        raise ValueError("Pin exactly one completed source081 attribution report per selected case")
    path = helper.authenticated_artifact(case, paths[0])
    reference = helper._read(path)
    run = helper._read(helper.authenticated_artifact(case, path.parent / "run.json"))
    configuration = run["config"]
    if (run.get("config_hash_version") != "canonical-json-config-v1"
            or run.get("config_sha256") != helper._digest(configuration)
            or run.get("source_archive_sha256") != SOURCE081_ARCHIVE
            or any(configuration.get(key) != plan[key] for key in
                   ("production_release", "production_archive_sha256", "production_manifest_sha256"))):
        raise ValueError("Source081 replay provenance differs from the original production plan")
    old_case = configuration["case"]
    if (any(old_case.get(key) != case[key] for key in ("id", "parent_run", "arm", "step"))
            or {key: value for key, value in case["parent_artifacts_sha256"].items()
                if key not in (str(path), str(path.parent / "run.json"))} != old_case["parent_artifacts_sha256"]):
        raise ValueError("Source081 replay changed the original case or authenticated asset set")
    if (run.get("status") != "completed" or reference.get("status") != "completed"
            or reference.get("frozen_state_unchanged") is not True or reference.get("parent_unchanged") is not True
            or reference.get("identities_before") != reference.get("identities_after")
            or reference.get("target") != configuration.get("target")
            or reference.get("original_item") != configuration.get("original_item")
            or reference.get("native_seed_parity") is not True
            or reference.get("controls", {}).get("native", {}).get("status") != "completed"):
        raise ValueError("Source081 reference is not a completed unchanged-state replay")
    return reference, {"path": str(path), "sha256": case["parent_artifacts_sha256"][str(path)]}


def probe_case(decoder, point, branch, config, counters, target, weight, item_name, item, *, reference,
               reference_asset, persist=lambda: None, report=None):
    from latentmol.boltz.local_fd import endpoint_row, neighboring_passes
    helper = _helper()
    report = report if report is not None else {}
    report.update(status="probing", claim_limit=CLAIM_LIMIT, head_module_sha256=helper._sha(__file__),
        reference_replay=reference_asset, epsilons=list(EPSILONS), finite_differences=[], head_cut=list(HEADS),
        cached_controls_scope="FP64 scalar assembly/contact only; all original model forwards are unchanged.")
    persist()
    baseline, gradients, base, cotangents, chain, geometry = baseline_probe(decoder, point, branch, config, counters, target, weight)
    report["baseline"] = baseline
    report["engineering_head_cut_complete"] = all(
        value["direct_vs_cut_chain"]["engineering_chain_complete"] for value in baseline["cut_chain"].values())
    report["head_cut_status"] = "complete" if report["engineering_head_cut_complete"] else "incomplete"
    persist()
    if not report["engineering_head_cut_complete"]:
        report["causal_interpretation"] = "Unavailable: the observed cut does not reproduce the direct binding/contact derivative."
        persist()
        raise ValueError("Incomplete head-cut chain; baseline retained and no localization claimed")
    relevant = "combined" if item_name == "combined" else "contact"
    reference_gradient = reference["controls"]["native"]["gradients"]["combined" if item_name == "combined" else "dbmol_contact_loss"]
    direct_gradient = baseline["gradients"][relevant]
    if (direct_gradient["sha256"] != reference_gradient["sha256"]
            or direct_gradient["norm_float64"] != reference_gradient["norm"]):
        raise ValueError("Original failed derivative magnitude/hash differs from source081")
    report["original_failed_gradient_exact_source081_match"] = True
    directions = helper.reconstruct_directions(point, gradients[relevant], item)
    if set(directions) != set(reference["direction_replay"]):
        raise ValueError("Direction membership differs from prespecified source081 replay")
    report["direction_replay"] = {key: metadata for key, (_, metadata) in directions.items()}
    for key, (_, metadata) in directions.items():
        if (key != "gradient" and metadata["hash_matches_original"] is not True
                or metadata["replayed_sha256"] != reference["direction_replay"][key]["replayed_sha256"]):
            raise ValueError("Original/source081 direction identity differs: " + key)
    if report["identities_before"] != reference["identities_before"]:
        raise ValueError("Frozen point/context/module identities differ from source081")
    original_values = reference["controls"]["native"]["baseline"]
    if (baseline["values"]["combined"] != original_values["combined"]
            or any(value != original_values[key] for key, value in baseline["native_outputs"].items())):
        raise ValueError("Passive head capture changed source081 native scalar values")
    report["baseline_exact_source081_scalar_parity"] = True
    derivatives = {**gradients, **{"head_readout_" + key: value for key, value in chain.items()}}
    values = {**baseline["values"], "head_readout_contact": 0., "head_readout_binding": 0.}
    persist()
    for label, (direction, _) in directions.items():
        for epsilon in EPSILONS:
            plus = endpoint_probe(decoder, point + epsilon * direction, branch, config, counters, target, weight, base, cotangents, geometry)
            minus = endpoint_probe(decoder, point - epsilon * direction, branch, config, counters, target, weight, base, cotangents, geometry)
            if plus["target_sha256"] != baseline["target_sha256"] or minus["target_sha256"] != baseline["target_sha256"]:
                raise ValueError("Held repair target changed at an endpoint")
            rows = {key: endpoint_row(point, direction, derivative, epsilon,
                {"loss": plus["values"][key], "loss_dtype": plus["dtypes"][key], "natural_selector_crossed": plus["natural_selector_crossed"]},
                {"loss": minus["values"][key], "loss_dtype": minus["dtypes"][key], "natural_selector_crossed": minus["natural_selector_crossed"]},
                values[key], rtol=config.diagnostics.fd_rtol, atol=config.diagnostics.fd_atol) for key, derivative in derivatives.items()}
            contact_changed = any(plus["contact_selection"].get(key) != minus["contact_selection"].get(key)
                                  for key in ("anchor_tokens", "best_ligand_token_per_pocket"))
            remainder = {key: {sign: endpoint["values"][key] - values[key] - endpoint["values"]["head_readout_" + key]
                              for sign, endpoint in (("plus", plus), ("minus", minus))} for key in ("contact", "binding")}
            report["finite_differences"].append({"direction": label, "epsilon": epsilon, "terms": rows,
                "contact_winners_changed": contact_changed, "native_minus_fixed_head_first_order": remainder,
                "plus": plus, "minus": minus})
            persist()
    report["neighboring_pass_diagnostics"] = {key: {label: neighboring_passes(
        [row["terms"][key] for row in report["finite_differences"] if row["direction"] == label],
        minimum=item["policy"]["minimum_neighboring_passes"]) for label in directions} for key in derivatives}
    report.update(status="completed", diagnostic_completed=True, old_gate_certified_by_this_diagnostic=False,
                  no_new_production_gradient_policy=True)
    persist()
    return report


def run_case(plan, case, output_root):
    helper = _helper()
    reference, reference_asset = reference_replay(case, plan)
    helper.probe_case = lambda *args, **kwargs: probe_case(*args, reference=reference, reference_asset=reference_asset, **kwargs)
    helper.REPEATS, helper.CASE_TIMEOUT_SECONDS, helper.CLAIM_LIMIT = 1, CASE_TIMEOUT_SECONDS, CLAIM_LIMIT
    return helper.run_case(plan, case, output_root)


def selected_cases(plan):
    selected = [case for case in plan["cases"] if case["id"] in CASE_IDS]
    if tuple(case["id"] for case in selected) != CASE_IDS or any(case["step"] != 20 for case in selected):
        raise ValueError("Require prespecified prior-0/prior-5 saved step20 cases")
    return selected


def run_study(path, sha):
    helper = _helper()
    plan = helper.load_plan(path, sha)
    cases = selected_cases(plan)
    from latentmol.experiments import ExperimentRun, write_json
    from latentmol.boltz.workflow_process import run_logged
    calls = 0
    for case in cases:
        _, item = helper.failed_item(helper._read(helper.authenticated_artifact(case,
            Path(case["parent_run"]) / case["arm"] / f"gate-{case['step']:04d}.json")))
        randoms = sum(label != "gradient" and row.get("passed") is False for label, row in item["direction_results"].items())
        calls += 3 + (1 + randoms) * len(EPSILONS) * 2
        reference_replay(case, plan)
    if calls > 116:
        raise ValueError("Head-cut plan exceeds 116 native scorer forwards")
    config = {"plan": plan, "plan_sha256": sha, "module_sha256": helper._sha(__file__),
        "loader_sha256": helper._sha(Path(__file__).with_name("failed_gate_attribution.py")),
        "selected_cases": list(CASE_IDS), "excluded_cases": [case["id"] for case in plan["cases"] if case not in cases],
        "exclusion_reason": "Prior-3 decoder-local discrepancy is covered by the separate source082 diagnostic.",
        "maximum_scorer_forwards": calls, "case_timeout_seconds": CASE_TIMEOUT_SECONDS, "claim_limit": CLAIM_LIMIT}
    with ExperimentRun(Path(plan["output_root"]), plan["name"], config, Path.cwd()) as run:
        rows = []
        for case in cases:
            execution = run_logged([sys.executable, str(Path(__file__).resolve()), "--plan", str(Path(path).resolve()),
                "--sha256", sha, "--child", case["id"], "--child-output", str(run.path / case["id"])],
                run.path / f"{case['id']}.log", timeout_seconds=CASE_TIMEOUT_SECONDS)
            complete = False
            if not execution["timed_out"] and execution["returncode"] == 0 and len(execution["run_paths"]) == 1:
                child = Path(execution["run_paths"][0])
                evidence = helper._read(child / "attribution.json")
                complete = (helper._read(child / "run.json")["status"] == "completed" and evidence.get("diagnostic_completed") is True
                    and evidence.get("frozen_state_unchanged") is True and evidence.get("parent_unchanged") is True
                    and evidence.get("baseline_exact_source081_scalar_parity") is True
                    and evidence.get("engineering_head_cut_complete") is True
                    and evidence.get("original_failed_gradient_exact_source081_match") is True)
            rows.append({"case_id": case["id"], "diagnostic_completed": complete, "execution": execution, "old_gate_certified": False})
            write_json(run.path / "cases.json", rows)
        run.annotate(diagnostic_cases_completed=sum(row["diagnostic_completed"] for row in rows),
                     diagnostic_cases_requested=len(rows), old_gates_certified=False)
        if not all(row["diagnostic_completed"] for row in rows):
            raise ValueError("Head diagnostic incomplete; every outcome is retained in cases.json")
        return run.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--child")
    parser.add_argument("--child-output", type=Path)
    args = parser.parse_args()
    helper = _helper()
    if not os.environ.get("LSB_JOBID") or not helper._hash(os.environ.get("LATENTMOL_SOURCE_SHA256")):
        raise ValueError("Native head-cut probes require an immutable DTU HPC allocation")
    plan = helper.load_plan(args.plan, args.sha256)
    cases = selected_cases(plan)
    if args.child is None:
        if args.child_output is not None:
            raise ValueError("Child output requires a selected child")
        print(run_study(args.plan, args.sha256), flush=True)
    else:
        matches = [case for case in cases if case["id"] == args.child]
        if len(matches) != 1 or args.child_output is None:
            raise ValueError("Unknown selected saved case or missing child output")
        print(run_case(plan, matches[0], args.child_output), flush=True)


if __name__ == "__main__":
    main()
