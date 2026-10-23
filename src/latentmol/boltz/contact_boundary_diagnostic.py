"""Allocated exact-state localization of the remaining structural contact secant.

The shared molecule, frozen native models, pose, pocket, original directions
and tolerances remain unchanged. Contact-only traversals must reproduce the
original full DBMol contact values and derivatives before supplying evidence.
Identity-cloned complete cuts distinguish representation, atom embedding and
structural trunk intervals. Baseline-held input ReLU signs and a widened final
linear projection are explicit diagnostic counterfactuals, never production
patches or approval of a historical failed optimization gate.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import sys

CASE_TIMEOUT_SECONDS = 3450
CLAIM_LIMIT = __doc__
SOURCE102_ARCHIVE = "e16455777ade2c432944c57560995d577d1ce4a5f36ba5bd6df1378a9d5fa6f8"
SOURCE102_MANIFEST = "7035768125fc2e49a1872ddf22ea95f514e4326791df8139e229cbfc2779f0dd"
CONTACT_KEYS = ("dbmol_contact_loss", "dbmol_contact_pocket_loss", "dbmol_contact_anchor_loss")
_STRUCTURAL = None
_CAPTURE = None
_CASE = None
_CALLS = 0
_LIMIT = 0


def adjacent(name):
    """Load observer bytes before the original scientific-source bootstrap."""
    spec = importlib.util.spec_from_file_location("_contact_" + name,
        Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def structural():
    global _STRUCTURAL
    if _STRUCTURAL is None:
        _STRUCTURAL = adjacent("structural_loss_diagnostic")
    return _STRUCTURAL


def capture_tools():
    global _CAPTURE
    if _CAPTURE is None:
        _CAPTURE = adjacent("contact_boundary_observer")
    return _CAPTURE


def source102_reference(case, report):
    """A completed structural observer must authenticate the same failed state."""
    replay = structural().tools().helper()
    paths = [Path(path) for path in case["parent_artifacts_sha256"]
             if "source-102-structural-loss-diagnostic-attempt001-replay" in path
             and Path(path).name == "attribution.json"]
    if len(paths) != 1:
        raise ValueError("Require one authenticated source102 case reference")
    path = replay.authenticated_artifact(case, paths[0])
    run_path = replay.authenticated_artifact(case, path.with_name("run.json"))
    previous, run = replay._read(path), replay._read(run_path)
    if (run.get("status") != "completed"
            or run.get("source_archive_sha256") != SOURCE102_ARCHIVE
            or run.get("source", {}).get("sha256") != SOURCE102_MANIFEST
            or any(previous.get(key) is not True for key in
                   ("diagnostic_completed", "frozen_state_unchanged", "parent_unchanged", "native_seed_parity"))
            or previous.get("identities_before") != report["identities_before"]
            or previous.get("identities_before") != previous.get("identities_after")
            or previous.get("target") != report["target"]
            or previous.get("original_item") != report["original_item"]):
        raise ValueError("Source102 is incomplete or represents another scientific state")
    return previous, {"path": str(path), "sha256": replay._sha(path),
        "run_sha256": replay._sha(run_path), "archive_sha256": SOURCE102_ARCHIVE,
        "manifest_sha256": SOURCE102_MANIFEST, "historical_gate_approval": False}


def cached_projection(model, pair, *, widened=False):
    """Replay the native affine distogram head; FP64 is an isolated control.

    Native symmetrization precedes the learned linear map. Widen both steps
    only in the declared alternative; original upstream tensors remain FP32.
    """
    import torch
    head = model.distogram_module
    layer = head.distogram
    if widened:
        pair = pair.double()
        logits = torch.nn.functional.linear(pair + pair.transpose(1, 2),
            layer.weight.double(), None if layer.bias is None else layer.bias.double())
    else:
        logits = layer(pair + pair.transpose(1, 2))
    return logits.reshape(pair.shape[0], pair.shape[1], pair.shape[2], head.num_distograms, head.num_bins)


def contact_forward(decoder, point, branch, config, counters, capture, label):
    """Original shared graph and adapter, followed by the native contact path."""
    global _CALLS
    import torch
    from latentmol.boltz.active_set import graph_from_decoder
    from latentmol.boltz.adapter import adapt_graph
    from latentmol.boltz.model import attached_trunk
    from latentmol.boltz.dbmol_objective import (CONTACT64_ARITHMETIC, contact_loss,
        native_bin_centers, pocket_token_mask)
    from latentmol.multimodel.study import tensor_digest
    from latentmol.representation import RelaxedGraph
    if _CALLS >= _LIMIT:
        raise ValueError("Contact-only native forward budget exceeded")
    _CALLS += 1
    graph = graph_from_decoder(decoder.decode(point), decoder.spec,
        fixed_active=branch.prepared.active, temperature=config.temperature)
    support = branch.assess(graph)
    if support.status == "unsupported" or (support.status == "lossy" and not branch.allow_lossy_charge):
        raise ValueError("Original branch rejects this shared graph")
    graph = RelaxedGraph(*(capture.save("graph." + name, getattr(graph, name))
                           for name in ("atoms", "charges", "bonds")))
    features, bonds = adapt_graph(graph, branch.prepared)
    features = {**features, **{key: capture.save("adapter." + key, features[key])
        for key in ("ref_element", "ref_charge", "token_bonds")}}
    bonds = capture.save("adapter.bond_types", bonds)
    def native():
        _, _, pair = attached_trunk(branch.scorer.structure, features, bonds, config.recycling_steps)
        return branch.scorer.structure.distogram_module(pair)
    logits = counters.call("boltz", label, "forward", native, point.device)
    capture.require_complete()
    centers = native_bin_centers(logits)
    pocket, ligand = pocket_token_mask(features, config.dbmol)
    if config.dbmol.objective_arithmetic != CONTACT64_ARITHMETIC:
        raise ValueError("Require the original source091 contact arithmetic")
    outputs = contact_loss(logits.double(), pocket, ligand,
        cutoff=config.dbmol.distance_cutoff, epsilon=config.dbmol.epsilon,
        bin_centers=centers.double())
    capture.contact_selection = contact_selection(logits, pocket, ligand, config.dbmol, centers)
    return outputs, tensor_digest({key: getattr(graph, key) for key in ("atoms", "charges", "bonds")}), (pocket, ligand, centers)


def contact_selection(logits, pocket, ligand, config, centers):
    """Detached winner/tie witnesses for native and both counterfactual paths."""
    pairs = structural().contact_pairs(logits, pocket, ligand, config, bin_centers=centers).detach()
    def winner(value):
        flat = value.reshape(-1)
        index = int(flat.argmin())
        ranked = flat.sort().values
        return {"index": index, "tie_indices": (flat == flat[index]).nonzero().flatten().tolist(),
            "best_vs_second_margin": float(ranked[1] - ranked[0]) if flat.numel() > 1 else None}
    return {"per_pocket": [winner(row) for row in pairs], "anchor": winner(pairs)}


def check_contact_endpoint(observed, graph_sha, reference, scope):
    if graph_sha != reference["graph_sha256"] or any(
            float(observed[key]) != reference["values"][key] for key in CONTACT_KEYS):
        raise ValueError("Original full-DBMol contact/graph parity failed: " + scope)


def budget(direction_counts):
    """Bound fresh graphs and distinguish native calls from recomputation."""
    if tuple(direction_counts) != (2, 4):
        raise ValueError("Require the original source091 two/four directions")
    # 1 native baseline, at most 10 complete-cut graphs, 1 held-sign baseline,
    # nine original scales, two signs, native and held at each endpoint.
    calls = [12 + 36 * count for count in direction_counts]
    return {"direction_counts": list(direction_counts), "maximum_cuts": 10,
        "contact_only_forwards_per_case": calls, "maximum_contact_only_forwards": sum(calls),
        "auxiliary_full_dbmol_forwards_per_case": 3,
        "maximum_physical_trunks_excluding_checkpoint_recomputation": sum(calls) + 12,
        "cut_upstream_reverse_traversals_per_case": 30,
        "held_branch_baseline_upstream_reverse_traversals_per_case": 1,
        "checkpoint_recompute_included_in_runtime_not_forward_count": True,
        "serial_child_seconds": CASE_TIMEOUT_SECONDS, "controller_headroom_seconds": 300,
        "optimizer_updates": 0, "new_defog_calls": 0}


def probe_case(decoder, point, branch, config, counters, target, weight, item_name, item,
               *, persist=lambda: None, report=None):
    global _CALLS, _LIMIT
    import torch
    from latentmol.boltz.active_set import score_at
    from latentmol.boltz.local_fd import EPSILONS, endpoint_row
    from latentmol.multimodel.study import tensor_digest
    report = report if report is not None else {}
    tools = structural().tools()
    boundary = tools.boundary(_CASE)
    reference, receipt = boundary.authenticated_reference(_CASE, report)
    structural_reference, structural_receipt = source102_reference(_CASE, report)
    forward_base, analytic_base = tools.reference_baselines(reference)
    if config.recycling_steps != 0:
        raise ValueError("Complete contact frontiers require zero recycling")
    components = {}
    measured, total = score_at(decoder, point, branch, config, counters,
        "contact_original_full_analytic", gradients=True, component_gradients=components,
        repair_target=target, repair_weight=weight)
    if (measured["components"] != {key: analytic_base["values"][key] for key in measured["components"]}
            or measured["loss"] != analytic_base["values"]["combined"]
            or tensor_digest({"gradient": total}) != analytic_base["sum_gradient_identity"]["combined_gradient_sha256"]):
        raise ValueError("Original full scalar/backward differs from source093")
    directions = tools.helper().reconstruct_directions(point,
        total if item_name == "combined" else components[item_name], item)
    if any(meta["replayed_sha256"] != reference["direction_replay"][name]["replayed_sha256"]
           for name, (_, meta) in directions.items()):
        raise ValueError("Original failed direction identity differs")
    direct = components["dbmol_contact_loss"].detach().clone()
    del total, components
    _CALLS, _LIMIT = 0, 12 + 36 * len(directions)
    report.update(status="probing", diagnostic_completed=False, scope=CLAIM_LIMIT,
        reference093=receipt, reference102=structural_receipt,
        native_provenance=structural().structural_provenance(_CASE, branch.scorer.structure),
        original_gate_not_reclassified=True,
        original_gate_scope=structural_reference["original_gate_scope"],
        direction_replay={key: metadata for key, (_, metadata) in directions.items()},
        cuts={}, finite_differences=[], optimizer_updates=0, new_defog_calls=0)

    def base_capture():
        with torch.no_grad(), capture_tools().ContactCapture(branch.scorer.structure) as capture:
            outputs, graph_sha, geometry = contact_forward(decoder, point, branch, config, counters,
                capture, "contact_boundary_baseline")
            check_contact_endpoint(outputs, graph_sha, forward_base, "baseline")
            return {"values": {name: value.detach().clone() for name, value in capture.values.items()},
                "relus": {name: value.detach().clone() for name, value in capture.relus.items()},
                "masks": capture.relu_masks(), "cuts": deepcopy(capture.cuts),
                "topology": deepcopy(capture.topology), "geometry": tuple(x.detach().clone() for x in geometry),
                "outputs": {key: value.detach().clone() for key, value in outputs.items()},
                "contact_selection": deepcopy(capture.contact_selection)}
    base = boundary.phase("contact_native_boundaries", base_capture, point.device, report, persist)
    cuts = base["cuts"]
    if len(cuts) > 10:
        raise ValueError("Declared ten-cut workload exceeded")
    report.update(boundary_order=list(cuts), native_stack_topology=base["topology"],
        boundary_tensors=boundary.tensor_metadata(base["values"]),
        baseline_relu_tensors=boundary.tensor_metadata(base["relus"]))
    cotangents, chains = {}, {}
    for cut_name, members in cuts.items():
        def cut_probe():
            probe = point.detach().clone().requires_grad_()
            with torch.enable_grad(), capture_tools().ContactCapture(branch.scorer.structure, telemetry=False) as capture:
                outputs, graph_sha, _ = contact_forward(decoder, probe, branch, config, counters,
                    capture, "contact_cut:" + cut_name)
                boundary.same_values(capture.values, base["values"], cut_name)
                check_contact_endpoint(outputs, graph_sha, forward_base, cut_name)
                scalar = outputs["dbmol_contact_loss"]
                gradient, = counters.call("boltz", "contact_direct:" + cut_name, "backward",
                    lambda: torch.autograd.grad(scalar, probe, retain_graph=True), point.device)
                clone_parity = boundary.chain_completeness(direct, gradient.detach())
                fields = tuple(capture.values[name] for name in members)
                raw = counters.call("boltz", "contact_cotangent:" + cut_name, "backward",
                    lambda: torch.autograd.grad(scalar, fields, retain_graph=True, allow_unused=True), point.device)
                fixed = tuple(torch.zeros_like(value) if grad is None else grad.detach()
                              for value, grad in zip(fields, raw))
                chain, = counters.call("boltz", "contact_chain:" + cut_name, "backward",
                    lambda: torch.autograd.grad(fields, probe, grad_outputs=fixed), point.device)
                completeness = boundary.chain_completeness(direct, chain.detach())
                if not clone_parity["engineering_chain_complete"] or not completeness["engineering_chain_complete"]:
                    raise ValueError("Incomplete/contact-changing frontier: " + cut_name)
                return chain.detach(), dict(zip(members, fixed)), {
                    "members": members, "unused_members": [name for name, grad in zip(members, raw) if grad is None],
                    "identity_clone_backward_parity": clone_parity, "direct_vs_chain": completeness,
                    "chain_sha256": tensor_digest({"gradient": chain}),
                    "cotangents": boundary.tensor_metadata(dict(zip(members, fixed)))}
        chains[cut_name], cotangents[cut_name], report["cuts"][cut_name] = boundary.phase(
            "contact_cut:" + cut_name, cut_probe, point.device, report, persist)

    def held_baseline():
        probe = point.detach().clone().requires_grad_()
        with torch.enable_grad(), capture_tools().ContactCapture(branch.scorer.structure,
                held_masks=base["masks"]) as capture:
            outputs, graph_sha, _ = contact_forward(decoder, probe, branch, config, counters,
                capture, "contact_held_relu_baseline")
            check_contact_endpoint(outputs, graph_sha, forward_base, "held baseline")
            grad, = counters.call("boltz", "contact_held_baseline", "backward",
                lambda: torch.autograd.grad(outputs["dbmol_contact_loss"], probe), point.device)
            parity = boundary.chain_completeness(direct, grad.detach())
            if not parity["engineering_chain_complete"]:
                raise ValueError("Held baseline ReLU derivative differs")
            return parity
    report["held_relu_baseline_parity"] = boundary.phase("contact_held_relu_baseline",
        held_baseline, point.device, report, persist)
    last_pair = "trunk." + str(base["topology"]["trunk_block_indices"][-1]) + ".z"
    pocket, ligand, centers = base["geometry"]

    def endpoint(value, label, *, held=False):
        from latentmol.boltz.dbmol_objective import contact_loss
        with torch.no_grad(), capture_tools().ContactCapture(branch.scorer.structure,
                held_masks=base["masks"] if held else None) as capture:
            outputs, graph_sha, _ = contact_forward(decoder, value, branch, config, counters, capture, label)
            readouts = {name: float(sum(((capture.values[key].double() - base["values"][key].double()) * fixed.double()).sum()
                for key, fixed in weights.items())) for name, weights in cotangents.items()}
            native_head = cached_projection(branch.scorer.structure, capture.values[last_pair])
            if not torch.equal(native_head, capture.values["distogram"]):
                raise ValueError("Cached final affine head differs from native output")
            wide_head = cached_projection(branch.scorer.structure, capture.values[last_pair], widened=True)
            wide = contact_loss(wide_head, pocket, ligand, cutoff=config.dbmol.distance_cutoff,
                epsilon=config.dbmol.epsilon, bin_centers=centers.double())
            return {"values": {key: float(outputs[key]) for key in CONTACT_KEYS}, "graph_sha256": graph_sha,
                "readouts": readouts, "relu_crossings": capture_tools().relu_crossings(base["relus"], capture.relus),
                "widened_final_affine_contact": {key: float(wide[key]) for key in CONTACT_KEYS},
                "contact_selection": deepcopy(capture.contact_selection),
                "widened_final_affine_selection": contact_selection(wide_head, pocket, ligand, config.dbmol, centers),
                "held_input_relu_counterfactual": held, "native_upstream_precision": "float32",
                "held_activation_scope": "Input embedder only; excludes decoder, templates and native smooth trunk nonlinearities",
                "structure_use_templates": bool(branch.scorer.structure.use_templates)}

    for name, (direction, _) in directions.items():
        for epsilon in EPSILONS:
            prior = [row for row in reference["finite_differences"] if row["direction"] == name
                and row["epsilon"] == epsilon and row["grad_enabled"] is False]
            if len(prior) != 1:
                raise ValueError("Missing original source093 endpoint row")
            native, held = {}, {}
            for sign, multiplier in (("plus", 1), ("minus", -1)):
                candidate = point + multiplier * epsilon * direction
                native[sign] = boundary.phase("contact_endpoint:" + name + ":" + str(epsilon) + ":" + sign,
                    lambda: endpoint(candidate, "contact_endpoint_native"), point.device, report, persist)
                check_contact_endpoint(native[sign]["values"], native[sign]["graph_sha256"], prior[0][sign], sign)
                held[sign] = boundary.phase("contact_held_endpoint:" + name + ":" + str(epsilon) + ":" + sign,
                    lambda: endpoint(candidate, "contact_endpoint_held", held=True), point.device, report, persist)
                if held[sign]["graph_sha256"] != native[sign]["graph_sha256"]:
                    raise ValueError("Held activation experiment changed the shared graph")
            scalar = endpoint_row(point, direction, direct, epsilon,
                *[{"loss": native[sign]["values"]["dbmol_contact_loss"], "loss_dtype": "torch.float64"}
                  for sign in ("plus", "minus")], float(base["outputs"]["dbmol_contact_loss"]),
                rtol=config.diagnostics.fd_rtol, atol=config.diagnostics.fd_atol)
            cut_rows = {}
            for cut_name in cuts:
                pair = [{"loss": native[sign]["readouts"][cut_name], "loss_dtype": "torch.float64"}
                        for sign in ("plus", "minus")]
                original = endpoint_row(point, direction, direct, epsilon, *pair, 0.,
                    rtol=config.diagnostics.fd_rtol, atol=config.diagnostics.fd_atol)
                chain = endpoint_row(point, direction, chains[cut_name], epsilon, *pair, 0.,
                    rtol=config.diagnostics.fd_rtol, atol=config.diagnostics.fd_atol)
                cut_rows[cut_name] = {"original_native_reference": original, "measured_chain_reference": chain,
                    "attribution": boundary.qualify_residual(original, chain)}
            report["finite_differences"].append({"direction": name, "epsilon": epsilon,
                "native_contact": scalar, "cuts": cut_rows,
                "baseline_contact_selection": base["contact_selection"],
                "held_input_relu_secant": (held["plus"]["values"]["dbmol_contact_loss"] - held["minus"]["values"]["dbmol_contact_loss"]) / (2 * epsilon),
                "widened_final_affine_secant": (native["plus"]["widened_final_affine_contact"]["dbmol_contact_loss"]
                    - native["minus"]["widened_final_affine_contact"]["dbmol_contact_loss"]) / (2 * epsilon),
                "plus": native["plus"], "minus": native["minus"], "held_plus": held["plus"], "held_minus": held["minus"],
                "source093_endpoint_parity": True, "ordered_localization": first_interval(cut_rows, boundary)})
            persist()
    report.update(status="completed", diagnostic_completed=True, old_gate_certified_by_this_diagnostic=False,
        contact_only_forwards_used=_CALLS, contact_only_forwards_bound=_LIMIT,
        interpretation="Complete-cut residuals and held-input-ReLU/linear-head controls localize causes at saved states; they neither alter production nor approve historical gates.")
    persist()
    return report


def first_interval(rows, boundary):
    """Reuse cut qualification while naming the actual contact branch."""
    result = boundary.first_interval(rows)
    if result["status"] == "all_cut_readouts_pass":
        result["remaining_region"] = "distogram_to_contact_reduction"
    return result


def run_case(plan, case, output_root):
    global _CASE
    _CASE = case
    replay = structural().tools().helper()
    original = replay.probe_case
    replay.probe_case = probe_case
    replay.CASE_TIMEOUT_SECONDS, replay.REPEATS, replay.CLAIM_LIMIT = CASE_TIMEOUT_SECONDS, 1, CLAIM_LIMIT
    try:
        return replay.run_case(plan, case, output_root)
    finally:
        replay.probe_case = original


def validate_cases(plan):
    structural().validate_cases(plan)
    for case in plan["cases"]:
        paths = [Path(path) for path in case["parent_artifacts_sha256"]
                 if "source-102-structural-loss-diagnostic-attempt001-replay" in path
                 and Path(path).name == "attribution.json"]
        if len(paths) != 1 or str(paths[0].with_name("run.json")) not in case["parent_artifacts_sha256"]:
            raise ValueError("Require the completed source102 exact-state references")
    return plan


def run_study(path, expected):
    plan = validate_cases(structural().tools().helper().load_plan(path, expected))
    from latentmol.experiments import ExperimentRun, write_json
    from latentmol.boltz.workflow_process import run_logged
    with ExperimentRun(Path(plan["output_root"]), plan["name"], {"plan": plan, "plan_sha256": expected,
            "diagnostic_module_sha256": structural().tools().helper()._sha(__file__),
            "budget": budget((2, 4)), "claim_limit": CLAIM_LIMIT}, Path.cwd()) as run:
        rows = []
        for case in plan["cases"]:
            execution = run_logged([sys.executable, str(Path(__file__).resolve()), "--plan", str(Path(path).resolve()),
                "--sha256", expected, "--child", case["id"], "--child-output", str(run.path / case["id"])],
                run.path / (case["id"] + ".log"), timeout_seconds=CASE_TIMEOUT_SECONDS)
            completed, reason = structural().tools().boundary(case).execution_completion(execution)
            rows.append({"case_id": case["id"], "execution": execution, "diagnostic_completed": completed,
                "incomplete_reason": reason, "old_gate_certified": False})
            write_json(run.path / "cases.json", rows)
        run.annotate(diagnostic_cases_completed=sum(row["diagnostic_completed"] for row in rows),
            diagnostic_cases_requested=2, old_gates_certified=False)
        print(run.path, flush=True)
        structural().tools().boundary(plan["cases"][0]).require_completed_cases(rows)
        return run.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--child", help=argparse.SUPPRESS)
    parser.add_argument("--child-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID") or not structural().tools().helper()._hash(os.environ.get("LATENTMOL_SOURCE_SHA256")):
        raise ValueError("Contact diagnostic requires immutable allocated DTU HPC execution")
    plan = validate_cases(structural().tools().helper().load_plan(args.plan, args.sha256))
    if args.child is not None:
        selected = [case for case in plan["cases"] if case["id"] == args.child]
        if len(selected) != 1 or args.child_output is None:
            raise ValueError("Unknown child or missing unique output root")
        print(run_case(plan, selected[0], args.child_output), flush=True)
    else:
        if args.child_output is not None:
            raise ValueError("Child output is only valid for a selected child")
        print(run_study(args.plan, args.sha256), flush=True)


if __name__ == "__main__":
    main()
