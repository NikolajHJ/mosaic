"""Allocated native deletion equivalence and binary occupancy selection.

Z remains the element-mass mechanism. Negative absence/ghost findings do not
prevent its explicitly diagnostic selection probe, and never validate G.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

import torch

from latentmol.multimodel.external import file_sha256, tensor_tree_sha256, tree_to
from .native import load_frozen_tiny
from .selection import endpoint_pair_summary, select_scalar_step
from .study import (CountedForward, FD_POLICY, compare_parity_outputs,
    inject_occupancy, native_gates, selected_outputs, write_json)

PHASE = "protenix-deletion-selection-diagnostic-v1"
SELECTION_POLICY = {
    "version": "binary-occupancy-diagnostic-selection-v1",
    "starts": [0.2, 0.8], "updates": 8, "maximum_q_step": 0.1,
    "q_min": 0.01, "q_max": 0.99, "trial_count": 16,
    "acceptance": "best_trial_v2", "arms": ["gradient", "random_sign", "unchanged"],
    "case_seeds": [1729, 1730, 1731], "native_endpoint_repeats": 3,
    "diagnostic_Z_allowed_after_absence_failure": True,
}


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, allow_nan=False)+"\n")


def run_study(config, output):
    if (config.get("phase") != PHASE or config.get("selection_policy") != SELECTION_POLICY
            or config.get("finite_difference_policy") != FD_POLICY):
        raise ValueError("Require exact sealed deletion/selection policies")
    if not os.environ.get("LSB_JOBID", "").isdigit() or not torch.cuda.is_available():
        raise RuntimeError("Deletion study requires allocated DTU CUDA execution")
    if (config["max_model_calls"] != 3000 or config["internal_seconds"] != 3300
            or len(config["cases"]) != 3):
        raise ValueError("Unexpected staged-study budget or panel")
    started = time.monotonic()
    forward = None
    result = dict(version=PHASE, status="running", stage="loading", cases=[],
        jobid=os.environ["LSB_JOBID"], selection_policy=SELECTION_POLICY,
        G="not_implemented", variable_count_adapter_validated=False,
        optimization_scope="Direct single optional-halogen occupancy; no model weights or latent z",
        scope="Element-mass Z diagnostic, including failures of absence equivalence",
        additional_backward_calls=0)
    try:
        torch.set_num_threads(4)
        native = load_frozen_tiny(config["source_root"], config["source_sha256"],
            config["weights_path"], config["weights_sha256"], device="cuda", seed=17)
        state = native.snapshot_state()
        forward = CountedForward(native, 3000, started+3300)
        result["native_provenance"] = native.provenance
        result["stage"] = "fresh_C_Z_readiness"
        gates_dir = output/"readiness"
        gates_dir.mkdir()
        readiness = native_gates(config, gates_dir, native=native, forward=forward)
        if readiness["status"] != "native_parity_and_gradient_passed":
            raise RuntimeError("Fresh C/Z parity and derivative readiness failed")
        result["readiness_sha256"] = file_sha256(gates_dir/"native-gate-result.json")
        result["readiness_model_calls"] = forward.calls
        result["readiness_backward_calls"] = sum(len(c["derivatives"]) for c in readiness["cases"])

        def score(features, pairs, targets, label, *, attached=False):
            with torch.no_grad():
                values = selected_outputs(forward(features, attached=attached), pairs, targets)
            if not all(bool(torch.isfinite(v).all()) for v in values.values()):
                raise RuntimeError("Nonfinite native endpoint or ghost output")
            artifact = output/(label+f"-f{forward.calls:04d}.pt")
            torch.save({k:v.detach().cpu() for k,v in values.items()}, artifact)
            return values, dict(loss=float(values["loss"].cpu()), outputs=str(artifact))

        for case_index, item in enumerate(config["cases"]):
            result["stage"] = item["case_id"]+":deletion_references"
            if file_sha256(item["bundle_path"]) != item["bundle_sha256"]:
                raise ValueError("Prepared deletion bundle hash mismatch")
            bundle = torch.load(item["bundle_path"], map_location="cpu", weights_only=True)
            if bundle["case_id"] != item["case_id"]:
                raise ValueError("Prepared deletion identity mismatch")
            raw = bundle["features"]
            raw_hash = tensor_tree_sha256(raw)
            if raw_hash != bundle["metadata"]["native_feature_sha256"]:
                raise ValueError("Original full feature hash differs from preparation")
            pairs = bundle["pair_indices"].to("cuda")
            targets = bundle["target_bins"].to("cuda")
            optional, token = int(bundle["optional_atom"]), int(bundle["optional_token"])
            prepared = native.prepare_features(tree_to(raw, "cuda"))
            prepared_hash = tensor_tree_sha256(prepared)
            record = dict(case_id=item["case_id"], role=item["role"],
                bundle_sha256=item["bundle_sha256"], metadata=bundle["metadata"],
                endpoints={}, comparisons={}, ghosts={}, attribute_gradients={}, trajectories=[])
            result["cases"].append(record)
            references = {}
            fields = [
                ("full", "features", "pair_indices"),
                ("compact_deleted", "compact_features", "compact_pair_indices"),
                ("fresh_full", "fresh_full_features", "fresh_full_pair_indices"),
                ("fresh_deleted", "fresh_deleted_features", "fresh_pair_indices"),
            ]
            for label, feature_field, pair_field in fields:
                features = prepared if label == "full" else native.prepare_features(tree_to(bundle[feature_field], "cuda"))
                bound_hash = tensor_tree_sha256(features)
                mapped_pairs = bundle[pair_field].to("cuda")
                if mapped_pairs.shape != pairs.shape:
                    raise ValueError("Endpoint changed the retained-pair denominator")
                samples = []
                for repeat in range(3):
                    values, observation = score(features, mapped_pairs, targets,
                        item["case_id"]+f"-{label}-{repeat}")
                    samples.append(observation)
                    if repeat == 0:
                        references[label] = {k:v.detach() for k,v in values.items()}
                record["endpoints"][label] = dict(samples=samples,
                    feature_sha256=bound_hash, pair_sha256=tensor_tree_sha256(mapped_pairs))
                if tensor_tree_sha256(features) != bound_hash:
                    raise RuntimeError("Native endpoint context changed")
                del features
            record["compact_ranking"] = endpoint_pair_summary(
                [r["loss"] for r in record["endpoints"]["full"]["samples"]],
                [r["loss"] for r in record["endpoints"]["compact_deleted"]["samples"]])
            record["fresh_ranking"] = endpoint_pair_summary(
                [r["loss"] for r in record["endpoints"]["fresh_full"]["samples"]],
                [r["loss"] for r in record["endpoints"]["fresh_deleted"]["samples"]])
            for condition in ("C", "Z"):
                for q_value in (0., 1.):
                    features = inject_occupancy(prepared, optional, torch.tensor(q_value, device="cuda"), condition)
                    values, observation = score(features, pairs, targets,
                        item["case_id"]+f"-{condition}-q{q_value:g}", attached=True)
                    expected = "full" if q_value else "compact_deleted"
                    comparison = compare_parity_outputs(values, references[expected])
                    record["comparisons"][condition+f"_q{q_value:g}"] = dict(
                        native_reference=expected, observation=observation, **comparison)
                    if q_value == 1. and not comparison["passed"]:
                        raise RuntimeError("New stage occupied native parity failed")
                    if condition == "Z" and q_value == 0:
                        zero = {k:v.detach() for k,v in values.items()}
                        record["Z_zero_fresh_deleted"] = compare_parity_outputs(values, references["fresh_deleted"])

            result["stage"] = item["case_id"]+":ghost_interventions"
            for field in ("ref_element", "ref_charge", "ref_atom_name_chars", "ref_pos", "token_bonds"):
                changed = tree_to(raw, "cpu")
                tensor = changed[field]
                if field == "ref_element":
                    tensor[optional] = 0
                    tensor[optional,5] = 1
                    intervention = "Conditional element becomes carbon before zero mass"
                elif field == "ref_charge":
                    tensor[optional] += 1
                    intervention = "Conditional ghost charge increases by one"
                elif field == "ref_atom_name_chars":
                    tensor[optional] = tensor[optional].roll(1, dims=-1)
                    intervention = "Rotate the ghost name character one-hot categories"
                elif field == "ref_pos":
                    tensor[optional] += tensor.new_tensor([3., -2., 1.])
                    intervention = "Translate ghost reference position by (3,-2,1) Angstrom; rebuild caches"
                else:
                    tensor[token,:] = 0
                    tensor[:,token] = 0
                    intervention = "Remove all ghost incident native token-bond indicators"
                if tensor_tree_sha256(changed) == raw_hash:
                    raise RuntimeError("Declared ghost intervention did not change its input")
                ghost_prepared = native.prepare_features(tree_to(changed, "cuda"))
                ghost = inject_occupancy(ghost_prepared, optional, torch.tensor(0.,device="cuda"), "Z")
                values, observation = score(ghost, pairs, targets,
                    item["case_id"]+"-ghost-"+field, attached=True)
                record["ghosts"][field] = dict(intervention=intervention,
                    caches_rebuilt=True, observation=observation,
                    comparison=compare_parity_outputs(values, zero))
            record["ghost_capacity_and_order"] = "not_run; no G absence certificate"
            record["attribute_gradients"]["ref_pos"] = "not_validated: geometry caches are prepared under no_grad"
            for field in ("ref_element", "ref_charge", "ref_atom_name_chars", "token_bonds"):
                leaf = prepared[field].detach().float().clone().requires_grad_(True)
                attrs = dict(prepared, **{field:leaf})
                attrs = inject_occupancy(attrs, optional, torch.tensor(0.,device="cuda"), "Z")
                values = selected_outputs(forward(attrs, attached=True), pairs, targets)
                if not all(bool(torch.isfinite(v).all()) for v in values.values()):
                    raise RuntimeError("Nonfinite ghost-attribute selected outputs")
                result["additional_backward_calls"] += 1
                gradient, = torch.autograd.grad(values["loss"], leaf, allow_unused=True)
                if gradient is None:
                    record["attribute_gradients"][field] = dict(unused=True, norm=0.)
                    continue
                if not bool(torch.isfinite(gradient).all()):
                    raise RuntimeError("Nonfinite ghost-attribute derivative")
                selected = (torch.cat([gradient[token,:].flatten(),gradient[:,token].flatten()])
                    if field == "token_bonds" else gradient[optional])
                record["attribute_gradients"][field] = dict(unused=False,
                    norm=float(selected.double().norm().cpu()),
                    maximum_absolute=float(selected.abs().max().cpu()),
                    interpretation="Declared frozen-feature route; no full G certificate")
                torch.save(selected.detach().cpu(), output/(item["case_id"]+"-ghost-gradient-"+field+".pt"))
                del values, gradient, leaf, attrs, selected
            record["Z_absence_equivalent"] = record["comparisons"]["Z_q0"]["passed"]
            record["Z_ghost_invariant_on_tested_routes"] = all(r["comparison"]["passed"] for r in record["ghosts"].values())
            write_json(output/(item["case_id"]+"-deletion.json"), record)

            result["stage"] = item["case_id"]+":diagnostic_selection"
            def evaluate_q(q, *, gradient=False, label="selection"):
                variable = torch.tensor(q, device="cuda", dtype=torch.float32, requires_grad=gradient)
                with torch.set_grad_enabled(gradient):
                    attrs = inject_occupancy(prepared, optional, variable, "Z")
                    values = selected_outputs(forward(attrs, attached=True), pairs, targets)
                artifact = output/(item["case_id"]+f"-{label}-f{forward.calls:04d}.pt")
                torch.save({k:v.detach().cpu() for k,v in values.items()},artifact)
                loss = float(values["loss"].detach().cpu())
                if not all(bool(torch.isfinite(v).all()) for v in values.values()):
                    return dict(q=float(variable.detach().cpu()),loss=None,nonfinite=True,outputs=str(artifact))
                if gradient:
                    result["additional_backward_calls"] += 1
                derivative = float(torch.autograd.grad(values["loss"], variable)[0].detach().cpu()) if gradient else None
                if gradient and not math.isfinite(derivative):
                    raise RuntimeError("Nonfinite optimization baseline derivative")
                actual_q = float(variable.detach().cpu())
                present = actual_q > .5
                return dict(q=actual_q, loss=loss, gradient=derivative, outputs=str(artifact),
                    probabilities={"NO_ATOM":1-actual_q,"real_element":actual_q},
                    expected_atom_count=bundle["metadata"]["ligand_heavy_atoms"]-1+actual_q,
                    hard_atom_count=bundle["metadata"]["ligand_heavy_atoms"]-1+int(present),
                    hard_smiles=bundle["metadata"]["full_smiles" if present else "deleted_smiles"],
                    hard_validity="CPU-authenticated connected neutral endpoint",
                    compact_native_loss=record["compact_ranking"]["full_mean" if present else "deleted_mean"],
                    fresh_native_loss=record["fresh_ranking"]["full_mean" if present else "deleted_mean"],
                    compact_Bernoulli_analysis=(1-actual_q)*record["compact_ranking"]["deleted_mean"]+actual_q*record["compact_ranking"]["full_mean"],
                    fresh_Bernoulli_analysis=(1-actual_q)*record["fresh_ranking"]["deleted_mean"]+actual_q*record["fresh_ranking"]["full_mean"])

            for start_index, start in enumerate(SELECTION_POLICY["starts"]):
                for arm in SELECTION_POLICY["arms"]:
                    stream = random.Random(f"{SELECTION_POLICY['case_seeds'][case_index]}:{start_index}")
                    q = start
                    run = dict(arm=arm,start_q=start,status="running",states=[],steps=[],
                        applied=0,improving=0,forced_fallback=0,full_step_nonincrease=0,no_op=0,
                        starting_model_calls=forward.calls, random_stream=f"{SELECTION_POLICY['case_seeds'][case_index]}:{start_index}")
                    record["trajectories"].append(run)
                    baseline = evaluate_q(q,gradient=arm=="gradient",label=arm)
                    if baseline["loss"] is None:
                        raise RuntimeError("Nonfinite optimization start")
                    q = baseline["q"]
                    run["states"].append(baseline)
                    best = baseline
                    run["best_observed"] = best
                    for update in range(1,9):
                        if update > 1:
                            baseline = evaluate_q(q,gradient=arm=="gradient",label=arm)
                            if baseline["loss"] is None:
                                raise RuntimeError("Nonfinite optimization baseline")
                        direction = (-(1 if baseline["gradient"]>0 else -1) if baseline["gradient"] else 0) if arm=="gradient" else stream.choice([-1,1]) if arm=="random_sign" else 0
                        if baseline["loss"] < best["loss"]:
                            best = baseline
                            run["best_observed"] = best
                        if arm == "unchanged":
                            chosen = dict(q=q,loss=baseline["loss"],acceptance_kind="unchanged_control",
                                no_op=True,improving=False,trials=[],selected_index=None,payload=baseline)
                        else:
                            def trial_score(candidate):
                                nonlocal best
                                observed = evaluate_q(candidate,label=arm+"-trial")
                                if observed["loss"] is not None and observed["loss"] < best["loss"]:
                                    best = observed
                                    run["best_observed"] = best
                                return observed if observed["loss"] is not None else dict(observed,loss=float("nan"))
                            chosen = select_scalar_step(q, baseline["loss"], direction, trial_score,
                                persist=lambda trials: append_json(output/"selection-trials.jsonl",dict(
                                    case_id=item["case_id"],arm=arm,start_index=start_index,update=update,
                                    completed_trials=len(trials),trial=trials[-1])))
                        q = chosen["q"]
                        chosen_state = chosen.get("payload") or chosen["trials"][chosen["selected_index"]]["payload"]
                        run["states"].append(chosen_state)
                        step = dict(update=update,baseline=baseline,**chosen)
                        run["steps"].append(step)
                        run["applied"] += 1
                        run["improving"] += int(chosen["improving"])
                        run["forced_fallback"] += int(chosen["acceptance_kind"]=="best_finite_fallback")
                        run["full_step_nonincrease"] += int(chosen["acceptance_kind"]=="full_step_nonincrease")
                        run["no_op"] += int(chosen["no_op"])
                        append_json(output/"selection-updates.jsonl",dict(case_id=item["case_id"],arm=arm,start_index=start_index,**step))
                    final = run["states"][-1]
                    initial = run["states"][0]
                    run.update(status="completed_8_updates",best_observed=best,
                        model_calls=forward.calls-run["starting_model_calls"],
                        soft_loss_change=final["loss"]-initial["loss"],
                        hard_atom_count_change=final["hard_atom_count"]-initial["hard_atom_count"],
                        compact_native_change=final["compact_native_loss"]-initial["compact_native_loss"],
                        fresh_native_change=final["fresh_native_loss"]-initial["fresh_native_loss"])
            if tensor_tree_sha256(raw) != raw_hash or tensor_tree_sha256(prepared) != prepared_hash:
                raise RuntimeError("Bound optimization context changed")
            native.assert_unchanged(state)
            record["status"] = "diagnostic_completed_not_G_validated"
            record["frozen_context_and_model_unchanged"] = True
            write_json(output/(item["case_id"]+"-complete.json"), record)
        result.update(status="deletion_selection_diagnostic_completed",stage="complete",
            complete_trajectories=sum(len(c["trajectories"]) for c in result["cases"]))
    except BaseException as exc:
        result.update(status="failed_or_partial",error=str(exc),traceback=traceback.format_exc())
        raise
    finally:
        result.update(model_calls=forward.calls if forward else 0,
            elapsed_seconds=time.monotonic()-started,peak_cuda_memory_bytes=torch.cuda.max_memory_allocated())
        write_json(output/"deletion-selection-result.json",result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--config-sha256",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if file_sha256(args.config)!=args.config_sha256:
        raise ValueError("Execution configuration SHA differs")
    args.output.mkdir(parents=True,exist_ok=False)
    config=json.loads(Path(args.config).read_text())
    result=run_study(config,args.output)
    print(json.dumps({k:v for k,v in result.items() if k!="cases"},allow_nan=False))


if __name__ == "__main__":
    main()
