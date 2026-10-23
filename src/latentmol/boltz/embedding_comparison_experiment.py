"""Allocated matched mapping/relaxed trajectories with authenticated shared starts.

Original epoch-36 files are read-only inputs. No model optimizer is constructed;
only z changes. The independent source138 namespace preserves earlier mapping
and VAE campaigns. Every failed/partial stage remains an explicit outcome.
"""

import argparse
import gc
import json
import os
from pathlib import Path
import socket
import time
import traceback
from types import SimpleNamespace

from .embedding_experiment import (cpu_stage, load_native, load_vae, verified, write)


VERSION = "epoch36-matched-embedding-comparison-experiment-v1"
BRIDGE_SHA = "013834958d52f7aee999ca4c39496e14fc5333d69eb697d47433821f48290343"


def load_bridge(config, vae, bank, device):
    import torch
    from latentmol.train import state_digest
    from .embedding_bridge import HiddenEmbeddingBridge
    if config["bridge"]["sha256"] != BRIDGE_SHA:
        raise ValueError("Comparison must use the completed source137 mapping")
    saved = torch.load(verified(config["bridge"]), map_location=device, weights_only=True)
    if (saved["vae_checkpoint"] != vae.embedding_bridge_checkpoint_identity
            or saved["vae_state_sha256"] != state_digest(vae.state_dict())
            or saved["bank_metadata"] != bank.metadata()):
        raise ValueError("Mapping does not bind the same original VAE/native models")
    actual = bank.state_dict()
    if set(actual) != set(saved["bank_state"]) or any(
            not torch.equal(actual[key], saved["bank_state"][key]) for key in actual):
        raise ValueError("Mapping native-projection bank changed")
    dims = saved["bridge_configuration"]
    bridge = HiddenEmbeddingBridge(bank, **{key: dims[key] for key in
        ("atom_input_dim", "pair_input_dim", "charge_input_dim", "hidden_dim")}).to(device)
    bridge.load_state_dict(saved["bridge_state"], strict=True)
    return bridge.eval().requires_grad_(False)


def prepare_starts(config, output):
    """Choose the first distinct audit posterior-mean groups before any score."""
    import torch
    from latentmol.experiments import file_sha256
    from latentmol.representation import HardGraphBatch
    from latentmol.train import state_digest
    from .embedding_training import conditioned_hard
    from .active_set import ContextBank
    vae, checkpoint = load_vae(config, "cpu")
    del checkpoint
    manifest = json.loads(verified(config["start_data_manifest"]).read_text())
    samples_path = verified(config["start_samples"])
    if (manifest["samples_sha256"] != config["start_samples"]["sha256"]
            or manifest["vae_checkpoint"] != vae.embedding_bridge_checkpoint_identity
            or manifest["vae_state_sha256_after"] != state_digest(vae.state_dict())):
        raise ValueError("Starting samples do not belong to original frozen epoch36")
    samples = torch.load(samples_path, map_location="cpu", weights_only=True)
    rows, seen_groups, seen_identities = [], set(), set()
    for index, (split, origin, group, identity) in enumerate(zip(
            samples["split"], samples["origin"], samples["group"], samples["identity"], strict=True)):
        if (split != "audit" or origin != "posterior_mean" or group in seen_groups
                or identity in seen_identities):
            continue
        rows.append(index)
        seen_groups.add(group)
        seen_identities.add(identity)
        if len(rows) == config["start_count"]:
            break
    if len(rows) != config["start_count"]:
        raise ValueError("Insufficient distinct prescribed audit posterior means; no replacement policy")
    catalog_path = verified(config["context_manifest"])
    catalog = json.loads(catalog_path.read_text())
    counts = {int(samples["active"][index].sum()) for index in rows}
    contexts = {count: {**catalog["contexts"][str(count)],
        "path": str((catalog_path.parent / catalog["contexts"][str(count)]["path"]).resolve())}
        for count in counts}
    header = {key: catalog[key] for key in
              ("target_sequence_sha256", "msa_sha256", "fixed_msa_row_indices")}
    bank = ContextBank(SimpleNamespace(contexts=contexts, context_identity=header), vae.spec)
    cases, plain = [], []
    for ordinal, index in enumerate(rows, 1):
        z = samples["z"][index:index + 1].clone()
        active = samples["active"][index].clone()
        expected = HardGraphBatch(*(samples[key][index:index + 1].long() for key in ("atoms", "charges", "bonds")))
        with torch.no_grad():
            decoded = vae.decode(z)
            reproduced = conditioned_hard(decoded, active[None], vae.spec)
        if any(not torch.equal(a, b) for a, b in zip(
                (expected.atoms, expected.charges, expected.bonds),
                (reproduced.atoms, reproduced.charges, reproduced.bonds), strict=True)):
            raise ValueError("Stored starting hard molecule did not reproduce")
        if not torch.equal(decoded.harden(vae.spec).atoms != 0, active[None]):
            raise ValueError("Prescribed posterior mean's natural occupancy differs from stored root")
        count = int(active.sum())
        name = f"case-{ordinal:03d}"
        # Binding verifies native row correspondence without moving the pose.
        from .active_set import bind_slots
        bound = bind_slots(bank.templates[count], active)
        from .provenance import context_digest
        record = dict(case_id=name, sample_index=index, group=samples["group"][index],
            identity=samples["identity"][index], occupied_slots=count,
            active_slots=active.nonzero().flatten().tolist(), context_asset=contexts[count],
            bound_context_sha256=context_digest(bound), source_split="audit", origin="posterior_mean")
        cases.append(dict(record=record, z=z, active=active))
        plain.append(record)
    torch.save(dict(cases=cases, graph_spec=vae.spec.to_dict()), output / "starts.pt")
    report = dict(status="prepared_shared_starts_not_optimized", selection=config["start_selection"],
        cases=plain, starts_sha256=file_sha256(output / "starts.pt"),
        start_samples_sha256=config["start_samples"]["sha256"],
        context_manifest_sha256=config["context_manifest"]["sha256"],
        frozen_vae_state_sha256=state_digest(vae.state_dict()))
    write(output / "starts.json", report)
    return report


def thin_run(run):
    return {key: run[key] for key in ("start_id", "path", "status", "applied", "improving",
        "forced_fallback", "full_step_nonincrease", "no_op", "best_observed", "callback_calls")}


def native_members(case_id, comparison):
    """Predeclare all baselines, terminals and best applied states; never rank across paths."""
    members = []
    for index, run in enumerate(comparison["runs"]):
        checkpoints = {row["update"]: row for row in run["checkpoints"]}
        chosen = [("terminal", run["checkpoints"][-1]),
                  ("best", checkpoints[run["best_observed"]["update"]])]
        if index == 0:
            chosen.insert(0, ("baseline", checkpoints[0]))
        for state, row in chosen:
            classification = row["classifications"]["conditioned"]
            members.append(dict(case_id=case_id, arm="start" if state == "baseline" else run["path"],
                state=state, smiles=classification["smiles"] if classification["strict_teacher_valid"] else None,
                failure=None if classification["strict_teacher_valid"] else classification["reason"],
                latent_sha256=row["latent_sha256"], checkpoint_path=row["tensor_path"],
                checkpoint_sha256=row["tensor_sha256"], update=row["update"]))
    return members


def gpu_stage(config, output, deadline):
    import torch
    from latentmol.experiments import file_sha256
    from latentmol.optimization import StepControl
    from latentmol.train import state_digest
    from .active_set import bind_slots
    from .contracts import PreparedComplex
    from .embedding_validation import native_gate
    from .embedding_comparison import FrozenComparisonObjective, run_matched_optimization
    from .embedding_comparison_validation import learned_start_gate
    from .embedding_comparison_rescore import freeze_selection, load_target, run_followup
    cpu_root = output.parent / "cpu"
    cpu = json.loads((cpu_root / "result.json").read_text())
    if (cpu["status"] != "passed" or cpu["source_sha256"] != os.environ["LATENTMOL_SOURCE_SHA256"]
            or cpu["plan_sha256"] != config["_plan_sha256"]
            or cpu["junit"]["tests"] < 1 or any(cpu["junit"][key] for key in ("failures", "errors", "skipped"))
            or file_sha256(cpu_root / "cpu-tests.xml") != cpu["junit_sha256"]
            or file_sha256(cpu_root / "starts.pt") != cpu["starts"]["starts_sha256"]):
        raise ValueError("Matched GPU requires exact-release CPU pass and authenticated shared starts")
    if not torch.cuda.is_available():
        raise RuntimeError("No allocated usable CUDA device")
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    vae, checkpoint = load_vae(config, "cuda")
    del checkpoint
    before = state_digest(vae.state_dict())
    scorer, base, bank = load_native(config, "cuda")
    bridge = load_bridge(config, vae, bank, "cuda")
    payload = torch.load(cpu_root / "starts.pt", map_location="cpu", weights_only=True)
    from latentmol.representation import GraphSpec
    if GraphSpec.from_dict(payload["graph_spec"]) != vae.spec:
        raise ValueError("Shared start schema differs from frozen VAE")
    cases, gate_reports = [], []
    # All prescribed cases pass both gates before either trajectory is released.
    for item in payload["cases"]:
        record = item["record"]
        directory = output / record["case_id"]
        directory.mkdir(parents=True, exist_ok=False)
        template = PreparedComplex.load(verified(record["context_asset"]), "cpu")
        prepared = bind_slots(template, item["active"]).to("cuda")
        from .provenance import context_digest
        if context_digest(prepared) != record["bound_context_sha256"]:
            raise ValueError("GPU context differs from allocated shared-start preparation")
        z = item["z"].to("cuda")
        oracle = native_gate(scorer, prepared, bank, vae, directory / "native-oracle.json",
                             config["native_tolerances"], deadline=deadline)
        reports = {path: learned_start_gate(scorer, prepared, bridge, vae, z,
                   directory / (path + "-start-gate.json"), deadline, scoring_path=path)
                   for path in ("relaxed", "learned")}
        from latentmol.multimodel.study import tensor_digest
        objective = FrozenComparisonObjective(scorer, prepared, bridge, vae, bank=bank, deadline=deadline)
        production_binding = {}
        for path in ("relaxed", "learned"):
            evaluated = objective.evaluate(path, z, require_gradient=True)
            matched = (evaluated["scores"] == reports[path]["baseline_scores"]
                       and tensor_digest({"gradient": evaluated["gradient"]}) == reports[path]["production_gradient_sha256"])
            production_binding[path] = dict(passed=matched, gradient_sha256=tensor_digest({"gradient": evaluated["gradient"]}))
            write(directory / "production-binding.json", production_binding)
            if not matched:
                raise ValueError("Trajectory objective does not exactly match its start derivative certificate")
        del objective, evaluated
        cases.append((record, prepared, z, directory))
        gate_reports.append(dict(case_id=record["case_id"], native_oracle=oracle["status"],
                                  start_gates={path: value["status"] for path, value in reports.items()}))
        write(output / "start-gates.json", dict(status="gating_not_optimized", cases=gate_reports))
    members, summary = [], []
    for record, prepared, z, directory in cases:
        objective = FrozenComparisonObjective(scorer, prepared, bridge, vae, bank=bank, deadline=deadline)
        result = run_matched_optimization({record["case_id"]: z}, objective, directory / "optimization",
            **config["optimization"], control=StepControl.from_mapping(config["step_control"]), deadline=deadline)
        members.extend(native_members(record["case_id"], result))
        summary.append(dict(case_id=record["case_id"], start=record,
            status=result["status"], runs=[thin_run(run) for run in result["runs"]],
            actual_model_calls=result["actual_model_calls"],
            frozen_after=result["frozen_after"], full_record=str(directory / "optimization/comparison.json")))
        write(output / "case-summary.json", dict(status="optimization_cases_complete_native_pending", cases=summary))
        del objective
    if state_digest(vae.state_dict()) != before:
        raise RuntimeError("Original frozen epoch36 VAE changed")
    # Free the frozen two-trunk scorer before independent ordinary CLI predictions.
    del scorer, bridge, bank, base, cases, prepared, z, payload, vae
    gc.collect()
    torch.cuda.empty_cache()
    from scripts.hpc.embedding_operator import quota
    write(output / "native-followup-quota.json", quota())
    selection = freeze_selection(members, pose_seeds=tuple(config["fresh_native"]["pose_seeds"]),
                                 native_settings=config["fresh_native"]["native_settings"])
    target = load_target(verified(config["native_template"]),
        config["fresh_native"]["target_sequence_sha256"], config["fresh_native"]["msa_sha256"],
        template_sha256=config["native_template"]["sha256"])
    followup = run_followup(selection, target, Path(config["boltz_cache"]), output / "fresh-native",
        deadline=deadline, native_timeout_seconds=config["fresh_native"]["native_timeout_seconds"],
        audit_timeout_seconds=config["fresh_native"]["audit_timeout_seconds"])
    from .embedding_comparison_report import generate_report
    html_report = generate_report(output, summary, followup)
    partial = followup.get("status") == "partial_runtime_not_passed"
    return dict(status="optimization_completed_native_partial" if partial else "completed_matched_pilot",
        native_followup_status=followup.get("status"), frozen_vae_unchanged=True, cases=summary,
        html_report=str(html_report),
        native_followup=followup, start_gates=gate_reports,
        comparison_scope="three paired starts; fixed-pose DBMol optimization; fresh-pose ordinary native affinity confirmation",
        learned_mapping=config["bridge"], existing_vae_continuation="untouched",
        cpu_result_sha256=file_sha256(cpu_root / "result.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("cpu", "gpu"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID") or not os.environ.get("LATENTMOL_SOURCE_SHA256"):
        raise RuntimeError("Matched comparison numerical work requires a sealed LSF allocation")
    import torch
    from latentmol.experiments import file_sha256
    torch.set_num_threads(min(4, int(os.environ.get("LSB_DJOB_NUMPROC", "4"))))
    config = json.loads(args.plan.read_text())
    if config["version"] != VERSION:
        raise ValueError("Unknown matched comparison plan")
    config["_plan_sha256"] = file_sha256(args.plan)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + config["internal_seconds"][args.stage]
    provenance = dict(version=VERSION, stage=args.stage, job_id=os.environ["LSB_JOBID"],
        hostname=socket.gethostname(), plan_sha256=config["_plan_sha256"],
        source_sha256=os.environ["LATENTMOL_SOURCE_SHA256"], vae_checkpoint=config["vae"],
        learned_mapping=config["bridge"], existing_vae_continuation="untouched")
    write(args.output / "result.json", {**provenance, "status": "running_not_passed"})
    try:
        if args.stage == "cpu":
            result = cpu_stage(config, args.output, deadline)
            result["starts"] = prepare_starts(config, args.output)
        else:
            result = gpu_stage(config, args.output, deadline)
    except BaseException as error:
        write(args.output / "result.json", {**provenance, "status": "failed_not_passed",
            "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(),
            "elapsed_seconds": time.monotonic() - started})
        raise
    write(args.output / "result.json", {**provenance, **result, "elapsed_seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()
