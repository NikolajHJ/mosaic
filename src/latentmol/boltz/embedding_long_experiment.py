"""Independent allocated long-run stages with serial CUDA process ownership.

The coordinator stays CUDA-free while each isolated CUDA worker runs and exits.
Original VAE training and the previous native recovery remain read-only inputs.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
import traceback

VERSION = "epoch36-guarded-embedding-long-experiment-v1"
CASES = ("case-001", "case-002", "case-003")


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def asset(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def verify(value):
    path = Path(value["path"])
    if sha(path) != value["sha256"]:
        raise ValueError("Artifact hash mismatch: " + str(path))
    return path


def validate_config(config):
    if config["version"] != VERSION:
        raise ValueError("Unknown long-run plan")
    parent = json.loads(verify(config["parent_comparison"]).read_text())
    changed = {"version", "cpu_tests", "optimization", "internal_seconds", "vae_continuation"}
    added = {"parent_comparison", "long_campaign", "long_training"}
    if (set(config) != set(parent) | added or any(config[key] != value
            for key, value in parent.items() if key not in changed)
            or not set(parent["cpu_tests"]) <= set(config["cpu_tests"])):
        raise ValueError("Long run changed an inherited frozen scientific setting")
    if config["optimization"] != {**parent["optimization"], "steps": 10000}:
        raise ValueError("Long run changed inherited base optimizer settings")
    from .embedding_comparison_correction import verify_correction
    verify_correction(parent)
    campaign = config["long_campaign"]
    if (campaign["case_ids"] != list(CASES) or campaign["updates_per_case"] != 10000
            or campaign["observe_every"] != 10 or campaign["pilot_updates"] != 20
            or campaign["max_consecutive_noop"] != 25
            or config["long_training"]["calibration"] != dict(split="dev", quantile=.99,
                quantile_method="nearest_rank", floor=1e-8)):
        raise ValueError("Unrecognized guarded-long protocol")
    return parent


def frozen_identities(config):
    return {key: config[key]["sha256"] for key in ("vae", "affinity", "context_manifest", "bridge")} | {
        "structure": config["dbmol"]["confidence_checkpoint"]["sha256"],
        "starts": config["precision_correction"]["parent_starts"]["sha256"]}


def stage_result(campaign, stage, source, plan):
    path = Path(campaign) / stage / "result.json"
    value = json.loads(path.read_text())
    if value["status"] != "passed" or value["source_sha256"] != source or value["plan_sha256"] != plan:
        raise ValueError("Prerequisite is not this exact release's passed " + stage)
    for item in value["artifacts"].values():
        verify(item)
    return value


def cpu(config, output, deadline):
    import torch
    from .embedding_experiment import cpu_stage
    from .embedding_comparison_experiment import prepare_starts
    from .embedding_comparison_correction import verify_retry_starts, cpu_environment_receipt
    parent = validate_config(config)
    environment = cpu_environment_receipt(parent, os.environ)
    result = cpu_stage(config, output, deadline)
    starts = prepare_starts(config, output)
    exact = verify_retry_starts(parent, output)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU gate initialized CUDA")
    return dict(status="passed", tests_passed=result["junit"]["tests"],
        artifacts=dict(junit=asset(output / "cpu-tests.xml"), starts=asset(output / "starts.pt")),
        cpu_gate=result, starts=starts, starts_readiness=exact, cpu_execution=environment)


def fit(config, output, deadline, campaign, source, plan):
    import torch
    from .embedding_experiment import load_vae, load_native
    from .embedding_long_training import prepare_long_dataset, fit_long_bridge
    stage_result(campaign, "cpu", source, plan)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    vae, checkpoint = load_vae(config, "cuda")
    del checkpoint
    scorer, prepared, bank = load_native(config, "cuda")
    from .embedding_validation import native_gate
    native_gate(scorer, prepared, bank, vae, output / "native-oracle.json",
                config["native_tolerances"], deadline=deadline)
    del scorer, prepared
    settings = config["long_training"]
    prepare_long_dataset(vae, config["dataset"], output / "paired-data", settings["seed"],
        prior_dataset=config["start_data_manifest"],
        benchmark_starts=config["precision_correction"]["parent_starts"],
        **settings["sampling"], deadline=deadline)
    fitted = fit_long_bridge(vae, bank, output / "paired-data", config["bridge"],
        output / "fit", "cuda", bindings=dict(source_sha256=source, plan_sha256=plan),
        seed=settings["seed"], **settings["fitting"], deadline=deadline)
    if (fitted["status"] != "completed_long_bridge_fit_not_native_validated"
            or fitted["frozen_vae_and_bank_unchanged"] is not True
            or fitted.get("dev_improved") is not True
            or fitted.get("audit_consumed") is not True
            or fitted["bindings"] != dict(source_sha256=source, plan_sha256=plan)
            or fitted["checkpoint_sha256"] != sha(fitted["checkpoint_path"])
            or fitted["calibration_sha256"] != sha(fitted["calibration_path"])):
        raise RuntimeError("New bridge fit/calibration did not pass")
    model = dict(checkpoint_sha256=fitted["checkpoint_sha256"], calibration_sha256=fitted["calibration_sha256"])
    return dict(status="passed", fit_completed=True, calibration_passed=True, model_identity=model,
        artifacts=dict(checkpoint=asset(fitted["checkpoint_path"]), calibration=asset(fitted["calibration_path"]),
            training_plan=asset(output / "fit/training-plan.json"),
            paired_data_manifest=asset(output / "paired-data/manifest.json"),
            audit=asset(output / "fit/audit.json")),
        fit=fitted, scope="Improved held-out fit and dev calibration; fresh native/derivative gates remain mandatory")


def load_long_bridge(config, vae, bank, fitted, source, plan, device):
    import torch
    from latentmol.train import state_digest
    from .embedding_bridge import HiddenEmbeddingBridge
    from .embedding_training import TRAINING_VERSION, _vae_identity
    from .embedding_long_training import VERSION as FIT_VERSION, CALIBRATION_VERSION, DATA_EXTENSION, calibrate_caps
    checkpoint = verify(fitted["artifacts"]["checkpoint"])
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    calibration = json.loads(verify(fitted["artifacts"]["calibration"]).read_text())
    training_path = verify(fitted["artifacts"]["training_plan"])
    training = json.loads(training_path.read_text())
    manifest_path = verify(fitted["artifacts"]["paired_data_manifest"])
    manifest = json.loads(manifest_path.read_text())
    verify(fitted["artifacts"]["audit"])
    verify(config["bridge"])
    bindings = dict(source_sha256=source, plan_sha256=plan)
    bank_state = bank.state_dict()
    fit_receipt = fitted["fit"]
    if (fitted["status"] != "passed" or fitted["fit_completed"] is not True
            or fitted["calibration_passed"] is not True
            or fit_receipt["status"] != "completed_long_bridge_fit_not_native_validated"
            or fit_receipt["frozen_vae_and_bank_unchanged"] is not True
            or fit_receipt["dev_improved"] is not True or fit_receipt["audit_consumed"] is not True
            or fit_receipt["bindings"] != bindings
            or fit_receipt["checkpoint_sha256"] != fitted["model_identity"]["checkpoint_sha256"]
            or fitted["model_identity"]["checkpoint_sha256"] != sha(checkpoint)
            or fit_receipt["calibration_sha256"] != fitted["model_identity"]["calibration_sha256"]
            or fitted["model_identity"]["calibration_sha256"] != sha(fitted["artifacts"]["calibration"]["path"])
            or saved.get("version") != TRAINING_VERSION or saved.get("long_training_version") != FIT_VERSION
            or saved["vae_checkpoint"] != _vae_identity(vae)
            or saved["vae_state_sha256"] != state_digest(vae.state_dict())
            or saved["bank_metadata"] != bank.metadata()
            or set(saved["bank_state"]) != set(bank_state)
            or any(saved["bank_state"][key].dtype != bank_state[key].dtype
                   or not torch.equal(saved["bank_state"][key], bank_state[key]) for key in bank_state)
            or saved["training"] != training or training["bindings"] != bindings
            or training["version"] != FIT_VERSION or training["warm_start"] != config["bridge"]
            or training["desired_score_labels"] is not False
            or training["dataset_manifest_sha256"] != sha(manifest_path)
            or manifest["long_data"]["version"] != DATA_EXTENSION
            or manifest["long_data"]["prior_dataset"] != config["start_data_manifest"]
            or manifest["long_data"]["benchmark_starts"] != config["precision_correction"]["parent_starts"]
            or manifest["long_data"]["old_train_replay_rows"] != 0
            or fit_receipt["dataset_manifest_sha256"] != sha(manifest_path)
            or fit_receipt["training_plan_sha256"] != sha(training_path)
            or calibration["training_plan_sha256"] != sha(training_path)
            or calibration["version"] != CALIBRATION_VERSION or calibration["bindings"] != bindings
            or calibration["status"] != "calibrated_dev_only_not_native_validated"
            or calibration["bridge_sha256"] != sha(checkpoint)
            or calibration["dataset_manifest_sha256"] != saved["dataset_manifest_sha256"]
            or saved["dataset_manifest_sha256"] != sha(manifest_path)
            or calibration["vae_state_sha256"] != saved["vae_state_sha256"]
            or calibration["vae_checkpoint"] != saved["vae_checkpoint"]
            or calibration["bank_metadata"] != bank.metadata()
            or calibration["bank_state_sha256"] != state_digest(bank_state)
            or calibration["warm_start"] != config["bridge"]
            or calibration["audit_used_for_calibration"] is not False
            or calibration["desired_score_labels"] is not False
            or (calibration["split"], calibration["quantile"], calibration["quantile_method"],
                calibration["cap_floor"]) != ("dev", .99, "nearest_rank", 1e-8)):
        raise ValueError("New mapping/calibration does not bind this frozen campaign")
    dims = saved["bridge_configuration"]
    bridge = HiddenEmbeddingBridge(bank, **{key: dims[key] for key in
        ("atom_input_dim", "pair_input_dim", "charge_input_dim", "hidden_dim")}).to(device)
    if bridge.configuration() != dims:
        raise ValueError("New mapping configuration differs from saved native projection layout")
    bridge.load_state_dict(saved["bridge_state"], strict=True)
    caps = {key.removesuffix("/mse"): value for key, value in calibration["caps"].items()}
    if (len(caps) != len(calibration["caps"]) or set(caps) != {d.key for d in bank.descriptors}
            or any(not key.endswith("/mse") for key in calibration["caps"])
            or calibration["metric_names"] != sorted(calibration["caps"])
            or calibration["caps"] != calibrate_caps(calibration["dev_errors"])
            or any(len(values) != calibration["dev_examples"] for values in calibration["dev_errors"].values())
            or calibration["dev_examples"] < 1):
        raise ValueError("Calibration must cover every projection exactly once")
    return bridge.eval().requires_grad_(False), caps


def load_case(item, device):
    from .contracts import PreparedComplex
    from .active_set import bind_slots
    from .provenance import context_digest
    prepared = bind_slots(PreparedComplex.load(verify(item["record"]["context_asset"]), "cpu"), item["active"]).to(device)
    if context_digest(prepared) != item["record"]["bound_context_sha256"]:
        raise ValueError("Long-run case context changed")
    return prepared, item["z"].to(device)


def bind_production_gates(objective, z, gates, policy):
    """Bind BOTH fresh scalar/gradient gates to the actual production callbacks."""
    from latentmol.multimodel.study import tensor_digest
    from .embedding_comparison_policy import policy_binding
    expected_policy = policy_binding(policy)
    if set(gates) != {"relaxed", "learned"}:
        raise ValueError("Require both fresh derivative gates")
    result = {}
    for path in ("relaxed", "learned"):
        gate = gates[path]
        evaluated = (objective.evaluate(z, require_gradient=True) if path == "learned"
            else objective.base.evaluate(path, z, require_gradient=True))
        gradient_sha = tensor_digest({"gradient": evaluated["gradient"]})
        matched = (gate.get("passed") is True and gate.get("scoring_path") == path
            and evaluated["scores"] == gate["baseline_scores"]
            and gradient_sha == gate["production_gradient_sha256"]
            and evaluated["context_sha256"] == gate["context_sha256"] == objective.context_sha256
            and evaluated["comparison_graph_policy_sha256"]
                == gate["comparison_graph_policy_sha256"] == expected_policy)
        result[path] = dict(passed=matched, gradient_sha256=gradient_sha,
            context_sha256=evaluated["context_sha256"], comparison_graph_policy_sha256=expected_policy)
        if not matched:
            raise ValueError("Fresh " + path + " gate and production scalar/derivative/context/policy differ")
    return result


def publish_state(output, state, *, case, pilot):
    """Immutable node-local hard-link alias for the controller's artifact scope."""
    source = verify(state)
    destination = Path(output) / (case + "-state.pt" if pilot else "current-state.pt")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)
    result = asset(destination)
    if result["sha256"] != state["sha256"]:
        raise ValueError("Node state alias changed committed bytes")
    return result


def trajectory(config, output, deadline, campaign, source, plan, *, case=None, segment_updates=None, resume=None):
    startup = time.monotonic()
    import torch
    from latentmol.multimodel.study import tensor_digest
    from latentmol.optimization import StepControl
    from .embedding_experiment import load_vae, load_native
    from .embedding_comparison import FrozenComparisonObjective
    from .embedding_comparison_policy import policy_binding
    from .embedding_validation import native_gate
    from .embedding_comparison_validation import learned_start_gate
    from .embedding_long_optimization import GuardedEmbeddingObjective, run_guarded_optimization
    cpu_pass = stage_result(campaign, "cpu", source, plan)
    fitted = stage_result(campaign, "fit", source, plan)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    vae, checkpoint = load_vae(config, "cuda")
    del checkpoint
    scorer, base, bank = load_native(config, "cuda")
    bridge, caps = load_long_bridge(config, vae, bank, fitted, source, plan, "cuda")
    payload = torch.load(verify(cpu_pass["artifacts"]["starts"]), weights_only=True, map_location="cpu")
    if tuple(item["record"]["case_id"] for item in payload["cases"]) != CASES:
        raise ValueError("Unexpected prescribed case order")
    policy = config["comparison_graph_policy"]
    pending = []
    for item in payload["cases"]:
        name = item["record"]["case_id"]
        if case is not None and name != case:
            continue
        prepared, z = load_case(item, "cuda")
        objective = GuardedEmbeddingObjective(FrozenComparisonObjective(scorer, prepared, bridge, vae,
            bank=bank, deadline=deadline, comparison_graph_policy=policy), caps)
        gate_dir = Path(campaign) / "pilot" / "gates" / name
        if case is None:
            gate_dir.mkdir(parents=True, exist_ok=False)
            assessment = objective.assess(z)
            write(gate_dir / "start-eligibility.json", assessment["record"])
            if not assessment["record"]["eligible"]:
                raise RuntimeError("Prescribed start outside fixed dev calibration; no automatic loosening or replacement")
            native_gate(scorer, prepared, bank, vae, gate_dir / "native.json", config["native_tolerances"], deadline=deadline)
            gates = {path: learned_start_gate(scorer, prepared, bridge, vae, z,
                gate_dir / (path + ".json"), deadline, scoring_path=path, comparison_graph_policy=policy)
                for path in ("relaxed", "learned")}
            production = bind_production_gates(objective, z, gates, policy)
            write(gate_dir / "production-binding.json", production)
            learned = gates["learned"]
            binding = dict(source_sha256=source, plan_sha256=plan, vae_checkpoint_sha256=config["vae"]["sha256"],
                bridge_checkpoint_sha256=fitted["model_identity"]["checkpoint_sha256"],
                calibration_sha256=fitted["model_identity"]["calibration_sha256"], gate_sha256=sha(gate_dir / "learned.json"),
                context_sha256=item["record"]["bound_context_sha256"], comparison_graph_policy_sha256=policy_binding(policy),
                start_sha256=tensor_digest({"z": z}), gradient_sha256=learned["production_gradient_sha256"],
                baseline_scores=learned["baseline_scores"])
            write(gate_dir / "binding.json", binding)
        else:
            binding = json.loads((gate_dir / "binding.json").read_text())
            if sha(gate_dir / "learned.json") != binding["gate_sha256"]:
                raise ValueError("Original derivative gate changed")
        pending.append((name, z, objective, binding))
    if not pending:
        raise ValueError("Unknown long-run case")
    startup_seconds = time.monotonic() - startup
    cases, artifacts = [], {}
    for name, z, objective, binding in pending:
        started = time.monotonic()
        root = Path(campaign) / "trajectories" / name
        before = sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if root.exists() else 0
        count = config["long_campaign"]["pilot_updates"] if case is None else segment_updates
        result = run_guarded_optimization(z, objective, root, case_id=name, binding=binding,
            target_updates=10000, segment_updates=count, observation_every=10, resume=resume,
            learning_rate=config["optimization"]["learning_rate"], max_gradient_norm=1., trust_radius=3.,
            control=StepControl.from_mapping(config["step_control"]), deadline=deadline,
            reserve_seconds=config["long_campaign"]["segment_reserve_seconds"], max_consecutive_noop=25)
        after = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
        updates = result["completed_update"] - result["starting_update"]
        profile = dict(seconds_per_update=(time.monotonic() - started) / max(updates, 1),
            startup_seconds=startup_seconds, bytes_per_update=max(0, after - before) / max(updates, 1), fixed_bytes=before)
        state_asset = publish_state(output, result["current_state"], case=name, pilot=case is None)
        summary = dict(case=name, completed_updates=result["completed_update"],
            accepted_nonzero_steps=result["counts"]["applied"], gates_passed=True,
            strict_valid_best=result["best_valid"] is not None, profile=profile,
            state=state_asset, model_identity=fitted["model_identity"],
            core_result=result)
        cases.append(summary)
        artifacts[name + "_state" if case is None else "state"] = state_asset
        if result["status"] == "scientific_stall_target_incomplete" or result.get("scientific_stall") or updates < 1:
            raise RuntimeError("Guarded trajectory stalled or made no bounded progress; retain partial state, stop successors")
    if case is None:
        return dict(status="passed", model_identity=fitted["model_identity"], cases=cases, artifacts=artifacts)
    result = cases[0]["core_result"]
    return dict(status="completed_case" if result["status"] == "target_completed" else "completed_segment",
        case=case, start_update=result["starting_update"], completed_updates=result["completed_update"],
        resume_state_sha256=resume["sha256"], model_identity=fitted["model_identity"], guards_passed=True,
        guard_stall=False, observation_every=10, artifacts=artifacts, core_result=result)


def native_target(config):
    from .embedding_comparison_rescore import load_target, _verify_cli_defaults
    verify(config["native_template"])
    _verify_cli_defaults()
    fresh = config["fresh_native"]
    return load_target(config["native_template"]["path"], fresh["target_sequence_sha256"],
        fresh["msa_sha256"], template_sha256=config["native_template"]["sha256"])


def request_for(config, smiles, seed):
    value = dict(smiles=smiles, seed=seed, native_settings=config["fresh_native"]["native_settings"])
    return dict(id=hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24],
                target_key="shared", **value)


def validate_native_result(result, request):
    """The same exact association/success checks apply to new and reused work."""
    if (result.get("id"), result.get("smiles"), result.get("seed")) != (
            request["id"], request["smiles"], request["seed"]):
        raise ValueError("Native request association differs")
    if result.get("status") != "completed" or result.get("retention_eligible") is not True:
        raise RuntimeError("Actual native request failed; preserve result and stop")
    scores = result.get("affinity")
    if (not isinstance(scores, dict) or any(type(scores.get(key)) not in (int, float)
            or not math.isfinite(scores[key]) for key in ("affinity_probability_binary", "affinity_pred_value"))
            or not 0 <= scores["affinity_probability_binary"] <= 1):
        raise ValueError("Native affinity output missing, nonfinite or outside probability range")
    return result


def execute_native(config, target, request, output, deadline):
    from latentmol.cuda_process import require_cuda_free_coordinator
    from .fresh_trajectory import execute_request
    require_cuda_free_coordinator()
    fresh = config["fresh_native"]
    if deadline - time.monotonic() < fresh["native_timeout_seconds"] + fresh["audit_timeout_seconds"] + 120:
        raise TimeoutError("Insufficient bounded native request headroom")
    result = execute_request(request, target, Path(config["boltz_cache"]), output,
        {key: fresh[key] for key in ("native_timeout_seconds", "audit_timeout_seconds")})
    require_cuda_free_coordinator()
    write(output / "result.json", result)
    return validate_native_result(result, request)


def native_smoke(config, campaign, output, deadline):
    import torch  # CPU deserialization never creates a coordinator CUDA context.
    from latentmol.cuda_process import require_cuda_free_coordinator
    from .embedding_native_recovery import cuda_child_smoke
    require_cuda_free_coordinator()
    cuda_child_smoke(output, deadline)
    starts_asset = config["precision_correction"]["parent_starts"]
    starts = torch.load(verify(starts_asset), map_location="cpu", weights_only=True)
    if tuple(item["record"]["case_id"] for item in starts["cases"]) != CASES:
        raise ValueError("Native smoke requires the three exact prescribed cases")
    request = request_for(config, starts["cases"][0]["record"]["identity"], config["fresh_native"]["pose_seeds"][0])
    target = native_target(config)
    write(output / "smoke-request.json", request)
    result = execute_native(config, target, request, output / "native-smoke", deadline)
    write(output / "native-smoke-gate.json", dict(status="passed", request=request,
        result=asset(output / "native-smoke/result.json"), target_sha256=hashlib.sha256(
            json.dumps(target, sort_keys=True).encode()).hexdigest(), starts_asset=starts_asset, counted_once=True))
    require_cuda_free_coordinator()
    return result


def followup(config, campaign, output, deadline, final_states):
    import torch  # CPU tensor reads only; this process must remain CUDA-free.
    from latentmol.cuda_process import require_cuda_free_coordinator
    from latentmol.chemistry_classification import classify_repair_graph
    from latentmol.representation import GraphSpec
    require_cuda_free_coordinator()
    target = native_target(config)
    source, plan = final_states["identity"]["source_sha256"], final_states["identity"]["plan_sha256"]
    cpu_pass = stage_result(campaign, "cpu", source, plan)
    pilot = stage_result(campaign, "pilot", source, plan)
    starts = torch.load(verify(cpu_pass["artifacts"]["starts"]), map_location="cpu", weights_only=True)
    if (tuple(item["record"]["case_id"] for item in starts["cases"]) != CASES
            or set(final_states["cases"]) != set(CASES)):
        raise ValueError("Native followup requires all three exact prescribed cases")
    spec = GraphSpec.from_dict(starts["graph_spec"])
    members = []
    fitted = stage_result(campaign, "fit", source, plan)
    for item in starts["cases"]:
        name = item["record"]["case_id"]
        root = Path(campaign) / "trajectories" / name
        final = final_states["cases"][name]
        if final["completed_updates"] != 10000:
            raise ValueError("Native follow-up before prescribed endpoint")
        state_path = verify(dict(path=final["state"], sha256=final["state_sha256"]))
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        if state["update"] != 10000 or state["case_id"] != name:
            raise ValueError("Native terminal state binding differs")
        manifest = json.loads((root / "manifest.json").read_text())
        binding = manifest["signature"]["binding"]
        signature_sha = hashlib.sha256(json.dumps(manifest["signature"], sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        observations = {}
        for role, key, update in (("baseline", "baseline_observation", 0),
                                  ("terminal", "last_observation", 10000)):
            receipt = state[key]
            record = json.loads(verify(receipt["record"]).read_text())
            verify(receipt["tensors"])
            if (receipt["update"] != update or record["update"] != update
                    or record["tensors"] != receipt["tensors"]
                    or record["signature_sha256"] != state["signature_sha256"]):
                raise ValueError("Native observation differs from authenticated final state")
            observations[role] = receipt["tensors"]
        states = [(role, observations[role]) for role in ("baseline", "terminal")]
        if (state["signature_sha256"] != manifest["sha256"]
                or manifest["sha256"] != signature_sha
                or binding["source_sha256"] != final_states["identity"]["source_sha256"]
                or binding["plan_sha256"] != final_states["identity"]["plan_sha256"]
                or binding["bridge_checkpoint_sha256"] != fitted["model_identity"]["checkpoint_sha256"]
                or binding["calibration_sha256"] != fitted["model_identity"]["calibration_sha256"]
                or state["best_valid"] is None):
            raise ValueError("Native selection no longer binds guarded optimization")
        states.append(("best_valid", state["best_valid"]["tensors"]))
        for role, value in states:
            tensors = torch.load(verify(value), map_location="cpu", weights_only=True)
            expected_z = state["z0"] if role == "baseline" else state["z"] if role == "terminal" else None
            if expected_z is not None and not torch.equal(tensors["z"], expected_z):
                raise ValueError("Native checkpoint differs from authenticated terminal/original latent")
            graph = tensors["conditioned"]
            classified = classify_repair_graph(*(graph[key][0] for key in ("atoms", "charges", "bonds")), spec)
            members.append(dict(case=name, role=role, checkpoint=value, classification=classified,
                smiles=classified["smiles"] if classified["strict_teacher_valid"] else None))
    requests = [request_for(config, identity, seed)
        for identity in sorted({row["smiles"] for row in members if row["smiles"]})
        for seed in config["fresh_native"]["pose_seeds"]]
    if len(requests) > 18:
        raise ValueError("Long-run native request budget exceeded")
    write(output / "selection.json", dict(members=members, requests=requests,
        policy="all baseline/terminal/best-valid by hard fixed-context loss, deduplicated; seeds17/23"))
    smoke = json.loads(verify(pilot["artifacts"]["native_smoke"]).read_text())
    if (smoke["status"] != "passed" or smoke["counted_once"] is not True
            or smoke["target_sha256"] != hashlib.sha256(json.dumps(target, sort_keys=True).encode()).hexdigest()
            or smoke["starts_asset"] != config["precision_correction"]["parent_starts"]
            or smoke["request"] != request_for(config, starts["cases"][0]["record"]["identity"],
                                               config["fresh_native"]["pose_seeds"][0])
            or smoke["request"] not in requests):
        raise ValueError("Baseline native smoke target changed")
    results = []
    for request in requests:
        if request == smoke["request"]:
            results.append(validate_native_result(json.loads(verify(smoke["result"]).read_text()), request))
        else:
            results.append(execute_native(config, target, request, output / "requests" / request["id"], deadline))
    require_cuda_free_coordinator()
    write(output / "summary.json", dict(status="completed", members=members, requests=requests, results=results,
        all_requests_accounted=len(requests) == len(results), native_affinity_scope="fresh poses; distinct from full fixed-context DBMol"))
    return dict(status="passed", all_requests_accounted=True, artifacts=dict(summary=asset(output / "summary.json")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("cpu", "fit", "pilot", "segment", "followup"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--seconds", type=int, required=True)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--segment-updates", type=int)
    parser.add_argument("--resume-state", type=Path)
    parser.add_argument("--resume-sha256")
    parser.add_argument("--bridge", type=Path)
    parser.add_argument("--bridge-sha256")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--calibration-sha256")
    parser.add_argument("--states", type=Path)
    parser.add_argument("--states-sha256")
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    if (not os.environ.get("LSB_JOBID") or os.environ.get("LATENTMOL_SOURCE_SHA256") != args.source_sha256
            or sha(args.config) != args.plan_sha256 or args.seconds < 1):
        raise RuntimeError("Long-run numerical work requires this exact sealed LSF allocation")
    config = json.loads(args.config.read_text())
    validate_config(config)
    if args.stage in ("pilot", "segment", "followup"):
        fitted = stage_result(args.campaign, "fit", args.source_sha256, args.plan_sha256)
        for role, path, digest in (("checkpoint", args.bridge, args.bridge_sha256),
                                   ("calibration", args.calibration, args.calibration_sha256)):
            if path is None or dict(path=str(path), sha256=digest) != fitted["artifacts"][role]:
                raise ValueError("Allocated bridge/calibration CLI differs from passed fit")
            verify(dict(path=str(path), sha256=digest))
    final_states = None
    if args.stage == "followup":
        final_states = json.loads(verify(dict(path=str(args.states), sha256=args.states_sha256)).read_text())
        if final_states["identity"]["source_sha256"] != args.source_sha256 or final_states["identity"]["plan_sha256"] != args.plan_sha256:
            raise ValueError("Final state manifest belongs to another release")
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = dict(version=VERSION, stage=args.stage, source_sha256=args.source_sha256,
        manifest_sha256=args.manifest_sha256, plan_sha256=args.plan_sha256,
        frozen_identities=frozen_identities(config), job_id=os.environ["LSB_JOBID"], host=socket.gethostname(),
        existing_vae_continuation="untouched")
    started = time.monotonic()
    deadline = started + args.seconds
    result_path = args.output / ("child-result.json" if args.child else "result.json")
    try:
        if args.stage in ("fit", "pilot", "segment") and not args.child:
            from latentmol.cuda_process import run_cuda_child, require_cuda_child, require_cuda_free_coordinator
            require_cuda_free_coordinator()
            if args.stage == "pilot":
                native_smoke(config, args.campaign, args.output, deadline)
            command = [sys.executable, "-m", __package__ + ".embedding_long_experiment", *sys.argv[1:], "--child"]
            # Share the remaining allocation deadline; never restart a full clock after native smoke.
            command[command.index("--seconds") + 1] = str(max(1, int(deadline - time.monotonic()) - 30))
            child = run_cuda_child(command, args.output / "cuda-child.log", timeout_seconds=max(1, int(deadline-time.monotonic())-10))
            write(args.output / "cuda-child-process.json", child)
            require_cuda_child(child)
            result = json.loads((args.output / "child-result.json").read_text())
            if args.stage == "pilot":
                result["artifacts"]["native_smoke"] = asset(args.output / "native-smoke-gate.json")
        else:
            if args.stage != "followup":
                import torch
                torch.set_num_threads(4)
            if args.stage == "cpu":
                result = cpu(config, args.output, deadline)
            elif args.stage == "fit":
                result = fit(config, args.output, deadline, args.campaign, args.source_sha256, args.plan_sha256)
            elif args.stage in ("pilot", "segment"):
                resume = None if args.stage == "pilot" else dict(path=str(args.resume_state), sha256=args.resume_sha256)
                result = trajectory(config, args.output, deadline, args.campaign, args.source_sha256, args.plan_sha256,
                    case=args.case, segment_updates=args.segment_updates, resume=resume)
            else:
                result = followup(config, args.campaign, args.output, deadline, final_states)
        write(result_path, {**provenance, **result, "elapsed_seconds": time.monotonic() - started})
    except BaseException as error:
        write(result_path, {**provenance, "status": "failed_not_passed", "error_type": type(error).__name__,
            "error": str(error), "traceback": traceback.format_exc(), "elapsed_seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    main()
