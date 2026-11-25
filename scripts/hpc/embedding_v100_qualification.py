"""Isolated allocated V100 qualification of the unchanged source142 models.

Fresh same-device native/derivative gates are required. This worker never
adopts A100 bindings, caches, checkpoints or diagnostic candidates into a run.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

VERSION = "embedding-v100-qualification-v1"
CONFIG = "configs/embedding_v100_qualification.json"
SOURCE_SHA = "d2ac85ad644a94f3b5967537f1c777bb7318faf2a8507038888bc32728a0d805"
MANIFEST_SHA = "89dd0f43752a417cb4a4ab0658a4fba294eba72eb65fd5899036a42d8207c4f2"
PLAN_SHA = "35adaf248e9011ce81a5bcd165d1897e97901864695a10d56fad70eebe63f70e"
CASES = ("case-001", "case-002", "case-003")
ARTIFACT_KEYS = {"fit_result", "starts"} | {
    "case%03d_%s" % (index, field)
    for index in (1, 2, 3) for field in ("state", "manifest", "binding")}
STATE_UPDATES = dict(zip(CASES, (1899, 20, 20)))
WORK = Path("/work3/s234818/latentmol")


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify(asset):
    if (not isinstance(asset, dict) or set(asset) != {"path", "sha256"}
            or not isinstance(asset["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", asset["sha256"]) is None):
        raise ValueError("Require a hash-bound qualification asset")
    path = Path(asset["path"])
    if sha(path) != asset["sha256"]:
        raise ValueError("Qualification asset changed: " + str(path))
    return path


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def artifact(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def validate_output(release, output, stage):
    release, output = Path(release).resolve(), Path(output).resolve()
    if (stage not in ("cpu", "gpu") or release.parent != WORK / "releases"
            or not re.fullmatch(r"source-146-embedding-v100-qualification-attempt[0-9]{3}", release.name)):
        raise ValueError("Qualification requires its own immutable source146 release")
    expected = WORK / "runs" / release.name.replace("source-146", "source146", 1) / stage
    if output != expected:
        raise ValueError("Qualification output must remain in its independent campaign")


def validate_config(config):
    expected = dict(version=VERSION, gpu_profile="v10032", fixed_learning_rate=.0025,
        step_candidates=1, line_search=False, warmups=2, repeats=5,
        internal_seconds=13800, memory_headroom_gib=4)
    if any(type(config.get(key)) is not type(value) or config[key] != value
           for key, value in expected.items()):
        raise ValueError("Require the fixed V10032 qualification policy without search")
    if config.get("state_updates") != STATE_UPDATES:
        raise ValueError("Qualification must retain the declared immutable resume points")
    parent = config["parent"]
    if (parent.get("release") != str(WORK / "releases/source-142-epoch36-guarded-embedding-long-attempt001")
            or parent.get("campaign") != str(WORK / "runs/source142-epoch36-guarded-embedding-long-attempt001")
            or parent.get("source_sha256") != SOURCE_SHA
            or parent.get("manifest_sha256") != MANIFEST_SHA
            or parent.get("plan_sha256") != PLAN_SHA
            or set(parent.get("artifacts", {})) != ARTIFACT_KEYS):
        raise ValueError("Qualification must use the exact original source142 scientific parent")
    for asset in parent["artifacts"].values():
        if (not isinstance(asset, dict) or set(asset) != {"path", "sha256"}
                or not isinstance(asset["sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", asset["sha256"]) is None
                or not Path(asset["path"]).is_relative_to(Path(parent["campaign"]))):
            raise ValueError("Qualification parent assets must be pinned within the scientific campaign")
    return config


def _operator():
    spec = importlib.util.spec_from_file_location("_v100_operator", Path(__file__).with_name("embedding_operator.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def authenticate_parent(config):
    validate_config(config)
    parent = config["parent"]
    release = Path(parent["release"])
    _operator().authenticate(release, SOURCE_SHA, PLAN_SHA, "configs/epoch36_embedding_long.json")
    if sha(release / "embedding-release-manifest.json") != MANIFEST_SHA:
        raise ValueError("Original scientific manifest changed")
    for asset in parent["artifacts"].values():
        verify(asset)
    return release


def require_passed_cpu(cpu, result_path, source_sha, plan_sha, cpu_sha):
    if (cpu_sha is None or sha(result_path) != cpu_sha
            or cpu != json.loads(Path(result_path).read_text(encoding="utf-8"))
            or cpu.get("version") != VERSION or cpu.get("stage") != "cpu"
            or cpu.get("status") != "passed" or cpu.get("source_sha256") != source_sha
            or cpu.get("plan_sha256") != plan_sha
            or re.fullmatch(r"[0-9]+", str(cpu.get("job_id", ""))) is None):
        raise ValueError("GPU requires its exact authenticated allocated CPU result")
    counts = cpu.get("junit", {})
    if (set(counts) != {"tests", "failures", "errors", "skipped"}
            or any(type(value) is not int for value in counts.values())
            or counts["tests"] < 1 or any(counts[key] != 0 for key in ("failures", "errors", "skipped"))):
        raise ValueError("CPU qualification fixtures must pass without skips or failures")
    suites = list(ET.parse(verify(cpu["junit_asset"])).getroot().iter("testsuite"))
    actual = {key: sum(int(s.get(key, "0")) for s in suites) for key in counts}
    if actual != counts:
        raise ValueError("CPU result counters differ from its authenticated JUnit")
    return cpu


def require_exact_binding(evaluation, gate, gradient_sha, context_sha, policy_sha, scoring_path="learned"):
    if (gate.get("passed") is not True or gate.get("scoring_path") != scoring_path
            or evaluation["scores"] != gate["baseline_scores"]
            or gradient_sha != gate["production_gradient_sha256"]
            or evaluation["context_sha256"] != gate["context_sha256"] or gate["context_sha256"] != context_sha
            or evaluation["comparison_graph_policy_sha256"] != gate["comparison_graph_policy_sha256"]
            or gate["comparison_graph_policy_sha256"] != policy_sha):
        raise ValueError("Fresh same-device gate differs from the production callback")
    return dict(passed=True, gradient_sha256=gradient_sha, context_sha256=context_sha,
                comparison_graph_policy_sha256=policy_sha)


def quantiles(values):
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Timings must be finite nonnegative measurements")
    ordered = sorted(values)
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                minimum=ordered[0], p90=ordered[math.ceil(.9 * len(ordered)) - 1], samples=values)


def measure(torch, function, config, deadline):
    values, result = [], None
    for index in range(config["warmups"] + config["repeats"]):
        if time.monotonic() + 120 >= deadline:
            raise TimeoutError("V100 qualification internal deadline")
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = function()
        torch.cuda.synchronize()
        if index >= config["warmups"]:
            values.append(time.perf_counter() - started)
    return quantiles(values), result


def gpu_metadata():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version,pstate,power.draw,power.limit,clocks.sm,clocks.mem", "--format=csv"],
                                text=True, capture_output=True, timeout=20)
        return dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except (OSError, subprocess.TimeoutExpired) as error:
        return dict(metadata_unavailable=str(error))


def cpu_stage(config, output, source_sha, plan_sha):
    authenticate_parent(config)
    command = [sys.executable, "-m", "pytest", "-q", "tests/test_embedding_v100_qualification.py",
               "--junitxml=" + str(output / "cpu-tests.xml")]
    with (output / "cpu-tests.log").open("xb") as stream:
        done = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=1800)
    suites = list(ET.parse(output / "cpu-tests.xml").getroot().iter("testsuite"))
    counts = {key: sum(int(s.get(key, "0")) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
    if done.returncode or counts["tests"] < 1 or any(counts[key] for key in ("failures", "errors", "skipped")):
        raise RuntimeError("Allocated V100 qualification fixtures failed")
    return dict(status="passed", junit=counts, junit_asset=artifact(output / "cpu-tests.xml"),
                numerical_models="not_loaded", source_sha256=source_sha, plan_sha256=plan_sha)


def gpu_stage(config, output, deadline):
    release = authenticate_parent(config)
    sys.path.insert(0, str(release / "src"))
    import torch
    from latentmol.boltz import embedding_comparison as comparison
    from latentmol.boltz import embedding_long_optimization as long
    from latentmol.boltz import embedding_long_experiment as experiment
    from latentmol.boltz import embedding_validation as validation
    from latentmol.boltz import embedding_comparison_validation as derivative
    from latentmol.boltz.embedding_experiment import load_vae, load_native
    from latentmol.multimodel.study import tensor_digest
    from latentmol.train import state_digest
    if not all(Path(module.__file__).resolve().is_relative_to(release / "src")
               for module in (comparison, long, experiment, validation, derivative)):
        raise ValueError("Qualification imported scientific code outside original source142")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("Allocated V100 qualification has no CUDA device")
    properties = torch.cuda.get_device_properties(0)
    hardware = dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=properties.name,
        capability=list(torch.cuda.get_device_capability(0)), compiled_arches=torch.cuda.get_arch_list(),
        total_memory_bytes=properties.total_memory, threads=torch.get_num_threads(), metadata_before=gpu_metadata())
    write(output / "hardware.json", hardware)
    if "V100" not in properties.name or properties.total_memory < 30 * 1024**3:
        raise RuntimeError("Qualification requires a 32GB V100, not the 16GB model")
    native = json.loads((release / "configs/epoch36_embedding_long.json").read_text())
    parent = config["parent"]
    fitted = json.loads(verify(parent["artifacts"]["fit_result"]).read_text())
    vae, checkpoint = load_vae(native, "cuda")
    del checkpoint
    scorer, unused, bank = load_native(native, "cuda")
    del unused
    bridge, caps = experiment.load_long_bridge(native, vae, bank, fitted, SOURCE_SHA, PLAN_SHA, "cuda")
    starts = torch.load(verify(parent["artifacts"]["starts"]), map_location="cpu", weights_only=True)
    if tuple(item["record"]["case_id"] for item in starts["cases"]) != CASES:
        raise ValueError("Qualification must retain all three original starts")
    modules = dict(vae=vae, scorer=scorer, bank=bank, bridge=bridge)
    frozen_before = {key: state_digest(module.state_dict()) for key, module in modules.items()}
    environment = hardware
    write(output / "setup.json", dict(environment=environment, model_identity=fitted["model_identity"],
        vae=native["vae"], cases=list(CASES), original_A100_bindings_adopted=False))
    result = dict(cases=[], environment=environment, model_identity=fitted["model_identity"],
        vae=native["vae"], source142_frozen_before=frozen_before, production_mutation=False,
        A100_bindings_or_hard_score_caches_migrated=False, optimization_trajectory_claim=False)
    torch.cuda.reset_peak_memory_stats(0)
    for index, item in enumerate(starts["cases"], 1):
        case = item["record"]["case_id"]
        directory = output / case
        directory.mkdir()
        prepared, z0 = experiment.load_case(item, "cuda")
        objective = comparison.FrozenComparisonObjective(scorer, prepared, bridge, vae, bank=bank,
            deadline=deadline, comparison_graph_policy=native["comparison_graph_policy"])
        guard = long.GuardedEmbeddingObjective(objective, caps)
        prefix = "case%03d_" % index
        saved = torch.load(verify(parent["artifacts"][prefix + "state"]), map_location="cpu", weights_only=True)
        manifest = json.loads(verify(parent["artifacts"][prefix + "manifest"]).read_text())
        binding = json.loads(verify(parent["artifacts"][prefix + "binding"]).read_text())
        if (saved["case_id"] != case or saved["update"] != config["state_updates"][case]
                or not saved["gate_binding_verified"] or not torch.equal(saved["z0"], z0.cpu())
                or saved["signature_sha256"] != manifest["sha256"]
                or long._json_digest(manifest["signature"]) != manifest["sha256"]
                or manifest["signature"]["binding"] != binding
                or binding["source_sha256"] != SOURCE_SHA or binding["plan_sha256"] != PLAN_SHA
                or binding["vae_checkpoint_sha256"] != native["vae"]["sha256"]
                or binding["bridge_checkpoint_sha256"] != fitted["model_identity"]["checkpoint_sha256"]
                or binding["calibration_sha256"] != fitted["model_identity"]["calibration_sha256"]):
            raise ValueError("Qualification resume snapshot differs from the frozen parent binding")
        long._validate_binding(binding, guard, z0)
        resume = saved["z"].to(z0)
        assessment = guard.assess(resume)
        write(directory / "resume-eligibility.json", assessment["record"])
        if not assessment["record"]["eligible"]:
            raise ValueError("Pinned resume is outside unchanged V100 guard eligibility")
        native_gate = validation.native_gate(scorer, prepared, bank, vae, directory / "native.json",
                                             native["native_tolerances"], deadline=deadline)
        gates = {path: derivative.learned_start_gate(scorer, prepared, bridge, vae, z0,
            directory / ("start-" + path + ".json"), deadline, scoring_path=path,
            comparison_graph_policy=native["comparison_graph_policy"]) for path in ("relaxed", "learned")}
        production = experiment.bind_production_gates(guard, z0, gates, native["comparison_graph_policy"])
        write(directory / "start-production-binding.json", production)
        resume_gate = derivative.learned_start_gate(scorer, prepared, bridge, vae, resume,
            directory / "resume-learned.json", deadline, scoring_path="learned",
            comparison_graph_policy=native["comparison_graph_policy"])
        evaluated = guard.evaluate(resume, require_gradient=True)
        resume_binding = require_exact_binding(evaluated, resume_gate,
            tensor_digest({"gradient": evaluated["gradient"]}), guard.context_sha256,
            guard.comparison_graph_policy_sha256)
        write(directory / "resume-production-binding.json", resume_binding)
        original = gates["learned"]
        cross_device = dict(reference="original A100 certificate; reported only, never adopted",
            baseline_scores_bitwise_equal=original["baseline_scores"] == binding["baseline_scores"],
            gradient_digest_equal=original["production_gradient_sha256"] == binding["gradient_sha256"],
            score_differences={key: original["baseline_scores"][key] - value
                               for key, value in binding["baseline_scores"].items()})

        def fixed_candidate():
            baseline = guard.evaluate(resume, require_gradient=True)
            candidate = resume - config["fixed_learning_rate"] * baseline["gradient"]
            assessment = guard.assess(candidate)
            if not assessment["record"]["eligible"]:
                write(directory / "ineligible-fixed-candidate.json", assessment["record"])
                raise ValueError("The single fixed candidate failed unchanged guard eligibility")
            measured = guard.evaluate(candidate, assessment=assessment)
            return dict(candidates=1, eligible=True, candidate_loss=measured["loss"])

        gradient_timing, _ = measure(torch, lambda: guard.evaluate(resume, require_gradient=True), config, deadline)
        fixed_timing, candidate = measure(torch, fixed_candidate, config, deadline)
        frozen = guard.assert_unchanged(full=True)
        row = dict(case=case, resume_update=saved["update"], resume_sha256=parent["artifacts"][prefix + "state"]["sha256"],
            context_sha256=guard.context_sha256, active_slots=int(prepared.active.sum()),
            native_passed=native_gate["passed"], start_production_bindings=production,
            resume_production_binding=resume_binding, A100_reference_audit=cross_device,
            timings=dict(production_guarded_gradient=gradient_timing, single_fixed_candidate=fixed_timing),
            candidate=candidate, frozen_after=frozen, gates={name: artifact(directory / name) for name in
                ("native.json", "start-relaxed.json", "start-learned.json", "resume-learned.json")})
        write(directory / "result.json", row)
        result["cases"].append(row)
        del evaluated, objective, guard, prepared, saved
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info(0)
    memory = dict(total_bytes=total, free_bytes=free,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
    memory["headroom_bytes"] = total - memory["peak_reserved_bytes"]
    write(output / "memory.json", memory)
    if memory["headroom_bytes"] < config["memory_headroom_gib"] * 1024**3:
        raise ValueError("V100 qualification lacks the fixed 4GiB memory reserve")
    frozen_after = {key: state_digest(module.state_dict()) for key, module in modules.items()}
    if frozen_before != frozen_after:
        raise ValueError("Frozen model state changed during V100 qualification")
    authenticate_parent(config)
    result.update(status="passed_device_qualification_not_continuation_adoption", memory=memory,
        metadata_after=gpu_metadata(), frozen_unchanged=True, source142_frozen_after=frozen_after,
        fixed_policy=dict(learning_rate=.0025, candidates=1, line_search=False),
        continuation_and_VAE="untouched; separate migration is not authorized by this result")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("cpu", "gpu"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-sha256")
    args = parser.parse_args()
    release = Path(__file__).resolve().parents[2]
    if (not os.environ.get("LSB_JOBID") or os.environ.get("LATENTMOL_SOURCE_SHA256") != args.source_sha256
            or args.config.resolve() != release / CONFIG or sha(args.config) != args.plan_sha256):
        raise RuntimeError("Qualification requires the exact allocated immutable release")
    _operator().authenticate(release, args.source_sha256, args.plan_sha256, CONFIG)
    validate_output(release, args.output, args.stage)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    header = dict(version=VERSION, stage=args.stage, source_sha256=args.source_sha256,
                  plan_sha256=args.plan_sha256, job_id=os.environ["LSB_JOBID"])
    try:
        config = validate_config(json.loads(args.config.read_text()))
        if args.stage == "cpu":
            result = cpu_stage(config, args.output, args.source_sha256, args.plan_sha256)
        else:
            cpu_path = args.output.parent / "cpu/result.json"
            require_passed_cpu(json.loads(cpu_path.read_text()), cpu_path, args.source_sha256,
                               args.plan_sha256, args.cpu_sha256)
            result = gpu_stage(config, args.output, started + config["internal_seconds"])
    except BaseException as error:
        torch = sys.modules.get("torch")
        if args.stage == "gpu" and torch is not None and torch.cuda.is_initialized():
            try:
                free, total = torch.cuda.mem_get_info(0)
                failure_memory = dict(free_bytes=free, total_bytes=total,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(0), metadata=gpu_metadata())
            except BaseException as memory_error:
                failure_memory = dict(memory_collection_error=str(memory_error))
            write(args.output / "failed-memory.json", failure_memory)
        write(args.output / "result.json", dict(**header, status="failed_not_passed", error=str(error),
            error_type=type(error).__name__, traceback=traceback.format_exc(), elapsed_seconds=time.monotonic()-started))
        raise
    write(args.output / "result.json", {**header, **result, "elapsed_seconds": time.monotonic()-started})


if __name__ == "__main__":
    main()
