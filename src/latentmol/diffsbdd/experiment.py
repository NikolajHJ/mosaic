"""Allocated, fail-closed numerical stages for frozen VAE/DiffSBDD alternation.

This module never submits jobs, installs packages, resumes VAE training or edits
other campaigns. The separate standard-library operator authenticates the whole
source release and predecessor artifact chain before invoking one stage here.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET


VERSION = "diffsbdd-vae-alternation-protocol-v1"
STAGE_VERSION = "diffsbdd-stage-v1"
STAGES = ("cpu", "native", "scorer", "fit", "audit", "smoke", "pilot", "followup")
VAE_SHA256 = "e5d7f99b9eb0fe04cd8b6b800e4bd90b6f1c24a65368216ce384a95061a7f3b0"


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def validate_protocol(config):
    """Reject in-place relaxation of the bounded, predeclared experiment."""
    if config.get("version") != VERSION or config.get("existing_campaign_mutations") != "none":
        raise ValueError("Unsupported independent experiment protocol")
    required = ((config["vae"]["epoch"], 39), (config["vae"]["sha256"], VAE_SHA256),
        (config["vae"]["latent"], 256), (config["vae"]["atom_hidden"], 96),
        (config["vae"]["max_nodes"], 50), (config["upstream"]["timesteps"], 500),
        (config["upstream"]["joint_nf"], 128), (config["upstream"]["norm_values"], [1, 4]),
        (config["dataset"]["requested_sources"], 20000),
        (config["dataset"]["maximum_source_attempts"], 100000),
        (config["dataset"]["require_checkpoint_dataset_and_split_lineage"], True),
        (config["fitting"]["steps"], 10000), (config["fitting"]["batch_size"], 128),
        (config["fitting"]["learning_rate"], 0.001), (config["fitting"]["weight_decay"], 0.0001),
        (config["fitting"]["evaluate_every"], 250), (config["fitting"]["times"], [0.0, 0.04, 0.1, 0.2]),
        (config["fitting"]["calibrate_runtime"], True),
        (config["fitting"]["min_atom_agreement"], 0.995),
        (config["fitting"]["max_embedding_nrmse"], 0.05),
        (config["fitting"]["max_denoiser_nrmse"], 0.05),
        (config["fitting"]["max_denoiser_p95_nrmse"], 0.1),
        (config["fitting"]["desired_scores_are_labels"], False),
        (config["pocket"]["atom_count"], 25), (config["runtime_forecast_factor"], 1.5))
    if any(actual != expected for actual, expected in required):
        raise ValueError("Sealed checkpoint, cohort, mapping budget or acceptance settings changed")
    if config["storage"]["own_initial_cap_bytes"] != 2 * 1024**3:
        raise ValueError("The experiment's own initial artifact cap changed")
    return config


def configuration(path, stage, output):
    if stage not in STAGES or not os.environ.get("LSB_JOBID", "").isdigit():
        raise RuntimeError("Numerical stages require an identifiable allocated LSF job")
    path = Path(path).resolve(strict=True)
    expected = os.environ.get("DIFFSBDD_CONFIG_SHA256")
    source_sha = os.environ.get("DIFFSBDD_SOURCE_MANIFEST_SHA256")
    if not expected or len(expected) != 64 or sha(path) != expected or not source_sha:
        raise ValueError("Operator-authenticated source/config bindings are mandatory")
    cfg = validate_protocol(json.loads(path.read_text(encoding="utf-8")))
    source = path.parents[1]
    if (source != Path(cfg["release_root"]).resolve()
            or sha(source / "source-manifest.json") != source_sha
            or Path(output).resolve() != Path(cfg["run_root"]).resolve() / stage):
        raise ValueError("Release/output paths differ from their authenticated independent campaign")
    manifest = json.loads((source / "source-manifest.json").read_text())
    if manifest.get("version") != "diffsbdd-source-manifest-v1" or manifest["files"].get(
            path.relative_to(source).as_posix()) != expected:
        raise ValueError("Source manifest does not bind this exact config")
    unix_deadline = float(os.environ.get("DIFFSBDD_DEADLINE_UNIX", "nan"))
    remaining = unix_deadline - time.time()
    if not math.isfinite(remaining) or remaining <= 0:
        raise TimeoutError("No remaining authenticated operator deadline")
    cfg = dict(cfg, _source_root=source, _config_path=path,
               _config_sha256=expected, _source_manifest_sha256=source_sha)
    return cfg, time.monotonic() + min(remaining, cfg["stages"][stage]["internal_seconds"])


def barrier(cfg, deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("The allocated stage's internal deadline expired")
    root = Path(cfg["run_root"])
    if root.exists():
        if any(path.is_symlink() for path in root.rglob("*")):
            raise ValueError("Experiment artifacts cannot contain symlinks")
        total = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
        if total > cfg["storage"]["own_initial_cap_bytes"]:
            raise RuntimeError("Own experiment artifacts exceed the sealed 2 GiB cap")
        return total
    return 0


def prior(cfg, stage):
    path = Path(cfg["run_root"]) / stage / "stage.json"
    value = json.loads(path.read_text())
    if (value.get("version") != STAGE_VERSION or value.get("stage") != stage
            or value.get("passed") is not True or value.get("config_sha256") != cfg["_config_sha256"]
            or value.get("source_manifest_sha256") != cfg["_source_manifest_sha256"]):
        raise ValueError("Missing exact-release predecessor pass: " + stage)
    return value


def load_vae(cfg, device):
    from latentmol.evaluate import load_model
    from latentmol.data import GraphDataset
    from latentmol.train import state_digest
    import torch

    checkpoint_path = Path(cfg["vae"]["path"])
    manifest_path = Path(cfg["dataset"]["path"]) / "manifest.json"
    if sha(checkpoint_path) != VAE_SHA256 or sha(manifest_path) != cfg["dataset"]["manifest_sha256"]:
        raise ValueError("Frozen VAE or actual training dataset manifest SHA256 mismatch")
    model, checkpoint = load_model(checkpoint_path, device)
    manifest = json.loads(manifest_path.read_text())
    lineage = dict(dataset_sha256=checkpoint["dataset_sha256"],
                   split_sha256=checkpoint["continuation"]["split_sha256"])
    if (checkpoint.get("epoch") != 39 or checkpoint["train_config"].get("mode") != "vae"
            or checkpoint["continuation"].get("epoch_complete") is not True
            or lineage != {key: manifest[key] for key in lineage}
            or model.spec.max_nodes != 50 or model.config.latent != 256):
        raise ValueError("VAE checkpoint epoch/mode/schema or training/split lineage differs")
    model.eval().requires_grad_(False)
    model.diffsbdd_checkpoint_identity = dict(epoch=39, sha256=VAE_SHA256)
    model.diffsbdd_training_data_identity = lineage
    generator = torch.Generator(device="cpu").manual_seed(17)
    z = torch.randn((2, 256), generator=generator).to(device)
    before = state_digest(model.state_dict())
    with torch.no_grad():
        ordinary = model.decode(z)
        captured, hidden = model.decode_with_features(z)
    parity = all(torch.equal(getattr(ordinary, key), getattr(captured, key))
                 for key in ("atoms", "charges", "bonds"))
    if not parity or hidden.atom_hidden.shape != (2, 50, 96) or state_digest(model.state_dict()) != before:
        raise ValueError("Feature export changed the frozen ordinary VAE decoder")
    return model, dict(epoch=39, sha256=VAE_SHA256, data_identity=lineage,
        graph_spec=model.spec.to_dict(), model_config=checkpoint["model_config"],
        state_sha256=before, feature_capture_exact=True)


def load_native(cfg, device):
    from .native import load_native_core
    from .bridge import NativeAtomBoundary

    upstream, source = cfg["upstream"], cfg["_source_root"]
    core, info, receipt = load_native_core(source / upstream["source_relative"],
        source / upstream["checkpoint_relative"], upstream["checkpoint_sha256"], device,
        expected_source_manifest_sha256=upstream["source_manifest_sha256"])
    if (int(core.T) != 500 or list(core.norm_values) != [1, 4]
            or core.dynamics.atom_encoder[-1].out_features != 128):
        raise ValueError("Loaded native architecture/schedule/normalization differs from the sealed protocol")
    boundary = NativeAtomBoundary(core.dynamics.atom_encoder, info["atom_decoder"],
                                  core.norm_values[1], core.norm_biases[1]).to(device).eval().requires_grad_(False)
    return core, info, receipt, boundary


def pocket_assets(cfg, device):
    import torch

    root = Path(cfg["run_root"]) / "cpu"
    prior(cfg, "cpu")
    raw = torch.load(root / "raw-pocket.pt", map_location=device, weights_only=True)
    definition = json.loads((root / "pocket-definition.json").read_text())
    return raw, definition


def factory_builder(cfg, vae, boundary, bridge, core, info, receipt, pocket, definition, deadline):
    from .guidance import GuidanceFactory
    from .alternation import LoopCallbacks

    prep = dict(receptor_pdb=cfg["_source_root"] / cfg["pocket"]["receptor_relative"],
        msa_path=Path(cfg["run_root"]) / "cpu/receptor-query.a3m",
        molecule_dir=cfg["boltz"]["molecule_dir"], expected_pdb_sha256=cfg["pocket"]["receptor_sha256"],
        expected_msa_sha256=sha(Path(cfg["run_root"]) / "cpu/receptor-query.a3m"))

    class BudgetedFactory:
        def __init__(self, factory):
            self.factory = factory

            def wrap(function):
                def checked(*args, **kwargs):
                    barrier(cfg, deadline)
                    result = function(*args, **kwargs)
                    barrier(cfg, deadline)
                    return result
                return checked

            original = factory.callbacks
            self.callbacks = LoopCallbacks(wrap(original.admit), wrap(original.propose),
                wrap(original.native_builder), wrap(original.rebuilt_objective),
                admit_control=wrap(original.admit_control))
            self.admit_terminal = wrap(factory.admit_terminal)

        def __getattr__(self, key):
            return getattr(self.factory, key)

    def build(output, mode, transport_stage):
        barrier(cfg, deadline)
        return BudgetedFactory(GuidanceFactory(vae, boundary, bridge,
            dict(core=core, dataset_info=info, source=receipt["source"], pocket=pocket),
            definition, prep, cfg["boltz"], output, device=next(core.parameters()).device,
            deadline=deadline, mode=mode, transport_stage=transport_stage))

    return build


def cpu_stage(cfg, output, deadline):
    import torch
    from latentmol.data import GraphDataset
    from .native_checks import normalization_check
    from .pocket import prepare_pocket

    tests = sorted((cfg["_source_root"] / "tests").glob("test_diffsbdd_*.py"))
    tests += [cfg["_source_root"] / "tests/test_decoder_features.py",
              cfg["_source_root"] / "tests/test_refined_decoder.py"]
    if any(not path.is_file() for path in tests) or not tests:
        raise ValueError("CPU contract suite is incomplete")
    command = [sys.executable, "-m", "pytest", "-q", *map(str, tests),
               "--junitxml=" + str(output / "cpu-tests.xml")]
    with (output / "cpu-tests.log").open("xb") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                 timeout=max(1, deadline - time.monotonic()))
    suites = list(ET.parse(output / "cpu-tests.xml").getroot().iter("testsuite"))
    counts = {key: sum(int(suite.get(key, "0")) for suite in suites)
              for key in ("tests", "failures", "errors", "skipped")}
    write(output / "cpu-gate.json", dict(command=command, returncode=process.returncode, junit=counts))
    if process.returncode or counts["tests"] == 0 or any(counts[key] for key in ("failures", "errors", "skipped")):
        raise RuntimeError("Allocated CPU fixture gate did not pass completely")
    barrier(cfg, deadline)
    vae, vae_receipt = load_vae(cfg, "cpu")
    dataset = GraphDataset(cfg["dataset"]["path"], "train")
    if dataset.manifest["dataset_sha256"] != vae.diffsbdd_training_data_identity["dataset_sha256"]:
        raise ValueError("Validated source dataset differs from frozen VAE lineage")
    core, info, native_receipt, boundary = load_native(cfg, "cpu")
    normalization = normalization_check(core)
    pocket, definition = prepare_pocket(cfg["_source_root"] / cfg["pocket"]["receptor_relative"],
                                         info["atom_decoder"], device="cpu")
    from .scorer_context import read_receptor_pdb
    observed = read_receptor_pdb(cfg["_source_root"] / cfg["pocket"]["receptor_relative"])
    if observed["sequence"] != definition["protein_sequence"]:
        raise ValueError("Scorer crystal sequence differs from the native diffusion receptor construct")
    (output / "receptor-query.a3m").write_text(
        ">BRD4_3MXF_exact_observed_crystal_query\n" + observed["sequence"] + "\n", encoding="utf-8")
    torch.save(pocket, output / "raw-pocket.pt")
    write(output / "pocket-definition.json", definition)
    write(output / "vae-load.json", vae_receipt)
    write(output / "native-cpu-load.json", native_receipt)
    write(output / "native-normalization.json", normalization)
    molecules = Path(cfg["boltz"]["molecule_dir"]).resolve(strict=True)
    archive = Path(cfg["boltz"].get("mols_tar", str(molecules.parent / "mols.tar"))).resolve(strict=True)
    if not archive.is_file() or archive.stat().st_size == 0:
        raise ValueError("Ordinary native follow-up requires its real existing nonempty mols.tar asset")
    members = {}
    for index, path in enumerate(sorted(molecules.rglob("*"))):
        if path.is_symlink():
            raise ValueError("Canonical molecule base assets require real read-only members")
        if path.is_file():
            if index % 100 == 0:
                barrier(cfg, deadline)
            members[path.relative_to(molecules).as_posix()] = sha(path)
    if not members:
        raise ValueError("Native canonical molecule tree is empty")
    write(output / "native-molecules.json", dict(version="diffsbdd-readonly-native-molecules-v1", members=members))
    assets = {role: dict(cfg["boltz"][role]) for role in ("affinity", "confidence")}
    for role, asset in assets.items():
        if sha(asset["path"]) != asset["sha256"]:
            raise ValueError("Protected native checkpoint hash mismatch: " + role)
    assets.update(mols_tar=dict(path=str(archive), sha256=sha(archive)),
        mols_dir=dict(path=str(molecules), manifest=dict(path=str(output / "native-molecules.json"),
                                                       sha256=sha(output / "native-molecules.json"))))
    write(output / "native-followup-assets.json", dict(assets=assets,
        protein=dict(sequence=observed["sequence"], sequence_sha256=hashlib.sha256(observed["sequence"].encode()).hexdigest(),
            msa=dict(path=str(output / "receptor-query.a3m"), sha256=sha(output / "receptor-query.a3m")))))
    return dict(checks=dict(cpu_fixtures_all_passed=True, frozen_vae39_strict_load=True,
        decoder_feature_capture_exact=vae_receipt["feature_capture_exact"], dataset_lineage=True,
        native_cpu_strict_load=True, native_normalization=normalization["passed"],
        crystal_pocket_and_query_sequence=True, ordinary_followup_assets_ready=True), junit=counts,
        successor_forecast=dict(stage="native", factor=1.5, passed=False,
            status="native_original_runtime_and_GPU_timing_not_yet_measured",
            calibration_allocation_required=True))


def native_stage(cfg, output, deadline):
    # The original-runtime child owns CUDA first and exits completely before
    # this process creates its own CUDA context in the combined runtime.
    original_path = output / "original-reference.pt"
    command = [cfg["reference_runtime"], "-m", "latentmol.diffsbdd.reference", "--config",
               str(cfg["_config_path"]), "--output", str(original_path), "--device", "cuda"]
    with (output / "original-reference.log").open("xb") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                 timeout=max(1, deadline - time.monotonic()))
    if process.returncode:
        raise RuntimeError("Original-compatible native reference process failed")
    sidecar = json.loads(original_path.with_suffix(".json").read_text())
    if sidecar.get("passed") is not True or sidecar.get("sha256") != sha(original_path):
        raise ValueError("Original-compatible runtime reference is not an authenticated pass")
    import torch
    from .native_checks import compare_native_references, export_native_reference, qualify_native_core

    barrier(cfg, deadline)
    core, info, receipt, boundary = load_native(cfg, "cuda")
    pocket, definition = pocket_assets(cfg, "cuda")
    counts = torch.tensor([25], device="cuda", dtype=torch.long)
    original = torch.load(original_path, map_location="cpu", weights_only=True)
    qualification = original.get("runtime_qualification", {})
    if (qualification.get("passed") is not True or qualification.get("device") != "cuda"
            or qualification.get("checkpoint_sha256") != cfg["upstream"]["checkpoint_sha256"]
            or qualification.get("source_manifest_sha256") != cfg["upstream"]["source_manifest_sha256"]):
        raise ValueError("Original native reference has a different source/checkpoint/device")
    combined = export_native_reference(core, pocket, counts, seed=17)
    torch.save(combined, output / "combined-reference.pt")
    cross_runtime = compare_native_references(original, combined)
    write(output / "cross-runtime.json", cross_runtime)
    barrier(cfg, deadline)
    native = qualify_native_core(core, pocket, counts, check_derivatives=False)
    write(output / "native-qualification.json", native)
    write(output / "native-load.json", receipt)
    return dict(checks=dict(original_compatible_runtime=qualification["passed"],
        cross_runtime_parity=cross_runtime["passed"], native_replay_and_boundary=native["passed"],
        no_sampler_gradient_required=native["derivative_checks_requested"] is False),
        sampler_derivative_scope="detached sampling; required derivatives are native Boltz component/total gates",
        successor_forecast=dict(stage="scorer", factor=1.5, passed=False,
            status="new_native_Boltz_context_and_AD_cost_not_yet_measured", calibration_allocation_required=True))


def _state_payload(state):
    return {key: value.detach().cpu() if hasattr(value, "detach") else value
            for key, value in vars(state).items()}


def _qualified_start_payload(admission, state, seed):
    return dict(seed=seed, level=state.level, state=_state_payload(state),
        graph=asdict(admission.baseline.graph), positions=admission.baseline.positions.cpu(),
        admission_receipt=admission.receipt)


def scorer_stage(cfg, output, deadline):
    import torch
    from .native import NativeStepper, GaussianStream
    from .transport import HandoffRejected, TransportPolicy

    vae, _ = load_vae(cfg, "cuda")
    core, info, receipt, boundary = load_native(cfg, "cuda")
    pocket, definition = pocket_assets(cfg, "cuda")
    build = factory_builder(cfg, vae, boundary, None, core, info, receipt, pocket, definition, deadline)
    factory = build(output / "guidance", "exact", "T1")
    stepper, counts = NativeStepper(core), torch.tensor([25], device="cuda", dtype=torch.long)
    attempts, admitted_starts, transition_times, admission_times = [], [], [], []
    for seed in cfg["loop"]["smoke_seeds"]:
        stream = GaussianStream(seed)
        state = stepper.initialize(pocket, counts, stream)
        while state.level:
            barrier(cfg, deadline)
            torch.cuda.synchronize()
            tick = time.monotonic()
            state = stepper.reverse_step(state, stream)
            torch.cuda.synchronize()
            transition_times.append(time.monotonic() - tick)
            if state.level not in cfg["loop"]["smoke_arrival_levels"]:
                continue
            observation = dict(seed=seed, level=state.level, admitted=False)
            clean = stepper.clean_estimate(state)
            cloud_path = output / f"seed-{seed}-level-{state.level}.pt"
            torch.save(dict(state=_state_payload(state), positions=clean.positions.cpu(),
                            features=clean.features.cpu()), cloud_path)
            observation["cloud_path"] = cloud_path.name
            tick = time.monotonic()
            try:
                admission = factory.callbacks.admit(state, clean)
                admission.require(25)
                unchanged = TransportPolicy("T1").lift(admission.baseline, admission.baseline.graph,
                                                        factory.callbacks.native_builder)
                if unchanged.changed or unchanged.candidate.graph != admission.baseline.graph:
                    raise RuntimeError("T1 identity handoff changed the baseline molecule")
                path = output / f"qualified-start-{len(admitted_starts):03d}.pt"
                torch.save(_qualified_start_payload(admission, state, seed), path)
                admitted_starts.append(dict(seed=seed, level=state.level, path=path.name,
                    sha256=sha(path), receipt=admission.receipt))
                observation.update(admitted=True, admission=admission.receipt, T1_transport=unchanged.receipt)
            except HandoffRejected as error:
                observation.update(reason=str(error), rejection_receipt=error.receipt)
            admission_times.append(time.monotonic() - tick)
            observation["elapsed_seconds"] = admission_times[-1]
            attempts.append(observation)
            with (output / "attempts.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(observation, sort_keys=True, allow_nan=False) + "\n")
            barrier(cfg, deadline)
        final = stepper.finalize(state, stream)
        torch.save({key: value.detach().cpu() for key, value in vars(final).items()},
                   output / f"seed-{seed}-terminal.pt")
    write(output / "qualified-starts.json", dict(starts=admitted_starts, attempts=attempts,
        measured_max_native_transition_seconds=max(transition_times),
        measured_max_admission_seconds=max(admission_times)))
    passed = bool(admitted_starts)
    return dict(checks=dict(all_prespecified_opportunities_recorded=len(attempts) == 9,
        at_least_one_native_scorer_AD_qualified_start=passed, T1_identity_transport=passed),
        qualified_starts=len(admitted_starts),
        measured_max_native_transition_seconds=max(transition_times),
        measured_max_admission_seconds=max(admission_times),
        successor_forecast=dict(stage="fit", factor=1.5, passed=False,
            status="first_mapping_fit_cost_not_yet_measured", calibration_allocation_required=True))


def fit_stage(cfg, output, deadline):
    from .paired_data import prepare_dataset
    from .training import train_bridge
    from latentmol.train import state_digest

    vae, _ = load_vae(cfg, "cuda")
    core, info, receipt, boundary = load_native(cfg, "cuda")
    frozen_core_before = state_digest(core.state_dict())
    prepare_dataset(vae, cfg["dataset"]["path"], output / "paired-data",
        requested_sources=20000, max_source_attempts=100000, seed=17, heavy_atoms=None,
        deadline=deadline, artifact_cap_bytes=cfg["storage"]["own_initial_cap_bytes"])
    barrier(cfg, deadline)

    def schedule(times, reference):
        gamma = core.gamma(times)
        return core.alpha(gamma, reference), core.sigma(gamma, reference)

    fitting = cfg["fitting"]
    keys = ("steps", "batch_size", "learning_rate", "weight_decay", "evaluate_every", "seed",
            "hidden_width", "width", "min_atom_agreement", "max_embedding_nrmse")
    result = train_bridge(vae, boundary, output / "paired-data", output / "fit", "cuda", schedule,
        **{key: fitting[key] for key in keys}, deadline=deadline,
        calibrate_runtime=True,
        internal_seconds=cfg["stages"]["fit"]["internal_seconds"])
    barrier(cfg, deadline)
    frozen_core_after = state_digest(core.state_dict())
    core_unchanged = (frozen_core_before == frozen_core_after and not core.training
                      and not any(p.requires_grad or p.grad is not None for p in core.parameters()))
    audit_calls = 200 * 2 * 4 * 8  # epsilon and native zero-noise transition, four mappings.
    per_transition = prior(cfg, "scorer")["measured_max_native_transition_seconds"]
    seconds = 120 + 1.5 * audit_calls * per_transition
    return dict(checks=dict(mapping_feature_audit=result["status"] == "passed_feature_audit_not_native_validated",
        full_mapping_budget_completed=result["completed_steps"] == 10000,
        frozen_vae_and_native_boundary=result["frozen_vae_and_native_boundary_unchanged"] is True,
        full_frozen_diffusion_core_unchanged=core_unchanged),
        full_diffusion_state_sha256_before=frozen_core_before, full_diffusion_state_sha256_after=frozen_core_after,
        mapping_result=result, successor_forecast=dict(stage="audit", factor=1.5, seconds=seconds,
            passed=seconds <= cfg["stages"]["audit"]["internal_seconds"],
            basis="prespecified denoiser calls times slowest measured actual native transition plus120s startup; T2 cost separate"))


def load_bridge(cfg, vae, boundary, device):
    import torch
    from latentmol.train import state_digest
    from .bridge import ConstrainedAtomBridge

    prior(cfg, "fit")
    path = Path(cfg["run_root"]) / "fit/fit/best-bridge.pt"
    saved = torch.load(path, map_location=device, weights_only=True)
    if (saved["vae_checkpoint"] != vae.diffsbdd_checkpoint_identity
            or saved["vae_state_sha256"] != state_digest(vae.state_dict())
            or saved["native_boundary_state_sha256"] != state_digest(boundary.state_dict())
            or saved["dataset_manifest_sha256"] != sha(Path(cfg["run_root"]) / "fit/paired-data/manifest.json")):
        raise ValueError("Development-selected head has different frozen VAE/native/data bindings")
    config = saved["head_configuration"]
    bridge = ConstrainedAtomBridge(config["hidden_width"], config["width"], config["elements"]).to(device)
    bridge.load_state_dict(saved["bridge_state"], strict=True)
    bridge.eval().requires_grad_(False)
    return bridge


def _pose_from_saved(saved, device):
    from .transport import MolecularGraph, PoseCandidate

    graph = saved["graph"]
    return PoseCandidate(MolecularGraph(tuple(graph["elements"]), tuple(graph["charges"]),
        tuple(tuple(row) for row in graph["bonds"]), tuple(graph["atom_ids"])), saved["positions"].to(device))


def audit_stage(cfg, output, deadline):
    import torch
    from .audit import run_denoiser_audit
    from .native import NativeState, NativeStepper, GaussianStream
    from .runs import require_T2_bond_change_example
    from .transport import HandoffRejected, TransportPolicy

    vae, _ = load_vae(cfg, "cuda")
    core, info, receipt, boundary = load_native(cfg, "cuda")
    bridge = load_bridge(cfg, vae, boundary, "cuda")
    pocket, definition = pocket_assets(cfg, "cuda")
    audit = run_denoiser_audit(vae, boundary, bridge, Path(cfg["run_root"]) / "fit/paired-data",
        core, pocket, output / "denoiser", deadline=deadline)
    if audit["passed"] is not True:
        return dict(checks=dict(heldout_denoiser=False), denoiser=audit)
    barrier(cfg, deadline)
    build = factory_builder(cfg, vae, boundary, bridge, core, info, receipt, pocket, definition, deadline)
    factory = build(output / "T2-guidance", "learned", "T2")
    starts = json.loads((Path(cfg["run_root"]) / "scorer/qualified-starts.json").read_text())["starts"]
    attempts, examples, durations = [], [], []
    for index, row in enumerate(starts):
        barrier(cfg, deadline)
        path = Path(cfg["run_root"]) / "scorer" / row["path"]
        if sha(path) != row["sha256"]:
            raise ValueError("Prespecified qualified scorer start changed")
        saved = torch.load(path, map_location="cuda", weights_only=True)
        candidate, tick = _pose_from_saved(saved, "cuda"), time.monotonic()
        record = dict(index=index, seed=row["seed"], level=row["level"], applied=False)
        try:
            admission = factory.admit_terminal(candidate)
            proposal = factory.callbacks.propose(admission, 5)
            lifted = TransportPolicy("T2").lift(admission.baseline, proposal.graph, factory.callbacks.native_builder)
            before = dict(factory.callbacks.rebuilt_objective(admission.baseline))
            after = dict(factory.callbacks.rebuilt_objective(lifted.candidate))
            if after["total"] > before["total"]:
                raise HandoffRejected("rebuilt_native_objective_worsened")
            record.update(applied=lifted.changed, hard_graph_changed=proposal.graph != admission.baseline.graph,
                hard_bonds_changed=proposal.graph.bonds != admission.baseline.graph.bonds,
                pose_changed=not torch.equal(lifted.candidate.positions, admission.baseline.positions),
                transport=lifted.receipt, rebuilt_baseline=before, rebuilt_proposed=after,
                proposal=proposal.receipt, admission=admission.receipt)
            example = require_T2_bond_change_example(record)
            state = NativeState(**{key: value.to("cuda") if hasattr(value, "to") else value
                                   for key, value in saved["state"].items()})
            stepper = NativeStepper(core)
            baseline = stepper.clean_input(admission.baseline.positions, admission.native_features, state)
            proposed = stepper.clean_input(lifted.candidate.positions, proposal.native_features, state)
            corrected = stepper.correct(state, baseline, proposed)
            if torch.equal(state.ligand, corrected.ligand):
                raise RuntimeError("Qualified T2 molecule produced no native-state correction")
            next_state = stepper.reverse_step(corrected, GaussianStream(row["seed"]))
            torch.save(dict(before=_state_payload(state), corrected=_state_payload(corrected),
                            next=_state_payload(next_state)), output / f"T2-transition-{index:03d}.pt")
            examples.append(example)
        except HandoffRejected as error:
            record.update(reason=str(error), rejection_receipt=error.receipt)
        except RuntimeError as error:
            if str(error).startswith("Incomplete/failed T2 scientific transport example"):
                record.update(reason=str(error))
            else:
                raise
        durations.append(time.monotonic() - tick)
        record["elapsed_seconds"] = durations[-1]
        attempts.append(record)
        write(output / f"T2-attempt-{index:03d}.json", record)
    if examples:
        write(output / "T2-example.json", examples[0]["example"])
    maximum_native = prior(cfg, "scorer")["measured_max_native_transition_seconds"]
    # T1/T2 smoke: 2 transports *3seeds *4arms, conservatively charge all3
    # opportunities at the slowest measured complete T2 block per arm.
    forecast = 120 + 1.5 * 24 * (502 * maximum_native + 3 * max(durations, default=math.inf))
    if not math.isfinite(forecast):
        successor = dict(stage="smoke", factor=1.5, passed=False, status="no_measured_complete_T2_block")
    else:
        successor = dict(stage="smoke", factor=1.5, seconds=forecast,
            passed=forecast <= cfg["stages"]["smoke"]["internal_seconds"],
            basis="24 full500step arms and72 worst measured complete T2 guidance blocks plus120s startup")
    return dict(checks=dict(heldout_denoiser=audit["passed"] is True,
        actual_T2_bond_pose_native_recovery=bool(examples), all_prespecified_starts_recorded=len(attempts) == len(starts)),
        qualified_T2_examples=len(examples), T2_attempts=attempts, successor_forecast=successor)


def cohort_stage(cfg, output, deadline, stage):
    import torch
    from .native import NativeStepper
    from .runs import run_cohort, require_T2_bond_change_example

    vae, _ = load_vae(cfg, "cuda")
    core, info, receipt, boundary = load_native(cfg, "cuda")
    bridge = load_bridge(cfg, vae, boundary, "cuda")
    pocket, definition = pocket_assets(cfg, "cuda")
    native = prior(cfg, "native")["checks"]
    scorer = prior(cfg, "scorer")["checks"]
    audit = prior(cfg, "audit")["checks"]
    example = json.loads((Path(cfg["run_root"]) / "audit/T2-example.json").read_text())
    qualified = require_T2_bond_change_example(example)
    qualification = dict(native_replay=native["native_replay_and_boundary"],
        exact_boundary=native["native_replay_and_boundary"],
        codec=scorer["at_least_one_native_scorer_AD_qualified_start"],
        native_pose_context=scorer["at_least_one_native_scorer_AD_qualified_start"],
        derivatives=scorer["at_least_one_native_scorer_AD_qualified_start"],
        frozen_models=audit["heldout_denoiser"], T1_transport=scorer["T1_identity_transport"],
        T2_transport=qualified["passed"], T2_bond_change_example=example)
    build = factory_builder(cfg, vae, boundary, bridge, core, info, receipt, pocket, definition, deadline)
    results = {}
    for transport in (("T1", "T2") if stage == "smoke" else ("T2",)):
        barrier(cfg, deadline)
        results[transport] = run_cohort(NativeStepper(core), pocket, torch.tensor([25], device="cuda"),
            build, qualification, output / transport, stage=stage, deadline=deadline,
            transport_stage=transport, native_atom_decoder=info["atom_decoder"])
    barrier(cfg, deadline)
    checks = {transport + "_complete_with_effective_transfers":
              result["status"] == "completed_cohort_not_fresh_pose_validation"
              for transport, result in results.items()}
    result = dict(checks=checks, cohorts=results,
        fresh_pose_followup="pending_separate_required_stage_no_efficacy_claim")
    if stage == "smoke":
        times = [arm["elapsed_seconds"] for value in results.values() for arm in value["completed"]]
        seconds = 120 + 1.5 * 48 * max(times) * (20 / 3)
        result["successor_forecast"] = dict(stage="pilot", factor=1.5, seconds=seconds,
            passed=seconds <= cfg["stages"]["pilot"]["internal_seconds"],
            basis="48 arms, slowest completed smoke arm scaled from3 to20 guidance opportunities plus120s startup")
    return result


def followup_stage(cfg, output, deadline):
    """New CPU-only parent; sequential native children own CUDA individually."""
    from .followup import run_followup

    prior(cfg, "pilot")
    prepared = json.loads((Path(cfg["run_root"]) / "cpu/native-followup-assets.json").read_text())
    config = dict(python=cfg["runtime"], run_root=cfg["run_root"],
        owned_cache=str(Path("/work3/s234818/latentmol/cache") / cfg["campaign"] / "followup-native"),
        assets=prepared["assets"], protein=prepared["protein"],
        diffusion_source=dict(path=str(cfg["_source_root"] / cfg["upstream"]["source_relative"]),
                              manifest_sha256=cfg["upstream"]["source_manifest_sha256"]),
        prediction=cfg["boltz"]["fresh_pose_prediction"], startup_margin_seconds=300,
        bootstrap_forecast=cfg["followup"]["bootstrap_forecast"],
        artifact_cap_bytes=cfg["storage"]["own_initial_cap_bytes"])
    # This function creates its own subdirectory; the outer stage stays owned
    # by the standard stage inventory and has no asset symlinks.
    result = run_followup(config, Path(cfg["run_root"]) / "pilot/T2", output / "native", deadline)
    return dict(checks=dict(all_required_fresh_pose_predictions=result.get("passed") is True,
        full_pilot_membership=result.get("terminal_memberships") == 48), followup=result,
        interpretation="exploratory model confirmation; no experimental binding or hybrid efficacy claim")


def finalize_output(output, stage, cfg, result, started):
    """Complete inventory, avoiding a manifest/decision self-hash cycle."""
    checks = result.get("checks", {})
    if not isinstance(checks, dict) or not checks or any(type(value) is not bool for value in checks.values()):
        raise ValueError("Stage checks must be a nonempty mapping of actual boolean decisions")
    inventory = {path.relative_to(output).as_posix(): sha(path) for path in output.rglob("*") if path.is_file()}
    write(output / "output-manifest.json", dict(version="diffsbdd-stage-outputs-v1", files=inventory))
    receipt = dict(result, version=STAGE_VERSION, stage=stage, passed=all(checks.values()),
        config_sha256=cfg["_config_sha256"], source_manifest_sha256=cfg["_source_manifest_sha256"],
        elapsed_seconds=time.monotonic() - started,
        output_manifest=dict(path="output-manifest.json", sha256=sha(output / "output-manifest.json")))
    write(output / "stage.json", receipt)
    return receipt


def run(config, stage, output):
    cfg, deadline = configuration(config, stage, output)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = dict(checks={"stage_completed": False})
    try:
        # Torch is imported only after allocation, config/source and output
        # namespace validation. native_stage launches its original child before
        # any combined-runtime CUDA initialization.
        if stage != "followup":
            import torch
            torch.set_num_threads(min(4, int(os.environ.get("LSB_DJOB_NUMPROC", "4"))))
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        barrier(cfg, deadline)
        handlers = dict(cpu=cpu_stage, native=native_stage, scorer=scorer_stage,
                        fit=fit_stage, audit=audit_stage, followup=followup_stage)
        result = handlers[stage](cfg, output, deadline) if stage in handlers else cohort_stage(cfg, output, deadline, stage)
        barrier(cfg, deadline)
    except BaseException as error:
        result = dict(checks={"stage_completed": False}, error_type=type(error).__name__, error=str(error))
        (output / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
    receipt = finalize_output(output, stage, cfg, result, started)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config, args.stage, args.output)
    print(json.dumps(dict(stage=args.stage, passed=result["passed"], output=str(args.output))))
    if result["passed"] is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
