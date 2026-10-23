"""Replay a failed active-set derivative gate without optimizing a molecule.

This is a diagnostic, never a substitute validation gate. It retains the exact
saved latent, selected mask, native context, objective and frozen weights. Wider
finite-difference steps and grad-enabled forward probes distinguish cancellation,
local curvature, nondeterminism and different inference paths. Merely converting
a rounded float32 output to Python float/float64 does not recover lost precision.
Native execution requires an immutable release in a DTU allocation. The focused
CPU profile explicitly records its different backend; it cannot approve the
original CUDA gate. No model or optimizer weights are updated in either profile.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import time

import torch

from latentmol.evaluate import load_model
from latentmol.experiments import ExperimentRun, file_sha256, source_manifest, write_json
from latentmol.graph_policy import graph_from_decoder
from latentmol.multimodel.engine import Counters, scalar_loss
from latentmol.multimodel.study import module_digest, tensor_digest, verified_file
from .active_set import ActiveSetConfig, ContextBank, Selection, make_branch, select_slots
from .adapter import adapt_graph
from .model import FrozenBoltzAffinity, load_affinity_model


# Smaller steps diagnose sharp curvature in random-prior starts; larger steps
# diagnose cancellation in tiny projected derivatives. This grid is diagnostic
# only and cannot convert an already failed production gate into a pass.
EPSILONS = (0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1)


def probe_schedule(profile: str) -> dict:
    """Prespecified diagnostics; neither profile changes a production FD gate."""
    if profile == "extended":
        return {"epsilons": EPSILONS, "random_directions": 3, "base_repeats_per_mode": 3,
                "backward_repeats": 2, "both_modes_at_all_epsilons": False}
    if profile == "focused_cpu":
        return {"epsilons": (0.00001, 0.0001, 0.001), "random_directions": 0,
                "base_repeats_per_mode": 2, "backward_repeats": 1,
                "both_modes_at_all_epsilons": True}
    raise ValueError("Unknown diagnostic probe profile")


def runtime_description(device) -> dict:
    """CPU diagnostics must not query a CUDA device or initialize its RNG."""
    device = torch.device(device)
    result = {"backend": device.type, "torch_num_threads": torch.get_num_threads(),
              "torch_num_interop_threads": torch.get_num_interop_threads(),
              "torch_cuda_build_version": torch.version.cuda,
              "float32_matmul_precision": torch.get_float32_matmul_precision(),
              "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
              "cudnn_deterministic": torch.backends.cudnn.deterministic,
              "cudnn_benchmark": torch.backends.cudnn.benchmark,
              "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
              "autocast_enabled": torch.is_autocast_enabled(device.type)}
    if device.type == "cuda":
        result.update(cuda_device_name=torch.cuda.get_device_name(device),
                      cuda_compute_capability=list(torch.cuda.get_device_capability(device)),
                      cudnn_version=torch.backends.cudnn.version())
    return result


def rng_digest(device) -> str:
    """Record only RNG streams used by the selected diagnostic backend."""
    states = {"cpu": torch.get_rng_state()}
    if torch.device(device).type == "cuda":
        states["cuda"] = torch.cuda.get_rng_state(device)
    return tensor_digest(states)


def prepared_mask_report(features: dict) -> dict:
    """Reconstruct pinned trunk/affinity masks without altering native execution.

    An all-masked softmax row has no valid keys. Counts describe prepared masks,
    not whether its output influences the measured affinity. In particular,
    ordinary receptor rows have ligand keys and must not be misclassified as
    all-masked merely because receptor-to-receptor pairs are masked.
    """
    pad = features["token_pad_mask"].bool()
    receptor = (features["mol_type"] == 0) & pad
    ligand = features["affinity_token_mask"].bool() & pad
    trunk = pad[:, :, None] & pad[:, None, :]
    affinity = ((ligand[:, :, None] & receptor[:, None, :]) |
                (receptor[:, :, None] & ligand[:, None, :]) |
                (ligand[:, :, None] & ligand[:, None, :]))

    def describe(mask):
        empty = ~mask.any(dim=-1)
        return {"shape": list(mask.shape), "fully_masked_rows": int(empty.sum()),
                "fully_masked_nonpadding_rows": int((empty & pad).sum()),
                "fully_masked_receptor_rows": int((empty & receptor).sum()),
                "fully_masked_ligand_rows": int((empty & ligand).sum()),
                "fully_masked_token_indices": [row.nonzero().flatten().tolist() for row in empty]}

    atom_pad = features.get("atom_pad_mask")
    return {"token_pad_zero": int((~pad).sum()), "token_pad_nonzero": int(pad.sum()),
            "atom_pad_zero": int((~atom_pad.bool()).sum()) if atom_pad is not None else None,
            "atom_pad_nonzero": int(atom_pad.bool().sum()) if atom_pad is not None else None,
            "receptor_tokens": int(receptor.sum()), "ligand_tokens": int(ligand.sum()),
            "trunk_pair_attention_key_mask": describe(trunk),
            "affinity_pair_attention_key_mask": describe(affinity),
            "scope": "Token masks reconstructed from pinned native formulas; atom padding counted without reconstructing local atom-attention windows; no kernel hook or causal attribution."}


def tensor_description(value: torch.Tensor, *, scalar=False) -> dict:
    """Describe actual tensor arithmetic precision before widening for JSON."""
    result = {"dtype": str(value.dtype), "shape": list(value.shape),
              "device": str(value.device), "requires_grad": value.requires_grad}
    if scalar:
        x = value.detach().reshape(()).cpu()
        result["value"] = float(x)
        # Both actual-dtype and float32 ULPs are recorded, not an asserted error
        # bound for the entire network (accumulated arithmetic can exceed a ULP).
        for name, number in (("output_ulp", x), ("float32_ulp", x.float())):
            up = torch.nextafter(number, torch.full_like(number, float("inf")))
            down = torch.nextafter(number, torch.full_like(number, float("-inf")))
            result[name] = max(float(up.double() - number.double()),
                               float(number.double() - down.double()))
    return result


def finite_difference_row(plus: dict, minus: dict, analytic: float, epsilon: float,
                          *, rtol: float, atol: float, baseline=None) -> dict:
    """Analyze already measured scalar outputs; no tolerances are adapted."""
    fplus, fminus = plus["loss"]["value"], minus["loss"]["value"]
    difference = fplus - fminus
    numerical = difference / (2 * epsilon)
    error = abs(analytic - numerical)
    scale = max(abs(analytic), abs(numerical))
    ulp = max(plus["loss"]["float32_ulp"], minus["loss"]["float32_ulp"])
    result = {"epsilon": epsilon, "f_plus": fplus, "f_minus": fminus,
            "analytic": analytic, "central_difference": numerical,
            "absolute_error": error, "relative_error": error / scale if scale else 0.0,
            "rtol": rtol, "atol": atol, "passed_original_tolerance": error <= atol + rtol * scale,
            "output_difference": difference, "float32_output_ulp": ulp,
            "difference_in_float32_ulps": abs(difference) / ulp if ulp else None,
            "expected_signal_in_float32_ulps": abs(2 * epsilon * analytic) / ulp if ulp else None,
            "one_output_ulp_derivative_scale": ulp / (2 * epsilon),
            "plus": plus, "minus": minus}
    if baseline is not None:
        forward, backward = (fplus - baseline) / epsilon, (baseline - fminus) / epsilon
        result.update(baseline_loss=baseline, forward_slope=forward, backward_slope=backward,
                      forward_backward_slope_difference=forward - backward,
                      central_second_difference=(fplus - 2 * baseline + fminus) / epsilon ** 2)
    return result


def realized_step(point, direction, gradient, epsilon):
    """Measure representable latent displacements in widened arithmetic.

    Perturbations are formed in the production dtype first. Float64 is used only
    to measure them, so a rounded-away float32 perturbation stays visible as zero.
    """
    plus, minus = point + epsilon * direction, point - epsilon * direction
    pdelta, mdelta = plus.double() - point.double(), point.double() - minus.double()
    return plus, minus, {
        "analytic_dot_float64": float((gradient.double() * direction.double()).sum()),
        "realized_plus_projection": float((pdelta * direction.double()).sum()),
        "realized_minus_projection": float((mdelta * direction.double()).sum()),
        "realized_plus_distance": float(pdelta.norm()),
        "realized_minus_distance": float(mdelta.norm()),
        "actual_secant_linear_prediction": float((gradient.double() * (pdelta + mdelta)).sum()) / (2 * epsilon)}


def repeat_comparison(repeats, original_loss):
    """Report output drift independently of whether any finite difference passes."""
    no_grad = [row["loss"]["value"] for row in repeats if not row["grad_enabled"]]
    grad = [row["loss"]["value"] for row in repeats if row["grad_enabled"]]
    return {"no_grad_range": max(no_grad) - min(no_grad),
            "grad_enabled_range": max(grad) - min(grad),
            "first_grad_minus_no_grad": grad[0] - no_grad[0],
            "first_grad_minus_original": grad[0] - original_loss,
            "limit": "Three identical repeats cannot prove every kernel is deterministic."}


def _probe(decoder, point, branch, config, counters, *, grad_enabled, backward=False,
           reject_nonfinite_objective=False):
    """Grad probes activate production checkpoint wrappers; free each graph here."""
    if backward and not grad_enabled:
        raise ValueError("Backward requires a grad-enabled probe")
    started = time.perf_counter()
    probe = point.detach().clone().requires_grad_(grad_enabled)
    purpose = "analytic_repeat" if backward else ("grad_forward" if grad_enabled else "no_grad_forward")
    with torch.set_grad_enabled(grad_enabled):
        decoded = decoder.decode(probe)
        graph = graph_from_decoder(decoded, decoder.spec, temperature=config.temperature,
                                   fixed_active=branch.prepared.active)
        output = counters.call("boltz", purpose, "forward", lambda: branch(graph), probe.device)
        if reject_nonfinite_objective:
            # Comparison-only classification: malformed/missing heads retain the
            # original scalar_loss exception. Only an observed nonfinite scalar
            # tensor is a retryable numerical trial; scorer arithmetic is intact.
            value = output.get(config.objective.output)
            if isinstance(value, torch.Tensor) and value.numel() == 1 and not bool(value.isfinite().all()):
                from latentmol.optimization import NonfiniteObjective
                raise NonfiniteObjective("Observed nonfinite scalar objective tensor")
        loss, _ = scalar_loss(output, config.objective)
        report = {"grad_enabled": grad_enabled, "backward": backward,
                  "heads": {key: tensor_description(value, scalar=True) for key, value in output.items()},
                  "loss": tensor_description(loss, scalar=True),
                  "natural_selection": select_slots(decoded, temperature=config.temperature,
                        threshold=config.threshold, require_nonempty=False).record()}
        report["natural_selector_crossed"] = report["natural_selection"]["active_slots"] != (
            branch.prepared.active.nonzero().flatten().tolist())
        report["forward_seconds"] = time.perf_counter() - started
        print("DIAGNOSTIC_PROBE=" + json.dumps({"phase": "forward", "purpose": purpose,
            "grad_enabled": grad_enabled, "device": str(probe.device), "loss": report["loss"]["value"],
            "seconds": report["forward_seconds"]}), flush=True)
        gradient = None
        if backward:
            backward_started = time.perf_counter()
            gradient, = counters.call("boltz", purpose, "backward", lambda:
                torch.autograd.grad(loss, probe), probe.device)
            gradient = gradient.detach()
            if not bool(gradient.isfinite().all()):
                raise ValueError("Nonfinite replay gradient")
            report["gradient"] = {**tensor_description(gradient),
                                  "norm_float64": float(gradient.double().norm()),
                                  "sha256": tensor_digest({"gradient": gradient})}
            report["backward_seconds"] = time.perf_counter() - backward_started
            print("DIAGNOSTIC_PROBE=" + json.dumps({"phase": "backward", "purpose": purpose,
                "device": str(probe.device), "gradient_norm": report["gradient"]["norm_float64"],
                "seconds": report["backward_seconds"]}), flush=True)
        report["total_seconds"] = time.perf_counter() - started
    return report, gradient


@torch.no_grad()
def _dtype_report(decoder, point, branch, config):
    decoded = decoder.decode(point)
    graph = graph_from_decoder(decoded, decoder.spec, temperature=config.temperature,
                               fixed_active=branch.prepared.active)
    features, bonds = adapt_graph(graph, branch.prepared)
    return {"latent": tensor_description(point),
            "decoder_logits": {key: tensor_description(getattr(decoded, key)) for key in ("atoms", "charges", "bonds")},
            "relaxed_graph": {key: tensor_description(getattr(graph, key)) for key in ("atoms", "charges", "bonds")},
            "adapted_features": {key: tensor_description(value) for key, value in features.items() if isinstance(value, torch.Tensor)},
            "adapted_bond_probabilities": tensor_description(bonds),
            "coordinates": tensor_description(branch.prepared.coordinates),
            "model_state": {key: tensor_description(value) for key, value in branch.scorer.state_dict().items()},
            "decoder_state": {key: tensor_description(value) for key, value in decoder.state_dict().items()}}


def _stage_inputs(decoder, point, branch, config, *, grad_enabled):
    """Capture only varying channels to localize a grad/no-grad mode difference."""
    with torch.set_grad_enabled(grad_enabled):
        probe = point.detach().clone().requires_grad_(grad_enabled)
        decoded = decoder.decode(probe)
        graph = graph_from_decoder(decoded, decoder.spec, temperature=config.temperature,
                                   fixed_active=branch.prepared.active)
        features, bonds = adapt_graph(graph, branch.prepared)
        tensors = {f"decoder.{key}": getattr(decoded, key) for key in ("atoms", "charges", "bonds")}
        tensors.update({f"graph.{key}": getattr(graph, key) for key in ("atoms", "charges", "bonds")})
        tensors.update({f"adapter.{key}": features[key] for key in ("ref_element", "ref_charge", "token_bonds")})
        tensors["adapter.bond_probabilities"] = bonds
        descriptions = {key: tensor_description(value) for key, value in tensors.items()}
        return descriptions, {key: value.detach().cpu() for key, value in tensors.items()}


def _stage_comparison(decoder, point, branch, config):
    no_info, no_tensors = _stage_inputs(decoder, point, branch, config, grad_enabled=False)
    grad_info, grad_tensors = _stage_inputs(decoder, point, branch, config, grad_enabled=True)
    return {key: {"no_grad": no_info[key], "grad_enabled": grad_info[key],
                  "no_grad_sha256": tensor_digest({key: value}),
                  "grad_enabled_sha256": tensor_digest({key: grad_tensors[key]}),
                  "maximum_absolute_difference": float((value.double() - grad_tensors[key].double()).abs().max())}
            for key, value in no_tensors.items()}


def gate_paths(parent: Path, arm: str, step: int) -> tuple[Path, Path]:
    """Resolve an explicit saved gate/state pair without guessing a failed step."""
    if arm not in ("active_set", "fixed_initial_mask"):
        raise ValueError("Require a declared optimization arm")
    if type(step) is not int or step < 0 or step > 9999:
        raise ValueError("Require an integer saved step from 0 through 9999")
    return parent / arm / f"gate-{step:04d}.json", parent / arm / f"state-{step:04d}.pt"


def _parent_inputs(parent: Path, *, arm="active_set", step=0, production_root=None):
    """Verify the recorded point and original production code before replaying.

    A failed transition can occur after optimization has already changed z.
    Only an initial gate must equal start.pt. Fixed-mask replay retains the
    original mask even when the decoder's later natural selector differs.
    """
    record = json.loads((parent / "run.json").read_text())
    gate_path, state_path = gate_paths(parent, arm, step)
    gate = json.loads(gate_path.read_text())
    if record.get("status") != "failed" or gate.get("status") != "failed" or gate.get("passed") is not False:
        raise ValueError("Require a failed saved derivative gate, not an optimization endpoint")
    if not gate.get("native_seed_parity") or not gate.get("finite_differences") or "loss" not in gate.get("measured", {}):
        raise ValueError("Require a derivative-gate failure after native parity and analytic scoring")
    config_path = verified_file(record["config"]["config_path"], record["config"]["config_file_sha256"])
    config = ActiveSetConfig.load(config_path)
    for key, value in config.record().items():
        # Dataclass tuples serialize as lists in the parent JSON record.
        if json.loads(json.dumps(value)) != record["config"][key]:
            raise ValueError(f"Parent config record differs from verified config: {key}")
    current = source_manifest(Path(production_root) if production_root else Path.cwd())
    if production_root is not None:
        if current["sha256"] != record["source"]["sha256"] or current["files"] != record["source"]["files"]:
            raise ValueError("Original production release differs from the failed run")
    else:
        for name in record["source"]["files"]:
            if not name.startswith("src/latentmol/") or name == "src/latentmol/boltz/active_set_diagnostic.py":
                continue
            if current["files"].get(name) != record["source"]["files"].get(name):
                raise ValueError(f"Production code changed since failed run: {name}")
    paths = [config_path, *[parent / name for name in ("run.json", "start.pt", "start-audit.json")],
             state_path, gate_path]
    hashes = {str(path): file_sha256(path) for path in paths}
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    start = torch.load(parent / "start.pt", map_location="cpu", weights_only=True)
    audit = json.loads((parent / "start-audit.json").read_text())
    if not audit["passed"] or tensor_digest({"z": state["z"]}) != gate["point_sha256"]:
        raise ValueError("Failed point does not match the saved audited gate state")
    if step == 0 and not torch.equal(state["z"], start["z"]):
        raise ValueError("Failed initial gate state differs from saved start")
    if state["context_sha256"] != gate["context_sha256"]:
        raise ValueError("Failed gate context differs from saved state")
    if state["active"].nonzero().flatten().tolist() != gate["active_slots"]:
        raise ValueError("Saved mask differs from failed gate")
    return record, gate, config, state, hashes


def run(parent: Path, run_root: Path, *, arm="active_set", step=0,
        production_root=None, device_override=None, probe_profile="extended") -> Path:
    """Replay a saved failure; completion reports data, not a passing gate."""
    if not os.environ.get("LSB_JOBID") or not os.environ.get("LATENTMOL_SOURCE_SHA256"):
        raise ValueError("Require an HPC allocation and immutable source archive identity")
    record, old_gate, config, state, parent_hashes = _parent_inputs(
        parent.resolve(), arm=arm, step=step, production_root=production_root)
    schedule = probe_schedule(probe_profile)
    if device_override not in (None, "cpu"):
        raise ValueError("Only an explicit CPU diagnostic device override is supported")
    if probe_profile == "focused_cpu" and device_override != "cpu":
        raise ValueError("The focused CPU profile requires the recorded CPU override")
    inputs = {"parent_run": str(parent.resolve()), "parent_artifact_sha256": parent_hashes,
              "parent_arm": arm, "parent_step": step,
              "parent_source_sha256": record["source"]["sha256"],
              "parent_source_archive_sha256": record["source_archive_sha256"],
              "production_release": str(Path(production_root).resolve()) if production_root else str(Path.cwd()),
              "diagnostic_module_sha256": file_sha256(Path(__file__)),
              "original_gate": old_gate, "original_config": config.record(),
              "probe_schedule": schedule, "probe_profile": probe_profile,
              "diagnostic_device_override": device_override,
              "backend_claim_limit": "CPU uses different arithmetic kernels and cannot validate or approve the original CUDA gate." if device_override == "cpu" else "Original CUDA backend retained.",
              "epsilons": schedule["epsilons"],
              "random_seeds": [config.diagnostics.seed + index for index in range(schedule["random_directions"])],
              "claim_limit": "Diagnostic only: no optimizer, tolerance change, automatic gate approval or molecular result."}
    # When replaying an older release, its ExperimentRun also uses its older
    # metadata implementation. Normalize mapping keys before passing the record
    # so integer count keys cannot create a fresh legacy JSON hash false alarm.
    inputs = json.loads(json.dumps(inputs, allow_nan=False))
    if device_override is not None:
        # Loading/validating the original config above remains strict. Only this
        # diagnostic execution selects another backend, recorded independently
        # of the original objective, weights, graph and context identities.
        config = replace(config, device=device_override)
    with ExperimentRun(run_root, "active-set-fd-diagnostic", inputs, Path.cwd()) as experiment:
        report = {"status": "running", "repeats": [], "finite_differences": [], "original_gate": old_gate,
                  "claim_limit": inputs["claim_limit"], "float_widening_recovers_lost_precision": False}
        def persist():
            write_json(experiment.path / "diagnostic.json", report)
        counters = Counters(lambda value: write_json(experiment.path / "oracle_cost.json", value))
        persist()
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            decoder_path = verified_file(config.decoder["path"], config.decoder["sha256"])
            affinity_path = verified_file(config.affinity_checkpoint["path"], config.affinity_checkpoint["sha256"])
            decoder, training = load_model(decoder_path, config.device)
            if training["train_config"]["mode"] != config.mode:
                raise ValueError("Decoder mode differs from failed run")
            decoder.eval().requires_grad_(False)
            bank = ContextBank(config, decoder.spec)
            point = state["z"].to(config.device)
            selection = Selection(state["active"].to(config.device), state["no_atom"].to(config.device), config.threshold)
            with torch.no_grad():
                actual_selection = select_slots(decoder.decode(point), temperature=config.temperature,
                    threshold=config.threshold, require_nonempty=False)
            if (arm == "active_set" or step == 0) and not torch.equal(actual_selection.active, selection.active):
                raise ValueError("Replayed decoder selector differs from saved gate mask")
            if arm == "fixed_initial_mask":
                start_state = torch.load(parent / "start.pt", map_location="cpu", weights_only=True)
                with torch.no_grad():
                    starting_selection = select_slots(decoder.decode(start_state["z"].to(config.device)),
                        temperature=config.temperature, threshold=config.threshold)
                if not torch.equal(starting_selection.active, selection.active):
                    raise ValueError("Fixed-mask gate differs from the audited starting mask")
            scorer = FrozenBoltzAffinity(load_affinity_model(affinity_path).to(config.device), config.recycling_steps)
            branch = make_branch(scorer, bank, selection, config, experiment.record["source"]["sha256"])
            if branch.manifest.context_sha256 != old_gate["context_sha256"]:
                raise ValueError("Bound native context does not match failed gate")
            identities = {"decoder_state_sha256": module_digest(decoder), "scorer_state_sha256": module_digest(scorer),
                          "context_sha256": branch.verify_context_unchanged()}
            for key in ("decoder_state_sha256", "scorer_state_sha256"):
                if identities[key] != record[key]:
                    raise ValueError(f"Frozen state differs from failed run: {key}")
            report.update(identities_before=identities, activation_checkpoint_policy=scorer.activation_checkpoint_policy,
                          tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
                          runtime=runtime_description(point.device),
                          backend_claim_limit=inputs["backend_claim_limit"],
                          rng_policy="Continuous forward RNG stream, as in production; only direction generators have fixed seeds. Native eval dropout may consume RNG with an all-one mask; RNG movement alone is not output nondeterminism.")
            report["prepared_masks"] = prepared_mask_report(branch.prepared.features)
            # Reconfirm native hard-input parity independently of the old report.
            # The relaxed failed point is still probed using its exact saved mask.
            with torch.no_grad():
                native = counters.call("boltz", "replayed_seed_parity", "native_forward", lambda:
                    scorer.discrete_reference(branch.prepared.features, branch.prepared.coordinates), point.device)
                adapted = counters.call("boltz", "replayed_seed_parity", "forward", lambda:
                    branch(branch.prepared.seed.to_relaxed(decoder.spec)), point.device)
            if set(native) != set(adapted):
                raise ValueError("Replay native and adapted heads disagree")
            report["replayed_native_seed_parity"] = {}
            for key in native:
                torch.testing.assert_close(adapted[key], native[key],
                    rtol=config.diagnostics.parity_rtol, atol=config.diagnostics.parity_atol)
                if not bool(native[key].isfinite().all()) or not bool(adapted[key].isfinite().all()):
                    raise ValueError(f"Nonfinite replay native parity for {key}")
                report["replayed_native_seed_parity"][key] = True
            persist()
            write_json(experiment.path / "tensor-dtypes.json", _dtype_report(decoder, point, branch, config))
            report["input_stage_mode_comparison"] = _stage_comparison(decoder, point, branch, config)
            persist()
            report["repeat_rng_before_sha256"] = rng_digest(point.device)
            gradients = []
            for grad_enabled in (False, True):
                for repeat in range(schedule["base_repeats_per_mode"]):
                    measured, gradient = _probe(decoder, point, branch, config, counters,
                                               grad_enabled=grad_enabled,
                                               backward=grad_enabled and repeat < schedule["backward_repeats"])
                    measured["repeat"] = repeat
                    report["repeats"].append(measured)
                    if gradient is not None:
                        gradients.append(gradient)
                    persist()
            report["repeat_rng_after_sha256"] = rng_digest(point.device)
            gradient = gradients[0]
            norm = float(gradient.double().norm())
            if norm <= 1e-12:
                raise ValueError("Replay has no nonzero gradient")
            report["gradient_repeat"] = {"backward_repeats": len(gradients),
                                         "max_absolute_difference": float((gradients[0] - gradients[1]).abs().max()) if len(gradients) > 1 else None,
                                         "difference_norm_float64": float((gradients[0] - gradients[1]).double().norm()) if len(gradients) > 1 else None,
                                         "original_norm": old_gate["gradient_norm"], "replayed_norm": norm,
                                         "repeatability_measured": len(gradients) > 1}
            report["forward_repeat"] = repeat_comparison(report["repeats"], old_gate["measured"]["loss"])
            # Reproduce production normalization exactly, including float32 random
            # norm rather than silently changing its direction arithmetic.
            directions = {"gradient": gradient / norm}
            for index in range(schedule["random_directions"]):
                generator = torch.Generator(device="cpu").manual_seed(config.diagnostics.seed + index)
                random = torch.randn(point.shape, generator=generator).to(point)
                directions["random" if index == 0 else f"random_{index + 1}"] = random / random.norm()
            torch.save({"z": point.cpu(), "gradient_repeats": [x.cpu() for x in gradients],
                        "directions": {key: value.cpu() for key, value in directions.items()}}, experiment.path / "probe-state.pt")
            for label, direction in directions.items():
                analytic = float((gradient * direction).sum())
                for epsilon in schedule["epsilons"]:
                    # All directions cover the prespecified larger scales. Old
                    # directions additionally compare inference/autograd math at
                    # every original epsilon without retaining graphs between calls.
                    both_modes = (schedule["both_modes_at_all_epsilons"] or
                                  label in ("gradient", "random") and epsilon in config.diagnostics.fd_epsilons)
                    modes = (False, True) if both_modes else (False,)
                    for grad_enabled in modes:
                        plus_point, minus_point, realized = realized_step(point, direction, gradient, epsilon)
                        plus, _ = _probe(decoder, plus_point, branch, config, counters, grad_enabled=grad_enabled)
                        minus, _ = _probe(decoder, minus_point, branch, config, counters, grad_enabled=grad_enabled)
                        row = finite_difference_row(plus, minus, analytic, epsilon,
                              rtol=config.diagnostics.fd_rtol, atol=config.diagnostics.fd_atol,
                              baseline=next(item["loss"]["value"] for item in report["repeats"]
                                            if item["grad_enabled"] == grad_enabled))
                        row.update(direction=label, grad_enabled=grad_enabled,
                                   natural_selector_crossed=plus["natural_selector_crossed"] or minus["natural_selector_crossed"],
                                   **realized)
                        report["finite_differences"].append(row)
                        persist()
            after = {"decoder_state_sha256": module_digest(decoder), "scorer_state_sha256": module_digest(scorer),
                     "context_sha256": branch.verify_context_unchanged()}
            report["identities_after"] = after
            if after != identities or any(file_sha256(Path(path)) != sha for path, sha in parent_hashes.items()):
                raise ValueError("Frozen state, context or parent artifact changed during diagnosis")
            report.update(status="completed", frozen_state_unchanged=True, oracle_cost=counters.values)
            persist()
            experiment.annotate(diagnostic_completed=True, frozen_state_unchanged=True,
                                original_gate_not_reclassified=True)
            return experiment.path
        except Exception as exc:
            report.update(status="failed", error=str(exc), error_type=type(exc).__name__, oracle_cost=counters.values)
            persist()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--arm", choices=("active_set", "fixed_initial_mask"), default="active_set")
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--device-override", choices=("cpu",))
    parser.add_argument("--probe-profile", choices=("extended", "focused_cpu"), default="extended")
    args = parser.parse_args()
    print(run(args.parent_run, args.run_root, arm=args.arm, step=args.step,
              device_override=args.device_override, probe_profile=args.probe_profile), flush=True)


if __name__ == "__main__":
    main()
