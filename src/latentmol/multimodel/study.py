"""Load, audit and optimize frozen branches under an explicit common graph policy.

Native models and the full HPC workflow are UNTESTED in this integration. A
successful local fixture test establishes orchestration only. This module never
retrains a scorer or converts chemistry inside the differentiable path.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from latentmol.evaluate import load_model
from latentmol.design.latent import count_band_penalty
from latentmol.experiments import file_sha256, write_json
from latentmol.graph_policy import graph_from_decoder
from latentmol.representation import HardGraphBatch
from latentmol.sampling import sample_hard_graph
from latentmol.scoring import evaluate_branches, require_frozen_eval
from .config import digest
from .engine import Counters, scalar_loss, score_graph
from .registry import factories, load_branch
from latentmol.boltz.gradient_precision import relative_ad_consistency, scalar_replay_consistency


GRADIENT_POLICY = "multimodel-pointwise-ad-v2"
LEGACY_GRADIENT_POLICY = "multimodel-finite-difference-v1"


def verified_file(path, expected):
    """Placeholders can be planned but never loaded as verified assets."""
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError(f"Replace the SHA256 placeholder for {path}")
    path = Path(path)
    if file_sha256(path) != expected:
        raise ValueError(f"Asset SHA256 mismatch: {path}")
    return path


def tensor_digest(tensors):
    h = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        h.update(f"{name}|{value.dtype}|{tuple(value.shape)}".encode())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def module_digest(module):
    return tensor_digest(dict(module.state_dict()))


def hard_digest(hard):
    return tensor_digest({key: getattr(hard, key) for key in ("atoms", "charges", "bonds")})


def selected_models(study, arm_names):
    """Zero weight means disabled: do not load that model or spend oracle calls."""
    if not arm_names or len(set(arm_names)) != len(arm_names) or set(arm_names) - set(study.arms):
        raise ValueError("Select nonempty unique known arms")
    return sorted({name for arm_name in arm_names for name in study.arms[arm_name].models
                   if study.weight(study.arms[arm_name], name) > 0})


def plan(study, arm_names):
    lookup = factories(study.plugins)
    enabled = selected_models(study, arm_names)
    for name in enabled:
        if study.models[name].plugin not in lookup:
            raise ValueError(f"Unknown model plugin: {study.models[name].plugin}")
    return {"status": "planned_not_executed", "config_sha256": study.fingerprint(),
            "models_to_load": enabled, "arms": list(arm_names),
            "factories": {name: lookup[study.models[name].plugin] for name in enabled},
            "occupancy": study.occupancy, "latents": "raw decoder coordinates; no AE/VAE distance calibration",
            "native_validation": "required; model names alone do not establish compatibility",
            "config": study.to_dict()}


@dataclass
class Problem:
    decoder: object
    runtimes: dict
    seed: HardGraphBatch
    active: object
    start: torch.Tensor
    identity: dict

    @property
    def spec(self):
        return self.decoder.spec

    def graph(self, point, temperature):
        return graph_from_decoder(self.decoder.decode(point), self.spec,
                                  temperature=temperature, fixed_active=self.active)

    def verify(self):
        require_frozen_eval(self.decoder)
        if module_digest(self.decoder) != self.identity["decoder_state_sha256"]:
            raise ValueError("Frozen decoder state changed")
        if hard_digest(self.seed) != self.identity["seed_sha256"]:
            raise ValueError("Canonical seed changed")
        for name, runtime in self.runtimes.items():
            require_frozen_eval(runtime.branch)
            if module_digest(runtime.branch) != self.identity["branches"][name]["state_sha256"]:
                raise ValueError(f"{name}: frozen weights/buffers changed")
            if runtime.verify:
                runtime.verify()


def load_problem(study, arm_names, source_sha256):
    """Verify assets before native factory execution; every prepared seed must agree."""
    plan(study, arm_names)
    decoder_path = verified_file(study.decoder["checkpoint"], study.decoder["sha256"])
    decoder, _ = load_model(decoder_path, study.device)
    decoder.eval().requires_grad_(False)
    runtimes = {}
    for name in selected_models(study, arm_names):
        cfg = study.models[name]
        runtimes[name] = load_branch(cfg.plugin, spec=decoder.spec, device=study.device,
                                    options=cfg.options, source_sha256=source_sha256, extra=study.plugins)
    seed = next(iter(runtimes.values())).seed
    seed_sha = hard_digest(seed)
    active = seed.atoms[0] != 0 if study.occupancy == "fixed" else None
    for name, runtime in runtimes.items():
        if hard_digest(runtime.seed) != seed_sha:
            raise ValueError(f"{name}: native bundle does not map to the identical canonical seed")
        if runtime.branch.spec != decoder.spec:
            raise ValueError(f"{name}: native graph schema differs")
        if runtime.branch.manifest.capabilities.occupancy == "fixed":
            if active is None or runtime.active is None or not torch.equal(active, runtime.active):
                raise ValueError(f"{name}: fixed occupancy incompatible with shared policy")
        require_frozen_eval(runtime.branch)
        if runtime.verify:
            runtime.verify()
    with torch.no_grad():
        if study.start:
            path = verified_file(study.start["path"], study.start["sha256"])
            bundle = torch.load(path, map_location=study.device, weights_only=True)
            if (bundle.get("format_version") != 1 or bundle.get("decoder_sha256") != study.decoder["sha256"]
                    or bundle.get("seed_sha256") != seed_sha):
                raise ValueError("Raw latent start must name the same decoder and canonical seed")
            point = bundle["z"]
        else:
            point, _ = decoder.encode(seed.to_relaxed(decoder.spec))
        if point.shape != (1, decoder.config.latent) or not torch.isfinite(point).all():
            raise ValueError("Start must contain one finite raw molecular latent")
        point = point.detach().clone()
    from latentmol.train import runtime_identity
    identity = {"runtime": runtime_identity(study.device),
                "decoder_sha256": study.decoder["sha256"], "decoder_state_sha256": module_digest(decoder),
                "seed_sha256": seed_sha, "start_sha256": tensor_digest({"z": point}),
                "graph_spec_sha256": decoder.spec.fingerprint(), "source_sha256": source_sha256,
                "branches": {name: {"manifest": runtime.branch.manifest.to_dict(), "assets": runtime.assets,
                                    "state_sha256": module_digest(runtime.branch)}
                             for name, runtime in runtimes.items()}}
    return Problem(decoder, runtimes, seed, active, point, identity)


def harden(graph, spec, decoded=None):
    """Projection for logging only, outside the gradient path; never repair labels."""
    atoms = graph.atoms.argmax(-1)
    active = atoms != 0
    charges = graph.charges.argmax(-1).masked_fill(~active, spec.formal_charges.index(0))
    if decoded is None and not torch.all((graph.occupancy == 0) | (graph.occupancy == 1)):
        raise ValueError("Variable-occupancy argmax requires conditional decoder bond logits")
    bond_values = decoded.symmetric_bond_logits if decoded is not None else graph.bonds
    bonds = bond_values.argmax(-1).masked_fill(~(active[:, :, None] & active[:, None, :]), 0)
    bonds = bonds.masked_fill(torch.eye(spec.max_nodes, device=bonds.device, dtype=torch.bool)[None], 0)
    return HardGraphBatch(atoms, charges, bonds)


def score_point(problem, study, point, weights, counters, purpose, *, gradients=False,
                atom_budget=None, original_slots=None):
    point = point.detach().clone().requires_grad_(gradients)
    with torch.set_grad_enabled(gradients):
        graph = problem.graph(point, study.temperature)
        measured, gradient = score_graph(graph, point, runtimes=problem.runtimes, models=study.models,
                                         weights=weights, spec=problem.spec, counters=counters,
                                         purpose=purpose, gradients=gradients)
        if atom_budget is None:
            return measured, gradient
        expected = graph.expected_size.squeeze(0)
        outside = graph.occupancy[:, ~original_slots].sum()
        count_loss = atom_budget.weight * count_band_penalty(
            expected, atom_budget.target, atom_budget.tolerance)
        outside_loss = atom_budget.outside_weight * outside
        loss = count_loss + outside_loss
        if loss.ndim != 0 or not loss.isfinite():
            raise ValueError("Expected one finite atom budget loss")
        gradient_norm = None
        if gradients:
            if not loss.requires_grad:
                raise ValueError("Atom budget is detached from the latent")
            derivative, = torch.autograd.grad(loss, point, allow_unused=True)
            if derivative is None or not derivative.isfinite().all():
                raise ValueError("Missing or nonfinite atom budget gradient")
            gradient = gradient + derivative.detach()
            if not gradient.isfinite().all():
                raise ValueError("Nonfinite combined latent gradient with atom budget")
            gradient_norm = float(derivative.norm())
            measured["combined_gradient_norm"] = float(gradient.norm())
        measured["atom_budget"] = dict(expected_atoms=float(expected.detach()),
            outside_occupancy=float(outside.detach()), weighted_count_loss=float(count_loss.detach()),
            weighted_outside_loss=float(outside_loss.detach()), gradient_norm=gradient_norm)
        measured["weighted_loss"] += float(loss.detach())
        if not math.isfinite(measured["weighted_loss"]):
            raise ValueError("Nonfinite combined loss with atom budget")
        return measured, gradient


def exploratory_validate(problem, study, arm_names, counters, path):
    """Check executable finite attached paths once, without qualifying derivatives.

    Native parity, FD resolution, hard reconstruction and nonzero signal are
    separate observations or optional qualified work, never prerequisites for a
    short comparison. Actual detached/nonfinite paths and frozen-state mutation
    still fail, and the ordinary score path repeats those basic checks on updates.
    """
    report = {"version": 3, "research_profile": "exploratory", "passed": False,
              "config_sha256": study.fingerprint(), "identity_sha256": digest(problem.identity),
              "models": {}, "observations": [], "derivative_qualification": "not_run",
              "scope": "finite attached starting paths only; no correctness or improvement claim"}
    write_json(path, report)
    try:
        problem.verify()
        for name in selected_models(study, arm_names):
            measured, gradient = score_point(problem, study, problem.start, {name: 1.0}, counters,
                                            "exploratory_smoke", gradients=True)
            if gradient.shape != problem.start.shape:
                raise ValueError(f"{name}: latent gradient shape differs from start")
            norm = float(gradient.double().norm())
            report["models"][name] = {"finite_attached_latent_path": True, "gradient_norm": norm,
                                       "measured": measured}
            if norm <= 1e-12:
                report["observations"].append({"model": name,
                    "status": "flat_at_start" if norm == 0 else "low_signal",
                    "gradient_norm": norm, "threshold": 1e-12,
                    "threshold_scope": "observation only; not a correctness or stop criterion",
                    "message": ("Attached zero gradient" if norm == 0 else "Attached small nonzero gradient")
                               + "; useful optimization signal is not established"})
            write_json(path, report)
        problem.verify()
        report.update(passed=True, status="executable_exploratory_smoke_passed")
    except Exception as error:
        report.update(status="failed", error=str(error), error_type=type(error).__name__)
        write_json(path, report)
        raise
    write_json(path, report)
    return report


def validate(problem, study, arm_names, counters, path, *, gradient_policy=GRADIENT_POLICY):
    """Run native seed parity and multi-epsilon derivatives for each actual scalar.

    This is an engineering gate, not chemical validity or useful binding evidence.
    Results are written incrementally so a failed real-model attempt is retained.
    """
    if gradient_policy not in (GRADIENT_POLICY, LEGACY_GRADIENT_POLICY):
        raise ValueError("Unknown multimodel gradient policy")
    pointwise = gradient_policy == GRADIENT_POLICY
    report = {"version": 2 if pointwise else 1, "passed": False, "config_sha256": study.fingerprint(),
              "identity_sha256": digest(problem.identity), "identity": problem.identity,
              "models": {}, "hard_native_outcome": "not_tested",
              "scope": "one prepared seed and decoded starting point; fixed-context engineering"}
    if pointwise:
        report.update(gradient_policy=gradient_policy, useful_gradient_validated=False,
                      full_network_fd_required=False)
    write_json(path, report)
    problem.verify()
    with torch.no_grad():
        decoded = problem.decoder.decode(problem.start)
        actual = harden(problem.graph(problem.start, study.temperature), problem.spec, decoded)
        if hard_digest(actual) != hard_digest(problem.seed):
            raise ValueError("Starting decoder projection differs from native seed. Supply a decoder-only selected raw-z start or rebuild native bundles; do not silently switch identity.")
    for name, runtime in problem.runtimes.items():
        cfg = study.models[name]
        with torch.no_grad():
            native = counters.call(name, "seed_parity", "native_forward", runtime.native_reference, problem.start.device)
        with torch.no_grad():
            adapted = counters.call(name, "seed_parity", "forward", lambda: evaluate_branches(
                problem.seed.to_relaxed(problem.spec), {name: runtime.branch}, spec=problem.spec,
                allow_lossy={name: cfg.allow_lossy})[name], problem.start.device)
        scalar_loss(native, cfg.objective)
        scalar_loss(adapted, cfg.objective)
        if set(native) != set(adapted):
            raise ValueError(f"{name}: native and adapted output names differ")
        for key in native:
            torch.testing.assert_close(adapted[key], native[key], rtol=study.diagnostics.parity_rtol,
                                       atol=study.diagnostics.parity_atol)
        # Only advertised channels are audited. Missing charge/occupancy support
        # remains explicit in the manifest rather than receiving a fake zero.
        from latentmol.representation import RelaxedGraph
        soft = problem.graph(problem.start, study.temperature)
        fields = {key: getattr(soft, key).detach().clone().requires_grad_(True)
                  for key in ("atoms", "charges", "bonds")}
        probe = RelaxedGraph(**fields)
        outputs = counters.call(name, "channel_gradient", "forward", lambda: evaluate_branches(
            probe, {name: runtime.branch}, spec=problem.spec, allow_lossy={name: cfg.allow_lossy})[name], problem.start.device)
        channel_loss, _ = scalar_loss(outputs, cfg.objective)
        reverse_scalar_dtype = str(channel_loss.dtype)
        channels = runtime.branch.manifest.capabilities.differentiable_channels
        if not channels or set(channels) - set(fields) or not channel_loss.requires_grad:
            raise ValueError(f"{name}: unsupported or detached declared gradient channels")
        grads = counters.call(name, "channel_gradient", "backward", lambda: torch.autograd.grad(
            channel_loss, [fields[key] for key in channels], allow_unused=True), problem.start.device)
        channel_norms = {}
        for channel, gradient in zip(channels, grads):
            if (gradient is None or not gradient.isfinite().all()
                    or (not pointwise and float(gradient.norm()) <= 1e-12)):
                raise ValueError(f"{name}: missing/nonfinite or legacy nonzero gradient to declared channel {channel}")
            channel_norms[channel] = float(gradient.norm())
        del fields, probe, outputs, channel_loss, grads, soft
        measured, grad = score_point(problem, study, problem.start, {name: 1.0}, counters,
                                     "derivative_analytic", gradients=True)
        norm = float(grad.double().norm()) if pointwise else float(grad.norm())
        if not math.isfinite(norm):
            raise ValueError(f"{name}: nonfinite latent gradient")
        if not pointwise and norm <= 1e-12:
            raise ValueError(f"{name}: scalar objective has zero latent gradient at the start")
        generator = torch.Generator(device="cpu").manual_seed(study.diagnostics.seed)
        first = (torch.randn(problem.start.shape, generator=generator).to(problem.start)
                 if pointwise and norm <= 1e-12 else grad)
        directions = {"gradient": first / first.norm()}
        random = torch.randn(problem.start.shape, generator=generator).to(problem.start)
        directions["random"] = random / random.norm()
        if pointwise:
            model_report = {"native_seed_parity": True, "latent_gradient_norm": norm,
                "channel_gradient_norms": channel_norms, "measured": measured, "passed": False,
                "reverse_scalar_dtype": reverse_scalar_dtype,
                "optimization_qualification": {"informative_total": norm > 1e-12},
                "gradient_label_is_seeded_probe": norm <= 1e-12,
                "pointwise_ad": {"scope": "independent local forward/reverse AD at this start",
                                 "rows": [], "passed": False}}
            report["models"][name] = model_report
            write_json(path, report)
            try:
                for label, direction in directions.items():
                    with torch.no_grad(), torch.autograd.forward_ad.dual_level():
                        dual = torch.autograd.forward_ad.make_dual(problem.start, direction)
                        graph = problem.graph(dual, study.temperature)
                        outputs = counters.call(name, "independent_forward_ad", "forward", lambda:
                            evaluate_branches(graph, {name: runtime.branch}, spec=problem.spec,
                                allow_lossy={name: cfg.allow_lossy})[name], problem.start.device)
                        loss, _ = scalar_loss(outputs, cfg.objective)
                        primal, tangent = torch.autograd.forward_ad.unpack_dual(loss)
                        if tangent is None or not bool(tangent.isfinite().all()):
                            raise ValueError(f"{name}: missing/nonfinite independent forward tangent")
                        analytic = float((grad.double() * direction.double()).sum())
                        witness = relative_ad_consistency(float(tangent), analytic, norm)
                        replay = scalar_replay_consistency(float(primal), measured["weighted_loss"],
                                                           str(primal.dtype), reverse_scalar_dtype)
                        row = dict(direction=label, direction_sha256=tensor_digest({"direction": direction}),
                            primal=float(primal), forward_jvp=float(tangent), reverse_projection=analytic,
                            relative_ad_consistency=witness,
                            scalar_replay_consistency=replay,
                            primal_exact=float(primal) == measured["weighted_loss"],
                            passed=witness["passed"] and replay["passed"])
                    model_report["pointwise_ad"]["rows"].append(row)
                    write_json(path, report)
                model_report["pointwise_ad"]["passed"] = all(
                    row["passed"] for row in model_report["pointwise_ad"]["rows"])
                if not model_report["pointwise_ad"]["passed"]:
                    raise ValueError(f"{name}: pointwise AD failed or is numerically unresolved; see {path}")
            except Exception as error:
                model_report.update(error=str(error), error_type=type(error).__name__)
                write_json(path, report)
                raise
        rows = []
        for label, direction in directions.items():
            analytic = (float((grad.double() * direction.double()).sum()) if pointwise
                        else float((grad * direction).sum()))
            for epsilon in study.diagnostics.fd_epsilons:
                plus = score_point(problem, study, problem.start + epsilon * direction, {name: 1.0},
                                   counters, "finite_difference")[0]["weighted_loss"]
                minus = score_point(problem, study, problem.start - epsilon * direction, {name: 1.0},
                                    counters, "finite_difference")[0]["weighted_loss"]
                numerical = (plus - minus) / (2 * epsilon)
                error = abs(analytic - numerical)
                if pointwise and not all(math.isfinite(value) for value in (plus, minus, analytic, numerical, error, epsilon)):
                    raise ValueError(f"{name}: nonfinite finite-difference diagnostic")
                passed = error <= study.diagnostics.fd_atol + study.diagnostics.fd_rtol * max(abs(analytic), abs(numerical))
                rows.append({"direction": label, "epsilon": epsilon, "analytic": analytic,
                             **({"required_for_pass": False, "f_plus": plus, "f_minus": minus} if pointwise else {}),
                             "central_difference": numerical, "absolute_error": error, "passed": passed})
        passed = pointwise or all(sum(row["passed"] for row in rows if row["direction"] == label)
                     >= study.diagnostics.minimum_fd_passes for label in directions)
        report["models"][name] = {**(model_report if pointwise else {}),
                                  "native_seed_parity": True, "latent_gradient_norm": norm,
                                  "channel_gradient_norms": channel_norms,
                                  "finite_differences": rows, "passed": passed,
                                  "measured": measured}
        write_json(path, report)
        if not passed:
            raise ValueError(f"{name}: finite differences failed; see {path}")
    problem.verify()
    report["passed"] = True
    write_json(path, report)
    return report


def require_gate(problem, study, path, *, gradient_policy=GRADIENT_POLICY):
    gate = json.loads(Path(path).read_text())
    if gradient_policy not in (GRADIENT_POLICY, LEGACY_GRADIENT_POLICY):
        raise ValueError("Unknown multimodel gradient policy")
    pointwise = gradient_policy == GRADIENT_POLICY
    if (gate.get("version") != (2 if pointwise else 1) or gate.get("passed") is not True
            or gate.get("config_sha256") != study.fingerprint()
            or gate.get("identity_sha256") != digest(problem.identity)
            or set(gate.get("models", {})) != set(problem.runtimes)
            or not all(row.get("passed") is True for row in gate["models"].values())):
        raise ValueError("Require a passing engineering report for these exact assets, config, source and selected models")
    for name, model in gate["models"].items():
        channels = model.get("channel_gradient_norms")
        expected_channels = set(problem.runtimes[name].branch.manifest.capabilities.differentiable_channels)
        if (not isinstance(channels, dict) or set(channels) != expected_channels
                or any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                       or (not pointwise and value <= 1e-12) for value in channels.values())):
            raise ValueError("Missing, nonfinite or undeclared channel-gradient evidence")
    if pointwise:
        if (gate.get("gradient_policy") != gradient_policy or gate.get("full_network_fd_required") is not False
                or gate.get("useful_gradient_validated") is not False):
            raise ValueError("Require the declared prospective multimodel derivative scope")
        for name, model in gate["models"].items():
            norm = model.get("latent_gradient_norm")
            if (type(norm) not in (int, float) or not math.isfinite(norm) or norm < 0
                    or model.get("optimization_qualification") != {"informative_total": norm > 1e-12}
                    or model.get("gradient_label_is_seeded_probe") is not (norm <= 1e-12)
                    or model.get("native_seed_parity") is not True):
                raise ValueError("Invalid multimodel gradient signal evidence")
            evidence = model.get("pointwise_ad", {})
            rows = evidence.get("rows", [])
            if evidence.get("passed") is not True or [row.get("direction") for row in rows] != ["gradient", "random"]:
                raise ValueError("Missing independent multimodel AD evidence")
            for row in rows:
                expected = relative_ad_consistency(row["forward_jvp"], row["reverse_projection"], norm)
                replay = scalar_replay_consistency(row["primal"], model["measured"]["weighted_loss"],
                    row["scalar_replay_consistency"]["dtype"], model["reverse_scalar_dtype"])
                if (row.get("relative_ad_consistency") != expected or not expected["passed"]
                        or row.get("scalar_replay_consistency") != replay or not replay["passed"]
                        or row.get("passed") is not True
                        or row.get("primal_exact") is not (row["primal"] == model["measured"]["weighted_loss"])):
                    raise ValueError("Multimodel AD witness does not reproduce its acceptance")
            diagnostics = model.get("finite_differences", [])
            if [(row.get("direction"), row.get("epsilon")) for row in diagnostics] != [
                    (direction, epsilon) for direction in ("gradient", "random") for epsilon in study.diagnostics.fd_epsilons]:
                raise ValueError("Missing or changed prescribed multimodel FD diagnostics")
            tangents = {row["direction"]: row["reverse_projection"] for row in rows}
            for row in diagnostics:
                fields = ("analytic", "central_difference", "absolute_error", "f_plus", "f_minus", "epsilon")
                if any(type(row.get(key)) not in (int, float) or not math.isfinite(row[key]) for key in fields):
                    raise ValueError("Nonfinite or missing multimodel FD diagnostic")
                numerical = (row["f_plus"] - row["f_minus"]) / (2 * row["epsilon"])
                error = abs(row["analytic"] - numerical)
                expected_pass = error <= study.diagnostics.fd_atol + study.diagnostics.fd_rtol * max(
                    abs(row["analytic"]), abs(numerical))
                if (row.get("required_for_pass") is not False or row["analytic"] != tangents[row["direction"]]
                        or row["central_difference"] != numerical or row["absolute_error"] != error
                        or row.get("passed") is not expected_pass):
                    raise ValueError("Multimodel FD diagnostic arithmetic differs")
    return gate


def run_arm(problem, study, arm_name, counters, path, event, *, observation_interval=1):
    """One fresh trajectory; all arms start at identical raw z and sampling seeds.

    This is exploratory after engineering gates. Hard native confirmation remains
    a separate experiment. Global clipping and a projected raw-z trust radius are
    explicit; they do not supply a calibrated AE/VAE distance or a chemistry prior.
    """
    from latentmol.chemistry import inspect_graph
    arm = study.arms[arm_name]
    weights = {name: study.weight(arm, name) for name in arm.models}
    if type(observation_interval) is not int or observation_interval < 1:
        raise ValueError("observation_interval must be a positive integer")
    point = problem.start.clone()
    steps = arm.steps or study.optimizer.steps
    budget = arm.atom_budget
    original_slots = None
    if budget is not None:
        if problem.active is not None or budget.target > problem.spec.max_nodes:
            raise ValueError("Atom budget needs variable occupancy and a target within decoder capacity")
        with torch.no_grad():
            occupancy = problem.graph(point, study.temperature).occupancy
            if occupancy.shape != (1, problem.spec.max_nodes) or not occupancy.isfinite().all():
                raise ValueError("Atom budget needs one finite starting occupancy per slot")
            # Freeze the starting top slots so rank changes cannot erase outside mass.
            original = occupancy[0].argsort(descending=True, stable=True)[:budget.target]
            original_slots = torch.zeros_like(occupancy[0], dtype=torch.bool)
            original_slots[original] = True
    path.mkdir(parents=True, exist_ok=False)
    metadata = {"arm": arm_name, "status": "running", "control": arm.control,
                "weights": weights, "steps": steps, "raw_z_trust_radius": study.optimizer.trust_radius,
                "hard_native_confirmation": "pending", "scientific_validation": "not_established",
                "observation_interval": observation_interval,
                "validity_scope": "saved sampled observations; not all optimization steps"}
    if budget is not None:
        metadata["atom_budget"] = {**asdict(budget), "original_slots": original.tolist()}
    write_json(path / "arm.json", metadata)
    records, valid, changed, unique = [], 0, 0, set()
    start_identity = None
    try:
        for step in range(steps + 1):
            observe = step in (0, steps) or step % observation_interval == 0
            update = step < steps and arm.control == "optimize"
            measured, grad = score_point(problem, study, point, weights, counters, "optimization" if update else "diagnostic",
                                         gradients=update, atom_budget=budget, original_slots=original_slots)
            sampled = []
            with torch.no_grad():
                decoded = problem.decoder.decode(point)
                graph = graph_from_decoder(decoded, problem.spec, temperature=study.temperature, fixed_active=problem.active)
                from latentmol.reporting.capture import capture_shared_graph
                if observe:
                    capture_shared_graph(path, step, graph, problem.spec,
                        active_slots=problem.active.nonzero().flatten().tolist() if problem.active is not None else None,
                        temperature=study.temperature)
                generator = torch.Generator(device=str(point.device)).manual_seed(study.diagnostics.seed + step)
                hard_draws = [("argmax", harden(graph, problem.spec, decoded))] if observe else []
                hard_draws += [("categorical", sample_hard_graph(decoded, problem.spec, generator=generator,
                                 temperature=study.temperature, fixed_active=problem.active))
                               for _ in range(study.diagnostics.hard_samples if observe else 0)]
                for index, (kind, hard) in enumerate(hard_draws):
                    hard = hard.to("cpu")
                    inspected = inspect_graph(hard.atoms[0], hard.charges[0], hard.bonds[0], problem.spec)
                    if step == 0 and kind == "argmax":
                        start_identity = inspected.smiles
                    identity = hard_digest(hard)
                    if inspected.smiles:
                        valid += 1
                        unique.add(inspected.smiles)
                        changed += int(inspected.smiles != start_identity)
                    item = {"candidate_id": f"{arm_name}-s{step:04d}-d{index:03d}", "arm": arm_name,
                            "step": step, "draw": index, "kind": kind, "graph_sha256": identity,
                            "smiles": inspected.smiles, "failure": inspected.failure_reason,
                            "heavy_atoms": int((hard.atoms != 0).sum()), "graph_spec": problem.spec.to_dict(),
                            "labels": {key: getattr(hard, key)[0].tolist() for key in ("atoms", "charges", "bonds")}}
                    sampled.append(item)
                    records.append(item)
                    with (path / "candidates.jsonl").open("a") as handle:
                        handle.write(json.dumps(item, allow_nan=False) + "\n")
                if observe:
                    torch.save({"z": point.detach().cpu(), "graph": {key: getattr(graph, key).cpu() for key in ("atoms", "charges", "bonds")}},
                               path / f"state-{step:04d}.pt")
            measured.update(arm=arm_name, step=step, control=arm.control, sampled_observation=observe,
                            latent_displacement=float((point - problem.start).norm()),
                            expected_atoms=float(graph.expected_size[0]),
                            expected_charge=float(graph.expected_charge(problem.spec).sum()),
                            candidates=[{k: v for k, v in item.items() if k not in ("labels", "graph_spec")} for item in sampled])
            event(measured)
            write_json(path / f"metrics-{step:04d}.json", measured)
            if step == steps or arm.control == "no_update":
                continue
            if arm.control == "random_direction":
                generator = torch.Generator(device="cpu").manual_seed(study.diagnostics.seed + 100000 + step)
                grad = torch.randn(point.shape, generator=generator).to(point)
                grad = grad / grad.norm() * study.optimizer.max_gradient_norm
            norm = float(grad.norm())
            grad = grad * min(1.0, study.optimizer.max_gradient_norm / max(norm, 1e-12))
            proposal = point - study.optimizer.learning_rate * grad
            delta = proposal - problem.start
            point = (problem.start + delta * min(1.0, study.optimizer.trust_radius / max(float(delta.norm()), 1e-12))).detach()
        metadata.update(status="completed", attempted=len(records), valid=valid, valid_changed=changed,
                        unique_valid=len(unique), validity_fraction=valid / len(records),
                        valid_changed_fraction=changed / len(records))
    except Exception as exc:
        metadata.update(status="failed", error=str(exc), attempted=len(records))
        raise
    finally:
        write_json(path / "arm.json", metadata)
    return metadata
