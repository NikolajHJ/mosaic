"""Strict derivative checks for a frozen model's selected local ReLU branch.

Natural central differences can cross a native affinity-readout ReLU threshold
and need not equal the derivative at the baseline. The new policy requires
native forward-AD/reverse-AD agreement AND FD on the baseline readout branch,
with exact baseline scalar/gradient parity. It records natural secants as well.
No branch holding is allowed during optimization or Boltz/DeFoG target updates.
Historical local_v1 failures remain failures under their original contract.
"""

from contextlib import contextmanager
import math

import torch
from torch import nn

from .local_fd import direction_plan, validate_local_fd
from .dbmol_objective import CONTACT_HEAD64_ARITHMETIC

POLICY = "branch_local_v2"
JVP_RTOL = 1e-3
JVP_ATOL = 1e-6


class AffinityReadoutBranches:
    """Capture/hold only the pinned two-head, twelve-ReLU affinity readout.

    The native trunks and structural/confidence heads remain entirely natural.
    Hooks are scoped and removed on exceptions. Registered modules/parameters
    and their forward methods are never replaced. A zero preactivation retains
    PyTorch ReLU's chosen zero derivative; it is not a two-sided derivative at
    a kink. Unsupported native head topologies fail rather than being guessed.
    """

    def __init__(self, scorer):
        model = scorer.affinity.model
        if not model.affinity_ensemble:
            raise ValueError("Branch validation requires the pinned two-member affinity ensemble")
        self.layers, self.masks, self.baseline = {}, {}, {}
        for member, outer in enumerate((model.affinity_module1, model.affinity_module2)):
            head = outer.affinity_heads
            for field in ("affinity_out_mlp", "to_affinity_pred_value", "to_affinity_pred_score"):
                sequence = getattr(head, field)
                expected = (nn.Linear, nn.ReLU, nn.Linear, nn.ReLU)
                if field != "affinity_out_mlp": expected += (nn.Linear,)
                if (type(sequence) is not nn.Sequential
                        or tuple(type(layer) for layer in sequence) != expected
                        or any(layer.inplace for layer in sequence if isinstance(layer, nn.ReLU))):
                    raise ValueError("Unsupported native affinity readout topology")
                for index in (1, 3):
                    self.layers[f"head.{member}.{field}.{index}"] = sequence[index]
        if len(self.layers) != 12:
            raise ValueError("Require all twelve native readout ReLUs")

    @contextmanager
    def capture(self):
        if self.masks:
            raise ValueError("Capture the baseline exactly once")
        handles = []
        def observe(name):
            def hook(module, args, output):
                if name in self.masks or len(args) != 1 or args[0].dtype != torch.float32:
                    raise ValueError("Require one native FP32 call per readout ReLU")
                value = args[0].detach().clone()
                if not bool(value.isfinite().all()):
                    raise ValueError("Nonfinite readout preactivation")
                self.masks[name] = value > 0
                self.baseline[name] = {"shape": list(value.shape),
                    "minimum_absolute_preactivation": float(value.abs().min()),
                    "exact_zero_units": int((value == 0).sum())}
            return hook
        try:
            handles = [module.register_forward_hook(observe(name)) for name, module in self.layers.items()]
            yield
            if set(self.masks) != set(self.layers):
                raise ValueError("Incomplete baseline affinity readout capture")
        finally:
            for handle in handles:
                handle.remove()

    @contextmanager
    def hold(self):
        if set(self.masks) != set(self.layers):
            raise ValueError("Require a complete captured readout branch")
        handles = []
        def fixed(name):
            def hook(module, args, output):
                value = args[0]
                mask = self.masks[name]
                if mask.shape != value.shape or mask.device != value.device:
                    raise ValueError("Readout branch shape/device changed")
                # where preserves positive-zero outputs for inactive negative
                # inputs and ReLU's baseline derivative, including exact zero.
                return torch.where(mask, value, torch.zeros_like(value))
            return hook
        try:
            handles = [module.register_forward_hook(fixed(name)) for name, module in self.layers.items()]
            yield
        finally:
            for handle in handles:
                handle.remove()


def shared_scalars(decoder, point, branch, config, *, target=None, weight=0.):
    """Direct shared-graph forward, preserving a forward-AD dual latent tensor.

    score_at intentionally detaches its probe and therefore cannot be used for
    independent forward AD. This mirrors its scalar assembly, including the
    frozen repair target and exact native graph policy, without any sampling.
    """
    from latentmol.graph_policy import graph_from_decoder
    from latentmol.multimodel.engine import scalar_loss
    from .dbmol_objective import COMPONENT_OUTPUTS
    decoded = decoder.decode(point)
    graph = graph_from_decoder(decoded, decoder.spec, temperature=config.temperature,
                               fixed_active=branch.prepared.active)
    outputs = branch(graph)
    combined, _ = scalar_loss(outputs, config.objective)
    result = {key: outputs[key].reshape(()) for key in COMPONENT_OUTPUTS}
    if target is not None:
        if config.repair_guidance is None or not math.isfinite(weight) or weight <= 0:
            raise ValueError("A fixed repair target requires declared positive guidance")
        from latentmol.repair_guidance.loss import fixed_target_nll
        repair, _ = fixed_target_nll(decoded, graph, target, decoder.spec,
            temperature=config.temperature, normalization=config.repair_guidance.normalization)
        result["repair_only"] = repair
        combined = combined + weight * repair
    result["combined"] = combined
    return result


def native_jvp(decoder, point, direction, branch, config, counters, *, target=None, weight=0.):
    """Independent forward AD, with native forward and checkpoint bypass.

    no_grad disables reverse recording only. Low-level dual tensors propagate
    tangents through native operations. Unsupported operators are fatal; there
    is deliberately no double-backward or finite-difference fallback.
    """
    from torch.autograd import forward_ad
    with torch.no_grad(), forward_ad.dual_level():
        dual = forward_ad.make_dual(point, direction)
        outputs = counters.call("boltz", "branch_validation", "forward_ad",
            lambda: shared_scalars(decoder, dual, branch, config, target=target, weight=weight), point.device)
        result = {}
        for key, output in outputs.items():
            primal, tangent = forward_ad.unpack_dual(output)
            if tangent is None or not bool(primal.isfinite().all()) or not bool(tangent.isfinite().all()):
                raise ValueError("Missing/nonfinite independent forward tangent: " + key)
            result[key] = {"primal": float(primal), "tangent": float(tangent)}
        return result


def validate_derivative(point, gradient, natural_score, branch_score, forward_probe, *,
                        baseline, key, seed, rtol, atol, minimum_passes, report, persist):
    """Require independent native JVP and strict branch FD on predeclared directions.

    Directions and grid are exactly local_v1's; tolerance/grid are not tuned
    after seeing outcomes. Natural FD is retained with its own pass/fail result,
    never rewritten as branch evidence. The caller must first establish exact
    native/branch baseline forward and backward parity for this same scalar.
    """
    directions, _, selection = direction_plan(point, gradient, seed=seed)
    report.update(validation_contract=POLICY, baseline_loss=baseline, local_branch_fd={}, natural_secants={},
                  independent_forward_ad={"rtol": JVP_RTOL, "atol": JVP_ATOL, "rows": [], "passed": False},
                  direction_selection=selection, passed=False)
    persist()

    if selection["pool_exhausted"]:
        report["status"] = "failed_direction_pool"
        return
    for name, direction in directions.items():
        observation = forward_probe(direction)[key]
        analytic = float((gradient.double() * direction.double()).sum())
        error = abs(observation["tangent"] - analytic)
        tolerance = JVP_ATOL + JVP_RTOL * max(abs(analytic), abs(observation["tangent"]))
        row = {"direction": name, **observation, "reverse_projection": analytic,
               "absolute_error": error, "tolerance": tolerance,
               "primal_exact": observation["primal"] == baseline,
               "passed": observation["primal"] == baseline and error <= tolerance}
        report["independent_forward_ad"]["rows"].append(row)
        persist()
    report["independent_forward_ad"]["passed"] = all(row["passed"] for row in report["independent_forward_ad"]["rows"])
    # Preserve genuine natural secants even when they cross a branch boundary.
    validate_local_fd(point, gradient, natural_score, baseline=baseline, seed=seed,
        rtol=rtol, atol=atol, minimum_passes=minimum_passes, report=report["natural_secants"], persist=persist)
    validate_local_fd(point, gradient, branch_score, baseline=baseline, seed=seed,
        rtol=rtol, atol=atol, minimum_passes=minimum_passes, report=report["local_branch_fd"], persist=persist)
    report["passed"] = report["independent_forward_ad"]["passed"] and report["local_branch_fd"]["passed"]
    report["policy"] = {**report["local_branch_fd"]["policy"], "version": POLICY,
        "native_model_during_optimization": "unmodified ReLU branches",
        "finite_difference_target": "same baseline local affinity-readout branch, exact baseline scalar/VJP parity required",
        "natural_secants_required_to_be_recorded": True}
    report["status"] = "passed" if report["passed"] else "failed"
    persist()


def validate_branch_derivatives(decoder, point, branch, config, counters, report, persist,
                                measured, gradient, components, *, target=None, weight=0.):
    """Production gate for the explicitly opted-in DBMol branch policy.

    Every scalar is checked even if an earlier scalar fails, so one failed
    affinity check cannot hide the contact/confidence evidence. Genuine natural
    secants and branch-conditioned finite differences remain separate records.
    Only independent-AD and local-branch FD jointly certify this new contract.
    """
    from .active_set import score_at
    from latentmol.multimodel.study import tensor_digest
    if (config.dbmol is None or config.dbmol.objective_arithmetic != CONTACT_HEAD64_ARITHMETIC
            or config.fd_policy != POLICY or config.recycling_steps != 0):
        raise ValueError("Branch validation supports only explicit contact-head DBMol with zero recycles")
    branches = AffinityReadoutBranches(branch.scorer)
    report["derivative_contract"] = {"version": POLICY,
        "scope": "native forward AD plus baseline local affinity-readout branch finite differences",
        "branch_holding_used_in_optimization": False,
        "historical_gate_approval": False}
    with torch.no_grad(), branches.capture():
        base, _ = score_at(decoder, point, branch, config, counters, "branch_capture",
                          gradients=False, repair_target=target, repair_weight=weight)
    if base["loss"] != measured["loss"] or base["components"] != measured["components"]:
        raise ValueError("Native forward baseline is not repeatable")
    held_components = {}
    with branches.hold():
        held, held_gradient = score_at(decoder, point, branch, config, counters, "branch_baseline_parity",
            gradients=True, component_gradients=held_components, repair_target=target, repair_weight=weight)
    parity = (held["loss"] == measured["loss"]
              and held["components"] == measured["components"]
              and torch.equal(gradient, held_gradient)
              and set(components) == set(held_components)
              and all(torch.equal(value, held_components[key]) for key, value in components.items()))
    report["local_branch_baseline_parity"] = {"passed": bool(parity),
        "scalar_exact": held["loss"] == measured["loss"],
        "total_gradient_exact": torch.equal(gradient, held_gradient),
        "component_gradient_exact": {key: torch.equal(value, held_components[key]) for key, value in components.items()},
        "captured_readout_relus": branches.baseline}
    persist()
    if not parity:
        raise ValueError("Baseline branch does not have exact native scalar and backward parity")
    # No native parameters or state change; only detached observation rows are
    # cached across identical direction tensors to avoid repeated native JVPs.
    jvp_cache = {}
    def forward_probe(direction):
        identity = tensor_digest({"direction": direction})
        if identity not in jvp_cache:
            jvp_cache[identity] = native_jvp(decoder, point, direction, branch, config, counters,
                                           target=target, weight=weight)
        return jvp_cache[identity]
    derivatives = dict(components)
    baselines = {key: measured["components"][key] for key in components}
    report["component_gradients"] = {}
    destinations = report["component_gradients"]
    if target is not None:
        from latentmol.repair_guidance.loss import target_score_at
        repair_row, repair_gradient = target_score_at(decoder, point, branch.prepared.active, target,
            temperature=config.temperature, normalization=config.repair_guidance.normalization, gradients=True)
        derivatives["repair_only"] = repair_gradient
        baselines["repair_only"] = repair_row["loss"]
    derivatives["combined"], baselines["combined"] = gradient, measured["loss"]
    # Full native outputs do not depend on which component is selected for a
    # diagnostic loss. Cache detached rows at identical latent endpoints, with
    # separate namespaces for natural and held branches. No graph/gradient is
    # cached and no scorer or repair target is resampled inside this gate.
    endpoint_cache = {"native": {}, "held": {}}
    def endpoint(candidate, key, *, held=False):
        mode = "held" if held else "native"
        identity = tensor_digest({"point": candidate})
        if identity not in endpoint_cache[mode]:
            def evaluate():
                return score_at(decoder, candidate, branch, config, counters,
                    "branch_secant:" + mode, gradients=False,
                    repair_target=target, repair_weight=weight)[0]
            if held:
                with branches.hold(): row = evaluate()
            else: row = evaluate()
            endpoint_cache[mode][identity] = row
        row = endpoint_cache[mode][identity]
        value = (row["loss"] if key == "combined" else row["repair_guidance"]["normalized_nll"]
                 if key == "repair_only" else row["components"][key])
        return {"loss": value, "loss_dtype": "torch.float32" if key == "repair_only" else "torch.float64",
                "contact_selection": row["contact_selection"]}
    errors = []
    for key, derivative in derivatives.items():
        item = {"gradient_norm": float(derivative.double().norm()), "attached": True,
                "fixed_pose_and_mask": True}
        if key == "combined": report["local_fd_validation"] = item
        elif key == "repair_only": report["repair_only_validation"] = item
        else: destinations[key] = item
        if item["gradient_norm"] <= 1e-12:
            item["status"] = "zero_at_current_point_uninformative"
            if key in ("combined", "repair_only"):
                errors.append(key + ": zero derivative")
            persist()
            continue
        def natural(candidate, key=key):
            return endpoint(candidate, key)
        def held_score(candidate, key=key):
            return endpoint(candidate, key, held=True)
        try:
            validate_derivative(point, derivative, natural, held_score, forward_probe,
                baseline=baselines[key], key=key, seed=config.diagnostics.seed,
                rtol=config.diagnostics.fd_rtol, atol=config.diagnostics.fd_atol,
                minimum_passes=config.diagnostics.minimum_fd_passes, report=item, persist=persist)
            if key == "dbmol_contact_loss" and not item["natural_secants"]["passed"]:
                item["passed"] = False
                errors.append(key + ": natural contact FD still fails")
            if item["passed"]:
                item["status"] = "validated_nonzero"
            else:
                errors.append(key + ": derivative checks failed")
        except Exception as exc:
            item.update(passed=False, status="failed", error=str(exc), error_type=type(exc).__name__)
            errors.append(key + ": " + str(exc))
        persist()
    report["independent_forward_ad_direction_count"] = len(jvp_cache)
    report["endpoint_cache_counts"] = {mode: len(rows) for mode, rows in endpoint_cache.items()}
    report["finite_differences"] = report["local_fd_validation"].get("local_branch_fd", {}).get("finite_differences", [])
    report["derivative_contract"]["passed"] = not errors
    persist()
    if errors:
        raise ValueError("Branch derivative gate failed: " + "; ".join(errors))
