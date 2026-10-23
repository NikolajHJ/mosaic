"""HPC-only native affinity-MLP attribution at the original prior-5 failure.

Every scorer endpoint uses unchanged source079 arithmetic. Passive hooks expose
the two ensemble members' pooled/shared/regression cuts. Cached FP64 and held
ReLU-branch evaluations are counterfactual endpoints, never a production fix or
historical gate approval. Widening observed inputs cannot restore upstream bits.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import gc
import importlib.util
import inspect
import os
from pathlib import Path
import sys
import time


CASE_ID = "prior-5"
CASE_TIMEOUT_SECONDS = 1800
MAXIMUM_SCORER_FORWARDS = 83
NATIVE_SOURCE_SHA256 = "164b48db0ae40eab2318724ebafcd5ad92bbf0c38f8d4eb940dd25eac07540f4"
SOURCE086_REPORT = ("/work3/s234818/latentmol/runs/source-086-precision-head-attempt001-replay/"
    "20261010T075304Z_source086-precision-head_c4737535/prior-5/"
    "20261010T075952Z_prior-5_51e5ec7c/attribution.json")
SOURCE086_REPORT_SHA256 = "3f3beb26cf8b2830d70ae8c4e7fe9ff6e21353ffca5e0033d92f61f9369c0d74"
SOURCE086_RUN_SHA256 = "89121d4296583d502fe552849b37012fc890bdee13935b22a93a61d98c37cfd4"
SOURCE086_ARCHIVE = "487a74b8dab72f2ad701ba31c9998380e7dbb388a0179c9a89e23187d1427fee"
SOURCE086_GRADIENT = {"sha256": "db4c26bae5f2c41a8657c014d203e6f151b590bcbc93b936c93e892b0ba42bad",
    "norm_float64": 3.1264574862490377, "dtype": "torch.float32"}
MODES = ("native_pooled", "native_shared", "float64_pooled", "float64_shared",
         "frozen_shared", "frozen_regression", "frozen_both")
BASELINE_PHASES = ("original", "affinity_cuts", *MODES)
EXACT_MODES = ("native_pooled", "native_shared", "frozen_shared", "frozen_regression", "frozen_both")
LEVELS = ("pooled", "shared", "value")
CLAIM_LIMIT = __doc__
_HEAD_MODULE = None
_ORIGINAL_BASELINE = None


def head_helpers():
    """Load observers without importing latentmol before source079 bootstrap."""
    global _HEAD_MODULE, _ORIGINAL_BASELINE
    if _HEAD_MODULE is None:
        spec = importlib.util.spec_from_file_location("_affinity_head_observer",
            Path(__file__).with_name("head_causal_diagnostic.py"))
        _HEAD_MODULE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_HEAD_MODULE)
        _ORIGINAL_BASELINE = _HEAD_MODULE.baseline_probe
    return _HEAD_MODULE


def validate_architecture(heads):
    """Fail closed on any different small-head topology; no generic graph rewrite."""
    from torch import nn
    expected = {"affinity_out_mlp": (nn.Linear, nn.ReLU, nn.Linear, nn.ReLU),
        "to_affinity_pred_value": (nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear),
        "to_affinity_pred_score": (nn.Linear, nn.ReLU, nn.Linear, nn.ReLU, nn.Linear)}
    if len(heads) != 2:
        raise ValueError("Require both native affinity ensemble members")
    for head in heads:
        for name, classes in expected.items():
            layers = getattr(head, name, None)
            if type(layers) is not nn.Sequential or tuple(type(layer) for layer in layers) != classes:
                raise ValueError("Unexpected affinity architecture: " + name)
            if any(isinstance(layer, nn.ReLU) and layer.inplace for layer in layers):
                raise ValueError("In-place ReLU invalidates passive preactivation capture")
        if type(head.to_affinity_logits_binary) is not nn.Linear:
            raise ValueError("Unexpected binary affinity projection")
        if any(parameter.requires_grad for parameter in head.parameters()) or head.training:
            raise ValueError("Require frozen evaluation-only native heads")


def native_heads(scorer):
    model = scorer.affinity.model
    if not model.affinity_ensemble:
        raise ValueError("Original prior-5 model requires affinity ensembling")
    heads = tuple(getattr(model, name).affinity_heads for name in ("affinity_module1", "affinity_module2"))
    validate_architecture(heads)
    paths = sorted({str(Path(inspect.getfile(type(head))).resolve()) for head in heads})
    helper = head_helpers()._helper()
    if len(paths) != 1 or helper._sha(paths[0]) != NATIVE_SOURCE_SHA256:
        raise ValueError("Installed native affinity implementation differs from pinned source")
    return heads, {"path": paths[0], "sha256": NATIVE_SOURCE_SHA256, "members": 2,
        "shared_relu_per_member": 2, "regression_relu_per_member": 2,
        "scope": "Regression path only; binary score branch remains native and outside this cut."}


def key(member, label):
    return f"member_{member}_{label}"


@contextmanager
def capture_affinity(heads):
    """Observe exactly one native call per boundary, including backward scope."""
    captured, handles = {}, []
    def save(label, value):
        if label in captured:
            raise ValueError("Affinity head unexpectedly replayed: " + label)
        captured[label] = value
    def before(label):
        def observe(module, args):
            if len(args) != 1:
                raise ValueError("Unexpected native small-head call signature")
            save(label, args[0])
        return observe
    def after(label):
        def observe(module, args, output):
            save(label, output)
        return observe
    expected = set()
    try:
        for member, head in enumerate(heads):
            bindings = ((head.affinity_out_mlp, "pooled", True),
                        (head.to_affinity_pred_value, "shared", True),
                        (head.to_affinity_pred_value, "value", False))
            for module, label, pre in bindings:
                label = key(member, label)
                expected.add(label)
                handles.append(module.register_forward_pre_hook(before(label)) if pre else module.register_forward_hook(after(label)))
            for block, layers in (("shared", head.affinity_out_mlp), ("regression", head.to_affinity_pred_value)):
                for index, layer_index in enumerate((1, 3)):
                    label = key(member, f"{block}_relu_{index}")
                    expected.add(label)
                    handles.append(layers[layer_index].register_forward_pre_hook(before(label)))
        yield captured
        if set(captured) != expected:
            raise ValueError("Incomplete passive affinity capture")
    finally:
        for handle in handles:
            handle.remove()


def sequential(layers, value, *, double=False, masks=None):
    """Reevaluate only tiny cached Linear/ReLU blocks with original stored weights."""
    import torch
    from torch import nn
    from torch.nn import functional as F
    value = value.double() if double else value
    gate = 0
    for layer in layers:
        if type(layer) is nn.Linear:
            weight = layer.weight.double() if double else layer.weight
            bias = layer.bias.double() if double and layer.bias is not None else layer.bias
            value = F.linear(value, weight, bias)
        elif type(layer) is nn.ReLU:
            if masks is None:
                value = F.relu(value)
            else:
                if masks[gate].shape != value.shape:
                    raise ValueError("Held ReLU mask shape changed")
                value = torch.where(masks[gate], value, torch.zeros_like(value))
            gate += 1
        else:
            raise ValueError("Unexpected cached small-head layer")
    if masks is not None and len(masks) != gate:
        raise ValueError("Held ReLU mask count changed")
    return value


def evaluate_members(heads, captured, masks, mode):
    """Changed-precision controls have their own derivatives, never native labels."""
    if mode not in MODES:
        raise ValueError("Unknown small-head diagnostic mode")
    result = []
    for member, head in enumerate(heads):
        double = mode.startswith("float64")
        shared_masks = masks[member]["shared"] if mode in ("frozen_shared", "frozen_both") else None
        regression_masks = masks[member]["regression"] if mode in ("frozen_regression", "frozen_both") else None
        shared = captured[key(member, "shared")] if mode.endswith("_shared") and mode.startswith(("native", "float64")) else sequential(
            head.affinity_out_mlp, captured[key(member, "pooled")], double=double, masks=shared_masks)
        result.append(sequential(head.to_affinity_pred_value, shared, double=double, masks=regression_masks))
    return result


def fixed_linear(value, base, cotangent):
    return ((value.double() - base.double()) * cotangent.double()).sum()


def activation_observations(captured, baseline, cotangents):
    """Native preactivation sign crossings are observational, not a pass rule."""
    result = {}
    for label, value in captured.items():
        if "_relu_" not in label:
            continue
        flipped = (value > 0) != (baseline[label] > 0)
        cotangent = cotangents[label]
        result[label] = {"elements": value.numel(), "sign_crossings": int(flipped.sum()),
            "crossings_with_nonzero_baseline_cotangent": int((flipped & (cotangent != 0)).sum()),
            "baseline_zeros": int((baseline[label] == 0).sum()), "endpoint_zeros": int((value == 0).sum()),
            "minimum_absolute_baseline": float(baseline[label].abs().min()),
            "minimum_absolute_endpoint": float(value.abs().min()),
            "absolute_crossed_linear_signal": float(((value.double() - baseline[label].double()) * cotangent.double()).abs()[flipped].sum())}
    return result


def exact_parity(reference_value, value, reference_gradient, gradient):
    import torch
    return {"scalar_exact": torch.equal(reference_value.detach(), value.detach()),
        "gradient_exact": torch.equal(reference_gradient, gradient),
        "gradient_difference": head_helpers()._difference(reference_gradient, gradient)}


def verify_source086(gradient_record, report):
    """Bind the earlier isolated channel without weakening source081 asset checks."""
    helper = head_helpers()._helper()
    if (helper._sha(SOURCE086_REPORT) != SOURCE086_REPORT_SHA256
            or helper._sha(Path(SOURCE086_REPORT).parent / "run.json") != SOURCE086_RUN_SHA256):
        raise ValueError("Pinned source086 regression report changed")
    previous = helper._read(SOURCE086_REPORT)
    run = helper._read(Path(SOURCE086_REPORT).parent / "run.json")
    prior = previous["baseline"]["gradients"]["isolated_head_affinity_pred_value"]
    if (run.get("status") != "completed" or run.get("source_archive_sha256") != SOURCE086_ARCHIVE
            or previous.get("diagnostic_completed") is not True
            or previous.get("frozen_state_unchanged") is not True
            or previous.get("parent_unchanged") is not True
            or previous.get("target") != report["target"]
            or previous.get("identities_before") != report["identities_before"]
            or previous.get("identities_before") != previous.get("identities_after")
            or prior != SOURCE086_GRADIENT or gradient_record != prior):
        raise ValueError("Regression derivative/provenance differs from authenticated source086")
    return {"path": SOURCE086_REPORT, "sha256": SOURCE086_REPORT_SHA256,
            "run_sha256": SOURCE086_RUN_SHA256, "source_archive_sha256": SOURCE086_ARCHIVE,
            "gradient": prior, "exact_gradient_match": True}


def assert_detached_tree(value):
    """A phase may return observations, never a live native autograd graph."""
    import torch
    if isinstance(value, torch.Tensor):
        if value.requires_grad or value.grad_fn is not None:
            raise ValueError("Phase result retained a live autograd tensor")
    elif isinstance(value, dict):
        for item in value.values():
            assert_detached_tree(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            assert_detached_tree(item)


def cuda_memory(device):
    """Allocation telemetry only; no allocator tuning or empty_cache workaround."""
    import torch
    if torch.device(device).type != "cuda":
        return {"device": str(device), "cuda": False}
    torch.cuda.synchronize(device)
    return {"device": str(device), "cuda": True,
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def require_tensor_parity(observed, reference, scope):
    """Authenticate every observed native boundary before a phase's VJPs."""
    import torch
    if set(observed) != set(reference):
        raise ValueError("Fresh phase tensor membership changed: " + scope)
    for name, value in observed.items():
        original = reference[name]
        if (value.dtype != original.dtype or value.shape != original.shape or value.device != original.device
                or not torch.equal(value.detach(), original)):
            raise ValueError("Fresh phase native tensor changed: " + scope + ":" + name)
    # Byte digests also distinguish signed-zero payloads that torch.equal treats
    # as equal. These are observations only and never enter the model path.
    current, expected = head_helpers()._metadata(observed), head_helpers()._metadata(reference)
    if any(current[name]["sha256"] != expected[name]["sha256"] for name in reference):
        raise ValueError("Fresh phase native tensor bytes changed: " + scope)
    return current


def detached_phase(label, operation, device, progress):
    """Bound native graph lifetime to one phase and retain progress on failure.

    Every operation returns (detached scientific data, JSON summary). Collection
    between phases releases checkpoint/autograd cycles; it does not alter
    forwards, backward formulas, tensors, tolerances or scientific selection.
    """
    import torch
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    event = {"status": "running", "memory_before": cuda_memory(device)}
    started = time.perf_counter()
    progress(label, deepcopy(event))
    try:
        result, summary = operation()
        assert_detached_tree(result)
        assert_detached_tree(summary)
        # operation's frame has returned. Only detached data remains reachable.
        gc.collect()
        event.update(status="completed", elapsed_seconds=time.perf_counter() - started,
            memory_after_graph_release=cuda_memory(device), observations=summary,
            no_live_autograd_tensor_returned=True)
        progress(label, deepcopy(event))
        return result
    except Exception as error:
        event.update(status="failed", elapsed_seconds=time.perf_counter() - started,
            memory_at_failure=cuda_memory(device), error=str(error), error_type=type(error).__name__)
        progress(label, deepcopy(event))
        raise


def baseline_probe(decoder, point, branch, config, counters, target, weight, *, progress=lambda label, record: None):
    """Nine identical-point phases instead of one growing retained native graph.

    Source087 attempt001 retained one checkpointed graph through 29 completed
    native reverse passes and failed on pass30. The original baseline, native
    cuts and seven cached modes now have independent lifetimes. Every new
    forward must exactly reproduce all original native/head/capture tensors.
    """
    import torch
    head = head_helpers()
    modules, architecture = native_heads(branch.scorer)

    def original_phase():
        with capture_affinity(modules) as captured:
            original = _ORIGINAL_BASELINE(decoder, point, branch, config, counters, target, weight)
            small_base = {name: value.detach().clone() for name, value in captured.items()}
        return (original, small_base), {"baseline": original[0], "small_head_tensors": head._metadata(small_base),
            "affinity_architecture": architecture, "latent_vjp_calls": 11, "local_head_cotangent_calls": 2}

    original, small_base = detached_phase("original", original_phase, point.device, progress)
    report, gradients, base, cotangents, chain, geometry = original
    pocket, ligand, centers = geometry
    coefficient = cotangents["binding"]["affinity_pred_value"]
    masks = [{block: tuple(small_base[key(index, f"{block}_relu_{gate}")] > 0 for gate in (0, 1))
              for block in ("shared", "regression")} for index in range(2)]

    def phase_forward(probe, heads, captured, phase):
        _, _, outputs, terms, detail = head._forward(
            decoder, probe, branch, config, counters, target, weight, "affinity_causal_baseline:" + phase)
        controls = head.cached_controls(outputs, heads, config, pocket, ligand, centers, terms["repair_nll"], weight)
        values = {**terms, **controls}
        require_tensor_parity(heads, base, phase + ":native_heads")
        require_tensor_parity(captured, small_base, phase + ":pooled_shared_value_preactivations")
        if ({name: float(value.detach()) for name, value in values.items()} != report["values"]
                or {name: str(value.dtype) for name, value in values.items()} != report["dtypes"]
                or {name: float(value.detach()) for name, value in outputs.items()} != report["native_outputs"]
                or detail["target_sha256"] != report["target_sha256"]
                or not torch.equal(probe.detach(), point)):
            raise ValueError("Fresh baseline phase changed native tensors/scalars/target/point: " + phase)
        return heads["affinity_pred_value"]

    def derivative(value, probe, label, outputs=None, *, retain=True):
        gradient = counters.call("boltz", "affinity_phase_backward:" + label, "backward", lambda:
            torch.autograd.grad(value, probe, grad_outputs=outputs, retain_graph=retain)[0], point.device).detach()
        if not bool(gradient.isfinite().all()):
            raise ValueError("Nonfinite affinity derivative: " + label)
        return gradient

    def cuts_phase():
        probe = point.detach().clone().requires_grad_()
        with torch.enable_grad(), head.capture_heads(branch.scorer) as heads, capture_affinity(modules) as captured:
            mean = phase_forward(probe, heads, captured, "affinity_cuts")
            regression_gradient = derivative(mean, probe, "affinity_cuts:regression", coefficient)
            members = [captured[key(index, "value")] for index in range(2)]
            if not torch.equal(torch.stack([value.reshape_as(mean) for value in members]).mean(0), mean):
                raise ValueError("Native ensemble mean reconstruction differs")
            preactivations = {name: value for name, value in captured.items() if "_relu_" in name}
            pre_grads = torch.autograd.grad(mean, tuple(preactivations.values()), grad_outputs=coefficient, retain_graph=True)
            pre_cotangents = {name: value.detach() for name, value in zip(preactivations, pre_grads)}
            # Compute every tiny downstream cotangent before the final upstream
            # VJP frees saved tensors. Boundaries remain separate complete cuts.
            cut_cotangents = {}
            for level in LEVELS:
                tensors = tuple(captured[key(index, level)] for index in range(2))
                local = torch.autograd.grad(mean, tensors, grad_outputs=coefficient, retain_graph=True)
                cut_cotangents.update({key(index, level): value.detach() for index, value in enumerate(local)})
            cut_gradients, cut_meta = {}, {}
            for level in LEVELS:
                partial = []
                for index in range(2):
                    label = key(index, level)
                    grad = derivative(captured[label], probe, "affinity_cuts:" + label, cut_cotangents[label],
                        retain=not (level == LEVELS[-1] and index == 1))
                    partial.append(grad)
                    cut_gradients["cut_" + label] = grad
                cut_gradients["cut_" + level] = sum(partial)
                cut_meta[level] = head.chain_completeness(regression_gradient, sum(partial))
            result = {"regression_gradient": regression_gradient, "regression_value": (mean.detach() * coefficient).sum(),
                "cut_gradients": cut_gradients, "cut_cotangents": cut_cotangents, "pre_cotangents": pre_cotangents,
                "cut_completeness": cut_meta,
                "member_values": [(value.detach() * cut_cotangents[key(index, "value")]).sum() for index, value in enumerate(members)]}
        return result, {"fresh_original_forward_exact": True, "regression_gradient": head._gradient_record(regression_gradient),
            "cut_completeness": cut_meta, "latent_vjp_calls": 7, "local_head_cotangent_calls": 4}

    cut = detached_phase("affinity_cuts", cuts_phase, point.device, progress)
    regression_gradient = cut["regression_gradient"]
    regression_value = cut["regression_value"]
    member_cotangents = [cut["cut_cotangents"][key(index, "value")] for index in range(2)]
    # Keep the original baseline dictionaries unchanged while fresh phases are
    # authenticated. Extended FD channels are appended only after every phase.
    extra_values = {"affinity_regression": regression_value}
    extra_gradients = {"affinity_regression": regression_gradient, **cut["cut_gradients"]}
    for name in cut["cut_gradients"]:
        extra_values[name] = torch.zeros((), dtype=torch.float64, device=point.device)
    cache_parity, changed_baseline, member_sum = {}, {}, {}

    def mode_phase(mode):
        probe = point.detach().clone().requires_grad_()
        with torch.enable_grad(), head.capture_heads(branch.scorer) as heads, capture_affinity(modules) as captured:
            mean = phase_forward(probe, heads, captured, mode)
            fresh_gradient = derivative(mean, probe, mode + ":original_regression", coefficient)
            if not torch.equal(fresh_gradient, regression_gradient):
                raise ValueError("Fresh phase changed original regression derivative: " + mode)
            reevaluated = evaluate_members(modules, captured, masks, mode)
            native_dtype = not mode.startswith("float64")
            mode_coefficient = coefficient if native_dtype else coefficient.double()
            value = (torch.stack([item.reshape_as(mean) for item in reevaluated]).mean(0) * mode_coefficient).sum()
            gradient = derivative(value, probe, mode + ":cached_ensemble")
            values, mode_gradients = {"mlp_" + mode: value.detach()}, {"mlp_" + mode: gradient}
            separate = []
            for index, member in enumerate(reevaluated):
                c_i = member_cotangents[index] if native_dtype else member_cotangents[index].double()
                weighted = (member * c_i).sum()
                name = f"mlp_{mode}_member_{index}"
                values[name] = weighted.detach()
                mode_gradients[name] = derivative(weighted, probe, name, retain=index != 1)
                separate.append(mode_gradients[name])
            parity = exact_parity(regression_value, value, regression_gradient, gradient)
            if mode in EXACT_MODES:
                parity["member_values_exact"] = all(torch.equal(value.detach(), small_base[key(index, "value")])
                    for index, value in enumerate(reevaluated))
                parity["member_gradients_exact"] = all(torch.equal(separate[index], cut["cut_gradients"]["cut_" + key(index, "value")])
                    for index in range(2))
            else:
                parity.update(scalar_difference_float64=float(value.detach().double() - regression_value.double()),
                    scope="Own changed-arithmetic baseline and derivative; exact native parity is not assumed.")
            completeness = head.chain_completeness(gradient, sum(separate))
            result = {"values": values, "gradients": mode_gradients, "parity": parity, "completeness": completeness}
        return result, {"fresh_original_forward_exact": True, "fresh_original_regression_gradient_exact": True,
            "cached_baseline_parity": parity, "member_sum_completeness": completeness,
            "values": {name: float(value) for name, value in values.items()},
            "dtypes": {name: str(value.dtype) for name, value in values.items()},
            "gradients": {name: head._gradient_record(value) for name, value in mode_gradients.items()},
            "latent_vjp_calls": 4, "cached_member_path_evaluations": 2}

    for mode in MODES:
        result = detached_phase(mode, lambda mode=mode: mode_phase(mode), point.device, progress)
        extra_values.update(result["values"])
        extra_gradients.update(result["gradients"])
        (cache_parity if mode in EXACT_MODES else changed_baseline)[mode] = result["parity"]
        member_sum[mode] = result["completeness"]
    for index in range(2):
        name = f"affinity_regression_member_{index}"
        extra_values[name] = cut["member_values"][index]
        extra_gradients[name] = cut["cut_gradients"]["cut_" + key(index, "value")]
    gradients.update(extra_gradients)
    report["values"].update({name: float(value) for name, value in extra_values.items()})
    report["dtypes"].update({name: str(value.dtype) for name, value in extra_values.items()})
    report["gradients"].update({name: head._gradient_record(value) for name, value in extra_gradients.items()})
    report.update(affinity_architecture=architecture, affinity_cut_completeness=cut["cut_completeness"],
        cached_member_sum_completeness=member_sum, cached_exact_baseline_parity=cache_parity,
        changed_arithmetic_baseline_comparison=changed_baseline,
        native_ensemble_output_cotangent=coefficient.cpu().tolist(),
        native_member_output_cotangents=[value.cpu().tolist() for value in member_cotangents],
        small_head_tensors=head._metadata(small_base), small_head_cut_cotangents=head._metadata(cut["cut_cotangents"]),
        relu_baseline=activation_observations(small_base, small_base, cut["pre_cotangents"]),
        held_relu_masks=head._metadata({key(index, f"{block}_mask_{gate}"): mask
            for index, item in enumerate(masks) for block, masks_i in item.items() for gate, mask in enumerate(masks_i)}),
        cached_member_path_evaluations=len(MODES) * 2,
        native_baseline_forward_phases=list(BASELINE_PHASES),
        graph_lifetime_policy="Separate same-z native graph per phase; detached returns, explicit final VJP release and GC; no allocator workaround.",
        counterfactual_scope="Only cached small MLPs change; FP64 controls use own gradients; held signs change endpoints, not production.")
    state = {"modules": modules, "base": small_base, "cuts": cut["cut_cotangents"], "pre_cotangents": cut["pre_cotangents"],
             "masks": masks, "coefficient": coefficient, "member_cotangents": member_cotangents}
    return report, gradients, base, cotangents, chain, (*geometry, state)

def endpoint_probe(decoder, point, branch, config, counters, target, weight, base, cotangents, geometry):
    """One unchanged native endpoint supplies every cut and cached comparison."""
    import torch
    from latentmol.boltz.active_set import select_slots
    from latentmol.multimodel.study import tensor_digest
    head = head_helpers()
    pocket, ligand, centers, state = geometry
    with torch.no_grad(), head.capture_heads(branch.scorer) as heads, capture_affinity(state["modules"]) as captured:
        decoded, graph, outputs, terms, detail = head._forward(
            decoder, point, branch, config, counters, target, weight, "affinity_causal_endpoint")
        controls = head.cached_controls(outputs, heads, config, pocket, ligand, centers, terms["repair_nll"], weight)
        readouts, contributions = {}, {}
        for name, cotangent in cotangents.items():
            readouts[name], contributions[name] = head.fixed_readout(heads, base, cotangent)
        mean, coefficient = heads["affinity_pred_value"], state["coefficient"]
        values = {**terms, **controls, **{"head_readout_" + name: value for name, value in readouts.items()},
                  "affinity_regression": (mean * coefficient).sum()}
        for level in LEVELS:
            partial = []
            for index in range(2):
                label = key(index, level)
                value = fixed_linear(captured[label], state["base"][label], state["cuts"][label])
                partial.append(value)
                values["cut_" + label] = value
            values["cut_" + level] = sum(partial)
        member_values = [captured[key(index, "value")] for index in range(2)]
        if not torch.equal(torch.stack([value.reshape_as(mean) for value in member_values]).mean(0), mean):
            raise ValueError("Endpoint ensemble mean reconstruction differs")
        for index, value in enumerate(member_values):
            values[f"affinity_regression_member_{index}"] = (value * state["member_cotangents"][index]).sum()
        parity = {}
        for mode in MODES:
            members = evaluate_members(state["modules"], captured, state["masks"], mode)
            c = coefficient.double() if mode.startswith("float64") else coefficient
            values["mlp_" + mode] = (torch.stack([value.reshape_as(mean) for value in members]).mean(0) * c).sum()
            for index, value in enumerate(members):
                c_i = state["member_cotangents"][index]
                c_i = c_i.double() if mode.startswith("float64") else c_i
                values[f"mlp_{mode}_member_{index}"] = (value * c_i).sum()
            if mode.startswith("native"):
                parity[mode] = all(torch.equal(a, b) for a, b in zip(members, member_values))
        natural = select_slots(decoded, temperature=config.temperature, threshold=config.threshold, require_nonempty=False)
        result = {"values": {name: float(value) for name, value in values.items()},
            "dtypes": {name: str(value.dtype) for name, value in values.items()}, "head_contributions": contributions,
            "native_outputs": {name: float(value) for name, value in outputs.items()},
            "graph_sha256": tensor_digest({name: getattr(graph, name) for name in ("atoms", "charges", "bonds")}),
            "target_sha256": detail["target_sha256"], "contact_selection": deepcopy(branch.scorer.last_contact_selection),
            "natural_selector_crossed": not torch.equal(natural.active, branch.prepared.active),
            "cached_native_member_exact": parity, "small_head_tensors": head._metadata(captured),
            "relu_observations": activation_observations(captured, state["base"], state["pre_cotangents"]),
            "cached_member_path_evaluations": len(MODES) * 2}
    return result


@contextmanager
def diagnostic_helpers(progress=lambda label, record: None):
    """Replace observer functions temporarily; production code is never patched."""
    head = head_helpers()
    before = head.baseline_probe, head.endpoint_probe
    head.baseline_probe = lambda *args: baseline_probe(*args, progress=progress)
    head.endpoint_probe = endpoint_probe
    try:
        yield head
    finally:
        head.baseline_probe, head.endpoint_probe = before


def baseline_ready(baseline):
    return (set(baseline["affinity_cut_completeness"]) == set(LEVELS)
        and set(baseline["cached_member_sum_completeness"]) == set(MODES)
        and all(row["engineering_chain_complete"] for row in baseline["affinity_cut_completeness"].values())
        and all(row["engineering_chain_complete"] for row in baseline["cached_member_sum_completeness"].values())
        and all(row["scalar_exact"] and row["gradient_exact"] and row["member_values_exact"] and row["member_gradients_exact"]
                for row in baseline["cached_exact_baseline_parity"].values())
        and set(baseline["cached_exact_baseline_parity"]) == set(EXACT_MODES))


def probe_case(*args, persist=lambda: None, report=None, **kwargs):
    report = report if report is not None else {}
    report.update(affinity_module_sha256=head_helpers()._helper()._sha(__file__), affinity_claim_limit=CLAIM_LIMIT)
    report["baseline_phases"] = {}
    def phase_progress(label, event):
        if label not in BASELINE_PHASES:
            raise ValueError("Unexpected native baseline phase")
        report["baseline_phases"][label] = event
        # Persist immediately, before another graph or backward is attempted.
        persist()
    def checked_persist():
        report["cached_controls_scope"] = ("Inherited FP64 contact/scalar controls plus exact native cached MLP reevaluation, "
            "FP64 regression-only/full-small-MLP controls with own derivatives, and held-ReLU endpoint counterfactuals. "
            "All full native scorer forwards remain unchanged; changed endpoints never certify original production gates.")
        baseline = report.get("baseline")
        if baseline is not None:
            ready = baseline_ready(baseline)
            report["affinity_baseline_validated"] = ready
            if not ready:
                report["affinity_status"] = "baseline_parity_or_cut_incomplete"
                persist()
                raise ValueError("Affinity baseline parity/cut failed; retained without endpoint attribution")
            if "source086_regression_reference" not in report:
                # The tensor digest was measured by the baseline, not inferred
                # from equal normalized directions or an unrelated total VJP.
                try:
                    report["source086_regression_reference"] = verify_source086(baseline["gradients"]["affinity_regression"], report)
                except Exception:
                    persist()
                    raise
        for row in report.get("finite_differences", []):
            if any(not all(row[sign]["cached_native_member_exact"].values()) for sign in ("plus", "minus")):
                persist()
                raise ValueError("Cached FP32 MLP endpoint differs from unchanged native output")
        report["cached_member_path_evaluations"] = (len(MODES) * 2 if baseline else 0) + sum(
            row[sign]["cached_member_path_evaluations"] for row in report.get("finite_differences", []) for sign in ("plus", "minus"))
        persist()
    with diagnostic_helpers(progress=phase_progress) as head:
        result = head.probe_case(*args, persist=checked_persist, report=report, **kwargs)
    result["affinity_status"] = "completed"
    checked_persist()
    return result


def selected_case(plan):
    matches = [case for case in plan["cases"] if case["id"] == CASE_ID]
    if len(matches) != 1 or matches[0]["step"] != 20:
        raise ValueError("Require original prior-5 step20 only")
    return matches[0]


def run_case(plan, case, output_root):
    head = head_helpers()
    helper = head._helper()
    reference, asset = head.reference_replay(case, plan)
    helper.probe_case = lambda *args, **kwargs: probe_case(*args, reference=reference, reference_asset=asset, **kwargs)
    helper.REPEATS, helper.CASE_TIMEOUT_SECONDS, helper.CLAIM_LIMIT = 1, CASE_TIMEOUT_SECONDS, CLAIM_LIMIT
    return helper.run_case(plan, case, output_root)


def run_study(path, sha):
    head = head_helpers()
    helper = head._helper()
    plan = helper.load_plan(path, sha)
    case = selected_case(plan)
    _, item = helper.failed_item(helper._read(helper.authenticated_artifact(case,
        Path(case["parent_run"]) / case["arm"] / "gate-0020.json")))
    randoms = [name for name, row in item["direction_results"].items() if name != "gradient" and row.get("passed") is False]
    calls = 2 + len(BASELINE_PHASES) + (1 + len(randoms)) * len(head.EPSILONS) * 2
    if set(randoms) != {"random_1", "random_2", "random_3"} or calls != MAXIMUM_SCORER_FORWARDS:
        raise ValueError("Original directions/native budget changed")
    head.reference_replay(case, plan)
    from latentmol.experiments import ExperimentRun, write_json
    from latentmol.boltz.workflow_process import run_logged
    configuration = {"plan": plan, "plan_sha256": sha, "module_sha256": helper._sha(__file__),
        "head_helper_sha256": helper._sha(Path(__file__).with_name("head_causal_diagnostic.py")),
        "loader_sha256": helper._sha(Path(__file__).with_name("failed_gate_attribution.py")),
        "native_affinity_source_sha256": NATIVE_SOURCE_SHA256, "selected_cases": [CASE_ID],
        "maximum_scorer_forwards": calls, "baseline_phases": list(BASELINE_PHASES),
        "additional_same_point_forwards_vs_attempt001": len(BASELINE_PHASES) - 1,
        "case_timeout_seconds": CASE_TIMEOUT_SECONDS, "claim_limit": CLAIM_LIMIT}
    with ExperimentRun(Path(plan["output_root"]), plan["name"], configuration, Path.cwd()) as run:
        execution = run_logged([sys.executable, str(Path(__file__).resolve()), "--plan", str(Path(path).resolve()),
            "--sha256", sha, "--child", CASE_ID, "--child-output", str(run.path / CASE_ID)],
            run.path / (CASE_ID + ".log"), timeout_seconds=CASE_TIMEOUT_SECONDS)
        complete = False
        if not execution["timed_out"] and execution["returncode"] == 0 and len(execution["run_paths"]) == 1:
            child = Path(execution["run_paths"][0])
            evidence = helper._read(child / "attribution.json")
            complete = (helper._read(child / "run.json")["status"] == "completed"
                and evidence.get("affinity_status") == "completed"
                and all(evidence.get(name) is True for name in ("diagnostic_completed", "frozen_state_unchanged", "parent_unchanged",
                    "baseline_exact_source081_scalar_parity", "engineering_head_cut_complete", "affinity_baseline_validated",
                    "original_failed_gradient_exact_source081_match"))
                and evidence.get("source086_regression_reference", {}).get("exact_gradient_match") is True)
            phases = evidence.get("baseline_phases", {})
            complete = (complete and set(phases) == set(BASELINE_PHASES)
                and all(row.get("status") == "completed" and row.get("no_live_autograd_tensor_returned") is True
                        for row in phases.values()))
        write_json(run.path / "cases.json", [{"case_id": CASE_ID, "diagnostic_completed": complete,
            "execution": execution, "old_gate_certified": False}])
        run.annotate(diagnostic_cases_completed=int(complete), diagnostic_cases_requested=1, old_gates_certified=False)
        if not complete:
            raise ValueError("Affinity causal diagnostic incomplete; receipts retained")
        return run.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--child")
    parser.add_argument("--child-output", type=Path)
    args = parser.parse_args()
    helper = head_helpers()._helper()
    if not os.environ.get("LSB_JOBID") or not helper._hash(os.environ.get("LATENTMOL_SOURCE_SHA256")):
        raise ValueError("Affinity probes require an immutable DTU HPC allocation")
    plan = helper.load_plan(args.plan, args.sha256)
    case = selected_case(plan)
    if args.child is None:
        if args.child_output is not None:
            raise ValueError("Child output requires selected child")
        print(run_study(args.plan, args.sha256), flush=True)
    else:
        if args.child != CASE_ID or args.child_output is None:
            raise ValueError("Require original prior-5 child/output")
        print(run_case(plan, case, args.child_output), flush=True)


if __name__ == "__main__":
    main()
