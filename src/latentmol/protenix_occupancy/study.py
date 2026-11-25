"""Native parity and direct NO_ATOM derivatives on allocated HPC.

The first-stage conditions C/Z isolate element mass. They intentionally keep
other features fixed; they are not a complete variable-occupancy adapter. G may
not be silently substituted with element scaling or inferred from these gates.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from latentmol.multimodel.external import file_sha256, tensor_tree_sha256, tree_to

PHASE = "protenix-occupancy-native-parity-gradient-v1"
NATIVE_REPEAT_POLICY = "native-repeat-loss-tv-v1"
OUTPUT_PARITY_POLICY = "native-output-loss-tv-v1"
Q_GRID = (0., .01, .1, .25, .5, .75, .9, .99, 1.)
FD_POLICY = {
    "version": "attached-repeat-loss-difference-fd-v1",
    "baseline_repeats": 8,
    "endpoint_repeats": 3,
    "range_multiplier": 4.0,
    "loss_difference_floor": 1e-12,
    "maximum_loss_difference_uncertainty": 1e-6,
    "signal_to_uncertainty": 10.0,
    "derivative_atol": 1e-6,
    "derivative_rtol": 0.05,
    "minimum_agreeing_scales": 2,
    "steps": [0.01, 0.003, 0.001],
}
FD_STEPS = tuple(FD_POLICY["steps"])


def write_json(path: Path, value: dict) -> None:
    """Persist each completed stage without overwriting historical receipts."""
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def floating_element_features(features: dict, *, requires_grad=False) -> dict:
    """Copy native elements to one common FP32 layout for both mechanism arms.

    Native one-hot elements may be integers. Only the explicit gradient probe
    detaches the baseline and creates a fresh floating leaf; ordinary conversion
    preserves any existing graph. Every non-element field remains shared.
    """
    element = features["ref_element"]
    if (not isinstance(element, torch.Tensor) or element.ndim != 2
            or element.shape[1] != 128 or element.layout != torch.strided):
        raise ValueError("Require official unbatched ref_element[A,128]")
    copied = element.to(dtype=torch.float32).contiguous().clone()
    if requires_grad:
        copied = copied.detach().requires_grad_(True)
    return dict(features, ref_element=copied)


def inject_occupancy(features: dict, optional_atom: int, q: torch.Tensor,
                     condition: str) -> dict:
    """C discards q; Z retains its real-element mass without normalizing it.

    NO_ATOM has no native element column. Its contribution is a fixed zero
    vector. Joint-softmax dependence is supplied by q=sigmoid(real_logit).
    All non-element fields are deliberately clamped for these mechanism arms.
    """
    if condition not in ("C", "Z"):
        raise NotImplementedError("G requires separately audited occupancy propagation")
    if not isinstance(q, torch.Tensor) or q.numel() != 1 or not q.is_floating_point():
        raise ValueError("Occupancy must be one floating scalar tensor")
    if not bool(torch.isfinite(q).all()) or not bool(((q >= 0) & (q <= 1)).all()):
        raise ValueError("Occupancy must be finite in [0,1]")
    element = features["ref_element"]
    if element.ndim != 2 or element.shape[1] != 128:
        raise ValueError("Require official unbatched ref_element[A,128]")
    if isinstance(optional_atom, bool) or not isinstance(optional_atom, int) or not 0 <= optional_atom < element.shape[0]:
        raise ValueError("Optional atom index is outside native rows")
    if q.device != element.device:
        raise ValueError("Occupancy and native features must share a device")
    total = element[optional_atom].float().sum()
    if not torch.allclose(total, total.new_tensor(1.), atol=1e-6, rtol=0):
        raise ValueError("Optional native element must be a normalized hard reference")
    # Equal representation boundary and row operation in C and Z. C changes no
    # chemical values and discards q; only Z retains its real-element mass.
    result = floating_element_features(features)
    copied = result["ref_element"]
    multiplier = copied.new_tensor(1.) if condition == "C" else q.reshape(())
    copied[optional_atom] = copied[optional_atom].clone() * multiplier
    return result


def reduce_distogram(logits: torch.Tensor, pair_indices: torch.Tensor,
                      target_bins: torch.Tensor) -> torch.Tensor:
    """Same retained pairs/targets/denominator at every occupancy and endpoint."""
    if logits.ndim != 3 or logits.shape[0] != logits.shape[1] or logits.shape[-1] < 2:
        raise ValueError("Require native distogram logits[T,T,K>=2]")
    if pair_indices.ndim != 2 or pair_indices.shape[1] != 2 or pair_indices.dtype != torch.long or not pair_indices.shape[0]:
        raise ValueError("Require nonempty long pair_indices[M,2]")
    if target_bins.shape != (pair_indices.shape[0],) or target_bins.dtype != torch.long:
        raise ValueError("Require long target_bins[M]")
    if bool((pair_indices < 0).any()) or bool((pair_indices >= logits.shape[0]).any()):
        raise ValueError("Readout pair index out of range")
    canonical = pair_indices.sort(dim=-1).values
    if canonical.unique(dim=0).shape[0] != pair_indices.shape[0] or bool((canonical[:,0] == canonical[:,1]).any()):
        raise ValueError("Readout must contain each unordered cross pair once and no self-pairs")
    if bool((target_bins < 0).any()) or bool((target_bins >= logits.shape[-1]).any()):
        raise ValueError("Readout target bin out of range")
    # Keep the pretrained forward FP32; accumulate the fixed scalar in FP64 so
    # averaging/CE rounding does not hide small finite-difference changes.
    selected = logits[pair_indices[:, 0], pair_indices[:, 1]].double()
    return -selected.log_softmax(-1).gather(-1, target_bins[:, None]).mean() / math.log(logits.shape[-1])


def selected_outputs(logits, pair_indices, target_bins):
    selected = logits[pair_indices[:, 0], pair_indices[:, 1]].float()
    return dict(logits=selected, probabilities=selected.softmax(-1),
                loss=reduce_distogram(logits, pair_indices, target_bins))


def compare_outputs(actual: dict, reference: dict, *, atol=1e-5, rtol=1e-4) -> dict:
    for field in ("logits", "probabilities", "loss"):
        if (actual[field].shape != reference[field].shape or
                actual[field].numel() == 0):
            raise ValueError("Native comparison requires matching nonempty output shapes")
    result = {}
    for field in ("logits", "loss"):
        a, b = actual[field], reference[field]
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        result[field] = dict(finite=finite,
            passed=finite and bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
            max_absolute_error=float((a-b).abs().max().detach().cpu()) if finite else None)
    tv = (actual["probabilities"] - reference["probabilities"]).abs().sum(-1) / 2
    result["maximum_pair_total_variation"] = float(tv.max().detach().cpu()) if bool(torch.isfinite(tv).all()) else None
    result["passed"] = all(result[k]["passed"] for k in ("logits", "loss")) and result["maximum_pair_total_variation"] is not None and result["maximum_pair_total_variation"] <= 1e-4
    return result


def derivative_agreement(autograd: float, fd: float, *, atol=1e-6, rtol=.05) -> dict:
    finite = math.isfinite(autograd) and math.isfinite(fd)
    error = abs(autograd-fd) if finite else None
    return dict(autograd=autograd if math.isfinite(autograd) else None,
        finite_difference=fd if math.isfinite(fd) else None, absolute_error=error,
        passed=finite and error <= atol+rtol*max(abs(autograd), abs(fd)))


def calibrate_loss_noise(groups: list[list[float]]) -> dict:
    """Bound the declared empirical allowance using identical-input repeats.

    Each group contains repeated evaluations at ONE input, never values pooled
    across distinct q values. The fixed multiplier is a conservative empirical
    allowance, not a statistical confidence bound or a backend diagnosis.
    """
    if not groups or any(len(values) < 2 for values in groups):
        raise ValueError("Noise calibration requires repeated identical-input groups")
    if any(not math.isfinite(value) for values in groups for value in values):
        raise ValueError("Nonfinite repeat-noise calibration")
    ranges = [max(values)-min(values) for values in groups]
    uncertainty = max(FD_POLICY["loss_difference_floor"],
                      FD_POLICY["range_multiplier"]*max(ranges))
    return dict(version=FD_POLICY["version"], groups=groups, group_ranges=ranges,
        loss_difference_uncertainty=uncertainty,
        maximum_uncertainty=FD_POLICY["maximum_loss_difference_uncertainty"],
        passed=uncertainty <= FD_POLICY["maximum_loss_difference_uncertainty"],
        interpretation="Fixed empirical allowance; not a rigorous noise bound")


def loss_difference_agreement(autograd: float, plus: list[float],
                              minus: list[float], epsilon: float,
                              calibration: dict) -> dict:
    """Compare before dividing by 2h; preserve strict FD as a diagnostic.

    A resolved signal must exceed the allowance in both the measured and
    autograd-predicted loss differences, with the same sign. Unresolved values
    may establish consistency with zero, never useful Z sensitivity.
    """
    if not math.isfinite(epsilon) or epsilon <= 0 or not plus or not minus:
        raise ValueError("Require positive finite epsilon and endpoint samples")
    finite = math.isfinite(autograd) and all(math.isfinite(v) for v in plus+minus)
    uncertainty = calibration.get("loss_difference_uncertainty")
    calibrated = (calibration.get("version") == FD_POLICY["version"]
        and calibration.get("passed") is True
        and isinstance(uncertainty, (int, float)) and math.isfinite(uncertainty)
        and FD_POLICY["loss_difference_floor"] <= uncertainty
        <= FD_POLICY["maximum_loss_difference_uncertainty"])
    p = math.fsum(plus)/len(plus) if finite else float("nan")
    m = math.fsum(minus)/len(minus) if finite else float("nan")
    delta = p-m
    fd = delta/(2*epsilon)
    strict = derivative_agreement(autograd, fd)
    predicted = 2*epsilon*autograd
    residual = abs(delta-predicted)
    base = 2*epsilon*(FD_POLICY["derivative_atol"]
        + FD_POLICY["derivative_rtol"]*max(abs(autograd), abs(fd)))
    passed = finite and calibrated and residual <= base+uncertainty
    resolved = (passed and delta*predicted > 0
        and min(abs(delta),abs(predicted)) > FD_POLICY["signal_to_uncertainty"]*uncertainty)
    return dict(strict, strict_passed=strict["passed"],
        passed=bool(passed), epsilon=epsilon, plus=p if finite else None,
        minus=m if finite else None, plus_samples=plus, minus_samples=minus,
        loss_difference=delta if finite else None,
        predicted_loss_difference=predicted if finite else None,
        loss_difference_residual=residual if finite else None,
        strict_loss_difference_tolerance=base if finite else None,
        loss_difference_uncertainty=uncertainty,
        resolved_above_repeat_noise=bool(resolved),
        policy_version=FD_POLICY["version"])


def compare_parity_outputs(actual: dict, reference: dict) -> dict:
    """User-directed output policy: raw-logit closeness is diagnostic only.

    Require finite logits, unchanged scalar-loss tolerance and probability TV.
    Preserve the original strict outcome and every numerical observation.
    """
    result = compare_outputs(actual, reference)
    result.update(version=OUTPUT_PARITY_POLICY, logits_used_as_gate=False,
        strict_parity_passed=result["passed"])
    tv = result["maximum_pair_total_variation"]
    result["passed"] = (result["logits"]["finite"] and result["loss"]["passed"]
        and tv is not None and tv <= 1e-4)
    return result


def compare_native_repeat(actual: dict, reference: dict) -> dict:
    """Apply the same criterion to repeats with its declared repeat identity."""
    result = compare_parity_outputs(actual, reference)
    result["version"] = NATIVE_REPEAT_POLICY
    return result


class CountedForward:
    def __init__(self, native, budget, deadline):
        self.native, self.budget, self.deadline = native, budget, deadline
        self.calls = 0
        self.fixed_rng = self.snapshot_rng()

    @staticmethod
    def snapshot_rng():
        return random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all()

    @staticmethod
    def restore_rng(state):
        random.setstate(state[0]);np.random.set_state(state[1])
        torch.set_rng_state(state[2]);torch.cuda.set_rng_state_all(state[3])

    def __call__(self, features, *, attached, legacy=False):
        if self.calls >= self.budget or time.monotonic() >= self.deadline:
            raise RuntimeError("Declared model-call/time budget exhausted")
        self.calls += 1
        outside = self.snapshot_rng()
        self.restore_rng(self.fixed_rng)
        try:
            if legacy:
                if not attached:
                    raise ValueError("Legacy diagnostics require explicit attached path")
                return self.native.legacy_logits(features, cycles=1)
            return self.native.attached_logits(features, cycles=1) if attached else self.native.native_logits(features, cycles=1)
        finally:
            self.restore_rng(outside)


def trace_path_diagnostics(native, features, pairs, targets, forward):
    """Forensic module traces; hooks can affect allocation, so never gate on them."""
    names = ("input_embedder", "constraint_embedder", "linear_no_bias_sinit",
        "linear_no_bias_zinit1", "linear_no_bias_zinit2", "relative_position_encoding",
        "linear_no_bias_token_bond", "layernorm_z_cycle", "linear_no_bias_z_cycle",
        "msa_module", "layernorm_s", "linear_no_bias_s", "pairformer_stack", "distogram_head")

    def flatten(value):
        if isinstance(value, torch.Tensor):
            return [value.detach().clone()]
        if isinstance(value, (tuple, list)):
            return [tensor for item in value for tensor in flatten(item)]
        if value is None:
            return []
        raise ValueError("Unexpected traced native module output")

    traces = {}
    for path, attached, legacy in (("native", False, False), ("legacy_manual", True, True),
                                   ("native_ast", True, False)):
        tensors, order, handles = {}, [], []

        def hook_for(name):
            def hook(_module, _inputs, output):
                if name in tensors:
                    raise ValueError("Trace expected one evaluation per named native module")
                tensors[name] = flatten(output)
                order.append(name)
            return hook

        try:
            for name in names:
                handles.append(getattr(native.model, name).register_forward_hook(hook_for(name)))
            with torch.no_grad():
                values = selected_outputs(forward(features, attached=attached, legacy=legacy), pairs, targets)
            traces[path] = (tensors, order, {k: v.detach() for k, v in values.items()})
        finally:
            for handle in handles:
                handle.remove()
    base, order, reference = traces["native"]
    result = dict(instrumented=True, used_as_gate=False,
        numerical_differences_used_as_gate=False, instrumentation_errors_stop_execution=True,
        caveat="Trace clones alter allocation/lifetimes; uninstrumented parity remains authoritative",
        module_order=order, comparisons={})
    for path in ("legacy_manual", "native_ast"):
        tensors, actual_order, actual = traces[path]
        if actual_order != order:
            raise ValueError("Native execution module order differs")
        rows = []
        for name in order:
            if len(tensors[name]) != len(base[name]):
                raise ValueError("Native execution module tensor count differs")
            for index, (a, b) in enumerate(zip(tensors[name], base[name])):
                if a.shape != b.shape:
                    raise ValueError("Native execution module output shape differs")
                finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
                rows.append(dict(module=name, output_index=index, shape=list(a.shape), finite=finite,
                    exact=finite and bool(torch.equal(a, b)),
                    max_absolute_error=float((a-b).abs().max()) if finite else None))
        result["comparisons"][path] = dict(modules=rows,
            first_nonexact=next((row["module"] for row in rows if not row["exact"]), None),
            selected_outputs=compare_outputs(actual, reference))
    return result


def native_gates(config: dict, output: Path, *, native=None, forward=None) -> dict:
    """Authenticate inputs, run independent native parity, then interior FD.

    A failure is persisted and stops this phase. No occupancy optimization is
    invoked here, and a C/Z gradient gate cannot certify G's absence semantics.
    """
    from .native import load_frozen_tiny

    if config.get("element_boundary") != "shared-fp32-element-boundary-v1":
        raise ValueError("Require declared common FP32 element boundary")
    if config.get("native_repeat_policy") != NATIVE_REPEAT_POLICY:
        raise ValueError("Require declared native-repeat acceptance policy")
    if config.get("output_parity_policy") != OUTPUT_PARITY_POLICY:
        raise ValueError("Require declared native-output acceptance policy")
    if config.get("finite_difference_policy") != FD_POLICY:
        raise ValueError("Require exact declared attached-repeat finite-difference policy")
    if not output.is_dir() or any(output.iterdir()):
        raise ValueError("Use a fresh empty output directory")
    if not torch.cuda.is_available():
        raise RuntimeError("Native phase requires an allocated CUDA GPU")
    if not __import__("os").environ.get("LSB_JOBID"):
        raise RuntimeError("Numerical execution requires an allocated DTU LSF job")
    torch.set_num_threads(int(config.get("cpu_threads", 4)))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    started = time.monotonic()
    result = dict(version=PHASE, status="running", stage="loading",
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(), cases=[],
        model_calls=0, G="not_implemented_not_validated", optimization="not_run",
        scope="C/Z fixed native-geometry mechanism gates; no variable-count pass",
        model_precision="float32_no_TF32",scalar_reduction_precision="float64")
    if (native is None) != (forward is None):
        raise ValueError("Supply both resident model and shared forward authority, or neither")
    if native is None:
        native = load_frozen_tiny(source_root=config["source_root"],
            source_sha256=config["source_sha256"], weights_path=config["weights_path"],
            weights_sha256=config["weights_sha256"], device="cuda", seed=17)
        forward = CountedForward(native, int(config.get("max_model_calls", 3000)),
            started+float(config.get("internal_seconds", 19800)))
    elif (forward.native is not native or forward.calls != 0
            or forward.budget != int(config["max_model_calls"])
            or forward.deadline > started+float(config["internal_seconds"])
            or native.source_root != Path(config["source_root"]).resolve()
            or native.weights_path != Path(config["weights_path"]).resolve()
            or native.provenance["source_sha256"] != config["source_sha256"]
            or native.provenance["weights_sha256"] != config["weights_sha256"]):
        raise ValueError("Shared native readiness authority differs from authenticated configuration")
    result["native_provenance"] = native.provenance
    result["native_repeat_policy"] = NATIVE_REPEAT_POLICY
    result["output_parity_policy"] = OUTPUT_PARITY_POLICY
    result["finite_difference_policy"] = FD_POLICY
    state = native.snapshot_state()
    try:
        for item in config["cases"]:
            path = Path(item["bundle_path"])
            if file_sha256(path) != item["bundle_sha256"]:
                raise ValueError("Sealed case bundle hash mismatch")
            bundle = torch.load(path, map_location="cpu", weights_only=True)
            if bundle.get("schema_version") != 1 or bundle.get("case_id") != item["case_id"]:
                raise ValueError("Wrong prepared case schema/identity")
            features = native.prepare_features(tree_to(bundle["features"], "cuda"))
            context_digest = tensor_tree_sha256(features)
            pairs = bundle["pair_indices"].to(device="cuda", dtype=torch.long)
            targets = bundle["target_bins"].to(device="cuda", dtype=torch.long)
            optional = int(bundle["optional_atom"])
            optional_token = int(bundle["optional_token"])
            if (not 0 <= optional < features["ref_element"].shape[0] or
                    not bool(features["is_ligand"][optional]) or
                    int(features["atom_to_token_idx"][optional]) != optional_token):
                raise ValueError("Optional atom must be a native ligand atom mapped to its declared token")
            if bool((pairs == optional_token).any()):
                raise ValueError("Optional atom pairs cannot enter the fixed retained-core readout")
            atom_tokens = features["atom_to_token_idx"].long()
            token_count = features["token_bonds"].shape[0]
            ligand_tokens = torch.zeros(token_count,device="cuda",dtype=torch.bool)
            protein_tokens = torch.zeros_like(ligand_tokens)
            ligand_tokens[atom_tokens[features["is_ligand"].bool()]] = True
            protein_tokens[atom_tokens[features["is_protein"].bool()]] = True
            if not bool((ligand_tokens[pairs[:,0]] & protein_tokens[pairs[:,1]] |
                         protein_tokens[pairs[:,0]] & ligand_tokens[pairs[:,1]]).all()):
                raise ValueError("Fixed readout must select ligand-core/protein pairs only")
            record = dict(case_id=item["case_id"], role=item["role"],
                bundle_sha256=item["bundle_sha256"], context_sha256=context_digest,
                metadata=bundle["metadata"], parity={}, derivatives=[], sweep=[])
            result["cases"].append(record)
            result["stage"] = item["case_id"]+":native_parity"
            with torch.no_grad():
                raw = forward(features, attached=False)
                reference = selected_outputs(raw, pairs, targets)
                repeated = selected_outputs(forward(features, attached=False), pairs, targets)
            record["repeat"] = compare_native_repeat(repeated, reference)
            if not record["repeat"]["passed"]:
                raise RuntimeError("Native repeat variation exceeds prospective tolerance")
            float_features = floating_element_features(features)
            with torch.no_grad():
                float_native = selected_outputs(forward(float_features, attached=False), pairs, targets)
            record["native_float32_boundary"] = compare_parity_outputs(float_native, reference)
            record["element_boundary"] = dict(version="shared-fp32-element-boundary-v1",
                native_dtype=str(features["ref_element"].dtype), adapted_dtype="torch.float32",
                native_context_unchanged=True)
            del float_features, float_native
            if item["role"] == "development":
                with torch.no_grad():
                    legacy = selected_outputs(forward(features, attached=True, legacy=True), pairs, targets)
                record["legacy_manual_q1"] = compare_outputs(legacy, reference)
            saved = {"native_full":{k:v.detach().cpu() for k,v in reference.items()}}
            for condition in ("C", "Z"):
                q = torch.tensor(1., device="cuda")
                adapted = inject_occupancy(features, optional, q, condition)
                with torch.no_grad():
                    actual = selected_outputs(forward(adapted, attached=True), pairs, targets)
                record["parity"][condition] = compare_parity_outputs(actual, reference)
                saved[condition+"_q1"] = {k:v.detach().cpu() for k,v in actual.items()}
            write_json(output/(item["case_id"]+"-occupied-parity.json"), record)
            torch.save(saved, output/(item["case_id"]+"-occupied-parity-outputs.pt"))
            # Collect forensic evidence before any later probe can fail.
            if item["role"] == "development":
                record["path_diagnostics"] = trace_path_diagnostics(native, features, pairs, targets, forward)
                write_json(output/(item["case_id"]+"-path-diagnostics.json"), record["path_diagnostics"])
            if (not record["native_float32_boundary"]["passed"]
                    or not all(row["passed"] for row in record["parity"].values())):
                raise RuntimeError("Occupied native parity gate failed")
            # Test the actual gradient-enabled dispatch at the same occupied input,
            # before interior derivative checks. Parameters remain frozen/eval.
            result["stage"] = item["case_id"]+":gradient_enabled_parity"
            graph_features = floating_element_features(features, requires_grad=True)
            with torch.enable_grad():
                graph_actual = selected_outputs(forward(graph_features, attached=True), pairs, targets)
            record["gradient_enabled_q1"] = compare_parity_outputs(graph_actual, reference)
            saved["gradient_enabled_q1"] = {k:v.detach().cpu() for k,v in graph_actual.items()}
            del graph_actual, graph_features
            torch.save(saved, output/(item["case_id"]+"-parity-outputs.pt"))
            write_json(output/(item["case_id"]+"-parity.json"), record)
            if (not all(row["passed"] for row in record["parity"].values())
                    or not record["gradient_enabled_q1"]["passed"]):
                raise RuntimeError("Occupied native parity gate failed")
            del raw, reference, repeated, actual, saved
            result["stage"] = item["case_id"]+":derivatives"
            for condition in ("C", "Z"):
                control_hashes = set()
                def scalar(variable, parameterization):
                    occupancy = (-variable).sigmoid() if parameterization == "no_atom_logit" else variable
                    adapted = inject_occupancy(features, optional, occupancy, condition)
                    if condition == "C":
                        control_hashes.add(tensor_tree_sha256(adapted))
                        if len(control_hashes) != 1:
                            raise RuntimeError("Clamped C input changed across occupancy/FD trials")
                    value = reduce_distogram(forward(adapted, attached=True), pairs, targets)
                    # Explicit analytical zero for the q-independent C control.
                    return value + variable*0 if condition == "C" else value
                for q_value in (.1, .5, .9):
                    for parameterization in ("q", "no_atom_logit"):
                        initial = q_value if parameterization == "q" else math.log((1-q_value)/q_value)
                        variable = torch.tensor(initial, device="cuda", requires_grad=True)
                        value = scalar(variable, parameterization)
                        gradient = float(torch.autograd.grad(value, variable)[0].detach().cpu())
                        loss_value = float(value.detach().cpu())
                        if not math.isfinite(loss_value) or not math.isfinite(gradient):
                            raise RuntimeError("Nonfinite derivative baseline/gradient")
                        row = dict(condition=condition, q=q_value, parameterization=parameterization,
                            loss=loss_value, autograd=gradient, finite_differences=[])
                        del value
                        # Calibrate the actual attached dispatch at this exact
                        # baseline, not an earlier native q=1 proxy. Include the
                        # grad-enabled baseline to expose dispatch differences.
                        with torch.no_grad():
                            repeats = [float(scalar(variable.detach(), parameterization).cpu())
                                for _ in range(FD_POLICY["baseline_repeats"])]
                        groups = [[loss_value]+repeats]
                        observations = []
                        for epsilon in FD_STEPS:
                            with torch.no_grad():
                                plus, minus = [], []
                                # Interleave signs to expose order-dependent
                                # jitter without changing fixed RNG or inputs.
                                for _ in range(FD_POLICY["endpoint_repeats"]):
                                    plus.append(float(scalar(variable.detach()+epsilon, parameterization).cpu()))
                                    minus.append(float(scalar(variable.detach()-epsilon, parameterization).cpu()))
                            groups.extend([plus, minus])
                            observations.append((epsilon, plus, minus))
                        row["noise_calibration"] = calibrate_loss_noise(groups)
                        for epsilon, plus, minus in observations:
                            comparison = loss_difference_agreement(gradient, plus, minus,
                                epsilon, row["noise_calibration"])
                            row["finite_differences"].append(comparison)
                        row["control_input_invariance"] = (
                            dict(passed=len(control_hashes)==1, sha256=next(iter(control_hashes)))
                            if condition == "C" else None)
                        minimum = FD_POLICY["minimum_agreeing_scales"]
                        row["passed"] = (row["noise_calibration"]["passed"]
                            and sum(x["passed"] for x in row["finite_differences"]) >= minimum
                            and (condition != "C" or gradient == 0.))
                        row["sensitivity"] = "resolved_nonzero" if gradient != 0 and sum(x["resolved_above_repeat_noise"] for x in row["finite_differences"]) >= minimum else "zero_or_unresolved"
                        record["derivatives"].append(row)
                        with (output/"derivatives.jsonl").open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(dict(case_id=item["case_id"], **row),allow_nan=False)+"\n")
                        if not row["passed"]:
                            raise RuntimeError("Interior derivative gate failed; later optimization forbidden")
            record["Z_resolved_sensitivity"] = {
                parameterization:any(r["condition"] == "Z" and
                    r["parameterization"] == parameterization and
                    r["sensitivity"] == "resolved_nonzero" for r in record["derivatives"])
                for parameterization in ("q","no_atom_logit")}
            if not all(record["Z_resolved_sensitivity"].values()):
                raise RuntimeError("Z sensitivity is zero/unresolved; no useful occupancy-gradient pass")
            for condition in ("C", "Z"):
                for q_value in Q_GRID:
                    with torch.no_grad():
                        q = torch.tensor(q_value, device="cuda")
                        logits = forward(inject_occupancy(features,optional,q,condition),attached=True)
                        selected = selected_outputs(logits,pairs,targets)
                        loss = float(selected["loss"].cpu())
                        probabilities = selected["probabilities"]
                        entropy = float((-(probabilities*probabilities.clamp_min(1e-30).log()).sum(-1).mean()).cpu())
                    if not math.isfinite(loss):
                        raise RuntimeError("Nonfinite sweep loss")
                    record["sweep"].append(dict(condition=condition,q=q_value,
                        loss=loss,distogram_entropy=entropy,
                        proposed_graph_expected_size=int(bundle["metadata"]["ligand_heavy_atoms"])-1+q_value,
                        proposed_graph_hard_count=int(bundle["metadata"]["ligand_heavy_atoms"])-1+int(q_value>.5),
                        scorer_native_ligand_rows_fixed=int(bundle["metadata"]["ligand_heavy_atoms"]),
                        occupancy_semantics="clamped_C" if condition=="C" else "element_mass_only_Z"))
                    torch.save({k:v.detach().cpu() for k,v in selected.items()},
                        output/(item["case_id"]+f"-{condition}-q{q_value:g}.pt"))
            if tensor_tree_sha256(features) != context_digest:
                raise RuntimeError("Bound native feature context changed")
            native.assert_unchanged(state)
            record["status"] = "native_parity_and_derivatives_passed"
            record["absence_semantics"] = "not_validated_C_Z_are_clamped_controls"
            write_json(output/(item["case_id"]+"-complete.json"),record)
        result.update(status="native_parity_and_gradient_passed",
            stage="complete_C_Z_only", absent_endpoint="not_run", ghost_invariance="not_run")
        return result
    finally:
        result["model_calls"] = forward.calls
        result["elapsed_seconds"] = time.monotonic()-started
        result["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated()
        if result["status"] == "running":
            result.update(status="failed", traceback=traceback.format_exc())
        write_json(output/"native-gate-result.json",result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--config-sha256",required=True)
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    if file_sha256(args.config) != args.config_sha256:
        raise ValueError("Execution config SHA256 differs from sealed plan")
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    config = json.loads(Path(args.config).read_text())
    try:
        result = native_gates(config,output)
    except BaseException as exc:
        if not (output/"native-gate-result.json").exists():
            write_json(output/"native-gate-result.json",dict(version=PHASE,status="failed",
                stage="startup",error_type=type(exc).__name__,error=str(exc),
                traceback=traceback.format_exc(),optimization="not_run"))
        raise
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == "__main__":
    main()
