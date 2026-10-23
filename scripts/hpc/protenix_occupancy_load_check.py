"""Allocated CPU checkpoint loading and development-case feature preflight.

The strict-load receipt remains separate from the feature probe.  Neither CPU
receipt certifies CUDA native parity or the NO_ATOM derivative hypothesis.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

from latentmol.protenix_occupancy.native import load_frozen_tiny


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def feature_probe(native, args, plan):
    """Use the already loaded frozen model for six actual CPU forwards."""
    started = time.monotonic()
    report = dict(version="protenix-cpu-native-feature-probe-v1", status="failed",
        jobid=os.environ["LSB_JOBID"], device="cpu", model_forward=True,
        native_validation=False, cuda_validation=False, optimization="not_run",
        no_atom_gradient_validation=False, plan_sha256=args.plan_sha256,
        prepared_manifest_sha256=args.prepared_manifest_sha256,
        cpu_threads=4, model_call_budget=20, internal_seconds=600,
        model_calls=0, parity={}, context_unchanged=None, model_unchanged=None)
    forward = None
    features = None
    context_digest = None
    try:
        import torch
        from latentmol.multimodel.external import file_sha256, tensor_tree_sha256
        from latentmol.protenix_occupancy.study import (
        CountedForward, compare_parity_outputs, compare_native_repeat,
        NATIVE_REPEAT_POLICY, OUTPUT_PARITY_POLICY, floating_element_features,
            inject_occupancy, selected_outputs,
        )
        state = native.snapshot_state()
        if plan.get("native_repeat_policy") != NATIVE_REPEAT_POLICY:
            raise ValueError("CPU probe native-repeat policy differs from sealed plan")
        report["native_repeat_policy"] = NATIVE_REPEAT_POLICY
        if plan.get("output_parity_policy") != OUTPUT_PARITY_POLICY:
            raise ValueError("CPU output-parity policy differs from sealed plan")
        report["output_parity_policy"] = OUTPUT_PARITY_POLICY
        manifest_raw = Path(args.prepared_manifest).read_bytes()
        if hashlib.sha256(manifest_raw).hexdigest() != args.prepared_manifest_sha256:
            raise ValueError("Prepared manifest checksum differs")
        manifest = json.loads(manifest_raw)
        cases = manifest.get("cases", [])
        if (manifest.get("preparation_version") != "protenix-occupancy-native-case-preparation-v1"
                or manifest.get("status") != "prepared_not_native_validated"
                or manifest.get("all_three_inputs_supported") is not True
                or [item.get("role") for item in cases] !=
                    ["development", "confirmation_1", "confirmation_2"]):
            raise ValueError("Wrong completed prepared-manifest schema or case order")
        case = cases[0]
        path = Path(case["bundle_path"])
        if file_sha256(path) != case["bundle_sha256"]:
            raise ValueError("Development case bundle checksum differs")
        bundle = torch.load(path, map_location="cpu", weights_only=True)
        if (bundle.get("schema_version") != 1
                or bundle.get("case_id") != case["case_id"]
                or bundle.get("metadata", {}).get("role") != "development"
                or bundle["metadata"].get("native_source_sha256") != plan["source_sha256"]):
            raise ValueError("Development case schema, identity or source differs")
        raw_features = bundle["features"]
        if tensor_tree_sha256(raw_features) != bundle["metadata"]["native_feature_sha256"]:
            raise ValueError("Prepared development native-feature digest differs")
        element = raw_features["ref_element"]
        if (element.device.type != "cpu" or element.is_floating_point()
                or element.ndim != 2 or element.shape[1] != 128):
            raise ValueError("CPU preflight requires untouched native integer element features")
        features = native.prepare_features(raw_features)
        context_digest = tensor_tree_sha256(features)
        pairs, targets = bundle["pair_indices"], bundle["target_bins"]
        optional, optional_token = bundle["optional_atom"], bundle["optional_token"]
        if (isinstance(optional, bool) or not isinstance(optional, int)
                or not 0 <= optional < features["ref_element"].shape[0]
                or not bool(features["is_ligand"][optional])
                or int(features["atom_to_token_idx"][optional]) != optional_token
                or bool((pairs == optional_token).any())):
            raise ValueError("Optional ligand mapping or retained readout differs")
        report.update(case_id=case["case_id"], role=case["role"],
            bundle_sha256=case["bundle_sha256"], context_sha256=context_digest,
            native_element_dtype=str(features["ref_element"].dtype),
            native_element_shape=list(features["ref_element"].shape),
            floating_element_dtype="torch.float32")
        forward = CountedForward(native, 20, started + 600)
        with torch.no_grad():
            reference = selected_outputs(forward(features, attached=False), pairs, targets)
            repeated = selected_outputs(forward(features, attached=False), pairs, targets)
            report["parity"]["native_repeat"] = compare_native_repeat(repeated, reference)
            floating = floating_element_features(features)
            actual = selected_outputs(forward(floating, attached=False), pairs, targets)
            report["parity"]["native_floating_boundary"] = compare_parity_outputs(actual, reference)
            q = torch.tensor(1., dtype=torch.float32, device="cpu")
            for condition in ("C", "Z"):
                adapted = inject_occupancy(features, optional, q, condition)
                actual = selected_outputs(forward(adapted, attached=True), pairs, targets)
                report["parity"][condition + "_q1"] = compare_parity_outputs(actual, reference)
        graph_features = floating_element_features(features, requires_grad=True)
        leaf = graph_features["ref_element"]
        if not leaf.is_leaf or leaf.dtype != torch.float32 or not leaf.is_contiguous():
            raise ValueError("Gradient probe boundary must be a contiguous FP32 leaf")
        with torch.enable_grad():
            actual = selected_outputs(forward(graph_features, attached=True), pairs, targets)
            report["parity"]["gradient_enabled_q1"] = compare_parity_outputs(actual, reference)
            gradient = torch.autograd.grad(actual["loss"], leaf)[0]
        gradient_finite = bool(torch.isfinite(gradient).all())
        report["element_leaf_gradient"] = dict(finite=gradient_finite,
            nonzero=bool((gradient != 0).any()),
            norm=float(gradient.double().norm()) if gradient_finite else None,
            interpretation="FP32 element-leaf autograd smoke check; not a NO_ATOM gradient pass")
        report["context_unchanged"] = tensor_tree_sha256(features) == context_digest
        if not report["context_unchanged"]:
            raise RuntimeError("CPU native-feature context changed")
        native.assert_unchanged(state)
        report["model_unchanged"] = True
        if not gradient_finite:
            raise RuntimeError("CPU element-leaf gradient is nonfinite")
        if not all(item["passed"] for item in report["parity"].values()):
            raise RuntimeError("CPU native-feature parity probe failed")
        if forward.calls != 6:
            raise RuntimeError("CPU native-feature probe must make exactly six model calls")
        if time.monotonic() >= started + 600:
            raise RuntimeError("CPU native-feature probe exceeded its internal time budget")
        report["status"] = "cpu_native_feature_probe_passed_not_cuda_validated"
    except BaseException as exc:
        report.update(error=str(exc), error_type=type(exc).__name__,
            traceback=traceback.format_exc())
        raise
    finally:
        report["model_calls"] = forward.calls if forward is not None else 0
        report["elapsed_seconds"] = time.monotonic() - started
        _write_json(args.feature_probe_output, report)
    print(json.dumps(report, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepared-manifest", required=True)
    parser.add_argument("--prepared-manifest-sha256", required=True)
    parser.add_argument("--feature-probe-output", required=True)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Numerical checkpoint checks require allocated DTU HPC")
    raw = Path(args.plan).read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.plan_sha256:
        raise ValueError("Sealed plan checksum differs")
    plan = json.loads(raw)
    import torch
    torch.set_num_threads(4)
    started = time.monotonic()
    result = dict(status="failed", model_forward=False, native_validation=False,
        optimization="not_run", jobid=os.environ["LSB_JOBID"],
        plan_sha256=args.plan_sha256)
    try:
        native = load_frozen_tiny(source_root=plan["source_root"],
            source_sha256=plan["source_sha256"], weights_path=plan["weights_path"],
            weights_sha256=plan["weights_sha256"], device="cpu", seed=17)
        before = native.snapshot_state()
        native.assert_unchanged(before)
        result.update(status="strict_checkpoint_load_passed_not_native_validated",
            provenance=native.provenance, state=before)
    except BaseException as exc:
        result.update(error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        _write_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "state"}))
    # Preserve the completed strict-load receipt before any native forward.
    feature_probe(native, args, plan)


if __name__ == "__main__":
    main()
