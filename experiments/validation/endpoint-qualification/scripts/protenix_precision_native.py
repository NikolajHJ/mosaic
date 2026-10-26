"""Separate-process precision reference at the immutable Protenix20k endpoint.

The FP32 production process exits before an isolated FP64 reference is loaded.
No optimizer, saved-state writer, diffusion or continuation is reachable here.
Only a newly versioned prospective measurement can pass; original failures stay.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

VERSION = "protenix-endpoint-precision-plan-v2"
BASE = Path("/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01")
ROOT = BASE / "protenix-endpoint-fix-03"
SOURCE = BASE / "protenix-endpoint-fix-release-03/source"
FORENSICS = BASE / "protenix-endpoint-forensics-release-01/source"
FORENSICS_PIN = "d127d6abe87307a5d305edb6d04e3a32e3545c508e58b82ce45290e0f3fc03d8"
FORENSICS_PLAN_SHA = "14d0cf5b03af73ee5d87ece55b8c48616897521c50bbb40e8f76b6a9bb8f5861"
SCIENCE = BASE / "protenix-extended-release-03/source"
PYTHON = "/work3/s234818/mosaic/envs/protenix-tiny/bin/python"
CAP = 256 * 1024**2
PRIOR_REPORT = {"path": str(BASE / "protenix-endpoint-forensics-01/gpu/payload/forensics/report.json"),
                "bytes": 7849579, "sha256": "acbb5273e0c794d5719067210e27a8031717fb8a51a21dc0eb19988881ee8359"}
ORIGINAL_DIRECTIONS = {"X": [{"row": 3, "source": 1, "target": 2}], "E_upper": []}
SITE_PACKAGES = Path("/work3/s234818/mosaic/envs/protenix-tiny/lib/python3.12/site-packages")
RUNTIME_FILES = [
    {"path": str(SITE_PACKAGES / "jax/_src/nn/functions.py"), "bytes": 50244,
     "sha256": "12e8ef763d56f4cc92189101b0b01e1a458042f851274a5d3f4ad3042430536b"},
    {"path": str(SITE_PACKAGES / "protenix/protenij.py"), "bytes": 82950,
     "sha256": "37f337839034afd2a814c1ed6b76867b67918f73dd56cdd365e67b8fed882a6d"},
    {"path": str(SITE_PACKAGES / "protenix/backend.py"), "bytes": 11130,
     "sha256": "fb0bf601239eef711081d4e254f06208bcb36002d4e57a50b5a54135948fd12c"},
]


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))


def write(path, value):
    with Path(path).open("x", encoding="utf8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def binding(path):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and not path.is_symlink(), "Nonregular proof")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def original():
    require(sha(FORENSICS / "source_manifest.json") == FORENSICS_PIN, "Original forensic source changed")
    manifest = read(FORENSICS / "source_manifest.json")
    require(len(manifest["files"]) == 13, "Original forensic source scope changed")
    require({p.relative_to(FORENSICS).as_posix() for p in FORENSICS.rglob("*") if p.is_file()}
            == set(manifest["files"]) | {"source_manifest.json"}, "Unexpected forensic source member")
    for name, digest in manifest["files"].items():
        path = FORENSICS / name
        require(not path.is_symlink() and sha(path) == digest, "Original forensic member changed")
    return load(FORENSICS / "scripts/protenix_forensics_native.py", "precision_original_native")


def validate_plan(plan, output, startup=False):
    require(plan.get("version") == VERSION and plan.get("source_root") == str(SOURCE)
            and plan.get("control_root") == str(ROOT), "Wrong precision plan roots/version")
    for name, expected in (("original_results_changed", False), ("optimization_settings_changed", False),
                           ("can_certify_or_restart_original_study", False), ("monitor_remains_paused", True),
                           ("max_bytes", CAP), ("cap_bytes", CAP), ("metadata_cap_bytes", 16 * 1024**2),
                           ("internal_seconds", 5400)):
        require(plan.get(name) == expected and type(plan.get(name)) is type(expected), "Wrong scope: " + name)
    expected = ROOT / ("cpu/payload/native-startup" if startup else "gpu/payload")
    require(Path(output) == expected and plan.get("output") == str(ROOT / "gpu/payload"), "Wrong new output scope")
    old = plan.get("original_forensics_plan", {})
    require(old.get("path") == str(FORENSICS / "plan.json") and old.get("sha256") == FORENSICS_PLAN_SHA,
            "Original protected forensic plan differs")
    require(binding(old["path"]) == old, "Original protected forensic plan bytes differ")
    prior = read(old["path"])
    require(plan.get("arm") == prior.get("arm"), "Original optimization settings/endpoint differ")
    require(plan.get("prior_forensic_report") == PRIOR_REPORT
            and binding(PRIOR_REPORT["path"]) == PRIOR_REPORT, "Prior forensic report changed")
    require(plan.get("original_directions") == ORIGINAL_DIRECTIONS, "Authentic original probe changed")
    require(plan.get("runtime_files") == RUNTIME_FILES, "Exact native runtime bindings changed")
    return prior


def verify(plan, output, startup=False):
    prior = validate_plan(plan, output, startup)
    native = original()
    arm, hooks, failed, ready, before = native.verify_inputs(prior, prior["output"])
    before[str(FORENSICS / "plan.json")] = binding(FORENSICS / "plan.json")
    before[PRIOR_REPORT["path"]] = binding(PRIOR_REPORT["path"])
    for row in plan.get("runtime_files", []):
        require(binding(row["path"]) == row, "Precision runtime source changed")
        before[row["path"]] = row
    require(len(plan.get("runtime_files", [])) == 3, "Three authenticated runtime files required")
    return native, prior, arm, hooks, failed, ready, before


def local_module(name):
    expected = SOURCE / "scripts" / (name + ".py")
    value = __import__(name)
    require(Path(value.__file__).resolve() == expected.resolve(), "Precision policy/reference module origin changed: " + name)
    return value


def preserved(before):
    after = {path: binding(path) for path in before}
    require(after == before, "Protected original scientific files changed")
    return after


def worker_command(plan_path, plan_sha, output, worker):
    require(worker in ("fp32", "fp64"), "Unknown reference phase")
    return [PYTHON, str(SOURCE / "scripts/protenix_precision_native.py"), "--plan", str(plan_path),
            "--plan-sha256", plan_sha, "--output", str(output), "--worker", worker]


def audit_jaxpr(closed):
    """Reject any floating primitive input/output/constant below double precision."""
    counts, wrong, visited = Counter(), [], set()
    def aval(value, where):
        dtype = getattr(getattr(value, "aval", value), "dtype", None)
        if dtype is not None:
            text = str(dtype)
            counts[text] += 1
            if text.startswith(("float", "bfloat", "complex")) and text not in ("float64", "complex128"):
                wrong.append({"where": where, "dtype": text})
    def walk(value, where):
        if id(value) in visited:
            return
        # Current JAX Jaxpr objects expose ``jaxpr`` as a compatibility alias
        # returning themselves. Inspect an executable graph before unwrapping
        # ClosedJaxpr; otherwise cycle prevention skips every actual equation.
        if hasattr(value, "eqns") and all(hasattr(value, key) for key in ("constvars", "invars", "outvars")):
            visited.add(id(value))
            for const in getattr(value, "consts", ()):
                aval(const, where + ".const")
            for var in (*value.constvars, *value.invars, *value.outvars):
                aval(var, where + ".var")
            for i, eqn in enumerate(value.eqns):
                label = where + "." + str(i) + ":" + str(eqn.primitive)
                for var in (*eqn.invars, *eqn.outvars):
                    aval(var, label)
                walk(eqn.params, label)
        elif hasattr(value, "jaxpr"):
            visited.add(id(value))
            for const in getattr(value, "consts", ()):
                aval(const, where + ".const")
            walk(value.jaxpr, where + ".jaxpr")
        elif isinstance(value, dict):
            for key, item in value.items():
                walk(item, where + "." + str(key))
        elif isinstance(value, (tuple, list)):
            for i, item in enumerate(value):
                walk(item, where + "." + str(i))
    walk(closed, "forward")
    require(counts.get("float64", 0) > 0 and not wrong, "Reference native computation contains non-FP64 floating operations: " + repr(wrong[:8]))
    return {"status": "passed", "floating_operations_float64": True,
            "aval_counts": dict(counts), "narrow_float_avals": wrong, "nested_graphs": len(visited)}


def current_fp32_measurements(fields):
    """Qualify current execution; historical computed bits are observations."""
    import numpy as np
    require(all(v.dtype == np.float32 and np.isfinite(v).all() for v in fields.values()),
            "Production FP32 path changed/nonfinite")
    require(fields['primal'].shape == fields['ad_value'].shape == (3,), "Three independent baseline measurements required")
    scalar_spread = float(max(fields['primal'].max(), fields['ad_value'].max())
                          - min(fields['primal'].min(), fields['ad_value'].min()))
    require(scalar_spread <= 2e-5, "Current independent FP32 primal/AD parity or repeat stability failed")
    differences = {}
    for name in ('X', 'E_upper'):
        values = fields['gradient_' + name]
        require(values.shape == (3, *fields[name].shape), "Current gradient repeat shape changed")
        differences[name] = float(np.max(np.abs(values - values[0])))
        require(differences[name] <= 2e-5, "Current FP32 gradient repeats unstable")
        require(float(np.max(values.max(-1) - values.min(-1))) > 1e-7, "Current FP32 raw field has no nonzero contrast")
    return {'status':'passed_current_authenticated_FP32_baseline','scalar_spread':scalar_spread,
            'gradient_repeat_maximum_differences':differences,'historical_computed_bits_used_as_gate':False,
            'does_not_establish_derivative_correctness':True}


def _production(plan, output, lifecycle, protected):
    native, prior, arm, hooks, failed, ready, before = verify(plan, output)
    protected.update(before)
    directory = output / "production"
    directory.mkdir(exist_ok=False)
    evaluator, frozen = native.native_evaluator(prior, arm, hooks, ready, directory, lifecycle)
    import numpy as np
    with np.load(native.ENDPOINT, allow_pickle=False) as saved:
        values = {field: saved[field].copy() for field in ("X", "E_upper")}
    evaluations = [evaluator(values) for _ in range(3)]
    historical = read(plan["prior_forensic_report"]["path"])
    baseline_evaluations = historical["points"][0]["evaluations"]
    require(len(baseline_evaluations) == 3, "Prior FP32 baseline repetition count changed")
    comparison = []
    for actual, expected in zip(evaluations, baseline_evaluations):
        row = {'primal':float(actual['primal']), 'historical_primal':expected['primal'],
               'AD_scalar':float(actual['ad_value']), 'historical_AD_scalar':expected['ad_value'],
               'scalar_bits_equal':bool(float(actual['primal']) == expected['primal']
                                         and float(actual['ad_value']) == expected['ad_value']), 'gradients':{}}
        for field in ("X", "E_upper"):
            gradient = np.asarray(actual["gradients"][field])
            previous = expected["gradients"][field]
            row['gradients'][field] = {'dtype':str(gradient.dtype),'shape':list(gradient.shape),
                'sha256':hashlib.sha256(gradient.tobytes()).hexdigest(),'historical':previous,
                'exact_equal':bool(str(gradient.dtype) == previous['dtype'] and list(gradient.shape) == previous['shape']
                                  and hashlib.sha256(gradient.tobytes()).hexdigest() == previous['sha256'])}
        comparison.append(row)
    fields = {**values, "primal": np.asarray([e["primal"] for e in evaluations]),
              "ad_value": np.asarray([e["ad_value"] for e in evaluations])}
    for field in values:
        fields["gradient_" + field] = np.stack([e["gradients"][field] for e in evaluations])
    # Persist actual bytes and verbatim historical comparison before any current
    # measurement gate. A failure must not discard the values that caused it.
    with (directory / "production.npz").open("xb") as stream:
        np.savez(stream, **fields)
    write(directory/'historical-replay.json', {'status':'historical_observations_preserved',
        'rows':comparison,'can_certify_original_study':False,'current_derivative_gate_passed':False,
        'prior_forensic_report':plan['prior_forensic_report']})
    current = current_fp32_measurements(fields)
    write(directory/'current-baseline.json',current)
    require(frozen() == lifecycle["frozen_before"], "Production frozen identity changed")
    write(directory / "protected-after.json", preserved(before))
    write(directory / "assessment.json", {"status": "passed_fp32_measurements", "updates": 0,
          "original_frozen": frozen(), "data": binding(directory / "production.npz"),
          "production_x64_enabled": False, "separate_process": True,
          "baseline_scope": "fresh_authenticated_current_FP32_baseline_v2; historical comparison preserved separately",
          "current_baseline":binding(directory/'current-baseline.json'),
          "historical_observations":binding(directory/'historical-replay.json'),
          "prior_forensic_report": plan["prior_forensic_report"]})


def _reference(plan, output, lifecycle, protected):
    native, prior, arm, hooks, failed, ready, before = verify(plan, output)
    protected.update(before)
    directory = output / "reference"
    directory.mkdir(exist_ok=False)
    sys.path[:0] = [str(SCIENCE / "src"), str(SCIENCE / "scripts")]
    import jax
    import jax.numpy as jnp
    import numpy as np
    import equinox as eqx
    from mosaic.small_molecule.protenix import FrozenProtenixTiny
    from mosaic.small_molecule.features import bootstrap_native_assets
    from mosaic.small_molecule.io import load_context
    from mosaic.small_molecule.pocket import map_protenix_pocket
    from mosaic.small_molecule.losses import NativeContactLoss
    from mosaic.small_molecule.gradient_checks import array_digest
    from mosaic.small_molecule.optimizers import evaluate_graph, _nonfinite_paths
    from mosaic.small_molecule.state import RelaxedGraph
    precision = local_module("protenix_precision_reference")
    admission = local_module("protenix_precision_admission")
    require(jax.config.jax_enable_x64 and len(jax.devices("gpu")) == 1
            and "L40S" in jax.devices("gpu")[0].device_kind, "Fresh FP64 single L40S process required")
    jax.config.update("jax_default_matmul_precision", "highest")
    production_file = output / "production/production.npz"
    production_assessment = read(output / "production/assessment.json")
    require(production_assessment.get("status") == "passed_fp32_measurements"
            and production_assessment.get("data") == binding(production_file), "FP32 measurements not authenticated")
    with np.load(production_file, allow_pickle=False) as saved:
        fp32 = {name: saved[name].copy() for name in saved.files}
    with np.load(native.ENDPOINT, allow_pickle=False) as saved:
        for field in ("X", "E_upper"):
            np.testing.assert_array_equal(fp32[field], saved[field])
    bootstrap_native_assets(native.CACHE, include_model=True)
    with precision.native_precision_context() as precision_hooks:
        original_model = FrozenProtenixTiny.load(cache_dir=native.CACHE)
        lifecycle.update(model_frozen=lambda: {"model_digest": array_digest(original_model)},
                         model_before={"model_digest": native.MODEL_DIGEST})
        original_context = load_context(native.CONTEXT_READY.parent / "context", ready["context_files"])
        require(array_digest(original_model) == native.MODEL_DIGEST
                and array_digest(original_context) == native.CONTEXT_DIGEST, "Reference original bytes differ before lossless promotion")
        lifecycle.update(model_frozen=lambda: {"model_digest": array_digest(original_model),
                                              "context_digest": array_digest(original_context)},
                         model_before={"model_digest": native.MODEL_DIGEST, "context_digest": native.CONTEXT_DIGEST})
        model, model_audit = precision.promote_tree(original_model, label="model")
        context, context_audit = precision.promote_tree(original_context, label="context")
        require(lifecycle["model_frozen"]() == lifecycle["model_before"], "Promotion modified original frozen values")
        promoted_before = {"model": array_digest(model), "context": array_digest(context)}
        lifecycle.update(model_frozen=lambda: {"model": array_digest(model), "context": array_digest(context)},
                         model_before=promoted_before)
        del original_model, original_context
        write(directory / "promotion.json", {"model": model_audit, "context": context_audit})
        precision.floating_dtype_audit(model)
        precision.floating_dtype_audit(context)
        pocket = read(native.bound(arm["inputs"]["pocket"]))
        tokens = map_protenix_pocket(context, pocket)
        loss = model.build_loss(context=context, loss=NativeContactLoss(8.0, tokens))
        key = jax.random.key(17)
        gpu = jax.devices("gpu")[0]
        # Match the original atom CPU/bond GPU placement, without changing values.
        devices = {"X": jax.devices("cpu")[0], "E_upper": gpu}
        def graph(values):
            return RelaxedGraph(**{f: jax.device_put(jnp.asarray(values[f], dtype=jnp.float64), devices[f]) for f in ("X", "E_upper")})
        baseline = {f: fp32[f].astype(np.float64) for f in ("X", "E_upper")}
        tree = eqx.filter_make_jaxpr(lambda g: loss(g, key=key))(graph(baseline))[0]
        dtype_audit = audit_jaxpr(tree)
        write(directory / "native-dtype-audit.json", dtype_audit)
        identity_before = {"model": array_digest(model), "context": array_digest(context)}
        def frozen():
            result = {"model": array_digest(model), "context": array_digest(context)}
            require(result == identity_before, "Promoted reference model/context changed")
            return result
        lifecycle.update(frozen=frozen, frozen_before=identity_before)
        cache, calls = {}, []
        def digest(values):
            return hashlib.sha256(b"".join(np.asarray(values[f], dtype=np.float64).tobytes() for f in ("X", "E_upper"))).hexdigest()
        def compute(values):
            require(native.output_bytes(output) < CAP - 8 * 1024**2, "Reference output cap exhausted")
            point = graph(values)
            primal, aux = loss(point, key=key)
            raw = evaluate_graph(loss, point, key)
            require(not _nonfinite_paths((primal, aux, raw)), "Nonfinite native FP64 reference")
            row = {"primal": np.asarray(primal), "ad_value": np.asarray(raw.value),
                   "gradients": {f: np.asarray(getattr(raw.gradients, f)) for f in ("X", "E_upper")}}
            require(row["primal"].dtype == np.float64 and all(v.dtype == np.float64 for v in row["gradients"].values()), "Reference path not FP64")
            number = len(calls)
            path = directory / ("point-" + str(number).zfill(4) + ".npz")
            with path.open("xb") as stream:
                np.savez(stream, **{f: np.asarray(values[f], dtype=np.float64) for f in ("X", "E_upper")},
                         primal=row["primal"], ad_value=row["ad_value"],
                         **{"gradient_" + f: v for f, v in row["gradients"].items()})
            calls.append({"point_digest": digest(values), "raw": binding(path)})
            return row
        basekey = digest(baseline)
        cache[basekey] = [compute(baseline) for _ in range(3)]
        cursor = Counter()
        def ref(values, kind):
            index = digest(values)
            if index not in cache:
                cache[index] = [compute(values)]
            rows = cache[index]
            n = cursor[(index, kind)]
            cursor[(index, kind)] += 1
            row = rows[min(n, len(rows) - 1)]
            return row["primal"] if kind == "primal" else (row["ad_value"], row["gradients"])
        def prod(values, kind):
            for f in ("X", "E_upper"):
                np.testing.assert_array_equal(values[f], fp32[f])
            n = cursor[("production", kind)]
            cursor[("production", kind)] += 1
            i = n % 3
            return fp32["primal"][i] if kind == "primal" else (fp32["ad_value"][i], {f: fp32["gradient_" + f][i] for f in ("X", "E_upper")})
        metadata = {"full_model_float64": True, "parameters_float64": True, "native_features_float64": True,
                    "native_operations_float64": True, "final_reduction_only": False,
                    "dtype_audit": binding(directory / "native-dtype-audit.json"),
                    "promotion": binding(directory / "promotion.json"), "validation_only": True}
        directions = plan["original_directions"]
        def run_policy(label, factor=1.0):
            cursor.clear()
            def ad32(values):
                value, gradients = prod(values, "ad")
                return value, {f: (v * np.float32(factor)).astype(np.float32) for f, v in gradients.items()}
            return admission.evaluate({f: fp32[f] for f in ("X", "E_upper")},
                lambda v: prod(v, "primal"), ad32, lambda v: ref(v, "primal"), lambda v: ref(v, "ad"), frozen,
                original_directions=directions, reference_precision=metadata,
                original_observations=plan["prior_forensic_report"],
                seal_callback=lambda sealed: write(directory / (label + "-selection.json"), sealed))
        result = run_policy("native")
        write(directory / "admission.json", result)
        negatives = []
        for label, factor in (("zero", 0.0), ("reversed", -1.0), ("scaled", 1.5)):
            count = len(calls)
            control = run_policy(label, factor)
            write(directory / (label + "-control.json"), control)
            require(control.get("can_admit_new_child") is False and len(calls) == count,
                    "Cached-native wrong production gradient was admitted or caused unsealed model probes")
            negatives.append({"label": label, "status": "rejected", "model_evaluations_added": 0})
        write(directory / "oracle-points.json", {"calls": calls, "unique_points": len(cache), "negative_controls": negatives,
              "frozen_before": identity_before, "frozen_after": frozen(), "attention_and_mask_hooks": precision_hooks})
        require(result.get("can_admit_new_child") is True, "New prospective FP64 endpoint measurement did not pass")
    write(directory / "protected-after.json", preserved(before))
    write(directory / "assessment.json", {"status": "passed_new_prospective_precision_measurement", "updates": 0,
          "admission": binding(directory / "admission.json"), "oracle": binding(directory / "oracle-points.json"),
          "can_certify_or_restart_original_study": False, "original_results_changed": False})


def output_bytes(directory):
    total = 0
    for path in Path(directory).rglob("*"):
        require(not path.is_symlink() and (path.is_dir() or path.is_file()), "Special/linked precision output")
        if path.is_file(): total += path.stat().st_size
    return total


def worker_preservation(output, worker, before, lifecycle, primary=None):
    """Try each preservation proof independently; never mask a native failure."""
    result = {"worker": worker, "status": "failed", "updates": 0,
              "original_results_changed": False, "optimization_settings_changed": False,
              "can_certify_or_restart_original_study": False,
              "primary_error": repr(primary) if primary is not None else None,
              "preservation_errors": []}
    try:
        if before:
            result["protected_after"] = preserved(before)
            result["protected_inputs_unchanged"] = True
        else:
            result["protected_inputs_unchanged"] = "not_established_before_model"
    except BaseException as error:
        result["protected_inputs_unchanged"] = False
        result["preservation_errors"].append("protected: " + repr(error))
    try:
        if lifecycle.get("frozen") is not None:
            after = lifecycle["frozen"]()
            require(after == lifecycle["frozen_before"], "Full frozen model/context changed")
            result.update(frozen_unchanged=True, frozen_after=after)
        elif lifecycle.get("model_frozen") is not None:
            after = lifecycle["model_frozen"]()
            require(after == lifecycle["model_before"], "Loaded frozen model changed")
            result.update(frozen_unchanged="full_context_not_available", loaded_model_unchanged=True,
                          loaded_model_after=after)
        else:
            result["frozen_unchanged"] = "model_not_available"
    except BaseException as error:
        result["frozen_unchanged"] = False
        result["preservation_errors"].append("frozen: " + repr(error))
    try:
        result["persistent_bytes_before_proof"] = output_bytes(output)
        require(result["persistent_bytes_before_proof"] < CAP - 65536, "Precision output cap exhausted")
    except BaseException as error:
        result["preservation_errors"].append("budget: " + repr(error))
    if primary is not None:
        chain, seen, cursor = [], set(), primary
        while cursor is not None and id(cursor) not in seen:
            seen.add(id(cursor)); chain.append(repr(cursor)); cursor = cursor.__cause__ or cursor.__context__
        result["exception_chain"] = chain
    passed = (primary is None and not result["preservation_errors"]
              and result.get("protected_inputs_unchanged") is True and result.get("frozen_unchanged") is True)
    result["status"] = "passed" if passed else "failed_native_or_preservation"
    write(output / (worker + "-final-preservation.json"), result)
    return result


def run_worker(plan, output, worker):
    require(worker in ("fp32", "fp64"), "Unknown isolated worker")
    lifecycle, before, primary = {}, {}, None
    try:
        (_production if worker == "fp32" else _reference)(plan, output, lifecycle, before)
    except BaseException as error:
        primary = error
    final = worker_preservation(output, worker, before, lifecycle, primary)
    if primary is not None:
        raise primary
    require(final["status"] == "passed", "Worker preservation failed: " + repr(final["preservation_errors"]))
    return final


def production(plan, output):
    return run_worker(plan, output, "fp32")


def reference(plan, output):
    return run_worker(plan, output, "fp64")


def verified_worker(output, worker):
    """Returncodes alone do not establish a native precision result."""
    directory = output / ("production" if worker == "fp32" else "reference")
    assessment = read(directory / "assessment.json")
    expected = "passed_fp32_measurements" if worker == "fp32" else "passed_new_prospective_precision_measurement"
    require(assessment.get("status") == expected and type(assessment.get("updates")) is int
            and assessment["updates"] == 0, "Isolated worker assessment differs")
    proof = read(output / (worker + "-final-preservation.json"))
    require(proof.get("status") == "passed" and proof.get("worker") == worker
            and proof.get("primary_error") is None and proof.get("preservation_errors") == []
            and proof.get("protected_inputs_unchanged") is True and proof.get("frozen_unchanged") is True,
            "Isolated worker frozen/input preservation not established")
    for key in (("data",) if worker == "fp32" else ("admission", "oracle")):
        require(binding(assessment[key]["path"]) == assessment[key], "Worker raw evidence binding changed")
    if worker == "fp32":
        require(assessment.get("production_x64_enabled") is False and assessment.get("separate_process") is True,
                "Production FP32 reference execution changed")
        for key in ('current_baseline','historical_observations'):
            require(binding(assessment[key]['path']) == assessment[key], 'Current/historical baseline proof differs')
        require(read(assessment['current_baseline']['path'])['status']=='passed_current_authenticated_FP32_baseline',
                'Current FP32 baseline was not qualified')
    else:
        require(assessment.get("original_results_changed") is False
                and assessment.get("can_certify_or_restart_original_study") is False,
                "Reference scope cannot recertify saved original")
        admission = read(assessment["admission"]["path"])
        require(admission.get("status") == "passed" and admission.get("can_admit_new_child") is True
                and admission.get("frozen_unchanged") is True, "Native prospective precision admission failed")
    return {"assessment": binding(directory / "assessment.json"),
            "preservation": binding(output / (worker + "-final-preservation.json"))}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--plan-sha256", required=True)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--startup-only", action="store_true")
    p.add_argument("--worker", choices=("fp32", "fp64"))
    args = p.parse_args(argv)
    require(sha(args.plan) == args.plan_sha256, "Own precision plan bytes changed")
    plan = read(args.plan)
    prior = validate_plan(plan, args.output, args.startup_only)
    require(str(os.environ.get("LSB_JOBID", "")).isdigit(), "Allocated execution required")
    if args.startup_only:
        require(args.worker is None and os.environ.get("LSB_QUEUE") == "hpc", "CPU metadata startup only")
        verify(plan, args.output, True)
        args.output.mkdir(exist_ok=False)
        value = {"status": "passed_metadata_only", "numerical_setup_entered": False, "updates": 0,
                 "plan_sha256": args.plan_sha256, "job_id": os.environ["LSB_JOBID"]}
        write(args.output / "endpoint-startup.json", value)
        return value
    native = original()
    native.allocated_environment(os.environ)
    if args.worker:
        require(args.output.is_dir() and not args.output.is_symlink(), "Coordinator output absent")
        return run_worker(plan, args.output, args.worker)
    require(not args.output.exists(), "Existing/unknown precision output; never repeat")
    args.output.mkdir(exist_ok=False)
    before = verify(plan, args.output)[-1]
    before[str(args.plan)] = binding(args.plan)
    write(args.output / "protected-before.json", before)
    assessment = {"status": "failed", "updates": 0, "job_id": os.environ["LSB_JOBID"],
        "plan": binding(args.plan), "original_results_changed": False, "optimization_settings_changed": False,
        "can_certify_or_restart_original_study": False, "production_optimizer_executed": False}
    started = time.monotonic()
    error = None
    worker_proofs = {}
    preservation_errors = []
    try:
        for worker in ("fp32", "fp64"):
            command = worker_command(args.plan, args.plan_sha256, args.output, worker)
            env = dict(os.environ, JAX_ENABLE_X64="0" if worker == "fp32" else "1")
            write(args.output / (worker + "-invocation.json"), {"argv": command, "JAX_ENABLE_X64": env["JAX_ENABLE_X64"],
                  "previous_worker_exited_before_start": True, "updates": 0})
            timeout = 5400 - (time.monotonic() - started)
            require(timeout > 0, "Shared reference runtime exhausted before next isolated worker")
            run = subprocess.run(command, env=env, timeout=timeout)
            require(run.returncode == 0, "Isolated " + worker + " reference stage failed")
            worker_proofs[worker] = verified_worker(args.output, worker)
        assessment.update(status="prospective_precision_gate_passed_original_failures_preserved",
                          reference_assessment=binding(args.output / "reference/assessment.json"),
                          worker_proofs=worker_proofs, frozen_unchanged=True)
    except BaseException as caught:
        error = caught
        assessment["error"] = type(caught).__name__ + ": " + str(caught)
    finally:
        try:
            write(args.output / "protected-after.json", preserved(before))
            require(output_bytes(args.output) < CAP, "Persistent output budget exceeded")
        except BaseException as caught:
            preservation_errors.append(repr(caught))
            if error is None: error = caught
            assessment.update(status="failed", preservation_error=repr(caught))
        assessment["preservation_errors"] = preservation_errors
        if error is not None:
            assessment["primary_error"] = repr(error)
        assessment["elapsed_seconds"] = time.monotonic() - started
        write(args.output / "assessment.json", assessment)
    if error:
        raise error
    return assessment


if __name__ == "__main__":
    main()
