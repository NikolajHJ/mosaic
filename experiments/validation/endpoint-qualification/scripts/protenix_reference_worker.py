"""Bounded separate FP64 Protenix reference for current raw endpoint graphs.

The immutable Pro09 helpers are authenticated imports, never a fixed20k runner.
This process has no optimizer, saved-run writer, diffusion or submission path.
All numerical imports and execution require the actual allocated GPU phase.
"""
import argparse
from contextlib import ExitStack, redirect_stdout
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import sys
import time


VERSION = "protenix-generic-fp64-reference-rpc-v1"
BASE = Path("/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01")
ROOT = BASE / "nonchem-precision-continuation-18"
SOURCE = BASE / "nonchem-precision-release-18/source"
QUALIFIED = BASE / "protenix-endpoint-fix-release-09/source"
QUALIFIED_PIN = "f2043af74b8fa9ff485a44b80fcf96454bf8585868859389828c37f0073a3664"
SCIENCE = BASE / "protenix-extended-release-03/source"
SCIENCE_PIN = "da9db6821d26b884ca1749303fc8e0b0095bd9ee7cbd1b427cdcf1b22eec6562"
MODEL = "8d6926d5390fe682e193089f790ce53aa4011a6d5509ddb452a55db3d89b0ab2"
CONTEXT = "b28dfad8e5ede565071c70a61695f6a38f5cb6bd591705e32979e59050ff5ce4"
POCKET = "9376c715eaf5b48689fb3e995b00d7820601a7776970df66bd7e7cff3b13ee43"
MAX_BYTES = 64 * 1024**2
INTERNAL_SECONDS = 5400
MAX_REQUESTS = 128
MAX_LINE_BYTES = 256 * 1024
PRECISION = {"full_model_float64": True, "parameters_float64": True,
             "native_features_float64": True, "native_operations_float64": True,
             "final_reduction_only": False}


def require(value, message):
    if not value:
        raise ValueError(message)


def reference_spec():
    """Closed stdlib-only declaration for the new continuation plan builder."""
    prior = BASE / "protenix-endpoint-fix-09"
    return {"version": VERSION,
        "qualified_source": {"path": QUALIFIED.as_posix(), "files": 20, "manifest_sha256": QUALIFIED_PIN},
        "qualified_plan": {"path": (QUALIFIED / "plan.json").as_posix(), "bytes": 86914,
            "sha256": "7adf251df61bc8de66711769df417eed5d8ac9ca57d9686baf4011df9f7d09bf"},
        "qualified_native": {"path": (prior / "gpu/payload/assessment.json").as_posix(), "bytes": 2133,
            "sha256": "bb4ed22fb92d25e5c8de46ea3c4d68a8ed2162d5621b136966da37d36b42e82b"},
        "qualified_LSF": {"path": (prior / "gpu/stdout.29838596.log").as_posix(), "bytes": 2079,
            "sha256": "e91efc831eaed697db7724570fb3954ea509c4659544414f3c614ada45e3b8e7"},
        "qualified_CPU": {"path": (prior / "cpu-certificate.json").as_posix(), "bytes": 2559,
            "sha256": "338b236d46b7510955195c022308a058913cbe97102823c55b23d7819f671ffd"},
        "science_source": {"path": SCIENCE.as_posix(), "files": 643, "manifest_sha256": SCIENCE_PIN},
        "original_model_digest": MODEL, "original_context_digest": CONTEXT,
        "pocket_definition_sha256": POCKET,
        "objective": {"native_name": "contact", "cutoff": 8.0, "geometry": "off",
                      "scope": "unchanged_JQ1_pocket_contact", "coefficient": 1.0},
        "production_execution_changed": False, "optimizer_updates": 0,
        "max_bytes": MAX_BYTES, "internal_seconds": INTERNAL_SECONDS,
        "max_requests": MAX_REQUESTS, "max_request_line_bytes": MAX_LINE_BYTES,
        "graph_transport": {"dtype": "float64", "X": [25, 6], "E_upper": [300, 5],
                            "digest": "SHA256_concatenated_C_order_X_then_E_upper"},
        "key_transport": {"dtype": "uint32", "shape": [2], "implementation": "threefry2x32",
                          "selection": "actual_live_caller_per_request", "original_seed_provenance": 17}}


def _raw(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _read(path):
    return decode_json(Path(path).read_bytes())


def decode_json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON field: " + key)
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def _write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(_raw(value))


def binding(path):
    path = Path(path)
    require(path.is_absolute() and ".." not in path.parts
            and all(not p.is_symlink() for p in (path, *path.parents)), "Linked/nonabsolute reference input")
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode), "Nonregular reference input")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": info.st_size, "sha256": digest}


def bound(row):
    require(isinstance(row, dict) and set(row) == {"path", "bytes", "sha256"}
            and type(row["bytes"]) is int and row["bytes"] >= 0
            and isinstance(row["sha256"], str)
            and re.fullmatch("[a-f0-9]{64}", row["sha256"]), "Malformed exact reference binding")
    require(binding(row["path"]) == row, "Reference input bytes changed: " + str(row.get("path")))
    return Path(row["path"])


def verify_source(source, pin=None, count=None):
    source = Path(source)
    require(source.is_absolute() and source.is_dir()
            and all(not p.is_symlink() for p in (source, *source.parents)), "Linked/nonabsolute source root")
    manifest_file = binding(source / "source_manifest.json")
    require(pin is None or manifest_file["sha256"] == pin, "Reference source manifest identity differs")
    value = _read(manifest_file["path"]); files = value.get("files")
    require(isinstance(files, dict) and files and (count is None or len(files) == count), "Reference source scope differs")
    actual = set()
    for path in source.rglob("*"):
        require(not path.is_symlink() and (path.is_file() or path.is_dir()), "Linked/special source member")
        if path.is_file(): actual.add(path.relative_to(source).as_posix())
    require(actual == set(files) | {"source_manifest.json"}, "Unexpected reference source member")
    for name, digest in files.items():
        part = PurePosixPath(name)
        require(not part.is_absolute() and ".." not in part.parts and "\\" not in name
                and name not in ("", ".", "source_manifest.json"), "Unsafe reference source member")
        require(binding(source / name)["sha256"] == digest, "Reference source member changed")
    return value


def allocated_environment(startup=False):
    require(str(os.environ.get("LSB_JOBID", "")).isdigit(), "Real allocated reference process required")
    require(os.environ.get("LSB_QUEUE") == ("hpc" if startup else "gpul40s"), "Reference allocation queue differs")
    declared = os.environ.get("LSB_DJOB_NUMPROC", "")
    if not declared:
        fields = os.environ.get("LSB_MCPU_HOSTS", "").split()
        require(len(fields) == 2 and fields[1].isdigit(), "Reference CPU allocation unavailable")
        declared = fields[1]
    require(declared == "4", "Reference requires exactly four allocated CPU slots")


def validate_metadata(plan_path, plan_sha256):
    path = Path(plan_path)
    require(path == SOURCE / "plan.json" and re.fullmatch("[a-f0-9]{64}", plan_sha256 or ""),
            "Reference plan must be the exact owning immutable plan")
    require(binding(path)["sha256"] == plan_sha256, "Reference plan digest differs")
    own = verify_source(SOURCE)
    require(Path(__file__).resolve() == SOURCE / "scripts/protenix_reference_worker.py"
            and own["files"].get("scripts/protenix_reference_worker.py") == binding(__file__)["sha256"]
            and own["files"].get("plan.json") == plan_sha256, "Executing reference worker/plan is not source-bound")
    plan = _read(path)
    require(plan.get("root") == str(ROOT) and plan.get("source_root") == str(SOURCE)
            and plan.get("original_results_changed") is False
            and plan.get("optimization_settings_changed") is False,
            "Reference owner roots/preservation scope differ")
    require(_raw(plan.get("protenix_reference")) == _raw(reference_spec()), "Closed Protenix reference specification differs")
    spec = reference_spec()
    qualified = verify_source(QUALIFIED, QUALIFIED_PIN, 20)
    verify_source(SCIENCE, SCIENCE_PIN, 643)
    qualified_plan = _read(bound(spec["qualified_plan"]))
    native = _read(bound(spec["qualified_native"]))
    cpu = _read(bound(spec["qualified_CPU"]))
    require(native.get("status") == "prospective_precision_gate_passed_original_failures_preserved"
            and native.get("job_id") == "29838596" and native.get("updates") == 0
            and native.get("original_results_changed") is False
            and native.get("optimization_settings_changed") is False,
            "Qualified Pro09 native success provenance differs")
    require(cpu.get("status") == "passed" and cpu.get("cases") == 336
            and cpu.get("source_manifest_sha256") == QUALIFIED_PIN,
            "Qualified Pro09 CPU success provenance differs")
    lsf = " ".join(bound(spec["qualified_LSF"]).read_text().split())
    require("Subject: Job 29838596:" in lsf and "queue <gpul40s>" in lsf
            and "Successfully completed." in lsf, "Qualified same-job native LSF success missing")
    forensic_plan = _read(bound(qualified_plan["original_forensics_plan"]))
    context_ready = _read(bound(forensic_plan["context_ready"]))
    require(context_ready.get("status") == "passed" and context_ready.get("model_digest") == MODEL
            and context_ready.get("context_digest") == CONTEXT, "Original prepared context identity differs")
    pocket = qualified_plan["arm"]["inputs"]["pocket"]
    require(pocket["sha256"] == POCKET, "Original pocket identity differs")
    protected = {}
    def keep(row):
        p = bound(row); protected[str(p)] = row
    for source in (SOURCE, QUALIFIED, SCIENCE):
        manifest = _read(source / "source_manifest.json")
        for name in (*manifest["files"], "source_manifest.json"): keep(binding(source / name))
    for name in ("qualified_plan", "qualified_native", "qualified_CPU", "qualified_LSF"): keep(spec[name])
    keep(qualified_plan["original_forensics_plan"]); keep(forensic_plan["context_ready"])
    keep(pocket); keep(qualified_plan["arm"]["config"])
    for row in qualified_plan["runtime_files"]: keep(row)
    for row in cpu["proofs"]: keep(row)
    context_dir = Path(forensic_plan["context_ready"]["path"]).parent / "context"
    for name, digest in context_ready["context_files"].items():
        require(Path(name).name == name, "Prepared context file escaped its original directory")
        row = binding(context_dir / name); require(row["sha256"] == digest, "Original prepared context bytes changed"); keep(row)
    return {"plan": plan, "plan_sha256": plan_sha256, "qualified_plan": qualified_plan,
            "forensic_plan": forensic_plan, "context_ready": context_ready, "context_dir": context_dir,
            "pocket": pocket, "protected": protected,
            "own_source_manifest_sha256": binding(SOURCE / "source_manifest.json")["sha256"]}


def validate_output(output):
    path = Path(output)
    require(path.is_absolute() and path != ROOT and path.is_relative_to(ROOT)
            and ".." not in path.parts and not path.exists()
            and all(not p.is_symlink() for p in path.parents), "Reference output must be a fresh owned directory")
    require(path.parent.is_dir(), "Reference output parent must already be owned by the driver")
    return path


class Protocol:
    """Closed monotonic request protocol; no choice of graph/key or caching."""
    def __init__(self):
        self.next_id = 0; self.gradient_requests = 0; self.messages = 0; self.closed = False

    def accept(self, request):
        require(not self.closed and isinstance(request, dict), "Reference request after close or malformed")
        self.messages += 1
        require(self.messages <= 2 * MAX_REQUESTS + 8, "Reference message budget exhausted")
        if "action" in request:
            require(set(request) == {"action"} and request["action"] in ("verify_identity", "close"),
                    "Unknown reference control message")
            if request["action"] == "close":
                require(self.next_id > 0 and self.gradient_requests > 0, "Unevaluated reference cannot claim an audit pass")
                self.closed = True
            return request["action"]
        require(set(request) == {"id", "gradient", "X", "E_upper", "graph_sha256", "key_data"},
                "Reference evaluation request fields differ")
        require(type(request["id"]) is int and request["id"] == self.next_id
                and self.next_id < MAX_REQUESTS, "Reference request ID replay/gap/budget violation")
        require(type(request["gradient"]) is bool and isinstance(request["graph_sha256"], str)
                and re.fullmatch("[a-f0-9]{64}", request["graph_sha256"]),
                "Explicit reference gradient flag and graph digest required")
        key = request["key_data"]
        require(type(key) is list and len(key) == 2
                and all(type(x) is int and 0 <= x <= 4294967295 for x in key),
                "Exact actual-caller uint32[2] key transport required")
        for field, shape in (("X", (25, 6)), ("E_upper", (300, 5))):
            values = request[field]
            require(type(values) is list and len(values) == shape[0]
                    and all(type(row) is list and len(row) == shape[1] for row in values),
                    "Malformed reference graph shape: " + field)
            require(all(type(x) in (int, float) and math.isfinite(x) for row in values for x in row),
                    "Nonfinite/non-numeric reference graph: " + field)
        self.next_id += 1; self.gradient_requests += int(request["gradient"])
        return "evaluate"


def decode_graph(request):
    require(str(os.environ.get("LSB_JOBID", "")).isdigit(), "Reference numerical transport requires allocation")
    import numpy as np
    values = {field: np.asarray(request[field], dtype=np.float64) for field in ("X", "E_upper")}
    observed = hashlib.sha256(b"".join(values[f].tobytes(order="C") for f in ("X", "E_upper"))).hexdigest()
    require(observed == request["graph_sha256"], "Reference transport changed FP64 graph bytes")
    require(all(np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all()
                and np.allclose(v.sum(-1), 1, rtol=0, atol=2e-6) for v in values.values()),
            "Reference graph must remain finite categorical probabilities; no repair is permitted")
    return values


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); sys.modules[name] = value; spec.loader.exec_module(value)
    require(Path(value.__file__).resolve() == Path(path), "Qualified helper module origin differs")
    return value


def _size(output):
    total = 0
    for path in output.rglob("*"):
        require(not path.is_symlink() and (path.is_dir() or path.is_file()), "Linked/special reference artifact")
        if path.is_file(): total += path.stat().st_size
    return total


class NativeRuntime:
    def __init__(self, metadata, output):
        self.meta = metadata; self.output = output; self.stack = ExitStack()
        self.started = time.monotonic(); self.rows = []; self.audited = False; self.frozen_before = None

    def load(self):
        # Source and plan were fully authenticated before any numerical import.
        require(not any(x == "mosaic" or x.startswith("mosaic.") for x in sys.modules), "Fresh isolated reference process required")
        context_tools = _module(QUALIFIED / "scripts/protenix_replay_context.py", "_rpc_qualified_context")
        context_proof = context_tools.qualify(verify_source)
        environment = context_tools.environment(os.environ, context_proof)
        os.environ["XLA_FLAGS"] = environment["XLA_FLAGS"]
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        require(os.environ.get("JAX_ENABLE_X64") == "1", "Reference must enter a fresh x64 interpreter")
        sys.path[:0] = [str(SCIENCE / "src"), str(SCIENCE / "scripts"), str(QUALIFIED / "scripts")]
        import jax
        import equinox as eqx
        import numpy as np
        from mosaic.small_molecule.protenix import FrozenProtenixTiny
        from mosaic.small_molecule.features import bootstrap_native_assets
        from mosaic.small_molecule.io import load_context
        from mosaic.small_molecule.pocket import map_protenix_pocket
        from mosaic.small_molecule.losses import NativeContactLoss
        from mosaic.small_molecule.gradient_checks import array_digest
        require(jax.config.jax_enable_x64 and len(jax.devices("gpu")) == 1
                and "L40S" in jax.devices("gpu")[0].device_kind, "One allocated L40S reference device required")
        jax.config.update("jax_default_matmul_precision", "highest")
        self.jax, self.eqx, self.np, self.array_digest = jax, eqx, np, array_digest
        self.helpers = _module(QUALIFIED / "scripts/protenix_precision_native.py", "_rpc_qualified_placement")
        precision = _module(QUALIFIED / "scripts/protenix_precision_reference.py", "_rpc_qualified_precision")
        bootstrap_native_assets(Path(self.meta["forensic_plan"]["cache"]), include_model=True)
        hooks = self.stack.enter_context(precision.native_precision_context())
        original_model = FrozenProtenixTiny.load(cache_dir=Path(self.meta["forensic_plan"]["cache"]))
        original_context = load_context(self.meta["context_dir"], self.meta["context_ready"]["context_files"])
        require(array_digest(original_model) == MODEL and array_digest(original_context) == CONTEXT,
                "Original model/context differ before reference promotion")
        self.model, model_audit = precision.promote_tree(original_model, label="model")
        self.context, context_audit = precision.promote_tree(original_context, label="context")
        require(array_digest(original_model) == MODEL and array_digest(original_context) == CONTEXT,
                "Reference promotion changed original frozen values")
        del original_model, original_context
        before = self.frozen()
        self.gpu = jax.devices("gpu")[0]
        self.model, mp = self.helpers.place_reference_tree(self.model, self.gpu)
        self.context, cp = self.helpers.place_reference_tree(self.context, self.gpu)
        require(self.frozen() == before, "Reference device transfer changed frozen values")
        precision.floating_dtype_audit(self.model); precision.floating_dtype_audit(self.context)
        self.frozen_before = self.frozen()
        pocket = _read(bound(self.meta["pocket"]))
        tokens = map_protenix_pocket(self.context, pocket)
        self.loss = self.model.build_loss(context=self.context, loss=NativeContactLoss(8.0, tokens))
        self.identity = {"model_digest": MODEL, "context_digest": CONTEXT,
            "pocket_definition_sha256": POCKET, "scientific_source_manifest_sha256": SCIENCE_PIN,
            "qualified_source_manifest_sha256": QUALIFIED_PIN,
            "own_source_manifest_sha256": self.meta["own_source_manifest_sha256"],
            "device": self.helpers.reference_device_identity(self.gpu), "key_implementation": "threefry2x32",
            "key_scope": "actual_live_caller_per_request", "optimizer_updates": 0,
            "can_certify_or_restart_original_study": False}
        _write(self.output / "load-identity.json", self.identity)
        _write(self.output / "promotion.json", {"model": model_audit, "context": context_audit,
            "model_placement": mp, "context_placement": cp, "runtime_hooks": hooks,
            "compiler_context": context_proof, "production_execution_changed": False})

    def guard(self):
        require(time.monotonic() - self.started < INTERNAL_SECONDS - 60, "Reference evaluation deadline exhausted")
        require(_size(self.output) < MAX_BYTES - 2 * 1024**2, "Reference persistent-output budget exhausted")

    def frozen(self):
        return {"model": self.array_digest(self.model), "context": self.array_digest(self.context)}

    def ready(self):
        return {"status": "ready_full_double_reference", "frozen": self.frozen_before,
                "precision": dict(PRECISION), "identity": self.identity, "optimizer_updates": 0}

    def verify_identity(self):
        current = self.frozen(); require(current == self.frozen_before, "Reference frozen state changed")
        return {"status": "passed", "frozen": current, "identity": self.identity}

    def evaluate(self, request, values):
        self.guard()
        from mosaic.small_molecule.state import RelaxedGraph
        from mosaic.small_molecule.optimizers import evaluate_graph, _nonfinite_paths
        arrays, placement = self.helpers.reference_graph_arrays(values, self.gpu)
        point = RelaxedGraph(**arrays)
        key_values = self.np.asarray(request["key_data"], dtype=self.np.uint32)
        key = self.jax.random.wrap_key_data(self.jax.device_put(key_values, self.gpu), impl="threefry2x32")
        self.helpers.require_reference_device(key, self.gpu)
        key_digest = hashlib.sha256(key_values.tobytes(order="C")).hexdigest()
        if not self.audited:
            forward = self.eqx.filter_make_jaxpr(lambda graph: self.loss(graph, key=key))(point)[0]
            ad = self.eqx.filter_make_jaxpr(lambda graph: self.eqx.filter_value_and_grad(
                self.loss, has_aux=True)(graph, key=key))(point)[0]
            _write(self.output / "forward-dtype-audit.json", self.helpers.audit_jaxpr(forward))
            _write(self.output / "AD-dtype-audit.json", self.helpers.audit_jaxpr(ad))
            _write(self.output / "initial-point-placement.json", {**placement,
                "key_data": request["key_data"], "key_sha256": key_digest,
                "scope": "first_actual_request_graph_and_key_not_a_fixed_saved_endpoint"})
            self.audited = True
        primal, aux = self.loss(point, key=key)
        require(not _nonfinite_paths((primal, aux)), "Nonfinite independent reference primal/generated output")
        scalar = self.np.asarray(primal)
        require(scalar.dtype == self.np.float64 and scalar.shape == (), "Reference primal is not an FP64 scalar")
        result = {"id": request["id"], "graph_sha256": request["graph_sha256"],
                  "key_data": request["key_data"], "key_sha256": key_digest, "value": float(scalar)}
        data = {**values, "primal": scalar, "key_data": key_values}
        if request["gradient"]:
            raw = evaluate_graph(self.loss, point, key)
            require(not _nonfinite_paths(raw), "Nonfinite reference autodiff")
            ad_scalar = self.np.asarray(raw.value)
            require(ad_scalar.dtype == self.np.float64 and abs(float(ad_scalar) - float(scalar)) <= 1e-10,
                    "Independent reference primal/AD scalar parity failed")
            gradients = {f: self.np.asarray(getattr(raw.gradients, f)) for f in ("X", "E_upper")}
            require(all(v.dtype == self.np.float64 and v.shape == values[f].shape
                        and self.np.isfinite(v).all() for f, v in gradients.items()), "Invalid reference raw gradient")
            result.update(value=float(ad_scalar), gradients={f: v.tolist() for f, v in gradients.items()})
            data.update(AD_value=ad_scalar, **{"gradient_" + f: v for f, v in gradients.items()})
        path = self.output / ("point-" + str(request["id"]).zfill(4) + ".npz")
        with path.open("xb") as stream: self.np.savez_compressed(stream, **data)
        row = {"id": request["id"], "gradient": request["gradient"], "graph_sha256": request["graph_sha256"],
               "key_data": request["key_data"], "key_sha256": key_digest, "raw": binding(path)}
        self.rows.append(row); result["raw"] = row["raw"]
        self.guard()
        return result

    def finish(self):
        self.verify_identity()
        require(self.audited and self.rows and any(row["gradient"] for row in self.rows), "Missing actual reference primal/AD audit")
        _write(self.output / "oracle-points.json", {"requests": self.rows,
            "frozen_before": self.frozen_before, "frozen_after": self.frozen(), "cached_evaluations": 0})
        return {"status": "passed_full_FP64_reference_audit", "floating_operations_float64": True,
            "narrow_float_avals": [], "forward_dtype_audit": binding(self.output / "forward-dtype-audit.json"),
            "AD_dtype_audit": binding(self.output / "AD-dtype-audit.json"),
            "frozen_unchanged": True, "requests": len(self.rows), "raw_points": binding(self.output / "oracle-points.json"),
            "device": self.helpers.reference_device_identity(self.gpu),
            "original_results_changed": False, "optimizer_updates": 0,
            "can_certify_or_restart_original_study": False}


def serve(runtime, stdin, stdout):
    protocol = Protocol()
    def emit(value): stdout.write(json.dumps(value, allow_nan=False) + "\n"); stdout.flush()
    emit(runtime.ready())
    while True:
        runtime.guard()
        line = stdin.readline(MAX_LINE_BYTES + 1)
        require(line and line.endswith("\n") and len(line.encode()) <= MAX_LINE_BYTES,
                "Reference stream ended, was unterminated or exceeded its bounded line size")
        message = decode_json(line)
        action = protocol.accept(message)
        if action == "close": return protocol, runtime.finish()
        if action == "verify_identity": emit(runtime.verify_identity())
        else:
            with redirect_stdout(sys.stderr):
                result = runtime.evaluate(message, decode_graph(message))
            emit(result)


def worker(metadata, output):
    output.mkdir(exist_ok=False)
    _write(output / "input-preservation-before.json", metadata["protected"])
    runtime = NativeRuntime(metadata, output); primary = None; audit = None
    old_handler = signal.getsignal(signal.SIGALRM)
    def expired(*_): raise TimeoutError("Bounded reference runtime exhausted")
    signal.signal(signal.SIGALRM, expired); signal.alarm(INTERNAL_SECONDS - 60)
    try:
        with redirect_stdout(sys.stderr): runtime.load()
        protocol, audit = serve(runtime, sys.stdin, sys.stdout)
    except BaseException as error:
        primary = error
    finally:
        signal.alarm(60)
        errors = []
        if runtime.frozen_before is not None:
            try: runtime.verify_identity()
            except BaseException as error: errors.append("frozen: " + repr(error))
        try: runtime.stack.close()
        except BaseException as error: errors.append("hooks: " + repr(error))
        try:
            after = {name: binding(name) for name in metadata["protected"]}
            _write(output / "input-preservation-after.json", after)
            require(after == metadata["protected"], "Original/source/reference input bytes changed")
        except BaseException as error: errors.append("protected: " + repr(error))
        try: require(_size(output) < MAX_BYTES - 65536, "Reference output cap exceeded")
        except BaseException as error: errors.append("budget: " + repr(error))
        proof = {"status": "passed" if primary is None and not errors else "failed", "version": VERSION,
            "job_id": os.environ["LSB_JOBID"], "original_results_changed": False if not errors else None,
            "optimizer_updates": 0, "production_execution_changed": False,
            "primary_error": repr(primary) if primary is not None else None, "preservation_errors": errors}
        _write(output / "preservation.json", proof)
        signal.alarm(0); signal.signal(signal.SIGALRM, old_handler)
    if primary is not None: raise primary
    require(not errors and audit is not None, "Reference preservation or final audit failed: " + repr(errors))
    audit.update(runtime_hooks_restored=True, preservation=binding(output / "preservation.json"))
    _write(output / "close-audit.json", audit)
    print(json.dumps({"status": "closed", "frozen": runtime.frozen_before,
                      "identity": runtime.identity, "audit": audit}, allow_nan=False), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-plan", type=Path, required=True)
    parser.add_argument("--reference-plan-sha256", required=True)
    parser.add_argument("--startup-only", action="store_true")
    parser.add_argument("--startup-output", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    allocated_environment(args.startup_only)
    if args.startup_only:
        require(args.output is None and args.startup_output == ROOT / "cpu/payload/protenix-reference-startup.json",
                "Closed reference startup output scope differs")
    else:
        require(args.startup_output is None and args.output is not None, "Reference native output scope missing/differs")
    metadata = validate_metadata(args.reference_plan, args.reference_plan_sha256)
    if args.startup_only:
        _write(args.startup_output, {"status": "passed_metadata_only", "version": VERSION,
            "numerical_setup_entered": False, "optimizer_updates": 0, "job_id": os.environ["LSB_JOBID"],
            "plan_sha256": args.reference_plan_sha256, "own_source_manifest_sha256": metadata["own_source_manifest_sha256"]})
        return
    worker(metadata, validate_output(args.output))


if __name__ == "__main__": main()
