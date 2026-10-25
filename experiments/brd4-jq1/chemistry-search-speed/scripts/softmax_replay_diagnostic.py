"""Bounded GPU softmax replay forensics; no model, loss, repair or optimizer.

All observed equalities/mismatches are diagnostic controls only. A control that
matches an original snapshot is never selected as a replacement implementation
or used to reclassify the failed full-model comparison.
"""

from contextlib import ExitStack
from datetime import datetime, timezone
import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time


VERSION = "chemistry-softmax-replay-diagnostic-v1"
BASE = Path("/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01")
SCIENCE = BASE / "protenix-extended-release-03/source"
SCIENCE_PIN = "da9db6821d26b884ca1749303fc8e0b0095bd9ee7cbd1b427cdcf1b22eec6562"
FAILED_SOURCE = BASE / "chemistry-full-model-release-03/source"
FAILED_PIN = "8480a39c05d612d1a80ea1892d816c9429fcd471e97adb31655f8132d9007f31"
FAILED_ROOT = BASE / "chemistry-full-model-03/protenix/payload/comparison"
CONTROLS = ("plain", "matrix_import", "matrix_installed", "combined", "native_backend_import")
POINT_LABELS = ("early-0", "early-10", "high-0", "high-10")
MAX_BYTES = 64 * 1024**2
INTERNAL_SECONDS = 600
FIELD_NAMES = ("atom_logits", "bond_logits", "X", "E_upper", "key_data")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes(), parse_constant=lambda value:
                      (_ for _ in ()).throw(ValueError("Nonfinite JSON: " + value)))


def write(path, value):
    with Path(path).open("x", encoding="utf8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def binding_schema(row):
    require(type(row) is dict and set(row) == {"path", "bytes", "sha256"}, "Exact file binding required")
    require(type(row["path"]) is str and Path(row["path"]).is_absolute()
            and ".." not in Path(row["path"]).parts and "\\" not in row["path"], "Canonical file path required")
    require(type(row["bytes"]) is int and row["bytes"] > 0
            and type(row["sha256"]) is str and re.fullmatch("[a-f0-9]{64}", row["sha256"]), "Finite length/exact SHA required")
    return row


def point_paths(variant, label):
    require(variant in ("baseline", "combined") and label in POINT_LABELS, "Undeclared original point")
    pilot, step = label.split("-")
    root = FAILED_ROOT / variant / "study" / (pilot + "_pilot")
    number = format(int(step), "04d")
    return root / "snapshots" / ("step-" + number + ".npz"), root / ("exact-replay-" + number + ".json")


def validate_plan(plan):
    require(type(plan) is dict and set(plan) == {"version", "scientific_source", "failed_pilot_source", "points"},
            "Exact diagnostic plan fields required")
    require(plan["version"] == VERSION, "Wrong diagnostic version")
    for key, path, pin, count in (("scientific_source", SCIENCE, SCIENCE_PIN, 643),
                                  ("failed_pilot_source", FAILED_SOURCE, FAILED_PIN, 30)):
        require(plan[key] == {"path": str(path), "manifest_sha256": pin, "files": count}, "Original source binding changed")
    require(type(plan["points"]) is list and len(plan["points"]) == 8, "All eight original boundaries required")
    observed = []
    for row in plan["points"]:
        require(type(row) is dict and set(row) == {"variant", "label", "snapshot", "replay"}, "Exact point fields required")
        expected = point_paths(row["variant"], row["label"])
        for key, path in zip(("snapshot", "replay"), expected):
            binding_schema(row[key]); require(row[key]["path"] == str(path), "Original point path redirected")
        observed.append((row["variant"], row["label"]))
    require(observed == [(variant, label) for variant in ("baseline", "combined") for label in POINT_LABELS],
            "Original point order/identity changed")
    return plan


def check_binding(row):
    binding_schema(row)
    path = Path(row["path"]); info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and not path.is_symlink()
            and all(not parent.is_symlink() for parent in path.parents), "Original input is linked/special")
    require(info.st_size == row["bytes"] and sha(path) == row["sha256"], "Original input bytes changed")
    return row


def verify_source(row):
    root = Path(row["path"]); path = root / "source_manifest.json"
    require(root.is_dir() and not root.is_symlink() and not path.is_symlink()
            and all(not parent.is_symlink() for parent in root.parents)
            and sha(path) == row["manifest_sha256"], "Source manifest changed")
    files = read(path)["files"]
    require(len(files) == row["files"], "Source member count changed")
    require(all(not member.is_symlink() and (member.is_dir() or member.is_file())
                for member in root.rglob("*")), "Source has linked/special members")
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() or path.is_symlink()}
    require(actual == set(files) | {"source_manifest.json"}, "Source membership changed")
    for name, digest in files.items():
        target = root / name
        require(not Path(name).is_absolute() and ".." not in Path(name).parts and "\\" not in name
                and target.is_file() and not target.is_symlink() and sha(target) == digest,
                "Original source file changed")
    return {"path": str(root), "manifest_sha256": sha(path), "files": len(files)}


def authenticate(plan):
    validate_plan(plan)
    sources = {key: verify_source(plan[key]) for key in ("scientific_source", "failed_pilot_source")}
    proofs = []
    for point in plan["points"]:
        for key in ("snapshot", "replay"): proofs.append(check_binding(point[key]))
        replay = read(point["replay"]["path"])
        require(replay.get("status") == "passed_exact_five_field_replay"
                and replay.get("snapshot") == point["snapshot"]
                and replay.get("stored_probability_substitution") is False
                and replay.get("equality_tolerance") is None
                and set(replay.get("field_bytes_sha256", {})) == set(FIELD_NAMES), "Original replay evidence changed")
        require(set(replay.get("placement", {})) == {"atom_logits", "bond_logits"}, "Original placement missing")
    return {"sources": sources, "point_bindings": proofs}


def allocation_guard(environment=None):
    env = os.environ if environment is None else environment
    require(re.fullmatch("[1-9][0-9]*", env.get("LSB_JOBID", "")) is not None
            and env.get("LSB_QUEUE") == "gpul40s" and env.get("LSB_DJOB_NUMPROC") == "8",
            "Replay numerical work requires its actual eight-slot L40S allocation")
    require(not sys.flags.optimize, "Optimized Python cannot disable diagnostic checks")


def output_bytes(root):
    require(not Path(root).is_symlink(), "Linked diagnostic output root")
    total = 0
    for directory, folders, files in os.walk(root, followlinks=False):
        require(all(not (Path(directory) / name).is_symlink() for name in folders), "Linked diagnostic directory")
        for name in files:
            path = Path(directory) / name; info = path.lstat()
            require(stat.S_ISREG(info.st_mode) and not path.is_symlink(), "Linked/special diagnostic output")
            total += info.st_size
    return total


def forbid_scientific_execution(frame, event, arg):
    if event != "call": return
    name = frame.f_code.co_name; path = frame.f_code.co_filename.replace("\\", "/")
    blocked = {"load_model", "prepare_context", "prepare_failure_guidance", "search_atom_preserving_repair",
               "search_progress_repair", "evaluate_combined", "run_chemical_updates", "check_combined_directions",
               "forward_trunk", "forward_hard_geometry_off"}
    if name in blocked or (name == "load" and path.endswith("mosaic/small_molecule/protenix.py")):
        raise RuntimeError("Diagnostic forbids model/loss/search/optimizer execution: " + name)


def forbid_weight_open(event, args):
    if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)): return
    path = os.fsdecode(args[0]).lower()
    if "/cache/mosaic/protenix/" in path or path.endswith((".eqx", ".pt", ".pth", ".ckpt", ".safetensors", ".skeleton.pkl")):
        raise RuntimeError("Replay diagnostic cannot open model/context checkpoint assets")


def _field_record(value):
    import numpy as np
    array = np.asarray(value)
    require(np.isfinite(array).all(), "Nonfinite softmax diagnostic output")
    return {"dtype": str(array.dtype), "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}


def _compare(actual, expected):
    import numpy as np
    a, b = np.asarray(actual), np.asarray(expected)
    schema = a.dtype == b.dtype and a.shape == b.shape
    return {"same_dtype_shape": schema, "exact_bytes": schema and a.tobytes(order="C") == b.tobytes(order="C"),
            "mismatched_elements": int(np.count_nonzero(a != b)) if schema else None,
            "max_absolute_difference": float(np.max(np.abs(a - b))) if schema else None}


def _array_metadata(value):
    import numpy as np
    result = {**_field_record(value), "host_strides_bytes": list(np.asarray(value).strides)}
    if hasattr(value, "devices"):
        result.update(devices=[{"platform": d.platform, "id": int(d.id), "process_index": int(d.process_index)}
                               for d in value.devices()], committed=bool(value.committed), weak_type=bool(value.weak_type),
                      sharding=str(value.sharding))
        for attribute in ("format", "layout"):
            try: result[attribute] = str(getattr(value, attribute))
            except (AttributeError, RuntimeError, ValueError): result[attribute] = "unavailable"
    return result


def _runtime_record(jax):
    import numpy as np
    softmax = inspect.unwrap(jax.nn.softmax)
    filename = inspect.getsourcefile(softmax)
    return {
        "jax_config_repr": {key: repr(value) for key, value in sorted(jax.config.values.items())},
        "environment": {key: value for key, value in os.environ.items()
                        if key.startswith(("JAX", "XLA", "CUDA", "NVIDIA", "OMP", "OPENBLAS", "MKL", "TRITON"))},
        "versions": {name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "equinox", "numpy")},
        "devices": [{"platform": d.platform, "id": int(d.id), "kind": d.device_kind} for d in jax.devices()],
        "softmax": {"module": softmax.__module__, "qualname": softmax.__qualname__, "path": filename,
                    "file_sha256": sha(filename), "source": inspect.getsource(softmax)},
        "numpy": np.__version__, "native_backend_imported": "protenix.backend" in sys.modules,
        "relevant_import_order": [{"module": name, "path": getattr(module, "__file__", None)}
                                  for name, module in sys.modules.items()
                                  if name.startswith(("jax", "jaxlib", "protenix", "gpu_matrix", "full_model_pilot",
                                                      "chem_speed", "matrix_risk", "combined"))],
    }


def _compiled_text(output, name, text):
    require(type(text) is str, "Compiler did not expose its textual representation")
    raw = text.encode("utf8"); limit = 64 * 1024
    path = output / (name + ".txt")
    path.write_bytes(raw[:limit])
    return {"path": str(path), "full_text_bytes": len(raw), "full_text_sha256": hashlib.sha256(raw).hexdigest(),
            "stored_bytes": min(len(raw), limit), "stored_sha256": sha(path), "truncated": len(raw) > limit}


def validate_completed_controls(rows):
    expected = [(control, repeat) for control in CONTROLS for repeat in (0, 1)]
    require(type(rows) is list and len(rows) == len(expected)
            and [(row.get("control"), row.get("repeat")) for row in rows] == expected,
            "All ten declared diagnostic controls must be observed exactly once in order")
    require(all(row.get("returncode") == 0
                and row.get("status") == "diagnostic_control_completed_original_failure_preserved"
                and type(row.get("assessment_sha256")) is str
                and re.fullmatch("[a-f0-9]{64}", row["assessment_sha256"]) is not None
                for row in rows), "An incomplete/failed control cannot establish completed replay diagnostics")
    return True


def child(plan, control, repeat, output):
    allocation_guard(); require(control in CONTROLS and repeat in (0, 1), "Undeclared control/repetition")
    before = authenticate(plan); output.mkdir(exist_ok=False)
    record = {"version": VERSION, "control": control, "repeat": repeat, "status": "failed",
              "input_before": before, "points": [], "scientific_gate_passed": False,
              "optimizer_updates": 0, "model_evaluations": 0, "probability_substitution": False,
              "can_certify_or_restart_original_study": False, "original_gate_reclassified": False,
              "started_utc": datetime.now(timezone.utc).isoformat()}
    previous = sys.getprofile(); sys.setprofile(forbid_scientific_execution); sys.addaudithook(forbid_weight_open)
    try:
        sys.path[:0] = [str(FAILED_SOURCE / "scripts"), str(SCIENCE / "scripts"), str(SCIENCE / "src")]
        # Preserve the original child-run ordering: launcher module, JAX/loss,
        # optional adapters, then driver initialization. No model is loaded.
        adapter = importlib.import_module("full_model_pilot")
        import jax
        import jax.numpy as jnp
        import numpy as np
        importlib.import_module("mosaic.small_molecule.chemistry.loss")
        require(not jax.config.jax_enable_x64 and jax.default_backend() == "gpu"
                and all(d.platform == "gpu" for d in jax.devices()), "Actual original FP32 GPU backend required")
        jax.config.update("jax_default_matmul_precision", "highest")
        record["runtime_before_control"] = _runtime_record(jax)
        with ExitStack() as stack:
            if control == "matrix_import": importlib.import_module("gpu_matrix")
            elif control == "matrix_installed":
                stack.enter_context(importlib.import_module("gpu_matrix").install_matrix_loss("matrix"))
            elif control == "combined":
                stack.enter_context(adapter.combined_adapters("protenix-random50k", output))
            elif control == "native_backend_import": importlib.import_module("protenix.backend")
            logit_type = importlib.import_module("protenix_initialization").ProtenixReplayLogits
            require(Path(importlib.import_module("protenix_initialization").__file__).resolve()
                    == (SCIENCE / "scripts/protenix_initialization.py").resolve(), "Wrong original logit class")
            record["original_class"] = {"module": logit_type.__module__, "name": logit_type.__qualname__,
                "path": inspect.getsourcefile(logit_type), "source": inspect.getsource(logit_type),
                "file_sha256": sha(inspect.getsourcefile(logit_type))}
            record["runtime_after_control"] = _runtime_record(jax)
            points = []
            for point in plan["points"]:
                with np.load(point["snapshot"]["path"], allow_pickle=False) as saved:
                    arrays = {name: saved[name].copy() for name in FIELD_NAMES}
                expected = read(point["replay"]["path"])
                require({name: _field_record(arrays[name])["sha256"] for name in FIELD_NAMES}
                        == expected["field_bytes_sha256"], "Raw original NPZ differs from five-field byte proof")
                require(all(arrays[name].dtype == np.float32 for name in FIELD_NAMES[:-1])
                        and arrays["key_data"].dtype == np.uint32, "Original field precision changed")
                points.append((point, arrays, expected))
            for point, arrays, expected in points:
                row = {"variant": point["variant"], "label": point["label"], "input_fields": {
                    name: _field_record(value) for name, value in arrays.items()}, "evaluations": [], "compiled": {}}
                for placement in ("declared_committed", "default_uncommitted"):
                    leaves = []
                    for name in ("atom_logits", "bond_logits"):
                        declared = expected["placement"][name]
                        require(declared["platform"] == "gpu" and declared["device_id"] == 0
                                and declared["process_index"] == 0 and declared["weak_type"] is False,
                                "Undeclared original native placement")
                        leaf = (jax.device_put(arrays[name], jax.sharding.SingleDeviceSharding(jax.devices("gpu")[0],
                                             memory_kind=declared["memory_kind"]))
                                if placement == "declared_committed" else jnp.asarray(arrays[name]))
                        require(_field_record(leaf) == _field_record(arrays[name]), "Logit transport altered original bytes")
                        leaves.append(leaf)
                    logits = logit_type(*leaves)
                    for cache_phase in ("warm_repeats", "cleared_cache"):
                        if cache_phase == "cleared_cache": jax.clear_caches()
                        for index in range(3):
                            graph = logits.graph(); jax.block_until_ready((graph.X, graph.E_upper))
                            actual = {"atom_logits": logits.atoms, "bond_logits": logits.bonds,
                                      "X": graph.X, "E_upper": graph.E_upper,
                                      "key_data": arrays["key_data"]}
                            comparisons = {}
                            for other, target, _ in points:
                                if other["label"] != point["label"]: continue
                                same_input = all(_field_record(arrays[name]) == _field_record(target[name])
                                                 for name in ("atom_logits", "bond_logits", "key_data"))
                                comparisons[other["variant"]] = {"same_logit_and_key_input": same_input,
                                    "fields": {name: _compare(value, target[name]) for name, value in actual.items()}}
                            row["evaluations"].append({"placement_control": placement, "cache_control": cache_phase,
                                "repeat": index, "original_class_graph": True,
                                "outputs": {name: _array_metadata(value) for name, value in actual.items()},
                                "comparisons": comparisons})
                    # These additional compiled softmax controls do not replace
                    # the original class or constitute a five-field admission.
                    if placement == "declared_committed":
                        for mode, function, arguments in (
                            ("dynamic", jax.jit(lambda bonds: jax.nn.softmax(bonds, -1)), (logits.bonds,)),
                            ("closed_constant", jax.jit(lambda: jax.nn.softmax(logits.bonds, -1)), ())):
                            lowered = function.lower(*arguments); compiled = lowered.compile()
                            value = compiled(*arguments); value.block_until_ready()
                            prefix = point["variant"] + "-" + point["label"] + "-" + mode
                            row["compiled"][mode] = {"diagnostic_only_not_original_class_admission": True,
                                "stablehlo": _compiled_text(output, prefix + "-stablehlo", str(lowered.compiler_ir(dialect="stablehlo"))),
                                "optimized_hlo": _compiled_text(output, prefix + "-optimized", compiled.as_text()),
                                "output": _array_metadata(value), "own_saved_E_upper": _compare(value, arrays["E_upper"])}
                require({name: _field_record(value)["sha256"] for name, value in arrays.items()}
                        == expected["field_bytes_sha256"], "Original in-memory input bytes changed")
                record["points"].append(row)
                require(output_bytes(output) <= MAX_BYTES // 2, "Bounded child diagnostic output exceeded")
        record["runtime_after_restoration"] = _runtime_record(jax)
        record["status"] = "diagnostic_control_completed_original_failure_preserved"
    except BaseException as error:
        record["error"] = repr(error)
        raise
    finally:
        sys.setprofile(previous)
        try:
            record["input_after"] = authenticate(plan)
            require(record["input_after"] == before, "Original snapshot/source bytes changed")
        except BaseException as error:
            record.update(status="failed", input_after_error=repr(error))
            raise
        finally:
            record["finished_utc"] = datetime.now(timezone.utc).isoformat()
            write(output / "control.json", record)


def parent(plan, plan_path, plan_sha256, output):
    started = time.monotonic()
    allocation_guard(); before = authenticate(plan); output.mkdir(exist_ok=False)
    result = {"version": VERSION, "status": "failed", "input_before": before, "controls": [],
              "optimizer_updates": 0, "model_evaluations": 0, "original_gate_reclassified": False,
              "can_certify_or_restart_original_study": False, "matching_control_selected_as_fix": False,
              "all_controls_completed": False,
              "persistent_cap_bytes": MAX_BYTES, "internal_seconds": INTERNAL_SECONDS,
              "started_utc": datetime.now(timezone.utc).isoformat()}
    try:
        for control in CONTROLS:
            for repeat in (0, 1):
                directory = output / (control + "-" + str(repeat))
                command = [sys.executable, str(Path(__file__).resolve()), "--plan", str(plan_path),
                           "--plan-sha256", plan_sha256, "--output", str(directory),
                           "--control", control, "--repeat", str(repeat)]
                remaining = INTERNAL_SECONDS - 30 - (time.monotonic() - started)
                require(remaining > 0, "Diagnostic internal shared deadline exhausted")
                env = dict(os.environ, JAX_PLATFORMS="cuda,cpu", JAX_ENABLE_X64="0",
                           XLA_PYTHON_CLIENT_PREALLOCATE="false", JAX_ENABLE_COMPILATION_CACHE="false",
                           PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
                with (output / (control + "-" + str(repeat) + ".log")).open("xb") as stream:
                    process = subprocess.Popen(command, cwd=SCIENCE, env=env, stdout=stream,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        while process.poll() is None:
                            require(output_bytes(output) < MAX_BYTES - 1024**2, "Joint diagnostic cap exhausted")
                            require(time.monotonic() - started < INTERNAL_SECONDS - 30, "Shared diagnostic deadline exhausted")
                            try: process.wait(timeout=min(5, remaining))
                            except subprocess.TimeoutExpired: pass
                    except BaseException:
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGTERM)
                            try: process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid, signal.SIGKILL); process.wait(timeout=5)
                        raise
                control_path = directory / "control.json"
                result["controls"].append({"control": control, "repeat": repeat, "argv": command,
                    "returncode": process.returncode, "assessment": str(control_path),
                    "assessment_sha256": sha(control_path) if control_path.exists() else None,
                    "status": read(control_path)["status"] if control_path.exists() else "control_failed_before_receipt"})
        result["all_controls_completed"] = validate_completed_controls(result["controls"])
        result["status"] = "diagnostics_completed_original_failure_preserved"
    except BaseException as error:
        result["error"] = repr(error); raise
    finally:
        try:
            result["input_after"] = authenticate(plan)
            require(result["input_after"] == before, "Original source/snapshots changed during diagnostics")
        except BaseException as error:
            result.update(status="failed", input_after_error=repr(error))
            raise
        finally:
            result["elapsed_seconds"] = time.monotonic() - started
            result["finished_utc"] = datetime.now(timezone.utc).isoformat()
            result["output_bytes_before_assessment"] = output_bytes(output)
            write(output / "assessment.json", result)
            require(output_bytes(output) <= MAX_BYTES, "Final diagnostic output cap exceeded")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", choices=CONTROLS)
    parser.add_argument("--repeat", type=int, choices=(0, 1))
    args = parser.parse_args(argv)
    require((args.control is None) == (args.repeat is None), "Private control and repetition must be paired")
    return args


def main(argv=None):
    args = parse_args(argv)
    require(args.plan.is_absolute() and args.output.is_absolute() and not args.output.exists()
            and re.fullmatch("[a-f0-9]{64}", args.plan_sha256) is not None
            and sha(args.plan) == args.plan_sha256, "Fresh output and exact absolute bound plan required")
    plan = validate_plan(read(args.plan)); allocation_guard()
    if args.control is not None: child(plan, args.control, args.repeat, args.output)
    else: parent(plan, args.plan, args.plan_sha256, args.output)


if __name__ == "__main__": main()
