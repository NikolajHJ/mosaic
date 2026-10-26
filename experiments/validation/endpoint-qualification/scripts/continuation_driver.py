"""Explicit prospective endpoint hooks around authenticated original drivers.

Private allocated process only. Original optimizer/model/source/config/saved
results are not edited. Only the raw-FD loop and its explicit readiness predicate
are replaced by the separately versioned prospective admission policy. All other
original checker statements (including nulls and component transports) remain.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import runpy
import subprocess
import sys
import textwrap


VERSION = "endpoint-aware-continuation-result-v1"
HOOK_VERSION = "prospective-endpoint-driver-hook-v1"
READINESS_VERSION = "prospective-endpoint-readiness-adapter-v1"
BASE = Path("/work3/s234818/mosaic/runs/20261022-brd4-jq1-lrboost-analysis-01")
RETENTION = BASE / "nonchem-stride20-startup-release-03/source"
RETENTION_SHA = "704fab033d97342a14dc12b58dc32df44874232c5c30aad47c4d02041a502aab"
SOURCES = {
    "protenix": ("protenix-extended-release-03/source", 643,
                 "da9db6821d26b884ca1749303fc8e0b0095bd9ee7cbd1b427cdcf1b22eec6562",
                 "examples/brd4_25_atoms.py", .3, 10000, 50000),
    "boltz": ("boltz-lr0025-release-03/source", 635,
              "c214bccf2c94ac6829996bc0ca7d6415d3c19549be4fe7f874cba2b433901cc6",
              "examples/brd4_25_atoms.py", .0025, 10000, 30000),
    "dbmol": ("boltz-dbmol-pilot-release-02/source", 657,
              "600ecc3b33135141a57c029a28a2c72f29e788e3ecf5cb2dac1761146d1d6eed",
              "scripts/dbmol_driver.py", .0025, 2500, 10000),
}
REPLACED_OPTIONS = ("--mode", "--output", "--offset", "--segment-steps",
                    "--resume-snapshot", "--resume-sha256", "--ready", "--ready-sha256")


class PremodelBoundaryReached(RuntimeError):
    """Actual original argparse/metadata route reached its numerical-import boundary."""


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def binding(path):
    path = Path(path)
    require(path.is_absolute() and path.is_file() and not path.is_symlink(), "Nonregular absolute proof")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def bound(row):
    require(binding(row["path"]) == row, "Bound input changed: " + str(row.get("path")))
    return Path(row["path"])


def arm_record(plan, name):
    require(name in SOURCES and isinstance(plan.get("arms"), list), "Unknown continuation arm/plan")
    rows = [row for row in plan["arms"] if row.get("id") == name]
    require(len(rows) == 1, "Missing or duplicate arm")
    return rows[0]


def _option(argv, option, optional=False):
    require(argv.count(option) == (int(option in argv) if optional else 1), "Duplicated/missing option: " + option)
    if option not in argv:
        return None
    index = argv.index(option)
    require(index + 1 < len(argv), "Missing option value")
    return argv[index + 1]


def verify_source(source, pin, count=None):
    source = Path(source)
    require(source.is_absolute() and not source.is_symlink() and sha(source / "source_manifest.json") == pin,
            "Source identity changed")
    manifest = read(source / "source_manifest.json")
    files = manifest["files"]
    require(count is None or len(files) == count, "Original source member count changed")
    actual = {p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file() or p.is_symlink()}
    require(actual == set(files) | {"source_manifest.json"}, "Unexpected source member")
    for name, digest in files.items():
        part = PurePosixPath(name)
        path = source.joinpath(*part.parts)
        require(not part.is_absolute() and ".." not in part.parts and "\\" not in name
                and not path.is_symlink() and path.is_file() and sha(path) == digest,
                "Source member changed")
    return manifest


def build_original_argv(plan, name, phase, offset, output, snapshot, ready=None):
    """Preserve actual original argv except explicitly declared continuation fields.

    ``snapshot``/``ready`` are exact path/bytes/SHA bindings. This function is
    metadata-only; authentication occurs in the executing allocated process.
    """
    arm = arm_record(plan, name)
    require(phase in ("validate", "segment") and type(offset) is int, "Unknown phase/offset")
    rel, count, pin, driver, rate, steps, target = SOURCES[name]
    require(arm["segment_updates"] == steps and arm["target_global"] == target
            and offset >= arm["start_global"] and offset in range(arm["start_global"], target, steps)
            and offset + steps <= target, "Undeclared continuation boundary")
    source = arm["source"]
    require(source["path"] == str(BASE / rel) and source["manifest_sha256"] == pin
            and source["files"] == count, "Wrong original scientific source")
    argv = list(arm["original_argv"])
    require(argv == arm["inputs"]["original_driver_argv"], "Original argv aliases differ")
    require(len(argv) >= 4 and Path(argv[1]) == BASE / rel / driver
            and _option(argv, "--manifest") == pin
            and _option(argv, "--config") == arm["config"]["path"], "Original driver/config/manifest differs")
    for option in REPLACED_OPTIONS:
        _option(argv, option, optional=True)
    output = Path(output)
    startup = Path(plan["root"]) / "cpu/payload" / ("startup-" + name + "-" + phase) / "study"
    worker_startup_scope = Path(plan["root"]) / "cpu/payload" / ("worker-startup-" + name) / "study"
    require(output.is_absolute() and output.resolve().is_relative_to(Path(plan["root"]).resolve())
            and output.name == "study" and (output.parent.name == "payload" or output == startup
                or (name in ("boltz", "dbmol") and phase == "validate" and output == worker_startup_scope)),
            "Output outside declared new scientific/startup scope")
    require(isinstance(snapshot, dict) and set(snapshot) == {"path", "bytes", "sha256"}, "Exact snapshot binding required")
    require((ready is not None) == (phase == "segment"), "Fresh readiness required only for production")
    result = argv[:2]
    index = 2
    while index < len(argv):
        if argv[index] in REPLACED_OPTIONS:
            index += 2
        else:
            result.append(argv[index]); index += 1
    result += ["--mode", "validate" if phase == "validate" else "run", "--output", str(output),
               "--offset", str(offset), "--segment-steps", str(steps),
               "--resume-snapshot", snapshot["path"], "--resume-sha256", snapshot["sha256"]]
    if ready:
        result += ["--ready", ready["path"], "--ready-sha256", ready["sha256"]]
    return result


def replace_raw_loop(function, replacement, *, kind):
    """Replace exactly one original raw-FD loop; preserve every other AST node."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    require(len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef), "Unexpected original checker shape")
    found = []
    class Transform(ast.NodeTransformer):
        def visit_For(self, node):
            direct = (isinstance(node.target, ast.Name) and node.target.id == "field"
                      and ast.dump(node.iter) == ast.dump(ast.parse("('X','E_upper')", mode="eval").body))
            dbmol = (kind == "dbmol" and isinstance(node.target, ast.Tuple)
                     and [n.id for n in node.target.elts if isinstance(n, ast.Name)] == ["field", "values", "gradient"])
            if direct or dbmol:
                found.append(deepcopy(node))
                return ast.parse("report.update(_endpoint_raw_replacement(locals()))").body[0]
            return self.generic_visit(node)
    transformed = Transform().visit(deepcopy(tree))
    require(len(found) == 1, "Original checker raw loop shape changed")
    ast.fix_missing_locations(transformed)
    globals_copy = dict(function.__globals__)
    globals_copy["_endpoint_raw_replacement"] = replacement
    namespace = {}
    exec(compile(transformed, function.__code__.co_filename + "[endpoint-raw-loop-only]", "exec"), globals_copy, namespace)
    # The retained loop is evaluated separately as historical/coarse evidence.
    legacy = ast.Module(body=[deepcopy(found[0])], type_ignores=[])
    ast.fix_missing_locations(legacy)
    return namespace[function.__name__], compile(legacy, function.__code__.co_filename + "[legacy-raw-evidence]", "exec")


def startup_tree(raw, filename, *, defer_ready=False):
    """Keep the real parser/metadata prefix; stop before any numerical import."""
    tree = ast.parse(raw, filename=filename)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"]
    require(len(functions) == 1, "Original driver main shape differs")
    function = functions[0]
    boundary = next((i for i, node in enumerate(function.body) if isinstance(node, ast.Import)
                     and any(alias.name in ("jax", "equinox") for alias in node.names)), None)
    require(boundary is not None and any(isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "parse_args"
                for node in function.body[:boundary]), "Missing actual parser before backend import")
    if defer_ready:
        ready_blocks = [i for i, node in enumerate(function.body[:boundary])
                        if isinstance(node, ast.If) and "verify_readiness(" in ast.unparse(node)]
        require(len(ready_blocks) == 1, "Missing unique original readiness branch")
        boundary = ready_blocks[0]
    function.body = function.body[:boundary] + ast.parse(
        "raise PremodelBoundaryReached('Actual original parser/metadata accepted; no numerical backend/model loaded')").body
    tree.body = [node for node in tree.body if not (isinstance(node, ast.If) and
                  "__name__" in ast.unparse(node.test))]
    ast.fix_missing_locations(tree)
    return tree


def prospective_ready(ready_path, arm, arm_name, offset, snapshot, manifest, plan_sha):
    ready = binding(ready_path)
    own_ready = read(Path(ready_path).parent.parent / "assessment.json")
    require(own_ready.get("version") == VERSION and own_ready.get("status") == "passed"
            and own_ready.get("phase") == "validate" and own_ready.get("arm") == arm_name
            and own_ready.get("offset") == offset and own_ready.get("input_snapshot") == snapshot
            and own_ready.get("source_manifest_sha256") == manifest
            and own_ready.get("plan_sha256") == plan_sha
            and own_ready.get("config_sha256") == arm["config"]["sha256"]
            and own_ready.get("readiness") == ready,
            "Exact new prospective readiness assessment required")
    for row in own_ready["proofs"]: bound(row)
    return own_ready


@contextmanager
def execution_context(source, argv):
    old_path, old_argv, old_cwd = sys.path[:], sys.argv[:], Path.cwd()
    old_bytecode = sys.dont_write_bytecode
    old_pythonpath = os.environ.get("PYTHONPATH")
    old_bytecode_env = os.environ.get("PYTHONDONTWRITEBYTECODE")
    try:
        sys.dont_write_bytecode = True
        sys.path[:0] = [str(source / "scripts"), str(source / "src"), str(source / "tests")]
        # sys.path is process-local. The original frozen Boltz workers are new
        # interpreters and must inherit the same authenticated scientific src.
        os.environ["PYTHONPATH"] = worker_environment(source)["PYTHONPATH"]
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        sys.argv = argv[1:]
        os.chdir(source)
        yield
    finally:
        sys.path[:] = old_path
        sys.argv[:] = old_argv
        sys.dont_write_bytecode = old_bytecode
        if old_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = old_pythonpath
        if old_bytecode_env is None:
            os.environ.pop("PYTHONDONTWRITEBYTECODE", None)
        else:
            os.environ["PYTHONDONTWRITEBYTECODE"] = old_bytecode_env
        os.chdir(old_cwd)


def worker_environment(source):
    """Give child interpreters exact original src first, preserving old entries."""
    result = dict(os.environ)
    prefix = [str(Path(source) / "src"), str(Path(source) / "scripts")]
    previous = result.get("PYTHONPATH")
    result["PYTHONPATH"] = os.pathsep.join(prefix + ([previous] if previous else []))
    result["PYTHONDONTWRITEBYTECODE"] = "1"
    return result


def worker_startup(source, arm_name, original_argv):
    """Real worker-interpreter import/CLI checks; never enter model setup."""
    require(arm_name in ("boltz", "dbmol"), "No separate worker for this arm")
    source = Path(source)
    python = _option(original_argv, "--worker-python", optional=True) or \
        "/work3/s234818/latentmol/env/boltz311/bin/python"
    modules = ["mosaic", "mosaic.small_molecule.boltz2.native"]
    if arm_name == "dbmol":
        modules.append("mosaic.small_molecule.boltz2.dbmol")
    expected = {name: binding(source / "src" / (name.replace(".", "/") + ".py"))
                for name in modules if name != "mosaic"}
    expected["mosaic"] = binding(source / "src/mosaic/__init__.py")
    program = """import hashlib,importlib,json,pathlib,sys
expected=json.loads(sys.argv[1]); observed={}
for name,row in expected.items():
    module=importlib.import_module(name)
    path=pathlib.Path(module.__file__).resolve()
    assert str(path)==str(pathlib.Path(row['path']).resolve()), (name,str(path),row['path'])
    assert path.stat().st_size==row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'], name
    observed[name]={'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
print(json.dumps({'status':'passed_exact_original_imports','modules':observed},allow_nan=False))
"""
    environment = worker_environment(source)
    commands = [[python, "-c", program, json.dumps(expected, separators=(",", ":"))]]
    commands.append([python, "-u", "-m", "mosaic.small_molecule.boltz2.worker", "--help"]
                    if arm_name == "boltz" else
                    [python, "-u", str(source / "scripts/dbmol_worker.py"), "--help"])
    rows = []; imports = None
    for index, command in enumerate(commands):
        process = subprocess.run(command, cwd=source, env=environment, capture_output=True,
                                 text=True, timeout=180, check=False)
        row = {"argv": command, "returncode": process.returncode,
               "stdout": process.stdout, "stderr": process.stderr}
        rows.append(row)
        require(process.returncode == 0, "Original worker premodel startup failed: " + repr(row))
        if index == 0:
            imports = json.loads(process.stdout)
            require(imports.get("status") == "passed_exact_original_imports"
                    and imports.get("modules") == expected, "Actual worker scientific import origin differs")
    require("--context-sha256" in rows[1]["stdout"] and "--checkpoint" in rows[1]["stdout"],
            "Actual original worker help did not execute")
    return {"version": "original-worker-premodel-startup-v1", "status": "passed_metadata_only",
            "arm": arm_name, "worker_python": python, "pythonpath": environment["PYTHONPATH"],
            "mosaic_origin": imports["modules"]["mosaic"]["path"],
            "exact_original_modules": imports["modules"], "commands": rows,
            "original_scientific_source_unchanged": True, "numerical_setup_entered": False,
            "model_constructed": False, "updates": 0, "scientific_readiness_passed": False}


def _json_safe(value):
    return json.loads(json.dumps(value, allow_nan=False))


def restore_point_placement(values, template, graph_type, device_put):
    """Preserve each authenticated original leaf's execution device exactly."""
    return graph_type(device_put(values["X"], template.X.device),
                      device_put(values["E_upper"], template.E_upper.device))


class EndpointHooks:
    def __init__(self, arm, arm_name, output, offset):
        self.arm, self.name, self.output, self.offset = arm, arm_name, Path(output), offset
        self.records = []
        self.transforms = []

    def _scope(self, directory):
        directory = Path(directory)
        require(directory.parent == self.output and directory.name in ("native_gradient_checks", "terminal_gradient_checks"),
                "Endpoint checker output escapes entry/terminal scope")
        return directory.name

    def _plan(self, graph, directory):
        import numpy as np
        import admission
        stage = self._scope(directory)
        historical = stage == "native_gradient_checks" and self.offset == self.arm["start_global"]
        original = self.arm["original_directions"] if historical else {"X": [], "E_upper": []}
        # DBMol original directions are component-specific.
        return {"stage": stage, "values": {"X": np.asarray(graph.X), "E_upper": np.asarray(graph.E_upper)},
                "original": original, "scope": "saved_failed_endpoint" if historical
                else "not_preexisting_at_new_endpoint"}

    def _new_plan(self, point, objective=None):
        import admission
        original = point["original"]
        if objective is not None and objective in original:
            original = original[objective]
        scope = point["scope"]
        if not any(original.get(field, []) for field in ("X", "E_upper")):
            scope = "original_probe_not_reached" if scope == "saved_failed_endpoint" else scope
        return admission.plan(point["values"], original,
                              original_direction_scope=scope)

    def _callbacks(self, loss, key, directory, frozen, template):
        import jax
        import numpy as np
        from mosaic.small_molecule.state import RelaxedGraph
        from mosaic.small_molecule.optimizers import evaluate_graph
        from mosaic.small_molecule.gradient_checks import _value, _save_evaluation
        counters = {"primal": 0, "ad": 0}
        write(directory / "raw-point-placement.json", {field: {
            "device": str(getattr(template, field).device), "dtype": str(getattr(template, field).dtype),
            "shape": list(getattr(template, field).shape),
            "raw_sha256": hashlib.sha256(np.asarray(getattr(template, field)).tobytes()).hexdigest(),
        } for field in ("X", "E_upper")})
        def graph(values):
            return restore_point_placement(values, template, RelaxedGraph, jax.device_put)
        def primal(values):
            number = counters["primal"]; counters["primal"] += 1
            value = _value(loss, graph(values), key, directory / f"primal-{number:04d}.npz", compact=True, logits=False)
            return np.float64(value) if self.name == "dbmol" else np.float32(value)
        def ad(values):
            number = counters["ad"]; counters["ad"] += 1
            value = evaluate_graph(loss, graph(values), key)
            _save_evaluation(directory / f"autodiff-{number:04d}.npz", value, compact=True, logits=False)
            return np.asarray(value.value), {"X": np.asarray(value.gradients.X), "E_upper": np.asarray(value.gradients.E_upper)}
        return primal, ad, frozen

    def _frozen(self, loss, worker=None, directory=None):
        def worker_identity(producer):
            count = 0
            def identity():
                nonlocal count
                raw = _json_safe(producer.request("verify_identity"))
                if directory is not None:
                    write(Path(directory) / f"frozen-{count:04d}.json", raw)
                count += 1
                if self.name == "dbmol":
                    from dbmol_protocol import frozen
                    frozen(raw, producer.identity)
                    keys = ("model_digest", "affinity_model_digest", "structural_model_digest", "context_digest",
                            "full_context_digest", "context_file_sha256", "checkpoint_sha256", "structural_checkpoint_sha256")
                else:
                    from campaign25_contract import verify_frozen_response
                    verify_frozen_response(raw, producer.identity["model_digest"], producer.identity["context_digest"])
                    keys = ("model_digest", "context_digest")
                # Request ID and measured elapsed_seconds are transport metadata,
                # preserved above; they are not frozen model/context state.
                return {key: raw[key] for key in keys}
            return identity
        if worker is not None:
            return worker_identity(worker)
        if self.name == "protenix":
            from mosaic.small_molecule.gradient_checks import array_digest
            return lambda: {"model_digest": array_digest(loss.model), "context_digest": array_digest(loss.context)}
        # The authenticated original driver's live local worker remains the one
        # and only frozen native producer; do not search arbitrary closures.
        frame = inspect.currentframe()
        expected = str(BASE / SOURCES[self.name][0] / SOURCES[self.name][3])
        while frame is not None:
            if frame.f_code.co_filename == expected and frame.f_code.co_name == "main":
                found = frame.f_locals.get("worker")
                require(found is not None, "Live original native worker unavailable")
                return worker_identity(found)
            frame = frame.f_back
        raise ValueError("Could not bind the authenticated original driver worker")

    def _replacement(self, point, prospective, legacy_code, original_function, policy_dir, callbacks):
        def replace(context):
            import admission
            import numpy as np
            # Preserve the old finite, nonzero maximum-contrast requirement.
            if "evaluation" in context:
                gradients = context["evaluation"].gradients
                pairs = (("X", gradients.X), ("E_upper", gradients.E_upper))
            elif "raw" in context:
                gradients = context["raw"].gradients
                pairs = (("X", gradients.X), ("E_upper", gradients.E_upper))
            else:
                pairs = (("X", context["gx"]), ("E_upper", context["ge"]))
            for field, values in pairs:
                array = np.asarray(values)
                if "gx" in context:
                    expected = np.asarray(context["x" if field == "X" else "e"])
                    require(array.dtype == np.float32 and array.shape == expected.shape,
                            "Original DBMol raw FP32 gradient shape/type changed: " + field)
                require(np.isfinite(array).all() and float((array.max(-1) - array.min(-1)).max()) > 1e-7,
                        "Original finite/nonzero raw contrast failed: " + field)
            result = admission.evaluate(prospective, *callbacks,
                                        seal_callback=lambda seal: write(policy_dir / "pair-seal.json", seal))
            write(policy_dir / "admission.json", result)
            record = {"stage": point["stage"], "path": str(policy_dir / "admission.json"),
                      "status": result["status"], "can_admit_new_child": result["can_admit_new_child"]}
            self.records.append(record)
            # Keep original coarse checks as separate observations, after the
            # complete prospective plan/pair commitment and evaluations. They
            # can fail without being relabeled or used for favorable selection.
            legacy = {"status": "running", "directions": [], "rows": []}
            local = dict(context); local["report"] = legacy
            legacy_dir = policy_dir / "original-coarse"; legacy_dir.mkdir()
            if "directory" in local:
                local["directory"] = legacy_dir
            try:
                exec(legacy_code, original_function.__globals__, local)
                legacy["status"] = "passed_original_coarse_observation_only"
            except Exception as error:
                legacy.update(status="failed_original_coarse_observation_preserved", error=repr(error))
            write(policy_dir / "original-coarse.json", legacy)
            require(result["status"] == "passed" and result["can_admit_new_child"] is True,
                    "Prospective endpoint admission failed; original coarse observations preserved")
            return {"endpoint_policy": admission.VERSION, "prospective_endpoint_admission": result,
                    "original_coarse_observation": legacy, "original_failed_run_preserved": True,
                    "original_failed_parent_recertified": False,
                    "original_raw_fd_gate_replaced_for_new_child_only": True}
        return replace

    def simple(self, original, *, kind):
        def wrapped(loss, graph, *args, **kwargs):
            key = kwargs.get("key", args[0] if args else None)
            directory = kwargs.get("output_directory", kwargs.get("directory", args[1] if len(args) > 1 else None))
            require(key is not None and directory is not None, "Original derivative call signature differs")
            point = self._plan(graph, directory)
            policy_dir = Path(directory).with_name(Path(directory).name + "-prospective")
            policy_dir.mkdir(parents=True, exist_ok=False)
            prospective = self._new_plan(point)
            write(policy_dir / "plan.json", prospective)
            frozen = self._frozen(loss, directory=policy_dir)
            callbacks = self._callbacks(loss, key, policy_dir, frozen, graph)
            holder = {}
            def replacement(context):
                return self._replacement(point, prospective, holder["legacy"], original,
                                         policy_dir, callbacks)(context)
            adapted, holder["legacy"] = replace_raw_loop(original, replacement, kind=kind)
            self.transforms.append({"function": original.__name__, "kind": kind,
                                    "change": "only_original_raw_FD_for_loop", "source": inspect.getsourcefile(original)})
            result = adapted(loss, graph, *args, **kwargs)
            if "finite_difference_policy" in result:
                result["legacy_finite_difference_policy_not_acceptance"] = result.pop("finite_difference_policy")
            result["status_semantics"] = "new_prospective_endpoint_policy_and_unchanged_nonraw_checks_only"
            # This report belongs solely to this new child attempt. The original
            # historical checker/source/report is never edited.
            (Path(directory) / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8")
            return result
        return wrapped

    def dbmol(self, module, original):
        """Retain native oracle/component transport/recomposition/null/frozen code."""
        def wrapped(worker, graph, key, directory, *, hard_graph, make_loss):
            point = self._plan(graph, directory)
            # Construct every component's full point inventory before the original
            # native validator performs any component scalar/AD call.
            plans = {name: self._new_plan(point, name) for name in module.COMPONENTS}
            prospective_root = Path(directory).with_name(Path(directory).name + "-prospective")
            prospective_root.mkdir(parents=True, exist_ok=False)
            for name, planned in plans.items():
                sub = prospective_root / name; sub.mkdir()
                write(sub / "plan.json", planned)
            outer = ast.parse(textwrap.dedent(inspect.getsource(original)))
            changed = {"policy": 0, "flag": 0}
            class HonestPolicy(ast.NodeTransformer):
                def visit_Constant(self, node):
                    if node.value == "dbmol-native-raw-fd-v1":
                        changed["policy"] += 1
                        return ast.copy_location(ast.Constant("dbmol-prospective-endpoint-v1"), node)
                    return node
                def visit_keyword(self, node):
                    if node.arg == "all_components_and_total_raw_fd_passed":
                        changed["flag"] += 1
                        node.arg = "all_components_and_total_endpoint_policy_passed"
                    return self.generic_visit(node)
            outer = HonestPolicy().visit(outer); ast.fix_missing_locations(outer)
            require(changed == {"policy": 1, "flag": 1}, "DBMol validator policy shape changed")
            def raw(primal, x, e, gx, ge, *, objective):
                sub = prospective_root / objective
                callbacks = self._callbacks(make_loss(objective), key, sub, self._frozen(None, worker, sub), graph)
                # Independent native primal retains the original caller's exact
                # objective/graph identity checks; save its typed scalar as FP64.
                primal_count = 0
                def independent(values):
                    nonlocal primal_count
                    import numpy as np
                    value = np.float64(primal(values["X"], values["E_upper"]))
                    np.savez_compressed(sub / f"primal-{primal_count:04d}.npz",
                                        X=values["X"], E_upper=values["E_upper"], loss=value)
                    primal_count += 1
                    return value
                callbacks = (independent, callbacks[1], callbacks[2])
                original_raw = module.check_raw_finite_differences
                holder = {}
                def replacement(context):
                    result = self._replacement(point, plans[objective], holder["legacy"], original_raw,
                                               sub, callbacks)(context)
                    result["objective"] = objective
                    return result
                adapted, holder["legacy"] = replace_raw_loop(original_raw, replacement, kind="dbmol")
                result = adapted(primal, x, e, gx, ge, objective=objective)
                result["legacy_raw_policy_not_acceptance"] = {key: result.pop(key)
                    for key in ("epsilons", "relative_tolerance", "required_adjacent_passes")}
                result["status_semantics"] = "new_prospective_endpoint_policy_and_unchanged_nonraw_checks_only"
                return result
            namespace = {}; globals_copy = dict(original.__globals__)
            globals_copy["check_raw_finite_differences"] = raw
            exec(compile(outer, original.__code__.co_filename + "[prospective-policy-only]", "exec"), globals_copy, namespace)
            result = namespace[original.__name__](worker, graph, key, directory, hard_graph=hard_graph, make_loss=make_loss)
            require(result["policy"] == "dbmol-prospective-endpoint-v1"
                    and "all_components_and_total_raw_fd_passed" not in result,
                    "Legacy FD success must not be forged")
            return result
        return wrapped


def verify_dbmol_native_report(report):
    import admission
    require(report["status"] == "passed" and report["policy"] == "dbmol-prospective-endpoint-v1"
            and report["all_components_and_total_endpoint_policy_passed"] is True
            and "all_components_and_total_raw_fd_passed" not in report
            and report["hard_native_parity"]["status"] == "passed"
            and report["frozen_unchanged"] is True and report["key_and_logit_row_shift_nulls"] == "passed"
            and report["live_primitive_recomposition"] == "passed_exact_scalar_and_raw_gradients",
            "Prospective DBMol complete native checks failed")
    expected = {"dbmol_affinity_loss", "dbmol_contact_loss", "dbmol_confidence_loss", "dbmol_total_loss"}
    require(set(report["components"]) == expected, "Missing native components")
    for row in report["components"].values():
        require(row["status"] == "passed" and row["typed_bond_contrast_nonzero"] is True
                and row["production_jax_transport"] == "passed" and row["endpoint_policy"] == admission.VERSION
                and row["independent_primal"] is True and row["raw_unclipped_gradients"] is True
                and row["prospective_endpoint_admission"]["status"] == "passed"
                and row["prospective_endpoint_admission"]["can_admit_new_child"] is True,
                "New component admission/typed/transport check failed")
    return report


def adapt_dbmol_readiness(original):
    """Replace only the old raw-FD flag/predicate with explicit new-policy checks."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    removed = []
    class RawPredicate(ast.NodeTransformer):
        def visit_BoolOp(self, node):
            keep = []
            for value in node.values:
                if "all_components_and_total_raw_fd_passed" in ast.unparse(value):
                    removed.append(ast.dump(value))
                else:
                    keep.append(value)
            node.values = keep
            return self.generic_visit(node)
    tree = RawPredicate().visit(tree); ast.fix_missing_locations(tree)
    require(len(removed) == 1, "Original readiness raw-FD predicate shape changed")
    namespace = {}; globals_copy = dict(original.__globals__)
    globals_copy["verify_native_report"] = verify_dbmol_native_report
    exec(compile(tree, original.__code__.co_filename + "[versioned-raw-readiness-only]", "exec"), globals_copy, namespace)
    return namespace[original.__name__]


@contextmanager
def install_hooks(state):
    patched = []
    try:
        if state.name == "dbmol":
            protocol = importlib.import_module("dbmol_protocol")
            original = protocol.verify_readiness
            protocol.verify_readiness = adapt_dbmol_readiness(original)
            patched.append((protocol, "verify_readiness", original))
            module = importlib.import_module("dbmol_native_validation")
            original = module.validate_native_components
            module.validate_native_components = state.dbmol(module, original)
            patched.append((module, "validate_native_components", original))
        else:
            name = "mosaic.small_molecule.gradient_checks" if state.name == "protenix" else "mosaic.small_molecule.boltz2.checks"
            attr = "check_graph_gradients" if state.name == "protenix" else "check_affinity_gradients"
            module = importlib.import_module(name); original = getattr(module, attr)
            setattr(module, attr, state.simple(original, kind=state.name))
            patched.append((module, attr, original))
        yield
    finally:
        for module, attr, original in reversed(patched):
            setattr(module, attr, original)


def _boundary(snapshot, initialization):
    import numpy as np
    with np.load(snapshot, allow_pickle=False) as old, np.load(initialization, allow_pickle=False) as new:
        for key in ("X", "E_upper", "atom_logits", "bond_logits", "key_data"):
            require(old[key].dtype == new[key].dtype and np.array_equal(old[key], new[key]), "Exact boundary replay differs: " + key)
    return True


def endpoint_policy_version():
    policy = importlib.import_module('admission')
    require(Path(policy.__file__).resolve() == Path(__file__).with_name('admission.py').resolve(),
            'Endpoint policy was imported outside the authenticated driver capsule')
    require(policy.VERSION == 'prospective-endpoint-admission-v2', 'Undeclared prospective endpoint policy')
    return policy.VERSION


def assessment(state, plan, pin, phase, offset, output, snapshot, ready, error=None):
    """New certificate schema; no original completion certificate is produced."""
    arm = state.arm; summary = read(output / "summary.json")
    validation = summary if phase == "validate" else read(Path(ready["path"]).with_name("summary.json"))
    entry = summary.get("native_gradient_checks", {})
    terminal = summary.get("terminal_gradient_checks", {})
    require(summary.get("source_manifest_sha256") == arm["source"]["manifest_sha256"]
            and summary.get("config_sha256") == arm["config"]["sha256"]
            and summary.get("job_id") == os.environ["LSB_JOBID"]
            and summary.get("chemistry_enabled") is False, "Original execution source/config/job/scope differs")
    def admission_pass(stage):
        rows = [row for row in state.records if row["stage"] == stage]
        return len(rows) == (4 if state.name == "dbmol" else 1) and all(
            row["status"] == "passed" and row["can_admit_new_child"] is True for row in rows)
    if state.name == "dbmol":
        native = entry.get("all_components_and_total_endpoint_policy_passed") is True
        nulls = entry.get("key_and_logit_row_shift_nulls") == "passed" and entry.get("live_primitive_recomposition") == "passed_exact_scalar_and_raw_gradients"
        before, after = summary.get("native_identity", {}), summary.get("frozen_identity_after", {})
        from dbmol_protocol import frozen as verify_frozen
        verify_frozen(after, before)
        frozen = True
    elif state.name == "protenix":
        native = entry.get("status") == "passed"
        nulls = entry.get("bond_order_null_same_atom_and_existence_inputs") is True and entry.get("bond_order_null_max_output_difference", float("inf")) <= 2e-5
        frozen = summary.get("frozen_identity_after") == {"model_digest": summary.get("model_digest"), "context_digest": summary.get("context_digest")}
    else:
        native = entry.get("status") == "passed" and entry.get("typed_bond_contrast_nonzero") is True
        nulls = native  # Typed bond categories are explicitly NON-null in Boltz.
        from campaign25_contract import verify_frozen_response
        verify_frozen_response(summary.get("frozen_identity_after", {}), summary.get("model_digest"), summary.get("context_digest"))
        frozen = True
    pilot = read(Path(ready["path"]).parent / "early_pilot/report.json") if ready else read(output / "early_pilot/report.json")
    case_ok = phase == "validate"; terminal_snapshot = None
    if phase == "segment" and summary.get("status") == "passed":
        case_dir = output / "case-0"; case = read(case_dir / "report.json")
        from retention import verify_case
        retained = verify_case(case, case_dir)
        require(retained == read(case_dir / "retained-frames.json"), "Stride20 retained index differs")
        for field in ("states", "updates"):
            lines = (case_dir / (field + ".jsonl")).read_bytes().splitlines(keepends=True)
            require(len(lines) == len(case[field]) and all(line.endswith(b"\n") and json.loads(line) == row
                    for line, row in zip(lines, case[field])), "Original full scalar stream differs")
        case_ok = len(case["updates"]) == arm["segment_updates"] and len(case["states"]) == arm["segment_updates"] + 1
        require([row["global_step"] for row in case["states"]] == list(range(offset, offset + arm["segment_updates"] + 1)),
                "Global scalar state range differs")
        terminal_snapshot = binding(case_dir / case["states"][-1]["snapshot"])
    runtime = validation.get("runtime_gate", {})
    checks = {"entry_admission": admission_pass("native_gradient_checks"),
              "hard_parity": summary.get("native_hard_parity", {}).get("status") == "passed",
              "null_controls": bool(nulls), "native_components": bool(native),
              "pilot_early": pilot.get("status") == "passed" and len(pilot.get("updates", [])) == 10,
              "pilot_high": True, "runtime": runtime.get("status") == "passed"
                  and runtime.get("conservative_projected_seconds", float("inf")) < 82800,
              "terminal_admission": phase == "validate" or admission_pass("terminal_gradient_checks"),
              "case_jsonl": case_ok, "fp32_boundary": _boundary(snapshot["path"], output / "initialization.npz")}
    passed = error is None and summary.get("status") == "passed" and frozen and all(checks.values())
    result = {"version": VERSION, "status": "passed" if passed else "failed", "arm": state.name,
              "job_id": os.environ["LSB_JOBID"], "input_snapshot": snapshot,
              "phase": phase, "offset": offset, "global_end": offset + (arm["segment_updates"] if phase == "segment" else 0),
              "source_manifest_sha256": pin, "scientific_source_manifest_sha256": arm["source"]["manifest_sha256"],
              "plan_sha256": plan["_sha256"], "config_sha256": arm["config"]["sha256"],
              "original_results_changed": False, "original_failed_parent_preserved": True,
              "frozen_unchanged": bool(frozen), "checks": checks,
              "applicability": {"pilot_high": "not_applicable_chemistry_disabled",
                                "terminal_admission": "not_applicable_validation_phase" if phase == "validate" else "required",
                                "case_jsonl": "not_applicable_validation_phase" if phase == "validate" else "required",
                                "fp32_boundary": "required",
                                "null_controls": "typed_bonds_required_nonnull" if state.name == "boltz" else "original_null_controls"},
              "endpoint_policy": endpoint_policy_version(), "driver_hook": HOOK_VERSION,
              "readiness_adapter": READINESS_VERSION if state.name == "dbmol" else "original_metadata_checks_with_explicit_new_endpoint_report",
              "endpoint_records": state.records, "original_summary": binding(output / "summary.json"),
              "projections": {"maximum_pilot_update_bytes": validation["storage_projection"]["maximum_pilot_update_bytes"],
                              "bytes_per_update": validation["storage_projection"]["maximum_pilot_update_bytes"]},
              "proofs": [binding(path) for path in sorted(output.rglob("*")) if path.is_file() and not path.is_symlink()]
                  + [binding(output.parent / "endpoint-driver-invocation.json"),
                     binding(output.parent / "postexecution-immutability.json")]}
    if terminal_snapshot: result["terminal_snapshot"] = terminal_snapshot
    if phase == "validate": result["readiness"] = binding(output / "assessment.json")
    if error is not None: result["error"] = repr(error)
    return result


def failed_assessment(state, plan, pin, phase, offset, output, snapshot, error, *, additional_errors=()):
    """Keep the original exception; absence of later checks is not a new cause."""
    names = ("entry_admission", "hard_parity", "null_controls", "native_components",
             "pilot_early", "pilot_high", "runtime", "terminal_admission", "case_jsonl", "fp32_boundary")
    rows = state.records
    needed = 4 if state.name == "dbmol" else 1
    entry = [row for row in rows if row["stage"] == "native_gradient_checks"]
    outcomes = {name: "not_completed" for name in names}
    if entry:
        outcomes["entry_admission"] = "passed" if len(entry) == needed and all(
            row["status"] == "passed" and row["can_admit_new_child"] is True for row in entry) else "failed"
    outcomes["pilot_high"] = "not_applicable_chemistry_disabled"
    if phase == "validate":
        outcomes["terminal_admission"] = outcomes["case_jsonl"] = "not_applicable_validation_phase"
    summary_path = output / "summary.json"; summary = None; proofs = []
    secondary = list(additional_errors)
    try:
        if output.is_dir():
            proofs = [binding(path) for path in sorted(output.rglob("*"))
                      if path.is_file() and not path.is_symlink()]
        summary = read(summary_path) if summary_path.is_file() else None
    except BaseException as metadata_error:
        secondary.append(metadata_error)
    return {"version": VERSION, "status": "failed", "arm": state.name, "phase": phase, "offset": offset,
            "job_id": os.environ["LSB_JOBID"], "input_snapshot": snapshot,
            "source_manifest_sha256": pin, "scientific_source_manifest_sha256": state.arm["source"]["manifest_sha256"],
            "plan_sha256": plan["_sha256"], "config_sha256": state.arm["config"]["sha256"],
            "error": repr(error), "primary_error": repr(error), "primary_error_preserved": True,
            "additional_errors": [repr(item) for item in secondary],
            "original_results_changed": False, "original_failed_parent_preserved": True,
            "checks": {name: outcome == "passed" for name, outcome in outcomes.items()},
            "check_outcomes": outcomes, "frozen_unchanged": None,
            "frozen_check_status": "not_completed_not_a_new_frozen_failure",
            "endpoint_policy": endpoint_policy_version(), "endpoint_records": rows,
            "original_summary": binding(summary_path) if summary is not None else None,
            "original_summary_status": summary.get("status") if summary is not None else None,
            "proofs": proofs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "output", "snapshot"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("plan-sha256", "manifest", "snapshot-sha256", "arm", "phase"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--offset", type=int, required=True)
    parser.add_argument("--ready", type=Path); parser.add_argument("--ready-sha256")
    parser.add_argument("--startup-only", action="store_true")
    parser.add_argument("--worker-startup-only", action="store_true")
    parser.add_argument("--startup-defer-ready", action="store_true")
    args = parser.parse_args()
    require(not args.worker_startup_only or (args.phase == "validate" and not args.startup_only
            and not args.startup_defer_ready), "Worker probe is a distinct validation-only premodel phase")
    require(not args.startup_defer_ready or (args.startup_only and args.phase == "segment"),
            "Readiness deferral is restricted to CPU segment metadata sentinel")
    require(os.environ.get("LSB_JOBID"), "Allocated DTU HPC required")
    require(sha(args.plan) == args.plan_sha256, "Plan changed")
    plan = read(args.plan); plan["_sha256"] = args.plan_sha256
    require(args.arm in plan.get("dispatch_arms", []), "This arm is blocked in the declared continuation plan")
    own = Path(__file__).resolve().parents[1]
    verify_source(own, args.manifest)
    arm = arm_record(plan, args.arm); source = Path(arm["source"]["path"])
    verify_source(source, arm["source"]["manifest_sha256"], arm["source"]["files"])
    verify_source(RETENTION, RETENTION_SHA, 12)
    bound(arm["config"])
    invocation = read(bound(arm["inputs"]["original_driver_invocation"]))
    require(invocation.get("args") == arm["original_argv"] == arm["inputs"]["original_driver_argv"],
            "Actual original invocation/declared argv differs")
    for row in arm["inputs"].values():
        if isinstance(row, dict) and set(row) == {"path", "bytes", "sha256"}:
            bound(row)
    snapshot = binding(args.snapshot); require(snapshot["sha256"] == args.snapshot_sha256, "Snapshot changed")
    if args.offset == arm["start_global"]:
        require(snapshot == arm["parent_snapshot"], "First continuation must use its exact failed parent endpoint")
    ready = binding(args.ready) if args.ready else None
    require((ready is None and args.ready_sha256 is None) or ready["sha256"] == args.ready_sha256, "Readiness changed")
    argv = build_original_argv(plan, args.arm, args.phase, args.offset, args.output, snapshot, ready)
    require(not args.output.exists(), "Existing/unknown scientific output must not be replayed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.worker_startup_only:
        expected = Path(plan["root"]) / "cpu/payload" / ("worker-startup-" + args.arm) / "study"
        require(args.output == expected, "Worker probe outside declared allocated CPU scope")
        write(args.output.parent / "worker-startup.json", worker_startup(source, args.arm, argv))
        return
    if args.startup_only:
        if args.startup_defer_ready:
            require(ready == arm["inputs"]["ready"], "Deferred CPU sentinel must bind exact historical ready bytes")
            require(args.output == Path(plan["root"]) / "cpu/payload" / ("startup-" + args.arm + "-segment") / "study",
                    "Only declared CPU segment sentinel may defer readiness")
        elif args.phase == "segment":
            prospective_ready(args.ready, arm, args.arm, args.offset, snapshot, args.manifest, args.plan_sha256)
        with execution_context(source, argv):
            if args.arm == "dbmol" and not args.startup_defer_ready:
                protocol = importlib.import_module("dbmol_protocol")
                protocol.verify_readiness = adapt_dbmol_readiness(protocol.verify_readiness)
            namespace = {"__name__": "endpoint_startup_original", "__file__": argv[1],
                         "PremodelBoundaryReached": PremodelBoundaryReached}
            exec(compile(startup_tree(Path(argv[1]).read_bytes(), argv[1], defer_ready=args.startup_defer_ready), argv[1], "exec"), namespace)
            try:
                namespace["main"]()
            except PremodelBoundaryReached as error:
                record = {"version": HOOK_VERSION, "status": "passed_metadata_only",
                          "original_argv": argv, "job_id": os.environ["LSB_JOBID"], "boundary": str(error),
                          "numerical_setup_entered": False, "updates": 0, "scientific_readiness_passed": False,
                          "readiness_deferred": args.startup_defer_ready,
                          "bound_historical_ready_not_evaluated": ready if args.startup_defer_ready else None}
                write(args.output.parent / "endpoint-startup.json", record)
                return
        raise ValueError("Actual startup did not reach its intended pre-model boundary")
    write(args.output.parent / "endpoint-driver-invocation.json", {"argv": argv, "hook": HOOK_VERSION,
          "plan_sha256": args.plan_sha256, "capsule_manifest_sha256": args.manifest,
          "original_source_unchanged": True, "new_endpoint_policy": endpoint_policy_version()})
    if args.phase == "segment":
        prospective_ready(args.ready, arm, args.arm, args.offset, snapshot, args.manifest, args.plan_sha256)
    state = EndpointHooks(arm, args.arm, args.output, args.offset)
    error = None
    additional_errors = []
    result = {"version": VERSION, "status": "failed", "error": "Original driver produced no saved summary",
              "job_id": os.environ["LSB_JOBID"], "original_results_changed": False,
              "original_failed_parent_preserved": True}
    with execution_context(source, argv):
        sys.path.insert(0, str(RETENTION))
        import retention
        import mosaic.small_molecule.chemistry.trajectory as trajectory
        retention.install_trajectory(trajectory, args.output)
        try:
            with install_hooks(state):
                namespace = runpy.run_path(argv[1], run_name="prospective_original_driver")
                namespace["main"]()
        except BaseException as caught:
            error = caught
        finally:
            identity = {"status": "passed", "scope": "source/config/input bytes unchanged after execution",
                        "input_snapshot": snapshot, "original_failed_parent_preserved": True}
            try:
                verify_source(source, arm["source"]["manifest_sha256"], arm["source"]["files"])
                verify_source(RETENTION, RETENTION_SHA, 12)
                verify_source(own, args.manifest)
                bound(arm["config"]); bound(snapshot)
                for row in arm["inputs"].values():
                    if isinstance(row, dict) and set(row) == {"path", "bytes", "sha256"}: bound(row)
            except BaseException as caught:
                identity.update(status="failed", error=repr(caught))
                if error is None:
                    error = caught
                else:
                    additional_errors.append(caught)
            write(args.output.parent / "postexecution-immutability.json", identity)
        if error is not None:
            result = failed_assessment(state, plan, args.manifest, args.phase, args.offset,
                                       args.output, snapshot, error, additional_errors=additional_errors)
        elif (args.output / "summary.json").is_file():
            try:
                result = assessment(state, plan, args.manifest, args.phase, args.offset,
                                    args.output, snapshot, ready, error)
            except BaseException as caught:
                result = failed_assessment(state, plan, args.manifest, args.phase, args.offset,
                                           args.output, snapshot, caught)
        write(args.output.parent / "assessment.json", result)
        if error is not None:
            raise error
        require(result["status"] == "passed", "New continuation checks failed")


if __name__ == "__main__":
    main()
