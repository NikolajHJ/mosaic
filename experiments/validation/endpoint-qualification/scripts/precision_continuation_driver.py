"""Actual original continuation entry point with explicit reference policies.

This account-bound adapter authenticates a byte-identical inherited driver and
changes only its private endpoint-hook and metadata-verification namespace.
Original production execution, optimizer, scientific argv, boundary replay,
nonraw checks, pilots, retention and error preservation remain in that driver.
Numerical code is imported only by its real allocated execution path.
"""
from contextlib import contextmanager
from copy import deepcopy
import ast
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys
import textwrap
from types import SimpleNamespace


ADAPTER_VERSION = "precision-reference-continuation-adapter-v1"
RESULT_VERSION = "precision-reference-continuation-result-v1"
POLICY_VERSION = "precision-reference-continuation-policy-v1"
HOOK_VERSION = "precision-reference-continuation-hook-v1"
READINESS_VERSION = "precision-reference-readiness-adapter-v1"
RAW_POLICY = "prospective-endpoint-admission-v3"
PRO_POLICY = "protenix-separate-precision-admission-v1"
DB_POLICY = "dbmol-separate-precision-finer-admission-v1"
DB_REPORT_POLICY = "dbmol-mixed-precision-endpoint-v1"
POLICIES = {
    "protenix": {"protenix_contact_loss": PRO_POLICY},
    "dbmol": {"dbmol_affinity_loss": DB_POLICY, "dbmol_contact_loss": RAW_POLICY,
              "dbmol_confidence_loss": RAW_POLICY, "dbmol_total_loss": RAW_POLICY},
}
BOUNDARIES = {"protenix": (20000, 50000, 2500, list(range(20000, 50000, 2500))),
              "dbmol": (7500, 10000, 1250, [7500, 8750])}
SEGMENT_POLICY = "precision-bounded-continuation-segments-v1"
CHECKS = {"entry_admission", "hard_parity", "null_controls", "native_components",
          "pilot_early", "pilot_high", "runtime", "terminal_admission", "case_jsonl", "fp32_boundary"}
REFERENCE_RUNTIME_SECONDS = 2 * 5400
SEGMENT_INTERNAL_SECONDS = 82200
SEGMENT_WALLTIME_SECONDS = 82800
ORCHESTRATION_SECONDS = 600
SCHEDULER_MARGIN_SECONDS = 600
WALLTIME_QUANTUM_SECONDS = 900


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    def invalid(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_bytes(), parse_constant=invalid)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def bound(row):
    require(isinstance(row, dict) and set(row) == {"path", "bytes", "sha256"}
            and isinstance(row["path"], str) and type(row["bytes"]) is int and row["bytes"] > 0
            and re.fullmatch(r"[0-9a-f]{64}", row["sha256"] or ""), "Malformed exact proof binding")
    path = Path(row["path"])
    require(path.is_absolute() and path.is_file() and not path.is_symlink()
            and ".." not in path.parts and all(not parent.is_symlink() for parent in path.parents)
            and path.stat().st_size == row["bytes"] and sha(path) == row["sha256"],
            "Exact proof changed or linked: " + row["path"])
    return path


def check_child_proof(row, output):
    """Keep an honestly empty owned reference stderr, never an empty result."""
    if isinstance(row, dict) and type(row.get("bytes")) is int and row["bytes"] == 0:
        require(set(row) == {"path", "bytes", "sha256"} and isinstance(row["path"], str),
                "Malformed empty raw-log binding")
        path = Path(row["path"])
        require(path.is_absolute() and path.name == "stderr.log" and path.parent.name == "reference-session"
                and path.is_relative_to(Path(output)) and path.is_file() and not path.is_symlink()
                and all(not parent.is_symlink() for parent in path.parents)
                and path.stat().st_size == 0 and row["sha256"] == hashlib.sha256(b"").hexdigest()
                and sha(path) == row["sha256"], "Only the actual empty owned reference stderr may have zero bytes")
        return path
    return bound(row)


def runtime_budget(validation_summary, arm):
    """Derive an upward-rounded allocation from a complete, passed pilot.

    Only the measured per-update Protenix term is shortened. Its original
    1,800-second reserve and DBMol's measured startup/entry/terminal terms are
    retained in full, in addition to both bounded precision references.
    """
    require(arm in BOUNDARIES, "Unknown bounded runtime arm")
    path = Path(validation_summary)
    runtime = read(path).get("runtime_gate", {})
    original = runtime.get("conservative_projected_seconds")
    require(runtime.get("status") == "passed" and type(original) in (int, float)
            and math.isfinite(original) and original > 0, "Original validated conservative runtime missing")
    if arm == "protenix":
        means = runtime.get("mean_update_seconds")
        require(isinstance(means, list) and means and all(type(value) in (int, float)
                and math.isfinite(value) and value > 0 for value in means), "Complete measured Protenix pilot runtime missing")
        require(math.isclose(original, 1.5 * max(means) * 10000 + 1800, rel_tol=1e-12, abs_tol=1e-9),
                "Original Protenix runtime formula differs")
        projected = 1.5 * max(means) * BOUNDARIES[arm][2] + 1800
    else:
        keys = ("mean_update_seconds", "observed_startup_entry_check_seconds", "observed_native_check_seconds")
        require(all(type(runtime.get(key)) in (int, float) and math.isfinite(runtime[key])
                and runtime[key] > 0 for key in keys)
                and runtime.get("segment_updates") == BOUNDARIES[arm][2]
                and runtime.get("terminal_check_safety_factor") == 1.5
                and runtime.get("update_safety_factor") == 1.5 and runtime.get("reserve_seconds") == 1800,
                "Complete measured DBMol segment/check runtime missing")
        expected = (1.5 * runtime["mean_update_seconds"] * BOUNDARIES[arm][2]
                    + runtime["observed_startup_entry_check_seconds"]
                    + 1.5 * runtime["observed_native_check_seconds"] + 1800)
        require(math.isclose(original, expected, rel_tol=1e-12, abs_tol=1e-9), "Original DBMol runtime formula differs")
        projected = original
    # The inherited full-pilot guard remains mandatory even for shorter jobs.
    require(original < SEGMENT_WALLTIME_SECONDS, "Original validated runtime guard failed")
    required = projected + REFERENCE_RUNTIME_SECONDS + ORCHESTRATION_SECONDS
    walltime = math.ceil((required + SCHEDULER_MARGIN_SECONDS) / WALLTIME_QUANTUM_SECONDS) * WALLTIME_QUANTUM_SECONDS
    internal = walltime - SCHEDULER_MARGIN_SECONDS
    require(walltime <= SEGMENT_WALLTIME_SECONDS and required <= internal <= SEGMENT_INTERNAL_SECONDS,
            "Measured segment plus both bounded reference checks exceeds runtime")
    return {"version": SEGMENT_POLICY, "status": "passed", "arm": arm,
            "segment_updates": BOUNDARIES[arm][2], "original_projected_seconds": original,
            "bounded_segment_projected_seconds": projected,
            "reference_seconds": REFERENCE_RUNTIME_SECONDS, "required_total_seconds": required,
            "orchestration_seconds": ORCHESTRATION_SECONDS, "scheduler_margin_seconds": SCHEDULER_MARGIN_SECONDS,
            "walltime_quantum_seconds": WALLTIME_QUANTUM_SECONDS, "walltime_seconds": walltime,
            "walltime": f"{walltime // 3600:02d}:{walltime % 3600 // 60:02d}",
            "limit_seconds": internal, "original_runtime_guard_changed": False,
            "validation_summary": {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}}


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "Authenticated adapter module unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _replace_expression(tree, old, new, expected=1):
    """Replace one declared metadata expression, failing on source drift."""
    original = ast.dump(ast.parse(old, mode="eval").body)
    replacement = ast.parse(new, mode="eval").body
    found = 0
    class Replace(ast.NodeTransformer):
        def visit(self, node):
            nonlocal found
            if isinstance(node, ast.expr) and ast.dump(node) == original:
                found += 1
                return ast.copy_location(deepcopy(replacement), node)
            return super().visit(node)
    result = Replace().visit(deepcopy(tree))
    require(found == expected, "Declared short-segment metadata AST differs: " + old)
    ast.fix_missing_locations(result)
    return result


def short_segment_driver_tree(raw, filename, arm):
    """Preserve original computation; change only its completion-count guard."""
    require(arm in BOUNDARIES, "Unknown short-segment driver")
    tree = ast.parse(raw, filename=filename)
    mains = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"]
    require(len(mains) == 1, "Original short-segment main shape differs")
    if arm == "protenix":
        old = 'result["status"] != "passed" or len(result["updates"]) != 10000 or [r["step"] for r in result["states"]] != list(range(10001))'
        new = 'result["status"] != "passed" or len(result["updates"]) != args.segment_steps or [r["step"] for r in result["states"]] != list(range(args.segment_steps + 1))'
        tree = _replace_expression(tree, old, new)
    else:
        # DBMol already uses the actual segment argument for every completion
        # count. Require that exact guard rather than weakening or duplicating it.
        expected = ast.dump(ast.parse("case['status'] != 'passed' or len(case['updates']) != a.segment_steps or len(case['states']) != a.segment_steps + 1", mode="eval").body)
        require(sum(isinstance(node, ast.expr) and ast.dump(node) == expected for node in ast.walk(mains[0])) == 1,
                "Original DBMol dynamic completion guard differs")
    return tree


def short_segment_protocol_function(function, arm, name):
    """Compile only the original metadata function's declared range/count edit."""
    require(arm in BOUNDARIES and function.__name__ == name, "Wrong original segment metadata function")
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    require(len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
            and tree.body[0].name == name and not tree.body[0].decorator_list,
            "Original segment metadata function shape differs")
    steps = BOUNDARIES[arm][2]
    if (arm, name) == ("protenix", "segment_scope"):
        tree = _replace_expression(tree, "steps != 10000", f"steps != {steps}")
        tree = _replace_expression(tree, "range(start, 50000, 10000)", f"range(start, 50000, {steps})")
    elif (arm, name) == ("protenix", "require_boundary"):
        tree = _replace_expression(tree, "execution.get('updates') != 10000", f"execution.get('updates') != {steps}")
    elif (arm, name) == ("dbmol", "scope"):
        # Leave protocol.SEGMENT and config25 untouched: the original config
        # still has its original segment metadata and scientific parameters.
        tree = _replace_expression(tree, "range(0, TARGET, SEGMENT)", f"range(0, TARGET, {steps})")
        tree = _replace_expression(tree, "steps == SEGMENT", f"steps == {steps}")
    else:
        raise ValueError("Undeclared segment metadata adaptation")
    namespace = {}
    exec(compile(tree, function.__code__.co_filename + "[bounded-segment-metadata-only]", "exec"),
         dict(function.__globals__), namespace)
    return namespace[name]


@contextmanager
def short_segment_protocol(module, arm):
    names = ("segment_scope", "require_boundary") if arm == "protenix" else ("scope",)
    originals = {name: getattr(module, name) for name in names}
    replacements = {name: short_segment_protocol_function(function, arm, name) for name, function in originals.items()}
    try:
        for name, value in replacements.items():
            setattr(module, name, value)
        yield module
    finally:
        for name, value in originals.items():
            setattr(module, name, value)


@contextmanager
def short_segment_sources(driver):
    """Change only private per-job lengths; preserve every scientific source pin."""
    original = driver.SOURCES
    replacement = dict(original)
    for arm, (_, target, steps, _) in BOUNDARIES.items():
        row = original[arm]
        require(isinstance(row, tuple) and len(row) == 7 and row[6] == target
                and row[5] == (10000 if arm == "protenix" else 2500),
                "Original authenticated source/segment declaration differs")
        replacement[arm] = (*row[:5], steps, row[6])
    try:
        driver.SOURCES = replacement
        yield driver
    finally:
        driver.SOURCES = original


def validate_adapter_plan(plan):
    require(plan.get("precision_adapter_version") == ADAPTER_VERSION
            and plan.get("precision_policy_by_arm") == POLICIES,
            "Separate precision adapter/per-objective policies differ")
    require(plan.get("optimization_settings_changed") is False
            and plan.get("original_failures_preserved") is True
            and plan.get("matrix_stride") == 20 and plan.get("loss_stride") == 1,
            "Production preservation/retention scope differs")
    require(plan.get("dispatch_arms") == ["protenix", "dbmol"]
            and isinstance(plan.get("arms"), list)
            and {row.get("id") for row in plan["arms"]} == set(POLICIES)
            and len(plan["arms"]) == 2, "Only exact Protenix/DBMol arms may dispatch")
    for row in plan["arms"]:
        start, target, segment, offsets = BOUNDARIES[row["id"]]
        require(all(type(row.get(key)) is int for key in ("start_global", "target_global", "segment_updates"))
                and isinstance(row.get("offsets"), list) and all(type(value) is int for value in row["offsets"])
                and row.get("start_global") == start and row.get("target_global") == target
                and row.get("segment_updates") == segment and row.get("offsets") == offsets,
                "Saved boundary, segment or target differs: " + row["id"])
    config = plan.get("reference_configuration")
    require(isinstance(config, dict) and set(config) == {"arms"}
            and isinstance(config["arms"], dict) and set(config["arms"]) == set(POLICIES),
            "Exact separately authenticated reference configuration absent")
    for arm, item in config["arms"].items():
        require(isinstance(item, dict) and isinstance(item.get("command_prefix"), list)
                and item["command_prefix"] and all(isinstance(part, str) for part in item["command_prefix"])
                and isinstance(item.get("environment"), dict)
                and isinstance(item.get("expected_identity"), dict) and item["expected_identity"]
                and all(isinstance(k, str) and isinstance(v, str) for k, v in item["environment"].items()),
                "Reference command/environment configuration malformed")
        require(item.get("historical_global") == BOUNDARIES[arm][0]
                and item.get("policy_kind") == ("protenix-original-fp64" if arm == "protenix" else "dbmol-finer-fp64-affinity"),
                "Historical reference endpoint/policy kind differs")
    return plan


def load_base(plan, *, directory=None):
    """Authenticate historical bytes and the packaged sibling before import."""
    directory = Path(directory) if directory is not None else Path(__file__).resolve().parent
    historical = bound(plan["base_driver"])
    sibling = directory / "continuation_driver.py"
    require(sibling.is_file() and not sibling.is_symlink()
            and sibling.read_bytes() == historical.read_bytes(), "Packaged base driver is not byte-identical to its authenticated parent")
    module = _module(sibling, "_precision_continuation_authenticated_base")
    require(module.VERSION == "endpoint-aware-continuation-result-v1"
            and module.HOOK_VERSION == "prospective-endpoint-driver-hook-v1"
            and callable(module.main) and callable(module.build_original_argv), "Inherited actual-entry driver schema differs")
    return module


def _own_plan():
    return read(Path(__file__).resolve().parents[1] / "plan.json")


def build_original_argv(plan, name, phase, offset, output, snapshot, ready=None):
    """Metadata alias used by actual-command CPU and pre-submission checks."""
    validate_adapter_plan(plan)
    base = load_base(plan)
    with short_segment_sources(base):
        return base.build_original_argv(plan, name, phase, offset, output, snapshot, ready)


def worker_environment(source):
    return load_base(_own_plan()).worker_environment(source)


def worker_startup(source, arm_name, original_argv):
    require(arm_name == "dbmol", "Only DBMol has a separate worker in this capsule")
    return load_base(_own_plan()).worker_startup(source, arm_name, original_argv)


def verify_dbmol_native_report(report):
    require(report.get("status") == "passed" and report.get("policy") == DB_REPORT_POLICY
            and report.get("policy_by_objective") == POLICIES["dbmol"]
            and report.get("all_components_and_total_endpoint_policy_passed") is True
            and "all_components_and_total_raw_fd_passed" not in report
            and report.get("hard_native_parity", {}).get("status") == "passed"
            and report.get("frozen_unchanged") is True
            and report.get("key_and_logit_row_shift_nulls") == "passed"
            and report.get("live_primitive_recomposition") == "passed_exact_scalar_and_raw_gradients",
            "Complete mixed-precision DBMol native/parity/null/frozen report failed")
    components = report.get("components")
    require(isinstance(components, dict) and set(components) == set(POLICIES["dbmol"]), "Missing DBMol native component")
    for name, policy in POLICIES["dbmol"].items():
        row = components[name]
        admission = row.get("prospective_endpoint_admission", {})
        require(row.get("status") == "passed" and row.get("endpoint_policy") == policy
                and row.get("typed_bond_contrast_nonzero") is True
                and row.get("production_jax_transport") == "passed" and row.get("independent_primal") is True
                and row.get("raw_unclipped_gradients") is True
                and admission.get("version") == policy and admission.get("status") == "passed"
                and admission.get("can_admit_new_child") is True and admission.get("frozen_unchanged") is True,
                "DBMol component policy/transport/nonzero/frozen admission incomplete: " + name)
    return report


def _subset(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(key in actual and _subset(actual[key], value)
                                                for key, value in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _subset(value, wanted) for value, wanted in zip(actual, expected))
    return type(actual) is type(expected) and actual == expected


def _raw_reference_points(session, output):
    residency = read(bound(session["residency"]))
    index_binding = residency.get("reference_client_raw_points")
    index_path = bound(index_binding)
    require(index_path.is_relative_to(Path(output)), "Reference raw point inventory outside new output")
    invocation = read(bound(session["invocation"]))
    require(invocation.get("client_raw_point_index_path") == str(index_path),
            "Reference invocation/raw inventory identity differs")
    key_data = invocation.get("key_data")
    require(isinstance(key_data, list) and len(key_data) == 2
            and all(type(value) is int and 0 <= value < 2**32 for value in key_data),
            "Reference raw point inventory has no exact native key")
    index = read(index_path)
    require(index.get("status") == "complete_actual_reference_points"
            and type(index.get("requests")) is int and index["requests"] > 0
            and isinstance(index.get("points"), list) and len(index["points"]) == index["requests"]
            and index.get("optimizer_updates") == 0 and index.get("cached_or_interpolated_points") == 0
            and index.get("input_transport") == "exact_current_FP64_graph_and_actual_native_key"
            and index.get("reference_frozen") == read(bound(session["identity"]))["frozen"],
            "Complete actual FP64 reference raw point inventory missing")
    require([point.get("id") for point in index["points"]] == list(range(index["requests"])),
            "Reference raw request IDs are not contiguous")
    files = [index_binding]
    for point in index["points"]:
        require(type(point.get("id")) is int and type(point.get("gradient")) is bool
                and re.fullmatch(r"[0-9a-f]{64}", point.get("graph_sha256", ""))
                and point.get("key_data") == key_data, "Reference raw point/key metadata differs")
        for field in ("raw", "response"):
            path = bound(point[field])
            require(path.is_relative_to(Path(output)), "Reference raw point file escapes own scientific output")
            files.append(point[field])
        response = read(point["response"]["path"])
        require(type(response.get("id")) is int and response["id"] == point["id"]
                and response.get("graph_sha256") == point["graph_sha256"],
                "Untouched reference response/input identity differs")
    require(len({row["path"] for row in files}) == len(files), "Duplicate actual reference raw point binding")
    return index_binding, files


def _session(record, output, configuration, plan_sha=None, own_pin=None):
    session = record.get("reference_session")
    require(isinstance(session, dict) and set(session) == {"invocation", "identity", "close", "residency"},
            "Independent reference session proof set incomplete")
    values = {}
    for name, proof in session.items():
        path = bound(proof)
        require(path.is_relative_to(Path(output)), "Reference evidence is outside the new actual scientific output")
        values[name] = read(path)
    invocation, identity, close, residency = (values[name] for name in ("invocation", "identity", "close", "residency"))
    command = invocation.get("argv")
    hooks = _module(Path(__file__).with_name("precision_continuation_hooks.py"), "_precision_session_metadata_helpers")
    prefix = hooks.expanded_command(configuration, plan_sha)
    require(isinstance(command, list) and command[:len(prefix)] == prefix
            and invocation.get("environment") == configuration["environment"]
            and invocation.get("optimizer_updates") == 0, "Actual isolated reference invocation differs")
    if configuration["policy_kind"] == "protenix-original-fp64":
        require(len(command) == len(prefix) + 2 and command[-2] == "--output"
                and Path(command[-1]).is_relative_to(Path(output)), "Protenix reference output invocation differs")
        if own_pin is not None:
            require(identity.get("identity", {}).get("own_source_manifest_sha256") == own_pin,
                    "Actual Protenix reference worker belongs to another continuation capsule")
    else:
        require(command == prefix, "DBMol reference invocation differs")
    require(isinstance(identity.get("frozen"), dict) and identity["frozen"]
            and identity.get("optimizer_updates") == 0 and close.get("status") == "closed"
            and close.get("frozen") == identity["frozen"], "Actual reference identity/frozen close handshake failed")
    require(_subset(identity, configuration["expected_identity"]), "Actual reference source/model/context identity differs")
    audit = close.get("audit", {})
    if configuration["policy_kind"] == "protenix-original-fp64":
        precision = identity.get("precision", {})
        require(audit.get("status") == "passed_full_FP64_reference_audit"
                and audit.get("floating_operations_float64") is True and audit.get("narrow_float_avals") == []
                and audit.get("frozen_unchanged") is True and audit.get("original_results_changed") is False
                and audit.get("optimizer_updates") == 0
                and all(precision.get(name) is True for name in ("full_model_float64", "parameters_float64",
                                                               "native_features_float64", "native_operations_float64"))
                and precision.get("final_reduction_only") is False,
                "Full native Protenix FP64 reference audit missing")
    else:
        require(audit.get("status") == "passed_full_operation_FP64_audit"
                and audit.get("narrow_floating_operations") == 0 and audit.get("production_execution_changed") is False,
                "Full native DBMol FP64 operation audit missing")
    require(residency.get("status") == "passed_live_production_reference_co_residency"
            and residency.get("input_FP32_baseline_unchanged") is True
            and residency.get("production_optimizer_updates") == 0
            and residency.get("reference_closed_before_return") is True and residency.get("no_runtime_fallback") is True,
            "Live original production/reference co-residency did not pass")
    for prefix in ("production_measurements", "frozen"):
        before = residency.get(prefix + "_before")
        require(isinstance(before, dict) and before and before == residency.get(prefix + "_during")
                == residency.get(prefix + "_after"), "Production values/frozen identity changed during reference residency")
    _raw_reference_points(session, output)
    return deepcopy(session)


def policy_records(records, arm, output, plan):
    require(arm in POLICIES and isinstance(records, list), "Unknown/missing endpoint policy records")
    normalized = []
    for record in records:
        require(isinstance(record, dict) and record.get("stage") in ("native_gradient_checks", "terminal_gradient_checks"),
                "Endpoint record stage differs")
        path = Path(record.get("path", ""))
        require(path.is_absolute() and path.is_file() and not path.is_symlink()
                and path.is_relative_to(Path(output)), "Endpoint raw report outside its new scientific output")
        report = read(path)
        objective = record.get("objective")
        policy = POLICIES[arm].get(objective)
        require(policy is not None and record.get("endpoint_policy") == policy
                and record.get("status") == report.get("status") == "passed"
                and record.get("can_admit_new_child") is report.get("can_admit_new_child") is True
                and report.get("version") == policy and report.get("frozen_unchanged") is True
                and report.get("can_certify_original_study") is False,
                "Explicit per-objective endpoint policy did not qualify")
        item = {"stage": record["stage"], "objective": objective, "endpoint_policy": policy,
                "report": {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}}
        if policy == RAW_POLICY:
            require(record.get("reference_session") is None
                    and record.get("reference_applicability") == "not_applicable_unchanged_raw_v3",
                    "Unchanged raw component is mislabeled as a reference result")
            item.update(reference_session=None, reference_applicability=record["reference_applicability"])
        else:
            item["reference_session"] = _session(record, output, plan["reference_configuration"]["arms"][arm],
                                                  plan.get("_sha256"), plan.get("_manifest"))
            item["reference_raw_points"] = _raw_reference_points(item["reference_session"], output)[0]
        normalized.append(item)
    return normalized


def verify_precision_result(result, plan, pin, phase, arm, offset, snapshot=None):
    """Metadata-only independent admission for controller/readiness consumers."""
    validate_adapter_plan(plan)
    require(phase in ("validate", "segment") and arm in POLICIES and offset in BOUNDARIES[arm][3],
            "Undeclared precision child phase/boundary")
    rows = [row for row in plan["arms"] if row["id"] == arm]
    require(len(rows) == 1, "Missing exact scientific arm")
    row = rows[0]
    bound(plan["base_driver"])
    plan_sha = plan.get("_sha256") or sha(Path(plan["source_root"]) / "plan.json")
    require(result.get("version") == RESULT_VERSION and result.get("status") == "passed"
            and result.get("arm") == arm and result.get("phase") == phase and result.get("offset") == offset
            and result.get("source_manifest_sha256") == pin and result.get("plan_sha256") == plan_sha
            and result.get("scientific_source_manifest_sha256") == row["source"]["manifest_sha256"]
            and result.get("config_sha256") == row["config"]["sha256"]
            and result.get("global_end") == offset + (row["segment_updates"] if phase == "segment" else 0),
            "Actual precision child identity/source/config/global boundary differs")
    require(result.get("endpoint_policy") == POLICY_VERSION and result.get("driver_hook") == HOOK_VERSION
            and result.get("precision_adapter_version") == ADAPTER_VERSION
            and result.get("readiness_adapter") == READINESS_VERSION
            and result.get("policy_by_objective") == POLICIES[arm]
            and result.get("base_driver") == plan["base_driver"]
            and result.get("original_results_changed") is False and result.get("original_failed_parent_preserved") is True
            and result.get("frozen_unchanged") is True, "Explicit reference policy/preservation scope differs")
    require(isinstance(result.get("checks"), dict) and set(result["checks"]) == CHECKS
            and all(result["checks"][name] is True for name in CHECKS), "Required original/native/pilot/runtime/boundary checks incomplete")
    if snapshot is not None:
        require(result.get("input_snapshot") == snapshot, "Precision child input checkpoint differs")
    summary = read(bound(result["original_summary"]))
    require(summary.get("status") == "passed" and summary.get("job_id") == result.get("job_id"),
            "Actual original-driver summary did not pass")
    output = Path(result["original_summary"]["path"]).parent
    expected_validation = (output / "summary.json" if phase == "validate" else Path(plan["root"]) /
                           (arm + "-validate-" + str(offset).zfill(5)) / "payload/study/summary.json")
    require(result.get("precision_runtime_budget") == runtime_budget(expected_validation, arm),
            "Whole original segment and both reference runtime budgets differ")
    record_plan = {**plan, "_sha256": plan_sha, "_manifest": pin}
    normalized = policy_records(result.get("endpoint_records"), arm, output, record_plan)
    require(result.get("precision_policy_records") == normalized, "Bound precision policy/session records differ")
    stages = ("native_gradient_checks",) if phase == "validate" else ("native_gradient_checks", "terminal_gradient_checks")
    require(len(normalized) == len(stages) * len(POLICIES[arm])
            and {(item["stage"], item["objective"]) for item in normalized}
            == {(stage, objective) for stage in stages for objective in POLICIES[arm]},
            "Complete entry/terminal objective cohort absent/duplicated")
    proofs = result.get("proofs")
    require(isinstance(proofs, list) and proofs, "Native raw proof inventory absent")
    for proof in proofs:
        check_child_proof(proof, output)
    proof_names = [proof["path"] for proof in proofs]
    require(len(proof_names) == len(set(proof_names)), "Duplicate child proof inventory")
    for item in normalized:
        require(item["report"]["path"] in proof_names, "Endpoint report missing from actual child proof inventory")
        for proof in (item.get("reference_session") or {}).values():
            require(proof["path"] in proof_names, "Live reference session proof missing from child inventory")
        if item.get("reference_session"):
            index, raw_files = _raw_reference_points(item["reference_session"], output)
            require(item.get("reference_raw_points") == index, "Reference raw point index binding differs")
            require(all(proof["path"] in proof_names for proof in raw_files),
                    "Actual reference raw point/response inventory is not preserved in child proofs")
    if arm == "dbmol":
        verify_dbmol_native_report(summary["native_gradient_checks"])
        if phase == "segment":
            verify_dbmol_native_report(summary["terminal_gradient_checks"])
    return result


@contextmanager
def adapted_driver(driver, plan, hooks):
    """Restore every private metadata symbol even after startup/native failure."""
    validate_adapter_plan(plan)
    names = ("VERSION", "HOOK_VERSION", "READINESS_VERSION", "endpoint_policy_version",
             "verify_dbmol_native_report", "assessment", "failed_assessment", "startup_tree", "execution_context", "runpy")
    originals = {name: getattr(driver, name) for name in names}
    original_assessment = driver.assessment
    original_failed = driver.failed_assessment
    def source_arm(source):
        matches = [row["id"] for row in plan["arms"] if Path(row["source"]["path"]) == Path(source)]
        require(len(matches) == 1, "Original execution source is not an exact declared arm")
        return matches[0]
    @contextmanager
    def execution_context(source, argv):
        arm = source_arm(source)
        with originals["execution_context"](source, argv):
            protocol = importlib.import_module("extended_protocol" if arm == "protenix" else "dbmol_protocol")
            with short_segment_protocol(protocol, arm):
                yield
    def startup_tree(raw, filename, *, defer_ready=False):
        arm = source_arm(Path(filename).parents[1])
        tree = short_segment_driver_tree(raw, filename, arm)
        return originals["startup_tree"](ast.unparse(tree), filename, defer_ready=defer_ready)
    def run_path(filename, *, run_name):
        require(run_name == "prospective_original_driver", "Undeclared original driver execution")
        arm = source_arm(Path(filename).parents[1])
        tree = short_segment_driver_tree(Path(filename).read_bytes(), filename, arm)
        namespace = {"__name__": run_name, "__file__": str(filename), "__package__": "",
                     "__loader__": None, "__spec__": None, "__cached__": None}
        # The inherited frozen-worker hook authenticates the original main
        # frame by its exact filename. Metadata AST edits must preserve that
        # execution identity as well as __file__ and the source bytes.
        exec(compile(tree, str(filename), "exec"), namespace)
        return namespace
    def assessment(state, current_plan, pin, phase, offset, output, snapshot, ready, error=None):
        value = original_assessment(state, current_plan, pin, phase, offset, output, snapshot, ready, error)
        value.update(precision_adapter_version=ADAPTER_VERSION, readiness_adapter=READINESS_VERSION,
                     policy_by_objective=deepcopy(POLICIES[state.name]),
                     base_driver=deepcopy(plan["base_driver"]),
                     precision_policy_records=policy_records(state.records, state.name, output, plan))
        validated_summary = Path(output) / "summary.json" if phase == "validate" else Path(ready["path"]).with_name("summary.json")
        value["precision_runtime_budget"] = runtime_budget(validated_summary, state.name)
        runtime_proof = value["precision_runtime_budget"]["validation_summary"]
        if runtime_proof["path"] not in {proof["path"] for proof in value["proofs"]}:
            value["proofs"].append(runtime_proof)
        verify_precision_result(value, current_plan, pin, phase, state.name, offset, snapshot)
        return value
    def failed(state, current_plan, pin, phase, offset, output, snapshot, error, **kwargs):
        value = original_failed(state, current_plan, pin, phase, offset, output, snapshot, error, **kwargs)
        value.update(precision_adapter_version=ADAPTER_VERSION, readiness_adapter=READINESS_VERSION,
                     policy_by_objective=deepcopy(POLICIES[state.name]),
                     base_driver=deepcopy(plan["base_driver"]), precision_policy_records=[],
                     precision_child_admission_complete=False)
        return value
    try:
        driver.VERSION = RESULT_VERSION
        driver.HOOK_VERSION = HOOK_VERSION
        driver.READINESS_VERSION = READINESS_VERSION
        driver.endpoint_policy_version = lambda: POLICY_VERSION
        driver.verify_dbmol_native_report = verify_dbmol_native_report
        driver.assessment = assessment
        driver.failed_assessment = failed
        driver.startup_tree = startup_tree
        driver.execution_context = execution_context
        driver.runpy = SimpleNamespace(run_path=run_path)
        with short_segment_sources(driver):
            with hooks.install_precision_hooks(driver, plan, plan["reference_configuration"]):
                yield driver
    finally:
        for name, value in originals.items():
            setattr(driver, name, value)


def _argument(argv, name):
    require(argv.count(name) == 1 and argv.index(name) + 1 < len(argv), "Missing/duplicate actual argument: " + name)
    return argv[argv.index(name) + 1]


def verify_owned_source(directory, pin):
    source = Path(directory)
    require(re.fullmatch(r"[0-9a-f]{64}", pin or "") and sha(source / "source_manifest.json") == pin,
            "Own precision source manifest differs")
    manifest = read(source / "source_manifest.json")
    members = manifest.get("files")
    require(isinstance(members, dict) and members, "Own precision source inventory absent")
    for name, digest in members.items():
        part = PurePosixPath(name)
        require(not part.is_absolute() and ".." not in part.parts and "\\" not in name,
                "Unsafe owned source member")
        path = source.joinpath(*part.parts)
        require(path.is_file() and not path.is_symlink() and sha(path) == digest, "Own precision source member differs")
    require({path.relative_to(source).as_posix() for path in source.rglob("*") if path.is_file()}
            == set(members) | {"source_manifest.json"}, "Own precision source member inventory differs")
    return manifest


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    require(os.environ.get("LSB_JOBID"), "Real allocated DTU job required")
    source = Path(__file__).resolve().parents[1]
    plan_path = Path(_argument(argv, "--plan"))
    plan_sha = _argument(argv, "--plan-sha256")
    pin = _argument(argv, "--manifest")
    manifest = verify_owned_source(source, pin)
    require(plan_path == source / "plan.json" and sha(plan_path) == plan_sha
            and manifest["files"].get("plan.json") == plan_sha, "Actual source-bound plan differs")
    plan = validate_adapter_plan(read(plan_path))
    plan["_sha256"] = plan_sha
    plan["_manifest"] = pin
    require(plan.get("source_root") == str(source), "Actual precision source root differs")
    base = load_base(plan)
    hooks = _module(source / "scripts/precision_continuation_hooks.py", "_precision_continuation_actual_hooks")
    old_argv = sys.argv[:]
    try:
        sys.argv = [str(source / "scripts/continuation_driver.py"), *argv]
        with adapted_driver(base, plan, hooks):
            # Execute the original real parser/metadata path, including actual
            # startup sentinel and original native/production main, verbatim.
            return base.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
