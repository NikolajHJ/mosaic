"""Allocated, exact-state readiness for the opt-in additive derivative gate.

The original source105 failed points, scorer, decoder, occupied slots, pose,
stopped target, scalar objective and native derivatives remain unchanged. Only
the derivative certificate changes to additive_branch_v3. Completed source106
mechanism observations are a required prerequisite, not replacement endpoints
or retrospective approval of the old failures. This module makes no optimizer
updates and invokes no DeFoG worker. Native success is unknown until allocation.

Fresh children bootstrap the authenticated OLD scientific package. New gate
modules are then loaded from adjacent candidate bytes with temporary canonical
module AND package-attribute bindings; all bindings are restored on exceptions.
Readers recompute the new gate with an externally derived scientific profile.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from copy import deepcopy
from dataclasses import replace
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys


POLICY = "additive_branch_v3"
ARITHMETIC = "contact_head_and_assembly_float64_v1"
CASE_TIMEOUT_SECONDS = 5400
MAXIMUM_FULL_FORWARDS_PER_CASE = 800
SOURCE105_ARCHIVE = "26d81fc50e9ca53144ab412a4f23a217a0c46c8b3e31d196ddaa4390cc0b836e"
SOURCE105_MANIFEST = "cd4632d3e644a0df1d57a4016bd18b6b94658eaf3822c251f28f0acd45674c79"
SOURCE106_NAME = "source-106-trajectory-additive-diagnostic-attempt001"
SOURCE106_ARCHIVE = "f97785eb930d92b9142c6063321bcd4c6661bbb7ead9e21b5d656a4ed6c8ed65"
SOURCE106_MANIFEST = "96b5ec7ba7bd7c48bd0c8d526498473f712e2f788b4ed9dc824d411a844e5dd7"
CASE_IDS = ("boltz_only", "boltz_defog")
SOURCE_PATHS = ("src/latentmol/boltz/additive_fd.py", "src/latentmol/boltz/additive_contract.py",
                "src/latentmol/boltz/additive_branch_fd.py", "src/latentmol/boltz/additive_readiness.py",
                "src/latentmol/boltz/failed_gate_attribution.py",
                "src/latentmol/repair_guidance/integration_smoke.py")
_HELPER = None
_CONFIG_IDENTITY = None
_PLAN = _CASE = _REFERENCE = None


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def helper():
    """Load stdlib-only authentication before any historical package import."""
    global _HELPER
    if _HELPER is None:
        _HELPER = _load(Path(__file__).with_name("failed_gate_attribution.py"), "_additive_readiness_replay")
        _HELPER.CASE_TIMEOUT_SECONDS, _HELPER.REPEATS = CASE_TIMEOUT_SECONDS, 1
        _HELPER.CLAIM_LIMIT = __doc__
    return _HELPER


def config_identity():
    """The canonical JSON reader is stdlib-only; keep fresh-child bootstrap clean."""
    global _CONFIG_IDENTITY
    if _CONFIG_IDENTITY is None:
        _CONFIG_IDENTITY = _load(Path(__file__).parents[1] / "config_identity.py", "_additive_readiness_config_identity")
    return _CONFIG_IDENTITY


def _read(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate readiness JSON key: " + key)
            value[key] = item
        return value
    def invalid(value):
        raise ValueError("Nonfinite readiness JSON value: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique, parse_constant=invalid)


def _binding(binding):
    h = helper()
    if (not isinstance(binding, dict) or set(binding) != {"plan", "output_root"}
            or not isinstance(binding["plan"], dict) or set(binding["plan"]) != {"path", "sha256"}
            or not h._absolute(binding["plan"]["path"]) or not h._hash(binding["plan"]["sha256"])
            or not h._absolute(binding["output_root"])):
        raise ValueError("Require exactly an absolute immutable plan binding and output_root")


def diagnostic_plan(binding, expected_plan=None):
    """Authenticate the observer's immutable inputs without requiring completion."""
    _binding(binding)
    h = helper()
    plan = h.load_plan(Path(binding["plan"]["path"]), binding["plan"]["sha256"])
    if (plan["output_root"] != binding["output_root"]
            or plan["production_archive_sha256"] != SOURCE105_ARCHIVE
            or plan["production_manifest_sha256"] != SOURCE105_MANIFEST
            or [case["id"] for case in plan["cases"]] != list(CASE_IDS)
            or any(case["arm"] != "fixed_initial_mask" or case["step"] != 15 for case in plan["cases"])):
        raise ValueError("Diagnostic reference does not name both exact source105 saved cases")
    if expected_plan is not None and any(plan[key] != expected_plan[key] for key in
            ("production_release", "production_archive_sha256", "production_manifest_sha256", "cases")):
        raise ValueError("Diagnostic reference and readiness use different historical scientific states")
    return plan


def load_plan(path, expected):
    """Add explicit v3 policy/arithmetic and its required mechanism observation."""
    h = helper()
    if not h._hash(expected) or h._sha(path) != expected:
        raise ValueError("Additive readiness plan SHA256 mismatch")
    plan = _read(path)
    extras = {"fd_policy", "objective_arithmetic", "diagnostic_reference"}
    fields = {"version", "name", "output_root", "production_release", "production_manifest_sha256",
              "production_archive_sha256", "cases"} | extras
    if (not isinstance(plan, dict) or set(plan) != fields or plan["fd_policy"] != POLICY
            or plan["objective_arithmetic"] != ARITHMETIC):
        raise ValueError("Require exactly the declared additive readiness plan fields and policy")
    # Reuse the original schema validator on an in-memory view. No old plan or
    # persisted failure is rewritten, even temporarily.
    original = {key: value for key, value in plan.items() if key not in extras}
    saved_read = h._read
    try:
        h._read = lambda candidate: original if Path(candidate) == Path(path) else saved_read(candidate)
        h.load_plan(path, expected)
    finally:
        h._read = saved_read
    if (plan["production_archive_sha256"] != SOURCE105_ARCHIVE
            or plan["production_manifest_sha256"] != SOURCE105_MANIFEST
            or [case["id"] for case in plan["cases"]] != list(CASE_IDS)
            or any(case["arm"] != "fixed_initial_mask" or case["step"] != 15 for case in plan["cases"])):
        raise ValueError("Require both original source105 fixed-mask step15 failures")
    diagnostic_plan(plan["diagnostic_reference"], plan)
    return plan


def validate_readiness_binding(binding):
    """Prepare immutable binding shape/assets; this never claims a passed gate."""
    _binding(binding)
    plan = load_plan(Path(binding["plan"]["path"]), binding["plan"]["sha256"])
    if plan["output_root"] != binding["output_root"]:
        raise ValueError("Readiness output root differs from its sealed plan")
    return plan


def authenticate_execution_record(record, identity):
    """Require completed canonical config/source and one actual numeric LSF ID."""
    h, cfg = helper(), config_identity()
    files = record.get("source", {}).get("files")
    job = record.get("environment", {}).get("job_id")
    if not isinstance(job, str) or not re.fullmatch("[0-9]+", job):
        raise ValueError("Readiness execution lacks a numeric allocation job ID")
    if (record.get("status") != "completed" or record.get("config_hash_version") != cfg.CONFIG_HASH_VERSION
            or not isinstance(files, dict) or not files or h._digest(files) != identity["source_manifest_sha256"]
            or record.get("source", {}).get("sha256") != identity["source_manifest_sha256"]
            or record.get("source_archive_sha256") != identity["source_archive_sha256"]):
        raise ValueError("Execution is incomplete or belongs to another immutable source")
    return {**cfg.verify_config_identity(record), "allocation_job_id": job}


def _unique_parent(root, name, plan_sha):
    candidates = [(path, _read(path)) for path in Path(root).glob("*/run.json")]
    candidates = [(path, record) for path, record in candidates if record.get("name") == name
                  and record.get("config", {}).get("plan_sha256") == plan_sha]
    if len(candidates) != 1:
        raise ValueError("Require one unambiguous completed execution of the bound plan")
    return candidates[0]


def _child(parent, declared, row):
    execution = row.get("execution", {})
    paths = execution.get("run_paths", [])
    if (execution.get("returncode") != 0 or execution.get("timed_out") is not False or len(paths) != 1):
        raise ValueError("Case is failed, timed out or lacks one unique child")
    child = Path(paths[0]).resolve()
    others = list((parent / declared["id"]).glob("*/run.json"))
    if child.parent != parent / declared["id"] or len(others) != 1 or others[0].resolve() != child / "run.json":
        raise ValueError("Case has an escaped or ambiguous child execution")
    return child


def require_diagnostic_reference(binding, expected_plan):
    """Require completed source106 observations of these same failed states.

    The observation may retain arbitrary old derivative failures. It must have
    completed every declared measurement without changing state or old assets.
    This pure byte/metadata check is safe before a fresh child's old bootstrap.
    """
    h, cfg = helper(), config_identity()
    plan = diagnostic_plan(binding, expected_plan)
    identity = {"source_archive_sha256": SOURCE106_ARCHIVE, "source_manifest_sha256": SOURCE106_MANIFEST}
    path, record = _unique_parent(binding["output_root"], plan["name"], binding["plan"]["sha256"])
    execution_identity = authenticate_execution_record(record, identity)
    if cfg.canonical_config_json(record["config"].get("plan")) != cfg.canonical_config_json(plan):
        raise ValueError("Diagnostic controller recorded another immutable plan")
    release = Path(plan["production_release"]).parent / SOURCE106_NAME
    h.authenticate_source(release, record, {"production_archive_sha256": SOURCE106_ARCHIVE,
                                           "production_manifest_sha256": SOURCE106_MANIFEST})
    parent = path.parent.resolve()
    rows_path = parent / "cases.json"
    rows = _read(rows_path)
    if not isinstance(rows, list) or [row.get("case_id") for row in rows] != list(CASE_IDS):
        raise ValueError("Diagnostic lacks both declared completed observations")
    receipts, evidence = [], {}
    for case, row in zip(plan["cases"], rows):
        if row.get("diagnostic_completed") is not True:
            raise ValueError("Required mechanism diagnostic is incomplete")
        child = _child(parent, case, row)
        child_record = _read(child / "run.json")
        child_identity = authenticate_execution_record(child_record, identity)
        if child_identity["allocation_job_id"] != execution_identity["allocation_job_id"]:
            raise ValueError("Diagnostic child belongs to another allocation")
        recorded = child_record["config"]
        if recorded.get("case") != case or any(recorded.get(key) != plan[key] for key in
                ("production_release", "production_manifest_sha256", "production_archive_sha256")):
            raise ValueError("Diagnostic child authenticated different original cases/source")
        proof = _read(child / "attribution.json")
        if (proof.get("status") != "completed" or any(proof.get(key) is not True for key in
                ("diagnostic_completed", "frozen_state_unchanged", "parent_unchanged", "native_seed_parity",
                 "original_gate_not_reclassified"))
                or proof.get("identities_before") != proof.get("identities_after")
                or not proof.get("identities_before") or proof.get("optimizer_updates") != 0
                or proof.get("new_defog_calls") != 0 or proof.get("old_gate_certified_by_this_diagnostic") is not False):
            raise ValueError("Mechanism reference lacks complete unchanged-state observations")
        directions, endpoints = proof.get("directions", {}), proof.get("endpoint_rows", [])
        scales = (1e-5, 3e-5, 1e-4, 3e-4, .001, .003)
        expected_rows = {(direction, epsilon, mode) for direction in directions for epsilon in scales
                         for mode in ("natural", "held_readout")}
        actual_rows = {(row.get("direction"), row.get("epsilon"), row.get("mode")) for row in endpoints}
        if (not directions or "gradient" not in directions or proof.get("epsilons") != list(scales)
                or len(endpoints) != len(expected_rows) or actual_rows != expected_rows):
            raise ValueError("Mechanism reference is missing its complete declared endpoint grid")
        for artifact in case["parent_artifacts_sha256"]:
            h.authenticated_artifact(case, artifact)
        evidence[case["id"]] = proof
        receipts.append({"case_id": case["id"], "run": str(child), "run_sha256": h._sha(child / "run.json"),
            "attribution_sha256": h._sha(child / "attribution.json"), "config_identity": child_identity})
    return {"passed": True, "binding": binding, "controller_run": str(parent), "controller_run_sha256": h._sha(path),
            "cases_sha256": h._sha(rows_path), "allocation_job_id": execution_identity["allocation_job_id"],
            "source_identity": identity, "cases": receipts}, evidence


@contextmanager
def module_bindings(bindings):
    """Restore both import registries; relative 'from . import' uses attributes."""
    missing = object()
    saved = {}
    try:
        for name, module in bindings.items():
            parent_name, attribute = name.rsplit(".", 1)
            parent = importlib.import_module(parent_name)
            saved[name] = (sys.modules.get(name, missing), parent, attribute, getattr(parent, attribute, missing))
            sys.modules[name] = module
            setattr(parent, attribute, module)
        yield
    finally:
        for name, (old, parent, attribute, old_attribute) in reversed(list(saved.items())):
            if old is missing: sys.modules.pop(name, None)
            else: sys.modules[name] = old
            if old_attribute is missing:
                if hasattr(parent, attribute): delattr(parent, attribute)
            else: setattr(parent, attribute, old_attribute)


@contextmanager
def candidate_modules():
    """Bind the actual new validator/reader bytes after the original bootstrap."""
    folder = Path(__file__).parent
    locations = (("latentmol.boltz.additive_fd", folder / "additive_fd.py"),
                 ("latentmol.boltz.additive_contract", folder / "additive_contract.py"),
                 ("latentmol.boltz.additive_branch_fd", folder / "additive_branch_fd.py"),
                 ("latentmol.repair_guidance.integration_smoke", folder.parent / "repair_guidance/integration_smoke.py"))
    loaded = {}
    with ExitStack() as stack:
        for name, path in locations:
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            stack.enter_context(module_bindings({name: module}))
            spec.loader.exec_module(module)
            loaded[name.rsplit(".", 1)[1]] = module
        yield loaded


class FullForwardBudget:
    """Count real native evaluations, including the two original seed calls."""
    def __init__(self, counters, report):
        self.counters, self.report, self.used = counters, report, 2
        report["actual_full_forward_reservations"] = self.used

    def call(self, model, purpose, action, function, device):
        if model == "boltz" and action in ("forward", "native_forward", "forward_ad"):
            if self.used >= MAXIMUM_FULL_FORWARDS_PER_CASE:
                raise ValueError("Additive readiness full-forward reservation exhausted")
            self.used += 1
            self.report["actual_full_forward_reservations"] = self.used
        return self.counters.call(model, purpose, action, function, device)


def probe_case(decoder, point, branch, config, counters, target, weight, item_name, item, *, persist, report):
    """Keep old/native parity, then invoke the actual new full production gate."""
    import torch
    from latentmol.boltz.active_set import score_at
    from latentmol.graph_policy import graph_from_decoder
    from latentmol.multimodel.study import tensor_digest
    h, cfg = helper(), config_identity()
    counters = FullForwardBudget(counters, report)
    reference = _REFERENCE[_CASE["id"]]
    if (reference["identities_before"] != report["identities_before"] or reference["target"] != report["target"]
            or reference["original_item"] != item_name):
        raise ValueError("Mechanism reference has another point/context/frozen model/target")
    gate_path = h.authenticated_artifact(_CASE, Path(_CASE["parent_run"]) / _CASE["arm"] / f"gate-{_CASE['step']:04d}.json")
    old_gate = _read(gate_path)
    components = {}
    measured, gradient = score_at(decoder, point, branch, config, counters, "additive_original_baseline", gradients=True,
        component_gradients=components, repair_target=target, repair_weight=weight)
    if measured["loss"] != old_gate["measured"]["loss"] or measured["components"] != old_gate["measured"]["components"]:
        raise ValueError("Original source105 failed scalar/output baseline did not replay exactly")
    values = {**measured["components"], "combined": measured["loss"]}
    if target is not None: values["repair_nll"] = measured["repair_guidance"]["normalized_nll"]
    if values != reference["baseline"]:
        raise ValueError("Original scalar baseline differs from completed source106 mechanism observations")
    with torch.no_grad():
        graph = graph_from_decoder(decoder.decode(point), decoder.spec, temperature=config.temperature,
                                   fixed_active=branch.prepared.active)
    graph_sha = tensor_digest({key: getattr(graph, key) for key in ("atoms", "charges", "bonds")})
    if graph_sha != reference["baseline_graph_sha256"]:
        raise ValueError("Original shared molecule differs from source106")
    relevant = gradient if item_name == "combined" else components[item_name]
    replay = h.reconstruct_directions(point, relevant, item)
    if set(replay) != set(reference["directions"]):
        raise ValueError("Original failed direction coverage changed")
    for label, (direction, receipt) in replay.items():
        if (label != "gradient" and receipt["hash_matches_original"] is not True
                or receipt != reference["directions"][label]["replay"]
                or float((relevant.double() * direction.double()).sum()) != reference["directions"][label]["reverse_projection"]):
            raise ValueError("Original failed direction/derivative does not replay source106 exactly")
    original = config.record()
    candidate = replace(config, fd_policy=POLICY)
    expected = {**deepcopy(original), "fd_policy": POLICY}
    if config.fd_policy != "branch_local_v2" or config.dbmol.objective_arithmetic != ARITHMETIC or candidate.record() != expected:
        raise ValueError("Readiness must change only the declared derivative policy")
    candidate_components = {}
    current, current_gradient = score_at(decoder, point, branch, candidate, counters, "additive_candidate_baseline", gradients=True,
        component_gradients=candidate_components, repair_target=target, repair_weight=weight)
    if (current != measured or not torch.equal(current_gradient, gradient)
            or set(candidate_components) != set(components)
            or any(not torch.equal(candidate_components[key], value) for key, value in components.items())):
        raise ValueError("Changing the certificate changed native primal or derivative mathematics")
    report.update(original_config=original, original_config_sha256=cfg.config_sha256(original),
        candidate_config=candidate.record(), candidate_config_sha256=cfg.config_sha256(candidate.record()),
        candidate_source_files={name: h._sha(Path.cwd() / name) for name in SOURCE_PATHS},
        old_gate={"path": str(gate_path), "sha256": h._sha(gate_path)},
        original_gate_not_reclassified=True, old_gate_certified_by_this_diagnostic=False,
        optimizer_updates=0, new_defog_calls=0, repair_target_available=target is not None,
        effective_repair_weight=weight, baseline_graph_sha256=graph_sha,
        baseline_primal_and_gradient_exact=True, diagnostic_reference=_PLAN["diagnostic_reference"],
        original_direction_replay={label: receipt for label, (_, receipt) in replay.items()})
    branch_report = {"passed": False, "status": "running", "native_seed_parity": report["native_seed_parity"],
        "point_sha256": report["identities_before"]["point_sha256"],
        "context_sha256": report["identities_before"]["context_sha256"],
        "active_slots": branch.prepared.active.nonzero().flatten().tolist(), "measured": current}
    report["branch_validation"] = branch_report
    persist()
    with candidate_modules() as modules:
        validator, contract, reader = (modules[name] for name in ("additive_branch_fd", "additive_contract", "integration_smoke"))
        if validator.POLICY != POLICY:
            raise ValueError("Loaded candidate does not implement the declared policy")
        profile = contract.profile_record(config.dbmol, config.objective, repair_weight=weight if target is not None else None)
        report["expected_profile"] = profile
        try:
            validator.validate_branch_derivatives(decoder, point, branch, candidate, counters, branch_report, persist,
                current, current_gradient, candidate_components, target=target, weight=weight)
            branch_report.update(passed=True, status="passed")
        except ValueError as error:
            branch_report.update(passed=False, status="failed", gate_error=str(error))
        passed = reader.full_gate_passed(branch_report, require_repair=target is not None, expected_profile=profile)
    report.update(status="completed", diagnostic_completed=True, validation_passed=passed,
        candidate_gate_passed=passed, declared_policy_only_changed=True)
    persist()


def run_case(plan, case, output_root):
    global _PLAN, _CASE, _REFERENCE
    _PLAN, _CASE = plan, case
    _, _REFERENCE = require_diagnostic_reference(plan["diagnostic_reference"], plan)
    return helper().run_case(plan, case, output_root, expected_policy="branch_local_v2", probe=probe_case)


def _verify_gate(gate, profile, require_repair):
    with candidate_modules() as modules:
        return modules["integration_smoke"].full_gate_passed(gate, require_repair=require_repair, expected_profile=profile)


def require_readiness(binding, identity):
    """Authenticate both actual native passes before any later optimization job."""
    from latentmol.experiments import source_manifest
    h, cfg = helper(), config_identity()
    if (not isinstance(identity, dict) or set(identity) != {"source_archive_sha256", "source_manifest_sha256"}
            or any(not h._hash(value) for value in identity.values())
            or source_manifest(Path.cwd())["sha256"] != identity["source_manifest_sha256"]
            or os.environ.get("LATENTMOL_SOURCE_SHA256") != identity["source_archive_sha256"]):
        raise ValueError("Readiness caller is not the declared immutable candidate source")
    plan = validate_readiness_binding(binding)
    reference_receipt, references = require_diagnostic_reference(plan["diagnostic_reference"], plan)
    path, record = _unique_parent(binding["output_root"], plan["name"], binding["plan"]["sha256"])
    execution_identity = authenticate_execution_record(record, identity)
    if cfg.canonical_config_json(record["config"].get("plan")) != cfg.canonical_config_json(plan):
        raise ValueError("Readiness controller recorded another plan")
    parent = path.parent.resolve()
    rows_path, summary_path = parent / "cases.json", parent / "validation.json"
    rows, summary = _read(rows_path), _read(summary_path)
    if (not isinstance(rows, list) or [row.get("case_id") for row in rows] != list(CASE_IDS)
            or summary.get("cases") != rows or summary.get("status") != "completed"
            or summary.get("validation_passed") is not True or summary.get("plan_sha256") != binding["plan"]["sha256"]
            or summary.get("fd_policy") != POLICY or summary.get("objective_arithmetic") != ARITHMETIC
            or summary.get("diagnostic_reference") != plan["diagnostic_reference"]
            or summary.get("original_gates_not_reclassified") is not True
            or summary.get("optimizer_updates") != 0 or summary.get("new_defog_calls") != 0):
        raise ValueError("Readiness lacks the complete passing two-case summary")
    receipts = []
    for case, row in zip(plan["cases"], rows):
        if any(row.get(key) is not True for key in ("validation_completed", "validation_passed")) or row.get("old_gate_certified") is not False:
            raise ValueError("Readiness has a failed or incomplete native case")
        child = _child(parent, case, row)
        child_record = _read(child / "run.json")
        child_identity = authenticate_execution_record(child_record, identity)
        if child_identity["allocation_job_id"] != execution_identity["allocation_job_id"]:
            raise ValueError("Readiness child belongs to a different native allocation")
        recorded = child_record["config"]
        if recorded.get("case") != case or any(recorded.get(key) != plan[key] for key in
                ("production_release", "production_manifest_sha256", "production_archive_sha256")):
            raise ValueError("Readiness child authenticated different historical inputs")
        for artifact in case["parent_artifacts_sha256"]:
            h.authenticated_artifact(case, artifact)
        original_record = _read(h.authenticated_artifact(case, Path(case["parent_run"]) / "run.json"))
        h.authenticate_source(plan["production_release"], original_record, plan)
        old_gate_path = h.authenticated_artifact(case, Path(case["parent_run"]) / case["arm"] / f"gate-{case['step']:04d}.json")
        old_gate = _read(old_gate_path)
        h.failed_item(old_gate, policy="branch_local_v2")
        proof = _read(child / "attribution.json")
        if (proof.get("status") != "completed" or any(proof.get(key) is not True for key in
                ("diagnostic_completed", "validation_passed", "candidate_gate_passed", "frozen_state_unchanged",
                 "parent_unchanged", "native_seed_parity", "original_gate_not_reclassified",
                 "baseline_primal_and_gradient_exact", "declared_policy_only_changed"))
                or proof.get("identities_before") != proof.get("identities_after")
                or proof.get("identities_before") != references[case["id"]]["identities_before"]
                or proof.get("target") != references[case["id"]]["target"]
                or proof.get("baseline_graph_sha256") != references[case["id"]]["baseline_graph_sha256"]
                or proof.get("old_gate_certified_by_this_diagnostic") is not False
                or proof.get("optimizer_updates") != 0 or proof.get("new_defog_calls") != 0
                or proof.get("diagnostic_reference") != plan["diagnostic_reference"]
                or proof.get("old_gate") != {"path": str(old_gate_path), "sha256": h._sha(old_gate_path)}):
            raise ValueError("Readiness child lacks unchanged exact-state provenance")
        gate = proof.get("branch_validation", {})
        if (gate.get("point_sha256") != old_gate.get("point_sha256")
                or gate.get("context_sha256") != old_gate.get("context_sha256")
                or gate.get("active_slots") != old_gate.get("active_slots")
                or gate.get("measured", {}).get("loss") != old_gate.get("measured", {}).get("loss")
                or gate.get("measured", {}).get("components") != old_gate.get("measured", {}).get("components")):
            raise ValueError("Readiness derivative gate differs from the original point/context/native scalar")
        original, candidate = proof.get("original_config"), proof.get("candidate_config")
        if (not isinstance(original, dict) or not isinstance(candidate, dict)
                or cfg.config_sha256(original) != proof.get("original_config_sha256")
                or cfg.config_sha256(candidate) != proof.get("candidate_config_sha256")
                or original.get("fd_policy") != "branch_local_v2"
                or original.get("dbmol", {}).get("objective_arithmetic") != ARITHMETIC
                or candidate != {**deepcopy(original), "fd_policy": POLICY}
                or cfg.canonical_config_json({key: original_record["config"].get(key) for key in original}) != cfg.canonical_config_json(original)):
            raise ValueError("Readiness changed science outside the declared validation policy")
        weight = recorded.get("effective_weight")
        target = proof["target"].get("target_sha256") is not None
        if (target != (case["id"] == "boltz_defog") or proof.get("repair_target_available") is not target
                or proof.get("effective_repair_weight") != weight
                or weight != old_gate.get("effective_repair_weight", 0)
                or proof["target"] != recorded.get("target")):
            raise ValueError("Readiness target or effective weight differs from the historical objective")
        with candidate_modules() as modules:
            profile = modules["additive_contract"].profile_record(original["dbmol"], original["objective"],
                repair_weight=weight if target else None)
        if proof.get("expected_profile") != profile or not _verify_gate(gate, profile, target):
            raise ValueError("Readiness lacks the full externally bound native derivative certificate")
        files = proof.get("candidate_source_files", {})
        if set(files) != set(SOURCE_PATHS) or any(child_record["source"]["files"].get(name) != digest for name, digest in files.items()):
            raise ValueError("Readiness candidate source coverage differs from the executed manifest")
        calls = proof.get("actual_full_forward_reservations")
        if type(calls) is not int or not 2 <= calls <= MAXIMUM_FULL_FORWARDS_PER_CASE:
            raise ValueError("Readiness lacks its declared actual full-forward budget")
        costs = _read(child / "oracle_cost.json").get("boltz", {})
        attempted = [value for row in costs.values() for key, value in row.items()
                     if key in ("forward_attempted", "native_forward_attempted", "forward_ad_attempted")]
        if any(type(value) is not int or value < 0 for value in attempted) or sum(attempted) != calls:
            raise ValueError("Readiness forward reservation differs from actual attempted-call counters")
        receipts.append({"case_id": case["id"], "run": str(child), "run_sha256": h._sha(child / "run.json"),
            "attribution_sha256": h._sha(child / "attribution.json"), "config_identity": child_identity,
            "candidate_config_sha256": proof["candidate_config_sha256"], "full_forward_reservations": calls,
            "oracle_cost_sha256": h._sha(child / "oracle_cost.json")})
    return {"passed": True, "plan": binding["plan"], "output_root": binding["output_root"],
        "controller_run": str(parent), "controller_run_sha256": h._sha(path),
        "cases_sha256": h._sha(rows_path), "summary_sha256": h._sha(summary_path),
        "allocation_job_id": execution_identity["allocation_job_id"], "source_identity": identity,
        "objective_arithmetic": ARITHMETIC, "fd_policy": POLICY, "cases": receipts,
        "diagnostic_reference": reference_receipt, "original_gates_not_reclassified": True}


def run_study(path, expected, plan):
    """Serial complete observations; failed scientific checks remain failures."""
    from latentmol.experiments import ExperimentRun, write_json
    from latentmol.boltz.workflow_process import run_logged
    require_diagnostic_reference(plan["diagnostic_reference"], plan)
    with ExperimentRun(Path(plan["output_root"]), plan["name"], {"plan": plan, "plan_sha256": expected,
            "case_timeout_seconds": CASE_TIMEOUT_SECONDS, "maximum_full_forwards_per_case": MAXIMUM_FULL_FORWARDS_PER_CASE,
            "optimizer_updates": 0, "new_defog_calls": 0}, Path.cwd()) as run:
        rows = []
        for case in plan["cases"]:
            execution = run_logged([sys.executable, str(Path(__file__).resolve()), "--plan", str(Path(path).resolve()),
                "--sha256", expected, "--child", case["id"], "--child-output", str(run.path / case["id"])],
                run.path / (case["id"] + ".log"), timeout_seconds=CASE_TIMEOUT_SECONDS)
            complete = passed = False
            if execution["returncode"] == 0 and execution["timed_out"] is False and len(execution["run_paths"]) == 1:
                child = Path(execution["run_paths"][0])
                proof, record = _read(child / "attribution.json"), _read(child / "run.json")
                complete = (record.get("status") == "completed" and proof.get("status") == "completed"
                    and all(proof.get(key) is True for key in ("diagnostic_completed", "frozen_state_unchanged", "parent_unchanged")))
                passed = complete and proof.get("validation_passed") is True
            rows.append({"case_id": case["id"], "validation_completed": complete, "validation_passed": passed,
                         "execution": execution, "old_gate_certified": False})
            write_json(run.path / "cases.json", rows)
        passed = len(rows) == 2 and all(row["validation_passed"] for row in rows)
        write_json(run.path / "validation.json", {"status": "completed", "validation_passed": passed, "cases": rows,
            "plan_sha256": expected, "fd_policy": POLICY, "objective_arithmetic": ARITHMETIC,
            "diagnostic_reference": plan["diagnostic_reference"], "original_gates_not_reclassified": True,
            "optimizer_updates": 0, "new_defog_calls": 0})
        run.annotate(validation_passed=passed, validation_cases_completed=sum(row["validation_completed"] for row in rows),
                     validation_cases_passed=sum(row["validation_passed"] for row in rows), old_gates_certified=False)
        return run.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--child")
    parser.add_argument("--child-output", type=Path)
    args = parser.parse_args()
    if not re.fullmatch("[0-9]+", os.environ.get("LSB_JOBID", "")) or not helper()._hash(os.environ.get("LATENTMOL_SOURCE_SHA256")):
        raise ValueError("Additive readiness requires an immutable allocated DTU HPC job")
    plan = load_plan(args.plan, args.sha256)
    if args.child is not None:
        selected = [case for case in plan["cases"] if case["id"] == args.child]
        if len(selected) != 1 or args.child_output is None:
            raise ValueError("Unknown readiness child or missing output directory")
        print(run_case(plan, selected[0], args.child_output), flush=True)
    else:
        if args.child_output is not None:
            raise ValueError("Child output is only valid with --child")
        print(run_study(args.plan, args.sha256, plan), flush=True)


if __name__ == "__main__":
    main()
