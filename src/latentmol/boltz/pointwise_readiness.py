"""Focused allocated readiness for pointwise_native_ad_v4; native results pending.

Only the two authenticated source105 step-15 failures are replayed. The
candidate changes the derivative certificate, not the molecule, native model,
objective, gradients or optimizer. The immutable existing-evidence review
explains that change; no new broad observer is a prerequisite. Each case has
a 1,200-second/80-native-forward cap, zero updates and zero new DeFoG calls.

Fresh children bootstrap old source105 before importing scientific modules.
Authentication and temporary import binding reuse the adjacent readiness
helpers by explicit file load. No old failure is retrospectively certified.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from copy import deepcopy
from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import re
import sys


POLICY = "pointwise_native_ad_v4"
ARITHMETIC = "contact_head_and_assembly_float64_v1"
CASE_TIMEOUT_SECONDS = 1200
MAXIMUM_FULL_FORWARDS_PER_CASE = 80
EVIDENCE_REVIEW_SHA256 = "63305fb4aea0d4776f461c7552d7739c39e761022c234799702eb598a23f3045"
SOURCE105_CAPSULE = "1725b668a0fccf46bb4ecb3f4e3f28303ea70bbe055bf98723a1e24bd700712a"
SOURCE105_ARCHIVE = "26d81fc50e9ca53144ab412a4f23a217a0c46c8b3e31d196ddaa4390cc0b836e"
SOURCE105_MANIFEST = "cd4632d3e644a0df1d57a4016bd18b6b94658eaf3822c251f28f0acd45674c79"
CASE_IDS = ("boltz_only", "boltz_defog")
OLD_CASES = {
    "boltz_only": ("20261014T205327Z_matched-pilot-boltz_only_f632abf8",
        "b0150f29ae1873a76c8fdfc198db8d2ae0996764df429b063bf0da5a2d7804c8", "control_failed_combined_gradient"),
    "boltz_defog": ("20261014T213903Z_matched-pilot-boltz_defog_65ba68e5",
        "5b42f08d274eba1b696167260b87328de504c6a4841add74f1009d7e1eb6dadc", "guided_failed_contact_random_2"),
}
SOURCE_PATHS = (
    "src/latentmol/boltz/pointwise_readiness.py", "src/latentmol/boltz/pointwise_ad.py",
    "src/latentmol/boltz/pointwise_cuts.py", "src/latentmol/boltz/pointwise_contract.py",
    "src/latentmol/boltz/additive_readiness.py", "src/latentmol/boltz/additive_fd.py",
    "src/latentmol/boltz/additive_contract.py", "src/latentmol/boltz/additive_branch_fd.py",
    "src/latentmol/boltz/failed_gate_attribution.py", "src/latentmol/repair_guidance/integration_smoke.py",
)
_COMMON = None
_PLAN = _CASE = None


def common():
    """Load shared byte/authentication helpers without importing latentmol."""
    global _COMMON
    if _COMMON is None:
        path = Path(__file__).with_name("additive_readiness.py")
        spec = importlib.util.spec_from_file_location("_pointwise_readiness_common", path)
        _COMMON = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_COMMON)
        h = _COMMON.helper()
        h.CASE_TIMEOUT_SECONDS, h.REPEATS, h.CLAIM_LIMIT = CASE_TIMEOUT_SECONDS, 1, __doc__
    return _COMMON


def helper():
    return common().helper()


def config_identity():
    return common().config_identity()


def _read(path):
    return common()._read(path)


def review_asset(reference, cases):
    """Bind the immutable review to the exact two old failed gate hashes."""
    h = helper()
    if (not isinstance(reference, dict) or set(reference) != {"path", "sha256"}
            or not h._absolute(reference["path"]) or reference["sha256"] != EVIDENCE_REVIEW_SHA256
            or h._sha(reference["path"]) != reference["sha256"]):
        raise ValueError("Require the immutable existing-evidence review bytes")
    review = _read(reference["path"])
    if review.get("source_capsule_sha256") != SOURCE105_CAPSULE:
        raise ValueError("Evidence review refers to another source105 capsule")
    for case in cases:
        run_id, gate_sha, key = OLD_CASES[case["id"]]
        relative = "/".join(("source-105-contact-precision-branch-validation-attempt002-pilot",
                              case["id"], "optimization", run_id))
        parent = str(case["parent_run"]).replace("\\", "/")
        gate = str(Path(case["parent_run"]) / "fixed_initial_mask/gate-0015.json")
        observed = review.get(key, {})
        local = str(observed.get("path", "")).replace("\\", "/")
        if (not parent.endswith("/runs/" + relative)
                or case["parent_artifacts_sha256"].get(gate) != gate_sha
                or observed.get("sha256") != gate_sha
                or not local.endswith("/runs/" + relative + "/fixed_initial_mask/gate-0015.json")):
            raise ValueError("Review and readiness must bind the exact two source105 failures")
    return review


def load_plan(path, expected):
    """Authenticate new certificate fields and the unchanged old case schema."""
    h = helper()
    if not h._hash(expected) or h._sha(path) != expected:
        raise ValueError("Pointwise readiness plan SHA256 mismatch")
    plan = _read(path)
    extras = {"fd_policy", "objective_arithmetic", "evidence_review"}
    fields = {"version", "name", "output_root", "production_release", "production_manifest_sha256",
              "production_archive_sha256", "cases"} | extras
    if (not isinstance(plan, dict) or set(plan) != fields or plan["fd_policy"] != POLICY
            or plan["objective_arithmetic"] != ARITHMETIC):
        raise ValueError("Require exactly the declared pointwise readiness fields/policy")
    original = {key: value for key, value in plan.items() if key not in extras}
    saved = h._read
    try:
        h._read = lambda candidate: original if Path(candidate) == Path(path) else saved(candidate)
        h.load_plan(path, expected)
    finally:
        h._read = saved
    if (plan["production_archive_sha256"] != SOURCE105_ARCHIVE
            or plan["production_manifest_sha256"] != SOURCE105_MANIFEST
            or [case["id"] for case in plan["cases"]] != list(CASE_IDS)
            or any(case["arm"] != "fixed_initial_mask" or case["step"] != 15 for case in plan["cases"])):
        raise ValueError("Require both exact original source105 fixed-mask step15 failures")
    review_asset(plan["evidence_review"], plan["cases"])
    return plan


def validate_readiness_binding(binding):
    common()._binding(binding)
    plan = load_plan(Path(binding["plan"]["path"]), binding["plan"]["sha256"])
    if plan["output_root"] != binding["output_root"]:
        raise ValueError("Readiness output root differs from its sealed plan")
    return plan


@contextmanager
def candidate_modules():
    """Bind actual candidate bytes after old bootstrap, restoring every alias."""
    folder = Path(__file__).parent
    locations = [("latentmol.boltz." + name, folder / (name + ".py")) for name in
        ("additive_fd", "additive_contract", "additive_branch_fd", "pointwise_cuts",
         "pointwise_contract", "pointwise_ad")]
    locations.append(("latentmol.repair_guidance.integration_smoke",
                      folder.parent / "repair_guidance/integration_smoke.py"))
    loaded = {}
    with ExitStack() as stack:
        for name, path in locations:
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            stack.enter_context(common().module_bindings({name: module}))
            spec.loader.exec_module(module)
            loaded[name.rsplit(".", 1)[1]] = module
        yield loaded


class FullForwardBudget:
    """Count native calls including the old helper's two seed-parity forwards."""
    def __init__(self, counters, report):
        self.counters, self.report, self.used = counters, report, 2
        report["actual_full_forward_reservations"] = self.used

    def call(self, model, purpose, action, function, device):
        if model == "boltz" and action in ("forward", "native_forward", "forward_ad"):
            if self.used >= MAXIMUM_FULL_FORWARDS_PER_CASE:
                raise ValueError("Pointwise readiness full-forward reservation exhausted")
            self.used += 1
            self.report["actual_full_forward_reservations"] = self.used
        return self.counters.call(model, purpose, action, function, device)


def probe_case(decoder, point, branch, config, counters, target, weight, item_name, item, *, persist, report):
    """Replay old native science, change only policy, then call the real gate."""
    import torch
    from latentmol.boltz.active_set import score_at
    from latentmol.graph_policy import graph_from_decoder
    from latentmol.multimodel.study import tensor_digest
    h, cfg = helper(), config_identity()
    counters = FullForwardBudget(counters, report)
    old_path = h.authenticated_artifact(_CASE, Path(_CASE["parent_run"]) / _CASE["arm"] / "gate-0015.json")
    old = _read(old_path)
    components = {}
    measured, gradient = score_at(decoder, point, branch, config, counters, "pointwise_original_baseline", gradients=True,
        component_gradients=components, repair_target=target, repair_weight=weight)
    if measured["loss"] != old["measured"]["loss"] or measured["components"] != old["measured"]["components"]:
        raise ValueError("Original source105 failed scalar/output baseline did not replay exactly")
    relevant = gradient if item_name == "combined" else components[item_name]
    replay = h.reconstruct_directions(point, relevant, item)
    if any(label != "gradient" and meta["hash_matches_original"] is not True for label, (_, meta) in replay.items()):
        raise ValueError("Original failed random direction no longer replays exactly")
    with torch.no_grad():
        graph = graph_from_decoder(decoder.decode(point), decoder.spec, temperature=config.temperature,
                                   fixed_active=branch.prepared.active)
    graph_sha = tensor_digest({key: getattr(graph, key) for key in ("atoms", "charges", "bonds")})
    original = config.record()
    candidate = replace(config, fd_policy=POLICY)
    if (config.fd_policy != "branch_local_v2" or config.dbmol.objective_arithmetic != ARITHMETIC
            or candidate.record() != {**deepcopy(original), "fd_policy": POLICY}):
        raise ValueError("Readiness must change only the declared derivative policy")
    current_components = {}
    current, current_gradient = score_at(decoder, point, branch, candidate, counters, "pointwise_candidate_baseline", gradients=True,
        component_gradients=current_components, repair_target=target, repair_weight=weight)
    if (current != measured or not torch.equal(current_gradient, gradient)
            or set(current_components) != set(components)
            or any(not torch.equal(value, current_components[key]) for key, value in components.items())):
        raise ValueError("Changing the certificate changed native primal or derivative mathematics")
    report.update(original_config=original, original_config_sha256=cfg.config_sha256(original),
        candidate_config=candidate.record(), candidate_config_sha256=cfg.config_sha256(candidate.record()),
        candidate_source_files={name: h._sha(Path.cwd() / name) for name in SOURCE_PATHS},
        old_gate={"path": str(old_path), "sha256": h._sha(old_path)}, evidence_review=_PLAN["evidence_review"],
        baseline_primal_and_gradient_exact=True, declared_policy_only_changed=True,
        original_gradient_sha256=tensor_digest({"gradient": gradient}), baseline_graph_sha256=graph_sha,
        repair_target_available=target is not None, effective_repair_weight=weight,
        original_direction_replay={label: meta for label, (_, meta) in replay.items()},
        optimizer_updates=0, new_defog_calls=0, old_gate_certified_by_this_diagnostic=False,
        original_gate_not_reclassified=True)
    gate = {"passed": False, "status": "running", "native_seed_parity": report["native_seed_parity"],
        "point_sha256": report["identities_before"]["point_sha256"],
        "context_sha256": report["identities_before"]["context_sha256"],
        "active_slots": branch.prepared.active.nonzero().flatten().tolist(), "measured": current}
    report["branch_validation"] = gate
    persist()
    with candidate_modules() as modules:
        validator, contract, reader = (modules[name] for name in ("pointwise_ad", "pointwise_contract", "integration_smoke"))
        if validator.POLICY != POLICY:
            raise ValueError("Loaded candidate does not implement the declared pointwise policy")
        profile = contract.profile_record(config.dbmol, config.objective, repair_weight=weight if target is not None else None)
        report["expected_profile"] = profile
        try:
            validator.validate_branch_derivatives(decoder, point, branch, candidate, counters, gate, persist,
                current, current_gradient, current_components, target=target, weight=weight)
            gate.update(passed=True, status="passed")
        except ValueError as error:
            gate.update(passed=False, status="failed", gate_error=str(error))
        passed = reader.full_gate_passed(gate, require_repair=target is not None, expected_profile=profile)
    report.update(status="completed", diagnostic_completed=True, validation_passed=passed, candidate_gate_passed=passed)
    persist()


def run_case(plan, case, output_root):
    global _PLAN, _CASE
    _PLAN, _CASE = plan, case
    review_asset(plan["evidence_review"], plan["cases"])
    return helper().run_case(plan, case, output_root, expected_policy="branch_local_v2", probe=probe_case)


def require_readiness(binding, identity):
    """Reauthenticate actual completed native cases and externally bound profiles."""
    from latentmol.experiments import source_manifest
    shared, h, cfg = common(), helper(), config_identity()
    if (not isinstance(identity, dict) or set(identity) != {"source_archive_sha256", "source_manifest_sha256"}
            or any(not h._hash(value) for value in identity.values())
            or source_manifest(Path.cwd())["sha256"] != identity["source_manifest_sha256"]
            or os.environ.get("LATENTMOL_SOURCE_SHA256") != identity["source_archive_sha256"]):
        raise ValueError("Readiness caller is not the declared immutable candidate source")
    plan = validate_readiness_binding(binding)
    path, record = shared._unique_parent(binding["output_root"], plan["name"], binding["plan"]["sha256"])
    execution_identity = shared.authenticate_execution_record(record, identity)
    if (cfg.canonical_config_json(record["config"].get("plan")) != cfg.canonical_config_json(plan)
            or any(record["config"].get(key) != value for key, value in
                (("case_timeout_seconds", CASE_TIMEOUT_SECONDS), ("maximum_full_forwards_per_case", MAXIMUM_FULL_FORWARDS_PER_CASE),
                 ("optimizer_updates", 0), ("new_defog_calls", 0)))):
        raise ValueError("Readiness controller recorded another plan")
    parent = path.parent.resolve()
    rows_path, summary_path = parent / "cases.json", parent / "validation.json"
    rows, summary = _read(rows_path), _read(summary_path)
    if (not isinstance(rows, list) or [row.get("case_id") for row in rows] != list(CASE_IDS)
            or summary.get("cases") != rows or summary.get("status") != "completed"
            or summary.get("validation_passed") is not True or summary.get("plan_sha256") != binding["plan"]["sha256"]
            or summary.get("fd_policy") != POLICY or summary.get("objective_arithmetic") != ARITHMETIC
            or summary.get("evidence_review") != plan["evidence_review"]
            or summary.get("original_gates_not_reclassified") is not True
            or summary.get("optimizer_updates") != 0 or summary.get("new_defog_calls") != 0):
        raise ValueError("Readiness lacks the complete passing two-case summary")
    receipts = []
    for case, row in zip(plan["cases"], rows):
        if any(row.get(key) is not True for key in ("validation_completed", "validation_passed")) or row.get("old_gate_certified") is not False:
            raise ValueError("Readiness has a failed or incomplete native case")
        child = shared._child(parent, case, row)
        child_record = _read(child / "run.json")
        child_identity = shared.authenticate_execution_record(child_record, identity)
        if child_identity["allocation_job_id"] != execution_identity["allocation_job_id"]:
            raise ValueError("Readiness child belongs to a different native allocation")
        recorded = child_record["config"]
        if (recorded.get("case") != case or any(recorded.get(key) != plan[key] for key in
                ("production_release", "production_manifest_sha256", "production_archive_sha256"))
                or recorded.get("case_timeout_seconds") != CASE_TIMEOUT_SECONDS or recorded.get("repeats_per_mode") != 1):
            raise ValueError("Readiness child authenticated different historical inputs")
        for artifact in case["parent_artifacts_sha256"]:
            h.authenticated_artifact(case, artifact)
        original_record = _read(h.authenticated_artifact(case, Path(case["parent_run"]) / "run.json"))
        h.authenticate_source(plan["production_release"], original_record, plan)
        old_path = h.authenticated_artifact(case, Path(case["parent_run"]) / case["arm"] / "gate-0015.json")
        old = _read(old_path)
        h.failed_item(old, policy="branch_local_v2")
        proof = _read(child / "attribution.json")
        if (proof.get("status") != "completed" or any(proof.get(key) is not True for key in
                ("diagnostic_completed", "validation_passed", "candidate_gate_passed", "frozen_state_unchanged",
                 "parent_unchanged", "native_seed_parity", "original_gate_not_reclassified",
                 "baseline_primal_and_gradient_exact", "declared_policy_only_changed"))
                or proof.get("identities_before") != proof.get("identities_after") or not proof.get("identities_before")
                or proof.get("old_gate_certified_by_this_diagnostic") is not False
                or proof.get("optimizer_updates") != 0 or proof.get("new_defog_calls") != 0
                or proof.get("evidence_review") != plan["evidence_review"]
                or proof.get("old_gate") != {"path": str(old_path), "sha256": h._sha(old_path)}):
            raise ValueError("Readiness child lacks unchanged exact-state provenance")
        identities = proof["identities_before"]
        gate = proof.get("branch_validation", {})
        if (gate.get("point_sha256") != old.get("point_sha256") or identities.get("point_sha256") != old.get("point_sha256")
                or gate.get("context_sha256") != old.get("context_sha256") or identities.get("context_sha256") != old.get("context_sha256")
                or any(identities.get(key) != original_record.get(key) for key in ("decoder_state_sha256", "scorer_state_sha256"))
                or gate.get("active_slots") != old.get("active_slots")
                or gate.get("measured", {}).get("loss") != old.get("measured", {}).get("loss")
                or gate.get("measured", {}).get("components") != old.get("measured", {}).get("components")):
            raise ValueError("Readiness gate differs from original point/context/native scalar/frozen state")
        if (not h._hash(proof.get("baseline_graph_sha256"))
                or gate.get("derivative_contract", {}).get("graph_sha256") != proof["baseline_graph_sha256"]):
            raise ValueError("Readiness validator did not certify the original shared graph")
        original, candidate = proof.get("original_config"), proof.get("candidate_config")
        if (not isinstance(original, dict) or not isinstance(candidate, dict)
                or cfg.config_sha256(original) != proof.get("original_config_sha256")
                or cfg.config_sha256(candidate) != proof.get("candidate_config_sha256")
                or original.get("fd_policy") != "branch_local_v2"
                or original.get("dbmol", {}).get("objective_arithmetic") != ARITHMETIC
                or candidate != {**deepcopy(original), "fd_policy": POLICY}
                or cfg.canonical_config_json({key: original_record["config"].get(key) for key in original}) != cfg.canonical_config_json(original)):
            raise ValueError("Readiness changed science outside the declared validation policy")
        weight, target = recorded.get("effective_weight"), proof["target"].get("target_sha256") is not None
        if (target != (case["id"] == "boltz_defog") or proof.get("repair_target_available") is not target
                or proof.get("effective_repair_weight") != weight or weight != old.get("effective_repair_weight", 0)
                or proof["target"] != recorded.get("target")
                or proof["target"].get("target_sha256") != old.get("measured", {}).get("repair_guidance", {}).get("target_sha256")):
            raise ValueError("Readiness target or effective weight differs from historical objective")
        with candidate_modules() as modules:
            profile = modules["pointwise_contract"].profile_record(original["dbmol"], original["objective"],
                repair_weight=weight if target else None)
            valid = modules["integration_smoke"].full_gate_passed(gate, require_repair=target, expected_profile=profile)
        if proof.get("expected_profile") != profile or not valid:
            raise ValueError("Readiness lacks the full externally bound pointwise derivative certificate")
        files = proof.get("candidate_source_files", {})
        if set(files) != set(SOURCE_PATHS) or any(child_record["source"]["files"].get(name) != digest for name, digest in files.items()):
            raise ValueError("Readiness source coverage differs from executed manifest")
        calls = proof.get("actual_full_forward_reservations")
        if type(calls) is not int or not 2 <= calls <= MAXIMUM_FULL_FORWARDS_PER_CASE:
            raise ValueError("Readiness lacks its declared actual native-forward budget")
        costs = _read(child / "oracle_cost.json").get("boltz", {})
        attempted = [value for cost in costs.values() for key, value in cost.items()
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
        "evidence_review": plan["evidence_review"], "original_gates_not_reclassified": True}


def run_study(path, expected, plan):
    """Run the two focused checks serially and retain every failed result."""
    from latentmol.experiments import ExperimentRun, write_json
    from latentmol.boltz.workflow_process import run_logged
    review_asset(plan["evidence_review"], plan["cases"])
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
            "evidence_review": plan["evidence_review"], "original_gates_not_reclassified": True,
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
        raise ValueError("Pointwise readiness requires an immutable allocated DTU HPC job")
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
