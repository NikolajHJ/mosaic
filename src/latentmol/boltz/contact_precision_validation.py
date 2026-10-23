"""Allocated validation of a declared FP64 contact-head arithmetic policy.

Authenticate and replay the historical source091 molecule/pose/target before
binding the NEW production scorer methods. Contact symmetrization, its affine
projection and scalar assembly change precision. Confidence consumes the
original FP32 distogram. The new explicit branch_local_v2 validation policy
requires independent native forward/reverse AD, exact baseline branch parity,
and strict finite differences on the same selected local affinity-readout
branch. Natural secants remain separate preserved observations. Frozen
weights, shared chemistry and finite-difference tolerances remain unchanged.
Every component, held-target repair and combined objective is checked;
scientific failures are retained together instead of stopping after the first.

Forward-mode dual tensors additionally verify each selected directional
derivative independently of reverse AD. They travel through the direct shared
graph path under no_grad, which preserves forward AD and avoids reverse-mode
checkpoint graphs. Unsupported forward AD fails the new readiness gate. It
never falls back to double-backward JVP or approves a historical failure.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import sys
from types import MethodType


ARITHMETIC = "contact_head_and_assembly_float64_v1"
POLICY = "branch_local_v2"
SOURCE103_ARCHIVE = "b0587cf657a09c156a5e43e5ba7337d566adee5e66371f6ea14ae2f6f54fde07"
SOURCE103_MANIFEST = "90ffe1ccda6ea9c01b3487a191c7029ccdcbd91d6f767d5e2456b0d3a41889a5"
CASE_TIMEOUT_SECONDS = 9000
MAXIMUM_FULL_FORWARDS_PER_CASE = 800
CONTACT_KEYS = ("dbmol_contact_loss", "dbmol_contact_pocket_loss", "dbmol_contact_anchor_loss")
COMPONENTS = ("dbmol_contact_loss", "dbmol_affinity_loss", "dbmol_confidence_loss")
SMALL_EPSILONS = (1e-5, 3e-5, 1e-4)
CLAIM_LIMIT = __doc__
_HELPER = None
_PLAN = None
_CASE = None


def adjacent(filename, name, *, register=False):
    """Load new release bytes explicitly, including after old-source bootstrap."""
    path = Path(__file__).with_name(filename + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if register:
        sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if register:
            sys.modules.pop(name, None)
        raise
    return module


def helper():
    """Reuse historical authentication without importing scientific modules early."""
    global _HELPER
    if _HELPER is None:
        _HELPER = adjacent("failed_gate_attribution", "_contact_precision_replay_helper")
    return _HELPER


def source103_path(case):
    paths = [Path(path) for path in case["parent_artifacts_sha256"]
             if "source-103-contact-boundary-diagnostic-attempt001-replay" in path
             and Path(path).name == "attribution.json"]
    if len(paths) != 1 or str(paths[0].with_name("run.json")) not in case["parent_artifacts_sha256"]:
        raise ValueError("Require one completed source103 run/attribution per exact saved case")
    return paths[0]


def load_plan(path, expected):
    """Add one explicit arithmetic field to the original hash-pinned case schema."""
    h = helper()
    if not h._hash(expected) or h._sha(path) != expected:
        raise ValueError("Contact precision plan SHA256 mismatch")
    plan = h._read(path)
    fields = {"version", "name", "output_root", "production_release", "production_manifest_sha256",
              "production_archive_sha256", "cases", "objective_arithmetic", "fd_policy"}
    if (not isinstance(plan, dict) or set(plan) != fields
            or plan["objective_arithmetic"] != ARITHMETIC or plan["fd_policy"] != POLICY):
        raise ValueError("Require the declared contact-head precision plan fields/arithmetic")
    # The parent loader owns the unchanged original source/case schema. Validate
    # its mapping through a temporary in-memory view, never a rewritten plan.
    original_read, original_sha = h._read, h._sha
    old_plan = {k: value for k, value in plan.items() if k not in ("objective_arithmetic", "fd_policy")}
    try:
        h._read = lambda candidate: old_plan if Path(candidate) == Path(path) else original_read(candidate)
        h._sha = lambda candidate: expected if Path(candidate) == Path(path) else original_sha(candidate)
        h.load_plan(path, expected)
    finally:
        h._read, h._sha = original_read, original_sha
    if ([case["id"] for case in plan["cases"]] != ["boltz_only", "boltz_defog"]
            or any(case["arm"] != "fixed_initial_mask" or case["step"] != 15 for case in plan["cases"])):
        raise ValueError("Require both original source091 fixed-mask step15 cases")
    for case in plan["cases"]:
        source103_path(case)
    return plan


def authenticate_reference(case, report):
    """Source103 must establish the same unchanged, completed scientific state."""
    h = helper()
    path = h.authenticated_artifact(case, source103_path(case))
    record = h._read(h.authenticated_artifact(case, path.with_name("run.json")))
    previous = h._read(path)
    if (record.get("status") != "completed"
            or record.get("source_archive_sha256") != SOURCE103_ARCHIVE
            or record.get("source", {}).get("sha256") != SOURCE103_MANIFEST
            or any(previous.get(k) is not True for k in
                   ("diagnostic_completed", "frozen_state_unchanged", "parent_unchanged", "native_seed_parity"))
            or previous.get("identities_before") != report["identities_before"]
            or previous.get("identities_after") != report["identities_before"]
            or previous.get("target") != report["target"]
            or previous.get("original_item") != report["original_item"]
            or previous.get("optimizer_updates") != 0 or previous.get("new_defog_calls") != 0):
        raise ValueError("Source103 does not authenticate this unchanged completed saved state")
    return previous, {"path": str(path), "sha256": h._sha(path), "run_sha256": h._sha(path.with_name("run.json")),
                      "archive_sha256": SOURCE103_ARCHIVE, "manifest_sha256": SOURCE103_MANIFEST}


def candidate_modules():
    """Resolve ACTUAL new production methods while retaining historical imports.

    The old release is first on sys.path by design. Temporary canonical imports
    allow the new dbmol_model globals to bind to the new objective/projection;
    canonical module entries are then restored. No historical module object,
    native file or checkpoint is replaced. The candidate globals are recorded.
    """
    objective = adjacent("dbmol_objective", "latentmol.boltz._contact_precision_objective", register=True)
    projection = adjacent("contact_projection", "latentmol.boltz._contact_precision_projection", register=True)
    names = {"latentmol.boltz.dbmol_objective": objective,
             "latentmol.boltz.contact_projection": projection}
    previous = {name: sys.modules.get(name) for name in names}
    try:
        sys.modules.update(names)
        model = adjacent("dbmol_model", "latentmol.boltz._contact_precision_model", register=True)
        validator = adjacent("branch_local_fd", "latentmol.boltz._contact_precision_branch_fd", register=True)
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    return objective, model, validator


@contextmanager
def candidate_scorer(branch, config):
    """Bind only the actual new production methods and an authenticated manifest."""
    from latentmol.boltz.scoring import boltz_branch_manifest
    from latentmol.experiments import source_manifest
    objective, model, validator = candidate_modules()
    if objective.CONTACT_HEAD64_ARITHMETIC != ARITHMETIC or validator.POLICY != POLICY:
        raise ValueError("Candidate production arithmetic differs from the sealed plan")
    candidate = objective.DBMolObjectiveConfig.from_mapping(
        {**config.dbmol.record(), "objective_arithmetic": ARITHMETIC}, Path.cwd())
    active_config = replace(config, dbmol=candidate, fd_policy=POLICY)
    before = deepcopy(config.record())
    after = deepcopy(active_config.record())
    after["dbmol"] = before["dbmol"]
    after["fd_policy"] = before["fd_policy"]
    if after != before:
        raise ValueError("Candidate changed settings outside the declared arithmetic/validation policy")
    if active_config.recycling_steps != 0:
        raise ValueError("The new branch validation requires original zero recycling")
    scorer = branch.scorer
    original_config, original_manifest = scorer.dbmol_config, branch.manifest
    methods = ("forward", "_reduce", "discrete_reference")
    saved = {name: (name in scorer.__dict__, scorer.__dict__.get(name)) for name in methods}
    old_selection = scorer.last_contact_selection
    try:
        scorer.dbmol_config = candidate
        for name in methods:
            setattr(scorer, name, MethodType(getattr(model.FrozenBoltzDBMol, name), scorer))
        branch.manifest = boltz_branch_manifest(branch.prepared,
            adapter_source_sha256=source_manifest(Path.cwd())["sha256"],
            recycling_steps=scorer.recycling_steps,
            validation_report_sha256=None, dbmol_profile=candidate.record())
        branch.verify_context_unchanged()
        yield active_config, validator
        branch.verify_context_unchanged()
    finally:
        scorer.dbmol_config = original_config
        branch.manifest = original_manifest
        scorer.last_contact_selection = old_selection
        for name, (present, value) in saved.items():
            if present:
                setattr(scorer, name, value)
            else:
                scorer.__dict__.pop(name, None)


def direct_forward(decoder, point, branch, config, counters, target, weight, label):
    """Return live tensors; NEVER detach the point, including forward-AD duals."""
    from latentmol.graph_policy import graph_from_decoder
    from latentmol.multimodel.engine import scalar_loss
    from latentmol.repair_guidance.loss import fixed_target_nll
    decoded = decoder.decode(point)
    graph = graph_from_decoder(decoded, decoder.spec, temperature=config.temperature,
                               fixed_active=branch.prepared.active)
    outputs = counters.call("boltz", label, "forward", lambda: branch(graph), point.device)
    binding, _ = scalar_loss(outputs, config.objective)
    terms = {key: outputs[key] for key in COMPONENTS}
    if target is not None:
        repair, _ = fixed_target_nll(decoded, graph, target, decoder.spec,
            temperature=config.temperature, normalization=config.repair_guidance.normalization)
        terms["repair_nll"] = repair
        combined = binding + weight * repair
    else:
        if weight != 0 or config.repair_guidance.provider_activation_weight != 0:
            raise ValueError("A missing target must be a declared zero-activation control")
        combined = binding
    terms["combined"] = combined
    return decoded, graph, outputs, terms


def metadata(graph, outputs, terms, branch):
    from latentmol.multimodel.study import tensor_digest
    return {"values": {key: float(value.detach()) for key, value in terms.items()},
        "dtypes": {key: str(value.dtype) for key, value in terms.items()},
        "outputs": {key: float(value.detach()) for key, value in outputs.items()},
        "graph_sha256": tensor_digest({k: getattr(graph, k) for k in ("atoms", "charges", "bonds")}),
        "contact_selection": deepcopy(branch.scorer.last_contact_selection)}


def baseline(decoder, point, branch, config, counters, target, weight):
    """Detached endpoint metadata for exact contact-counterfactual parity only."""
    import torch
    with torch.no_grad():
        _, graph, outputs, terms = direct_forward(decoder, point, branch, config, counters,
                                                 target, weight, "candidate_baseline")
        return metadata(graph, outputs, terms, branch)


def endpoint_plan(point, gradients, original_directions, *, seed):
    """Declare the full endpoint union and independent JVP directions in advance."""
    from latentmol.boltz.local_fd import EPSILONS, direction_plan
    from latentmol.multimodel.study import tensor_digest
    plans, unique, jvp = {}, set(), {}
    def include(direction, epsilons):
        for epsilon in epsilons:
            for sign in (-1, 1):
                unique.add(tensor_digest({"z": point + sign * epsilon * direction}))
    for key, gradient in gradients.items():
        if float(gradient.double().norm()) <= 1e-12:
            plans[key] = {"status": "zero_at_current_point_uninformative"}
            continue
        selected, legacy, plan = direction_plan(point, gradient, seed=seed)
        plans[key] = plan
        if plan["pool_exhausted"]:
            continue
        for label, direction in selected.items():
            include(direction, EPSILONS)
            jvp.setdefault(tensor_digest({"direction": direction}), direction)
        include(legacy, EPSILONS)
    for direction, _ in original_directions.values():
        include(direction, SMALL_EPSILONS)
    # Reserve all parity/baseline/production-AD confirmation calls as well as
    # every forward-mode traversal before evaluating any native endpoint.
    maximum = 2 * len(unique) + len(jvp) + 12
    if maximum > MAXIMUM_FULL_FORWARDS_PER_CASE:
        raise ValueError("Complete preselected validation exceeds the declared scorer-call budget")
    return plans, unique, jvp, maximum


def all_checks_pass(report):
    """Fail closed; uninformative components are retained, never declared useful."""
    gates = report.get("candidate_gates", {})
    required = {*COMPONENTS, "combined"}
    if report.get("repair_target_available") is True:
        required.add("repair_only")
    return (report.get("candidate_native_seed_parity") is True
            and report.get("candidate_repeatability") is True
            and report.get("native_unchanged_outputs") is True
            and report.get("source103_counterfactual_exact_parity") is True
            and report.get("branch_validation", {}).get("local_branch_baseline_parity", {}).get("passed") is True
            and report.get("branch_validation", {}).get("derivative_contract", {}).get("passed") is True
            and set(gates) == required
            and all(row.get("passed") is True
                    and row.get("independent_forward_ad", {}).get("passed") is True
                    and row.get("local_branch_fd", {}).get("passed") is True
                    for row in gates.values())
            and gates.get("dbmol_contact_loss", {}).get("natural_secants", {}).get("passed") is True)


class FullForwardBudget:
    """Bound real scorer traversals, including independent forward AD/parity.

    The historical bootstrap already executed two hard-input parity calls.
    Caches may save calls; no outcome can expand this immutable reservation.
    """
    def __init__(self, counters, report):
        self.counters, self.report = counters, report
        self.used = 2

    def call(self, model, purpose, action, function, device):
        if model == "boltz" and action in ("forward", "native_forward", "forward_ad"):
            if self.used >= MAXIMUM_FULL_FORWARDS_PER_CASE:
                raise ValueError("Contact precision full-forward reservation exhausted")
            self.used += 1
            self.report["actual_full_forward_reservations"] = self.used
        return self.counters.call(model, purpose, action, function, device)


def probe_case(decoder, point, branch, config, counters, target, weight, item_name, item,
               *, persist=lambda: None, report=None):
    """Validate both new local calculus and the previously measured mechanism."""
    import torch
    from latentmol.boltz.active_set import score_at, select_slots
    from latentmol.boltz.local_fd import endpoint_row
    from latentmol.repair_guidance.loss import target_score_at
    from latentmol.multimodel.study import tensor_digest
    from latentmol.config_identity import config_sha256, normalize_json
    report = report if report is not None else {}
    counters = FullForwardBudget(counters, report)
    reference, asset = authenticate_reference(_CASE, report)
    components = {}
    measured, total = score_at(decoder, point, branch, config, counters, "original_analytic",
        gradients=True, component_gradients=components, repair_target=target, repair_weight=weight)
    if (tensor_digest({"gradient": components["dbmol_contact_loss"]})
            != reference["cuts"]["graph"]["chain_sha256"]
            or any(row["native_contact"]["baseline_loss"] != measured["components"]["dbmol_contact_loss"]
                   for row in reference["finite_differences"])):
        raise ValueError("Original contact baseline/backward differs from authenticated source103")
    old_directions = helper().reconstruct_directions(point, total if item_name == "combined" else components[item_name], item)
    if {key: value[1] for key, value in old_directions.items()} != reference["direction_replay"]:
        raise ValueError("Original direction replay differs from source103")
    report.update(status="validating", diagnostic_completed=False, validation_passed=False,
        candidate_gate_passed=False, source103_reference=asset, original_contact_gradient_sha256=
        tensor_digest({"gradient": components["dbmol_contact_loss"]}),
        original_direction_replay={key: value[1] for key, value in old_directions.items()},
        optimizer_updates=0, new_defog_calls=0, original_gate_not_reclassified=True,
        repair_target_available=target is not None,
        original_objective_arithmetic=config.dbmol.objective_arithmetic,
        original_config=normalize_json(config.record()),
        original_config_sha256=config_sha256(config.record()),
        candidate_gates={}, old_direction_precision_mechanism=[])
    persist()
    with candidate_scorer(branch, config) as (candidate, validator):
        report["candidate_config"] = normalize_json(candidate.record())
        report["candidate_config_sha256"] = config_sha256(candidate.record())
        report["candidate_branch_manifest"] = branch.manifest.to_dict()
        report["candidate_branch_manifest_sha256"] = config_sha256(branch.manifest.to_dict())
        report["candidate_source_files"] = {name: helper()._sha(Path(__file__).with_name(name + ".py"))
            for name in ("dbmol_objective", "dbmol_model", "contact_projection", "branch_local_fd", "contact_precision_validation")}
        with torch.no_grad():
            native = counters.call("boltz", "candidate_seed_parity", "native_forward", lambda:
                branch.scorer.discrete_reference(branch.prepared.features, branch.prepared.coordinates), point.device)
            adapted = counters.call("boltz", "candidate_seed_parity", "forward", lambda:
                branch(branch.prepared.seed.to_relaxed(decoder.spec)), point.device)
        if set(native) != set(adapted):
            raise ValueError("Candidate hard-input native/adapted output membership differs")
        for key in native:
            torch.testing.assert_close(native[key], adapted[key], rtol=config.diagnostics.parity_rtol,
                                       atol=config.diagnostics.parity_atol)
        report["candidate_native_seed_parity"] = True
        candidate_components = {}
        production, production_gradient = score_at(decoder, point, branch, candidate, counters,
            "candidate_production_backward", gradients=True, component_gradients=candidate_components,
            repair_target=target, repair_weight=weight)
        gradients = {key: candidate_components[key] for key in COMPONENTS}
        if target is not None:
            _, gradients["repair_only"] = target_score_at(decoder, point, branch.prepared.active, target,
                temperature=candidate.temperature, normalization=candidate.repair_guidance.normalization, gradients=True)
        gradients["combined"] = production_gradient
        repeats = [baseline(decoder, point, branch, candidate, counters, target, weight)
                   for _ in range(2)]
        base = repeats[0]
        report["candidate_repeatability"] = (repeats[0] == repeats[1]
            and production["loss"] == base["values"]["combined"]
            and production["components"] == base["outputs"])
        if not report["candidate_repeatability"]:
            raise ValueError("Candidate native grad/no-grad baseline is not exactly repeatable")
        changed = set(CONTACT_KEYS) | {"dbmol_total_loss", "dbmol_weighted_contact_loss"}
        # Affinity/confidence scalar assembly was already FP64 in source091.
        # Every output outside contact and total must remain exact.
        report["native_unchanged_outputs"] = all(value == measured["components"][key]
            for key, value in base["outputs"].items() if key not in changed)
        if not report["native_unchanged_outputs"]:
            raise ValueError("Contact arithmetic changed an affinity/confidence output")
        if any(not torch.equal(gradients[key], components[key]) for key in
               ("dbmol_affinity_loss", "dbmol_confidence_loss")):
            raise ValueError("Contact arithmetic changed an affinity/confidence derivative")
        report["candidate_baseline"] = base
        plans, planned, jvp_directions, bound = endpoint_plan(point, gradients, old_directions,
                                                               seed=candidate.diagnostics.seed)
        report["endpoint_plan"] = {"direction_plans": plans, "unique_native_endpoints": len(planned),
            "unique_independent_jvp_directions": len(jvp_directions), "maximum_full_forwards": bound,
            "maximum_full_forwards_per_case": MAXIMUM_FULL_FORWARDS_PER_CASE,
            "selection_completed_before_endpoint_scores": True}
        cache = {}
        def endpoint(value):
            identity = tensor_digest({"z": value})
            if identity not in planned:
                raise ValueError("Unplanned finite-difference endpoint")
            if identity not in cache:
                with torch.no_grad():
                    decoded, graph, outputs, terms = direct_forward(decoder, value, branch, candidate,
                        counters, target, weight, "candidate_endpoint")
                    row = metadata(graph, outputs, terms, branch)
                    selection = select_slots(decoded, temperature=candidate.temperature,
                        threshold=candidate.threshold, require_nonempty=False)
                    row["natural_selector_crossed"] = not torch.equal(selection.active, branch.prepared.active)
                    cache[identity] = row
                report["unique_endpoints_scored"] = len(cache)
            return cache[identity]
        # Reuse the actual production policy, including its strict independent
        # JVP, branch-baseline scalar/VJP parity and separately retained natural
        # secants. No duplicate validator can drift from the subsequent pilot.
        branch_report = {"status": "validating", "passed": False, "native_seed_parity": True,
                         "measured": production, "gradient_norm": float(production_gradient.double().norm())}
        report["branch_validation"] = branch_report
        try:
            validator.validate_branch_derivatives(decoder, point, branch, candidate, counters,
                branch_report, persist, production, production_gradient,
                {key: candidate_components[key] for key in COMPONENTS}, target=target, weight=weight)
        except ValueError as error:
            # A failed scientific gate is a completed readiness observation.
            # Authentication/baseline failures cannot be reclassified this way.
            if branch_report.get("local_branch_baseline_parity", {}).get("passed") is not True:
                raise
            branch_report["gate_error"] = str(error)
        gates = {**branch_report.get("component_gradients", {}),
                 "combined": branch_report.get("local_fd_validation", {})}
        if target is not None:
            gates["repair_only"] = branch_report.get("repair_only_validation", {})
        report["candidate_gates"] = gates
        if set(gates) != set(gradients):
            raise ValueError("Production validator omitted a required scalar")
        for key, gate in gates.items():
            selection = gate.get("direction_selection")
            if selection is not None and selection != plans[key]:
                raise ValueError("Actual production plan differs from the pre-endpoint plan")
        # These original directions are mechanism evidence, not a replacement
        # passing direction set. Authenticate against source103's widened-head
        # counterfactual and preserve native roundoff/subterm cancellation.
        for label, (direction, _) in old_directions.items():
            for epsilon in SMALL_EPSILONS:
                old = [row for row in reference["finite_differences"]
                       if row["direction"] == label and row["epsilon"] == epsilon]
                if len(old) != 1:
                    raise ValueError("Missing unique original source103 endpoint")
                fresh = {sign: endpoint(point + multiplier * epsilon * direction)
                         for sign, multiplier in (("plus", 1), ("minus", -1))}
                for sign in fresh:
                    if (fresh[sign]["graph_sha256"] != old[0][sign]["graph_sha256"]
                            or any(fresh[sign]["outputs"][key] != old[0][sign]["widened_final_affine_contact"][key]
                                   for key in CONTACT_KEYS)):
                        raise ValueError("Actual contact policy differs from the source103 widened-head control")
                row = endpoint_row(point, direction, gradients["dbmol_contact_loss"], epsilon,
                    *[{"loss": fresh[sign]["values"]["dbmol_contact_loss"], "loss_dtype": "torch.float64"}
                      for sign in ("plus", "minus")], base["values"]["dbmol_contact_loss"],
                    rtol=candidate.diagnostics.fd_rtol, atol=candidate.diagnostics.fd_atol)
                subterms = {key: (fresh["plus"]["outputs"][key] - fresh["minus"]["outputs"][key]) / (2 * epsilon)
                            for key in CONTACT_KEYS[1:]}
                numerator = sum(abs(value) for value in subterms.values())
                denominator = abs(sum(subterms.values()))
                report["old_direction_precision_mechanism"].append({"direction": label, "epsilon": epsilon,
                    "original_native_contact": old[0]["native_contact"], "candidate_contact": row,
                    "contact_subterm_secants": subterms,
                    "directional_cancellation_ratio": numerator / denominator if denominator > 0 else None,
                    "native_minus_widened_secant": old[0]["native_contact"]["central_difference"] - row["central_difference"],
                    "source103_widened_counterfactual_exact": True})
                persist()
        report["source103_counterfactual_exact_parity"] = True
        report["candidate_gate_passed"] = all_checks_pass(report)
        branch_report.update(status="passed" if report["candidate_gate_passed"] else "failed",
                             passed=report["candidate_gate_passed"])
    report.update(status="completed", diagnostic_completed=True,
        validation_passed=report["candidate_gate_passed"], old_gate_certified_by_this_diagnostic=False,
        interpretation="Changed arithmetic must pass every component/combined gate and independent JVP; no old failure is approved.")
    persist()
    return report


def run_case(plan, case, output_root):
    """Fresh original-source bootstrap owns all state/asset/frozen-identity checks."""
    global _CASE, _PLAN
    _CASE, _PLAN = case, plan
    replay = helper()
    previous = replay.probe_case
    replay.probe_case = probe_case
    replay.CASE_TIMEOUT_SECONDS, replay.REPEATS, replay.CLAIM_LIMIT = CASE_TIMEOUT_SECONDS, 1, CLAIM_LIMIT
    try:
        return replay.run_case(plan, case, output_root)
    finally:
        replay.probe_case = previous


def validate_readiness_binding(binding):
    """Validate only immutable binding shape/assets, without requiring results.

    CPU preparation may call this before the native job runs. A valid binding
    is never itself a passed readiness certificate.
    """
    h = helper()
    if (not isinstance(binding, dict) or set(binding) != {"plan", "output_root"}
            or not isinstance(binding["plan"], dict) or set(binding["plan"]) != {"path", "sha256"}
            or not h._absolute(binding["plan"]["path"]) or not h._hash(binding["plan"]["sha256"])
            or not h._absolute(binding["output_root"])):
        raise ValueError("Readiness needs exactly an absolute immutable plan binding and output_root")
    plan = load_plan(Path(binding["plan"]["path"]), binding["plan"]["sha256"])
    if plan["output_root"] != binding["output_root"]:
        raise ValueError("Readiness output root differs from its sealed plan")
    return plan


def authenticate_execution_record(record, identity):
    """Check canonical config, executable identity and an actual allocation ID."""
    from latentmol.config_identity import CONFIG_HASH_VERSION, verify_config_identity
    source = record.get("source", {})
    files = source.get("files")
    job = record.get("environment", {}).get("job_id")
    if not isinstance(job, str) or not job or not job.isascii() or not job.isdigit():
        raise ValueError("Readiness execution lacks a nonempty numeric allocation job ID")
    if (record.get("status") != "completed" or record.get("config_hash_version") != CONFIG_HASH_VERSION
            or not isinstance(files, dict) or not files
            or helper()._digest(files) != identity["source_manifest_sha256"]
            or source.get("sha256") != identity["source_manifest_sha256"]
            or record.get("source_archive_sha256") != identity["source_archive_sha256"]):
        raise ValueError("Readiness execution is incomplete or belongs to a different immutable source")
    return {**verify_config_identity(record), "allocation_job_id": job}


def require_readiness(binding, identity):
    """Authenticate both completed native saved-state certificates before pilot.

    Each arm must reference exactly its sealed old case and actual completed
    child, with canonical configuration/source proofs, preserved original
    assets, frozen-state equality, the new full production gate and a passing
    natural contact check. Failed, partial, ambiguous or differently sourced
    records cannot trigger a pilot. Return byte-level receipts for its ledger.
    """
    from latentmol.config_identity import config_sha256, canonical_config_json
    from latentmol.experiments import source_manifest
    from latentmol.repair_guidance.integration_smoke import full_gate_passed
    h = helper()
    if (not isinstance(identity, dict) or set(identity) != {"source_manifest_sha256", "source_archive_sha256"}
            or any(not h._hash(value) for value in identity.values())):
        raise ValueError("Readiness needs exact immutable source manifest/archive identities")
    if (source_manifest(Path.cwd())["sha256"] != identity["source_manifest_sha256"]
            or os.environ.get("LATENTMOL_SOURCE_SHA256") != identity["source_archive_sha256"]):
        raise ValueError("Readiness caller is not running the declared immutable candidate source")
    plan = validate_readiness_binding(binding)
    root = Path(binding["output_root"]).resolve()
    candidates = []
    for path in root.glob("*/run.json"):
        record = h._read(path)
        if record.get("name") == plan["name"] and record.get("config", {}).get("plan_sha256") == binding["plan"]["sha256"]:
            candidates.append((path, record))
    if len(candidates) != 1:
        raise ValueError("Readiness requires one unambiguous parent execution of the sealed plan")
    parent_path, parent_record = candidates[0]
    parent = parent_path.parent
    parent_config = authenticate_execution_record(parent_record, identity)
    if canonical_config_json(parent_record["config"].get("plan")) != canonical_config_json(plan):
        raise ValueError("Readiness parent recorded another plan")
    rows_path, summary_path = parent / "cases.json", parent / "validation.json"
    rows, summary = h._read(rows_path), h._read(summary_path)
    if (summary.get("status") != "completed" or summary.get("validation_passed") is not True
            or summary.get("plan_sha256") != binding["plan"]["sha256"]
            or summary.get("objective_arithmetic") != ARITHMETIC or summary.get("fd_policy") != POLICY
            or summary.get("cases") != rows
            or [row.get("case_id") for row in rows] != [case["id"] for case in plan["cases"]]
            or summary.get("original_gates_not_reclassified") is not True
            or summary.get("optimizer_updates") != 0 or summary.get("new_defog_calls") != 0):
        raise ValueError("Readiness parent lacks the complete passing two-case summary")
    receipts = []
    for declared, row in zip(plan["cases"], rows):
        execution = row.get("execution", {})
        paths = execution.get("run_paths", [])
        if (row.get("validation_completed") is not True or row.get("validation_passed") is not True
                or execution.get("returncode") != 0 or execution.get("timed_out") is not False
                or len(paths) != 1 or row.get("old_gate_certified") is not False):
            raise ValueError("Readiness case is failed, incomplete or has no unique completed child")
        child = Path(paths[0]).resolve()
        if child.parent != parent / declared["id"]:
            raise ValueError("Readiness child escaped its sealed case output directory")
        other_children = list((parent / declared["id"]).glob("*/run.json"))
        if len(other_children) != 1 or other_children[0].resolve() != child / "run.json":
            raise ValueError("Readiness case contains ambiguous native executions")
        child_record = h._read(child / "run.json")
        config_identity = authenticate_execution_record(child_record, identity)
        if config_identity["allocation_job_id"] != parent_config["allocation_job_id"]:
            raise ValueError("Readiness child belongs to a different native allocation")
        config = child_record["config"]
        if (config.get("case") != declared or config.get("production_release") != plan["production_release"]
                or config.get("production_manifest_sha256") != plan["production_manifest_sha256"]
                or config.get("production_archive_sha256") != plan["production_archive_sha256"]):
            raise ValueError("Readiness child did not authenticate the declared historical case")
        proof = h._read(child / "attribution.json")
        if (proof.get("status") != "completed"
                or any(proof.get(key) is not True for key in ("diagnostic_completed", "validation_passed",
                       "candidate_gate_passed", "frozen_state_unchanged", "parent_unchanged", "original_gate_not_reclassified"))
                or proof.get("identities_before") != proof.get("identities_after")
                or proof.get("old_gate_certified_by_this_diagnostic") is not False
                or proof.get("optimizer_updates") != 0 or proof.get("new_defog_calls") != 0
                or not all_checks_pass(proof)
                or not full_gate_passed(proof.get("branch_validation", {}), require_repair=declared["id"] == "boltz_defog")):
            raise ValueError("Readiness child has no authenticated full native derivative certificate")
        original, candidate = proof.get("original_config"), proof.get("candidate_config")
        manifest = proof.get("candidate_branch_manifest")
        if (not isinstance(original, dict) or not isinstance(candidate, dict) or not isinstance(manifest, dict)
                or config_sha256(original) != proof.get("original_config_sha256")
                or config_sha256(candidate) != proof.get("candidate_config_sha256")
                or config_sha256(manifest) != proof.get("candidate_branch_manifest_sha256")
                or manifest.get("adapter_source_sha256") != identity["source_manifest_sha256"]):
            raise ValueError("Readiness candidate config/manifest identity changed")
        expected = deepcopy(original)
        expected["fd_policy"] = POLICY
        expected["dbmol"]["objective_arithmetic"] = ARITHMETIC
        if canonical_config_json(expected) != canonical_config_json(candidate):
            raise ValueError("Readiness changed science outside declared arithmetic/validation")
        for filename, expected_sha in proof.get("candidate_source_files", {}).items():
            if child_record["source"]["files"].get("src/latentmol/boltz/" + filename + ".py") != expected_sha:
                raise ValueError("Readiness candidate executable bytes differ from its manifest")
        if set(proof.get("candidate_source_files", {})) != {
                "dbmol_objective", "dbmol_model", "contact_projection", "branch_local_fd", "contact_precision_validation"}:
            raise ValueError("Readiness candidate source coverage is incomplete")
        # Recheck all immutable old inputs now, rather than trusting only the
        # saved post-run boolean. No model is loaded or numerically evaluated.
        for path in declared["parent_artifacts_sha256"]:
            h.authenticated_artifact(declared, path)
        original_record = h._read(h.authenticated_artifact(declared, Path(declared["parent_run"]) / "run.json"))
        h.authenticate_source(plan["production_release"], original_record, plan)
        original_projection = {key: original_record["config"].get(key) for key in original}
        if canonical_config_json(original_projection) != canonical_config_json(original):
            raise ValueError("Readiness original parsed config differs from its actual historical parent")
        receipts.append({"case_id": declared["id"], "run": str(child),
            "run_sha256": h._sha(child / "run.json"), "attribution_sha256": h._sha(child / "attribution.json"),
            "config_identity": config_identity, "candidate_config_sha256": proof["candidate_config_sha256"],
            "candidate_branch_manifest_sha256": proof["candidate_branch_manifest_sha256"]})
    return {"passed": True, "plan": binding["plan"], "output_root": str(root),
        "controller_run": str(parent), "controller_run_sha256": h._sha(parent_path),
        "cases_sha256": h._sha(rows_path), "summary_sha256": h._sha(summary_path),
        "controller_config_identity": parent_config, "source_identity": identity,
        "allocation_job_id": parent_config["allocation_job_id"],
        "objective_arithmetic": ARITHMETIC, "fd_policy": POLICY, "cases": receipts,
        "original_gates_not_reclassified": True}


def run_study(path, expected):
    """Retain both results; a completed failed readiness report is never promoted."""
    plan = load_plan(path, expected)
    from latentmol.experiments import ExperimentRun, write_json
    from latentmol.boltz.workflow_process import run_logged
    with ExperimentRun(Path(plan["output_root"]), plan["name"], {"plan": plan, "plan_sha256": expected,
            "validation_module_sha256": helper()._sha(__file__), "claim_limit": CLAIM_LIMIT,
            "maximum_full_forwards": 2 * MAXIMUM_FULL_FORWARDS_PER_CASE,
            "maximum_serial_child_seconds": 2 * CASE_TIMEOUT_SECONDS}, Path.cwd()) as run:
        rows = []
        for case in plan["cases"]:
            execution = run_logged([sys.executable, str(Path(__file__).resolve()), "--plan", str(Path(path).resolve()),
                "--sha256", expected, "--child", case["id"], "--child-output", str(run.path / case["id"])],
                run.path / (case["id"] + ".log"), timeout_seconds=CASE_TIMEOUT_SECONDS)
            completed, passed = False, False
            if not execution["timed_out"] and execution["returncode"] == 0 and len(execution["run_paths"]) == 1:
                child = Path(execution["run_paths"][0])
                evidence, record = helper()._read(child / "attribution.json"), helper()._read(child / "run.json")
                completed = (record.get("status") == "completed" and evidence.get("status") == "completed"
                    and all(evidence.get(key) is True for key in
                            ("diagnostic_completed", "frozen_state_unchanged", "parent_unchanged")))
                passed = completed and evidence.get("validation_passed") is True and all_checks_pass(evidence)
            rows.append({"case_id": case["id"], "validation_completed": completed, "validation_passed": passed,
                         "execution": execution, "old_gate_certified": False})
            write_json(run.path / "cases.json", rows)
        passed = len(rows) == 2 and all(row["validation_passed"] for row in rows)
        write_json(run.path / "validation.json", {"status": "completed", "validation_passed": passed,
            "plan_sha256": expected, "objective_arithmetic": ARITHMETIC, "fd_policy": POLICY, "cases": rows,
            "original_gates_not_reclassified": True, "optimizer_updates": 0, "new_defog_calls": 0})
        run.annotate(validation_cases_completed=sum(row["validation_completed"] for row in rows),
            validation_cases_passed=sum(row["validation_passed"] for row in rows),
            validation_cases_requested=2, validation_passed=passed, old_gates_certified=False)
        return run.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--child", help=argparse.SUPPRESS)
    parser.add_argument("--child-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID") or not helper()._hash(os.environ.get("LATENTMOL_SOURCE_SHA256")):
        raise ValueError("Contact precision validation requires immutable allocated DTU HPC execution")
    plan = load_plan(args.plan, args.sha256)
    if args.child is not None:
        selected = [case for case in plan["cases"] if case["id"] == args.child]
        if len(selected) != 1 or args.child_output is None:
            raise ValueError("Unknown child or missing unique child output root")
        print(run_case(plan, selected[0], args.child_output), flush=True)
    else:
        if args.child_output is not None:
            raise ValueError("Child output is only valid with --child")
        print(run_study(args.plan, args.sha256), flush=True)


if __name__ == "__main__":
    main()
