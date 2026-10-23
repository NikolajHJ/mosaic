"""Allocated engineering fixtures; no saved-state or native gradient approval."""
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SPEC = importlib.util.spec_from_file_location("_contact_precision_validation_fixture",
    Path(__file__).parents[1] / "src/latentmol/boltz/contact_precision_validation.py")
observer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(observer)


def sealed_plan(tmp_path):
    cases = []
    for name in ("boltz_only", "boltz_defog"):
        ref = tmp_path / "source-103-contact-boundary-diagnostic-attempt001-replay" / name / "attribution.json"
        cases.append({"id": name, "parent_run": str(tmp_path / "old" / name),
            "arm": "fixed_initial_mask", "step": 15,
            "parent_artifacts_sha256": {str(ref): "a" * 64, str(ref.with_name("run.json")): "b" * 64}})
    plan = {"version": 1, "name": "contact-precision-fixture", "output_root": str(tmp_path / "output"),
        "production_release": str(tmp_path / "original"), "production_manifest_sha256": "c" * 64,
        "production_archive_sha256": "d" * 64, "cases": cases, "objective_arithmetic": observer.ARITHMETIC,
        "fd_policy": observer.POLICY}
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path, plan


def test_plan_requires_exact_arithmetic_both_cases_and_source103_reference(tmp_path):
    path, plan = sealed_plan(tmp_path)
    assert observer.load_plan(path, observer.helper()._sha(path)) == plan
    for change in ({"objective_arithmetic": "native_v1"}, {"fd_policy": "local_v1"}, {"silent_waiver": True},
                   {"cases": plan["cases"][:1]}):
        path.write_text(json.dumps({**plan, **change}), encoding="utf-8")
        with pytest.raises(ValueError):
            observer.load_plan(path, observer.helper()._sha(path))
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        observer.load_plan(path, "f" * 64)


def test_plan_does_not_change_old_loader_or_plan_bytes(tmp_path):
    path, _ = sealed_plan(tmp_path)
    helper = observer.helper()
    before = (helper._read, helper._sha, path.read_bytes())
    observer.load_plan(path, helper._sha(path))
    assert (helper._read, helper._sha, path.read_bytes()) == before


def complete_report():
    return {"candidate_native_seed_parity": True,
        "candidate_repeatability": True, "native_unchanged_outputs": True,
        "source103_counterfactual_exact_parity": True,
        "branch_validation": {"local_branch_baseline_parity": {"passed": True}, "derivative_contract": {"passed": True}},
        "candidate_gates": {key: {"passed": True, "natural_secants": {"passed": True},
                                 "independent_forward_ad": {"passed": True}, "local_branch_fd": {"passed": True}}
                            for key in (*observer.COMPONENTS, "combined")}}


def test_readiness_requires_every_component_and_independent_native_jvp():
    report = complete_report()
    assert observer.all_checks_pass(report)
    for key in ("candidate_native_seed_parity", "candidate_repeatability",
                "native_unchanged_outputs", "source103_counterfactual_exact_parity"):
        assert not observer.all_checks_pass({**report, key: False})
    for key in report["candidate_gates"]:
        broken = {**report, "candidate_gates": {**report["candidate_gates"], key: {"passed": False}}}
        assert not observer.all_checks_pass(broken)
    assert not observer.all_checks_pass({**report, "candidate_gates": {}})
    assert not observer.all_checks_pass({**report, "branch_validation": {"derivative_contract": {"passed": False}}})
    contact = report["candidate_gates"]["dbmol_contact_loss"]
    assert not observer.all_checks_pass({**report, "candidate_gates": {**report["candidate_gates"],
        "dbmol_contact_loss": {**contact, "natural_secants": {"passed": False}}}})
    assert not observer.all_checks_pass({**report, "repair_target_available": True})


def test_full_forward_budget_counts_native_jvp_parity_and_fails_closed(monkeypatch):
    calls = []
    counters = SimpleNamespace(call=lambda *args: calls.append(args[:3]) or args[3]())
    report = {}
    bounded = observer.FullForwardBudget(counters, report)
    monkeypatch.setattr(observer, "MAXIMUM_FULL_FORWARDS_PER_CASE", 5)
    for action in ("native_forward", "forward", "forward_ad"):
        assert bounded.call("boltz", "fixture", action, lambda: 7, "cpu") == 7
    assert report["actual_full_forward_reservations"] == 5
    assert bounded.call("boltz", "fixture", "backward", lambda: 8, "cpu") == 8
    with pytest.raises(ValueError, match="reservation exhausted"):
        bounded.call("boltz", "fixture", "forward", lambda: 7, "cpu")
    assert len(calls) == 4


def test_endpoint_union_is_preselected_and_refuses_budget_growth(monkeypatch):
    point = torch.zeros(1, 4)
    gradients = {"contact": torch.tensor([[1., 2., 3., 4.]]),
                 "combined": torch.tensor([[4., 3., 2., 1.]])}
    old = {"original": (torch.tensor([[1., 0., 0., 0.]]), {})}
    plans, endpoints, duals, bound = observer.endpoint_plan(point, gradients, old, seed=17)
    assert set(plans) == set(gradients)
    assert endpoints and duals and bound == 2 * len(endpoints) + len(duals) + 12
    assert all(plan["selection_completed_before_endpoint_scores"] for plan in plans.values())
    monkeypatch.setattr(observer, "MAXIMUM_FULL_FORWARDS_PER_CASE", 1)
    with pytest.raises(ValueError, match="budget"):
        observer.endpoint_plan(point, gradients, old, seed=17)


def test_zero_component_is_retained_as_uninformative_not_a_success():
    plans, _, _, _ = observer.endpoint_plan(torch.zeros(1, 3), {"zero": torch.zeros(1, 3)}, {}, seed=17)
    assert plans["zero"]["status"] == "zero_at_current_point_uninformative"


def test_forward_ad_flows_through_direct_shared_graph_under_no_grad():
    """Small fake scorer verifies plumbing only; no native-Boltz claim follows."""
    from latentmol.representation import DecoderOutput, GraphSpec
    from latentmol.multimodel.config import Objective
    from latentmol.multimodel.engine import Counters
    spec = GraphSpec(max_nodes=2)
    class Decoder:
        saw_dual = False
        def decode(self, point):
            assert not torch.is_grad_enabled()
            self.saw_dual = torch.autograd.forward_ad.unpack_dual(point).tangent is not None
            x = point.reshape(1, 1, 1)
            atoms = x * torch.arange(len(spec.atomic_numbers))[None, None].expand(1, 2, -1)
            charges = x * torch.arange(len(spec.formal_charges))[None, None].expand(1, 2, -1)
            bonds = x[:, :, None] * torch.arange(len(spec.bond_names))[None, None, None].expand(1, 2, 2, -1)
            return DecoderOutput(atoms, charges, bonds)
    decoder = Decoder()
    decoder.spec = spec
    class Branch:
        prepared = SimpleNamespace(active=torch.tensor([True, True]))
        scorer = SimpleNamespace(last_contact_selection={})
        def __call__(self, graph):
            assert not torch.is_grad_enabled()
            value = graph.atoms[..., 1].sum() + graph.bonds[..., 1].sum()
            return {**{key: value for key in observer.COMPONENTS}, "dbmol_total_loss": 3 * value}
    config = SimpleNamespace(temperature=1., objective=Objective(output="dbmol_total_loss", direction="minimize"),
                             repair_guidance=SimpleNamespace(provider_activation_weight=0))
    from latentmol.boltz.branch_local_fd import native_jvp
    observation = native_jvp(decoder, torch.tensor([[.3]]), torch.tensor([[1.]]),
                             Branch(), config, Counters())
    assert decoder.saw_dual
    assert set(observation) == {*observer.COMPONENTS, "combined"}
    assert all(torch.isfinite(torch.tensor(row["tangent"])) for row in observation.values())


def test_candidate_parser_and_methods_restore_even_after_exception(tmp_path, monkeypatch):
    from latentmol.boltz import scoring
    from latentmol import experiments
    from latentmol.boltz.dbmol_objective import DBMolObjectiveConfig
    original = DBMolObjectiveConfig.from_mapping({"confidence_checkpoint": {"path": str(tmp_path / "confidence.ckpt"),
        "sha256": "a" * 64}, "pocket_residues": [{"asym_id": 0, "residue_index": 1}],
        "objective_arithmetic": "contact_and_assembly_float64_v1"}, tmp_path)
    @dataclass(frozen=True)
    class Config:
        dbmol: object
        fd_policy: str = "local_v1"
        recycling_steps: int = 0
        def record(self):
            return {"dbmol": self.dbmol.record(), "fd_policy": self.fd_policy,
                    "recycling_steps": self.recycling_steps, "other_setting": "unchanged"}
    class Scorer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dbmol_config = original
            self.recycling_steps = 0
            self.last_contact_selection = {"old": True}
        def forward(self, *args):
            return "old"
    scorer = Scorer()
    old_manifest = SimpleNamespace(adapter_source_sha256="b" * 64, validation_report_sha256=None)
    branch = SimpleNamespace(scorer=scorer, prepared=object(), manifest=old_manifest,
                             verify_context_unchanged=lambda: "same")
    monkeypatch.setenv("LATENTMOL_SOURCE_SHA256", "c" * 64)
    monkeypatch.setattr(experiments, "source_manifest", lambda _path: {"sha256": "c" * 64})
    monkeypatch.setattr(scoring, "boltz_branch_manifest", lambda *args, **kwargs: kwargs)
    with pytest.raises(RuntimeError, match="fixture"):
        with observer.candidate_scorer(branch, Config(original)) as (candidate, validator):
            assert candidate.dbmol.objective_arithmetic == observer.ARITHMETIC
            assert candidate.fd_policy == validator.POLICY == observer.POLICY
            assert scorer.dbmol_config is candidate.dbmol
            assert branch.manifest["adapter_source_sha256"] == "c" * 64
            assert all(name in scorer.__dict__ for name in ("forward", "_reduce", "discrete_reference"))
            raise RuntimeError("fixture")
    assert scorer.dbmol_config is original and branch.manifest is old_manifest
    assert scorer.last_contact_selection == {"old": True}
    assert all(name not in scorer.__dict__ for name in ("forward", "_reduce", "discrete_reference"))


def test_source103_reference_refuses_incomplete_or_wrong_state(tmp_path):
    path, plan = sealed_plan(tmp_path)
    case = plan["cases"][0]
    ref = observer.source103_path(case)
    ref.parent.mkdir(parents=True)
    report = {"identities_before": {"z": "a" * 64}, "target": {"disabled": True}, "original_item": "combined"}
    evidence = {**report, "identities_after": report["identities_before"], "diagnostic_completed": True,
        "frozen_state_unchanged": True, "parent_unchanged": True, "native_seed_parity": True,
        "optimizer_updates": 0, "new_defog_calls": 0}
    ref.write_text(json.dumps(evidence), encoding="utf-8")
    run = ref.with_name("run.json")
    run.write_text(json.dumps({"status": "completed", "source_archive_sha256": observer.SOURCE103_ARCHIVE,
                              "source": {"sha256": observer.SOURCE103_MANIFEST}}), encoding="utf-8")
    case["parent_artifacts_sha256"] = {str(ref): observer.helper()._sha(ref), str(run): observer.helper()._sha(run)}
    assert observer.authenticate_reference(case, report)[0] == evidence
    for key in ("diagnostic_completed", "frozen_state_unchanged", "parent_unchanged", "native_seed_parity"):
        bad = {**evidence, key: False}
        ref.write_text(json.dumps(bad), encoding="utf-8")
        case["parent_artifacts_sha256"][str(ref)] = observer.helper()._sha(ref)
        with pytest.raises(ValueError, match="authenticate"):
            observer.authenticate_reference(case, report)


def test_cli_refuses_unallocated_execution(monkeypatch, tmp_path):
    monkeypatch.delenv("LSB_JOBID", raising=False)
    monkeypatch.setenv("LATENTMOL_SOURCE_SHA256", "a" * 64)
    monkeypatch.setattr("sys.argv", ["validator", "--plan", str(tmp_path / "plan.json"), "--sha256", "b" * 64])
    with pytest.raises(ValueError, match="allocated DTU HPC"):
        observer.main()


def test_readiness_binding_can_prepare_before_gpu_but_never_claims_a_pass(tmp_path):
    path, plan = sealed_plan(tmp_path)
    binding = {"plan": {"path": str(path), "sha256": observer.helper()._sha(path)},
               "output_root": plan["output_root"]}
    assert observer.validate_readiness_binding(binding) == plan
    for bad in ({**binding, "output_root": str(tmp_path / "other")}, {**binding, "passed": True},
                {**binding, "plan": {**binding["plan"], "sha256": "f" * 64}}):
        with pytest.raises(ValueError):
            observer.validate_readiness_binding(bad)


def test_execution_record_requires_canonical_config_and_complete_source_identity():
    from latentmol.config_identity import config_sha256, CONFIG_HASH_VERSION
    files = {"src/fixture.py": "a" * 64}
    identity = {"source_manifest_sha256": observer.helper()._digest(files), "source_archive_sha256": "b" * 64}
    config = {"context": {2: "two", 10: "ten"}}
    record = {"status": "completed", "config": config, "config_sha256": config_sha256(config),
        "config_hash_version": CONFIG_HASH_VERSION, "source": {"files": files, "sha256": identity["source_manifest_sha256"]},
        "source_archive_sha256": identity["source_archive_sha256"], "environment": {"job_id": "29799999"}}
    assert observer.authenticate_execution_record(record, identity)["verified"]
    for bad in ({**record, "status": "failed"}, {**record, "source_archive_sha256": "c" * 64},
                {**record, "config_sha256": "d" * 64}, {**record, "source": {"sha256": identity["source_manifest_sha256"]}},
                {**record, "environment": {"job_id": ""}}, {**record, "environment": {"job_id": 29799999}},
                {**record, "environment": {"job_id": "unknown"}}):
        with pytest.raises(ValueError):
            observer.authenticate_execution_record(bad, identity)


def readiness_fixture(tmp_path, monkeypatch):
    """Metadata-only fabricated receipts exercise gates, never native models."""
    from latentmol.config_identity import config_sha256, CONFIG_HASH_VERSION
    from latentmol import experiments
    path, plan = sealed_plan(tmp_path)
    files = {"src/latentmol/boltz/" + name + ".py": "a" * 64 for name in
             ("dbmol_objective", "dbmol_model", "contact_projection", "branch_local_fd", "contact_precision_validation")}
    identity = {"source_manifest_sha256": observer.helper()._digest(files), "source_archive_sha256": "b" * 64}
    monkeypatch.setenv("LATENTMOL_SOURCE_SHA256", identity["source_archive_sha256"])
    monkeypatch.setattr(experiments, "source_manifest", lambda _root: {"sha256": identity["source_manifest_sha256"]})
    # Exact original source authentication has separate tests. This fixture
    # isolates the new receipt validator with tiny known metadata artifacts.
    monkeypatch.setattr(observer.helper(), "authenticated_artifact", lambda _case, artifact: Path(artifact))
    monkeypatch.setattr(observer.helper(), "authenticate_source", lambda *args: None)
    def write(target, data):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data), encoding="utf-8")
    def record(name, config):
        return {"status": "completed", "name": name, "config": config,
            "config_sha256": config_sha256(config), "config_hash_version": CONFIG_HASH_VERSION,
            "source": {"files": files, "sha256": identity["source_manifest_sha256"]},
            "source_archive_sha256": identity["source_archive_sha256"], "environment": {"job_id": "29799999"}}
    parent = Path(plan["output_root"]) / "unique-parent"
    rows = []
    for case in plan["cases"]:
        child = parent / case["id"] / "unique-child"
        original = {"fd_policy": "local_v1", "dbmol": {"objective_arithmetic": "contact_and_assembly_float64_v1"}}
        candidate = {"fd_policy": observer.POLICY, "dbmol": {"objective_arithmetic": observer.ARITHMETIC}}
        manifest = {"adapter_source_sha256": identity["source_manifest_sha256"]}
        write(Path(case["parent_run"]) / "run.json", {"config": original})
        config = {"case": case, "production_release": plan["production_release"],
                  "production_manifest_sha256": plan["production_manifest_sha256"],
                  "production_archive_sha256": plan["production_archive_sha256"]}
        write(child / "run.json", record(case["id"], config))
        proof = complete_report()
        if case["id"] == "boltz_defog":
            proof["repair_target_available"] = True
            proof["candidate_gates"]["repair_only"] = dict(proof["candidate_gates"]["combined"])
        for item in proof["candidate_gates"].values():
            item["status"] = "validated_nonzero"
            item["baseline_loss"] = 0.
            item["independent_forward_ad"].update(rtol=1e-3, atol=1e-6,
                rows=[{"direction": label, "primal": 0., "tangent": 1., "reverse_projection": 1.,
                       "absolute_error": 0., "tolerance": 1e-6 + 1e-3, "passed": True, "primal_exact": True}
                      for label in ("gradient", "random_1", "random_2", "random_3")])
        branch = proof["branch_validation"]
        branch.update(status="passed", passed=True, native_seed_parity=True,
            component_gradients={key: proof["candidate_gates"][key] for key in observer.COMPONENTS},
            local_fd_validation=proof["candidate_gates"]["combined"])
        branch["derivative_contract"].update(version=observer.POLICY, branch_holding_used_in_optimization=False)
        if case["id"] == "boltz_defog":
            branch["repair_only_validation"] = proof["candidate_gates"]["repair_only"]
        proof.update(status="completed", diagnostic_completed=True, validation_passed=True, candidate_gate_passed=True,
            frozen_state_unchanged=True, parent_unchanged=True, original_gate_not_reclassified=True,
            identities_before={"z": "a" * 64}, identities_after={"z": "a" * 64},
            old_gate_certified_by_this_diagnostic=False, optimizer_updates=0, new_defog_calls=0,
            original_config=original, original_config_sha256=config_sha256(original),
            candidate_config=candidate, candidate_config_sha256=config_sha256(candidate),
            candidate_branch_manifest=manifest, candidate_branch_manifest_sha256=config_sha256(manifest),
            candidate_source_files={Path(name).stem: sha for name, sha in files.items()})
        write(child / "attribution.json", proof)
        rows.append({"case_id": case["id"], "validation_completed": True, "validation_passed": True,
                     "execution": {"returncode": 0, "timed_out": False, "run_paths": [str(child)]},
                     "old_gate_certified": False})
    binding = {"plan": {"path": str(path), "sha256": observer.helper()._sha(path)}, "output_root": plan["output_root"]}
    write(parent / "run.json", record(plan["name"], {"plan": plan, "plan_sha256": binding["plan"]["sha256"]}))
    write(parent / "cases.json", rows)
    write(parent / "validation.json", {"status": "completed", "validation_passed": True,
        "plan_sha256": binding["plan"]["sha256"], "objective_arithmetic": observer.ARITHMETIC,
        "fd_policy": observer.POLICY, "cases": rows, "original_gates_not_reclassified": True,
        "optimizer_updates": 0, "new_defog_calls": 0})
    return binding, identity, parent


def test_readiness_accepts_only_complete_bound_pair_and_rejects_contact_failure(tmp_path, monkeypatch):
    binding, identity, parent = readiness_fixture(tmp_path, monkeypatch)
    receipt = observer.require_readiness(binding, identity)
    assert receipt["passed"] and len(receipt["cases"]) == 2
    assert receipt["allocation_job_id"] == "29799999"
    child = parent / "boltz_defog" / "unique-child"
    path = child / "attribution.json"
    proof = json.loads(path.read_text())
    proof["candidate_gates"]["dbmol_contact_loss"]["natural_secants"]["passed"] = False
    path.write_text(json.dumps(proof), encoding="utf-8")
    with pytest.raises(ValueError, match="full native derivative"):
        observer.require_readiness(binding, identity)


def test_readiness_refuses_ambiguous_native_executions(tmp_path, monkeypatch):
    binding, identity, parent = readiness_fixture(tmp_path, monkeypatch)
    second = parent / "boltz_only" / "unreferenced-second" / "run.json"
    second.parent.mkdir()
    second.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="ambiguous native"):
        observer.require_readiness(binding, identity)


def test_readiness_refuses_a_case_from_another_allocation(tmp_path, monkeypatch):
    binding, identity, parent = readiness_fixture(tmp_path, monkeypatch)
    path = parent / "boltz_defog" / "unique-child" / "run.json"
    record = json.loads(path.read_text())
    record["environment"]["job_id"] = "29799998"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="different native allocation"):
        observer.require_readiness(binding, identity)
