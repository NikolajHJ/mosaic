"""HPC-only engineering fixtures; no native-model scientific validation.

Run these in the submitted CPU gate. They exercise faithful objective assembly,
same-endpoint component attribution, source/target authentication, and retained
wrong-backward/mode-mismatch evidence. No pretrained assets or worker calls.
"""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest
import torch

from latentmol.boltz.active_set import ContextBank, make_branch, score_at, select_slots
from latentmol.boltz.dbmol_objective import DBMolObjectiveConfig
from latentmol.boltz.failed_gate_attribution import (
    REPEATS, _digest, _sha, _target_from_receipts, authenticate_source,
    authenticated_artifact, disabled_control_target, failed_item, forward_values, load_plan, probe_case,
    reconstruct_directions, sum_gradient_identity,
)
from latentmol.boltz.local_fd import direction_plan
from latentmol.multimodel.config import Objective
from latentmol.multimodel.engine import Counters
from latentmol.multimodel.study import tensor_digest
from latentmol.repair_guidance.loss import target_score_at
from test_boltz_active_set_runtime import SPEC, _Decoder
from test_dbmol_gradient_gate import CompositeFixture
from test_repair_guidance import _guided_cfg, _target


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return str(path.absolute()), _sha(path)


def _problem(tmp_path, *, wrong=False):
    profile = DBMolObjectiveConfig.from_mapping({
        "confidence_checkpoint": {"path": "fake.ckpt", "sha256": "a" * 64},
        "pocket_residues": [{"asym_id": 0, "residue_index": 0}]}, tmp_path)
    config = replace(_guided_cfg(tmp_path), dbmol=profile,
                     objective=Objective("dbmol_total_loss", direction="minimize"), element_diagnostics=False)
    decoder = _Decoder()
    point = torch.tensor([[-.2]], dtype=torch.float64)
    selection = select_slots(decoder.decode(point), temperature=1., threshold=.5)
    branch = make_branch(CompositeFixture(profile, wrong_contact=wrong), ContextBank(config, SPEC),
                         selection, config, "c" * 64)
    target = _target(SPEC, selection.active)
    return decoder, point, branch, config, target


def _item(point, derivative):
    _, _, plan = direction_plan(point, derivative, seed=1729)
    return {"status": "failed", "passed": False, "policy": {"version": "local_v1"},
            "finite_differences": [{"retained_original_failure": True}], "direction_selection": plan,
            "direction_results": {"gradient": {"passed": True}, "random_1": {"passed": False},
                                  "random_2": {"passed": True}, "random_3": {"passed": True}}}


def test_component_failure_is_read_with_empty_top_level_fd_and_no_total():
    item = {"status": "failed", "passed": False, "policy": {"version": "local_v1"}, "finite_differences": [1]}
    gate = {"status": "failed", "passed": False, "native_seed_parity": True,
            "measured": {"loss": 1.}, "finite_differences": [],
            "component_gradients": {"dbmol_contact_loss": item}}
    assert failed_item(gate) == ("dbmol_contact_loss", item)
    total = deepcopy(gate)
    total["component_gradients"]["dbmol_contact_loss"]["status"] = "validated_nonzero"
    total["local_fd_validation"] = item
    assert failed_item(total) == ("combined", item)
    gate["local_fd_validation"] = item
    with pytest.raises(ValueError, match="one completed failed"):
        failed_item(gate)


@pytest.mark.parametrize("bad", ["native", "passed", "policy", "rows"])
def test_non_derivative_or_wrong_policy_cases_are_rejected(bad):
    gate = {"status": "failed", "passed": False, "native_seed_parity": True, "measured": {"loss": 1.},
            "local_fd_validation": {"status": "failed", "passed": False,
                                    "policy": {"version": "local_v1"}, "finite_differences": [1]}}
    if bad == "native":
        gate["native_seed_parity"] = False
    elif bad == "passed":
        gate["passed"] = True
    elif bad == "policy":
        gate["local_fd_validation"]["policy"]["version"] = "unknown"
    else:
        gate["local_fd_validation"]["finite_differences"] = []
    with pytest.raises(ValueError):
        failed_item(gate)


def test_original_random_hash_replayed_without_new_score_based_selection():
    point, gradient = torch.zeros(1, 8), torch.arange(1., 9.).reshape(1, 8)
    item = _item(point, gradient)
    directions = reconstruct_directions(point, gradient, item)
    assert set(directions) == {"gradient", "random_1"}
    assert directions["random_1"][1]["hash_matches_original"] is True
    assert directions["gradient"][1]["hash_matches_original"] is None
    # A different analytic vector cannot replace the historically selected
    # Gaussian proposal. Its old direction remains exactly the same tensor.
    changed = reconstruct_directions(point, gradient.flip(-1), item)
    torch.testing.assert_close(changed["random_1"][0], directions["random_1"][0], rtol=0, atol=0)
    receipt = next(r for r in item["direction_selection"]["proposals"] if r["selected_label"] == "random_1")
    receipt["direction_sha256"] = "f" * 64
    assert reconstruct_directions(point, gradient, item)["random_1"][1]["hash_matches_original"] is False


def test_endpoint_modes_reuse_original_score_scalar_assembly_and_fixed_target(tmp_path):
    decoder, point, branch, config, target = _problem(tmp_path)
    observed, _ = score_at(decoder, point, branch, config, Counters(), "fixture", gradients=True,
                           repair_target=target, repair_weight=.2)
    for mode in (False, True):
        row = forward_values(decoder, point, branch, config, Counters(), target, .2, grad_enabled=mode)
        assert row["values"]["combined"] == observed["loss"]
        assert row["values"]["binding_loss"] == observed["binding_loss"]
        assert row["values"]["repair_nll"] == observed["repair_guidance"]["normalized_nll"]
        assert row["target_sha256"] == target.identity and row["weight"] == .2
        assert row["dtypes"]["combined"] == observed["loss_dtype"]


@pytest.mark.parametrize("wrong", [False, True])
def test_same_endpoint_attribution_preserves_wrong_backward_and_never_approves_gate(tmp_path, wrong):
    decoder, point, branch, config, target = _problem(tmp_path, wrong=wrong)
    components = {}
    _, gradient = score_at(decoder, point, branch, config, Counters(), "fixture", gradients=True,
        component_gradients=components, repair_target=target, repair_weight=.2)
    _, repair = target_score_at(decoder, point, branch.prepared.active, target, gradients=True)
    identity = sum_gradient_identity(gradient, components, repair, config, .2)
    assert identity["max_absolute_difference"] < 1e-12
    item = _item(point, components["dbmol_contact_loss"])
    counters = Counters()
    report = probe_case(decoder, point, branch, config, counters, target, .2,
                        "dbmol_contact_loss", item)
    assert report["diagnostic_completed"] is True
    assert report["old_gate_certified_by_this_diagnostic"] is False
    assert report["original_gate_not_reclassified"] is True
    assert report["random_directions_match_original"] is True
    assert report["maximum_scorer_forwards"] == 78
    assert len(report["baselines"]) == REPEATS * 2
    for row in report["baselines"]:
        if row["grad_enabled"]:
            assert row["scalar_assembly_identity"]["exact_combined_scalar_match"] is True
            assert row["scalar_assembly_identity"]["exact_component_scalars_match"] is True
            assert row["scalar_assembly_identity"]["exact_repair_scalar_match"] is True
    assert len(report["finite_differences"]) == 2 * 9 * 2
    assert report["gradient_repeat_max_absolute_difference"] == 0
    for row in report["finite_differences"]:
        assert set(row["terms"]) == {"dbmol_affinity_loss", "dbmol_contact_loss", "dbmol_confidence_loss",
                                      "binding_loss", "repair_nll", "combined"}
        assert row["plus"]["target_sha256"] == row["minus"]["target_sha256"] == target.identity
    # Deliberately incorrect contact backward remains an observed disagreement,
    # although all derivative sums can still be internally consistent.
    rows = [r["terms"]["dbmol_contact_loss"] for r in report["finite_differences"]
            if r["epsilon"] == .001]
    assert all(r["passed_tolerance"] is (not wrong) for r in rows)
    assert not decoder.training and not branch.training


def test_disabled_control_replays_production_binding_without_fabricated_repair(tmp_path):
    decoder, point, branch, config, _ = _problem(tmp_path)
    config = replace(config, repair_guidance=replace(config.repair_guidance, weight=0.))
    components = {}
    observed, gradient = score_at(decoder, point, branch, config, Counters(), 'control', gradients=True,
        component_gradients=components, repair_target=None, repair_weight=0.)
    for mode in (False, True):
        row = forward_values(decoder, point, branch, config, Counters(), None, 0., grad_enabled=mode)
        assert row['values']['combined'] == observed['loss'] == observed['binding_loss']
        assert row['dtypes']['combined'] == observed['loss_dtype']
        assert 'repair_nll' not in row['values'] and 'repair_nll' not in row['dtypes']
        assert row['target_sha256'] is None
    report = probe_case(decoder, point, branch, config, Counters(), None, 0., 'combined', _item(point, gradient))
    assert report['diagnostic_completed'] is True
    assert report['repair_gradient_repeat_max_absolute_difference'] is None
    assert report['old_gate_certified_by_this_diagnostic'] is False
    for baseline in report['baselines']:
        if baseline['grad_enabled']:
            assert baseline['scalar_assembly_identity']['exact_combined_scalar_match'] is True
            assert baseline['scalar_assembly_identity']['repair_term_present'] is False
    assert all('repair_nll' not in row['terms'] for row in report['finite_differences'])
    with pytest.raises(ValueError, match='disabled repair'):
        forward_values(decoder, point, branch, config, Counters(), None, .1, grad_enabled=False)


def _control_receipts(tmp_path):
    _, _, _, config, _ = _problem(tmp_path)
    config = replace(config, repair_guidance=replace(config.repair_guidance, weight=0., refresh_interval=5))
    case = {'parent_run': str(tmp_path), 'arm': 'fixed_initial_mask', 'step': 15, 'parent_artifacts_sha256': {}}
    for step in (0, 5, 10, 15):
        path, sha = _write(tmp_path / 'fixed_initial_mask' / f'repair-refresh-objective-{step:04d}.json',
            {'step': step, 'new_target_sha256': None,
             'receipt': {'status': 'disabled', 'guidance_weight': 0., 'attempts': []}})
        case['parent_artifacts_sha256'][path] = sha
    gate = {'repair_application': {'scheduled_weight': 0., 'target_available': False,
                'used_target': False, 'applied_weight': 0.},
            'measured': {'repair_guidance': {'status': 'disabled_weight_zero', 'weight': 0.}}}
    return case, config, gate


def test_disabled_control_requires_zero_activation_and_all_authenticated_empty_receipts(tmp_path):
    case, config, gate = _control_receipts(tmp_path)
    target, receipt = disabled_control_target(case, config, gate)
    assert target is None and receipt['new_worker_calls'] == 0
    assert len(receipt['checked_refresh_receipts']) == 4
    with pytest.raises(ValueError, match='zero-activation'):
        disabled_control_target(case, replace(config, repair_guidance=replace(config.repair_guidance, weight=.1)), gate)
    altered = deepcopy(gate)
    altered['repair_application']['target_available'] = True
    with pytest.raises(ValueError, match='disabled repair'):
        disabled_control_target(case, config, altered)
    path = tmp_path / 'fixed_initial_mask/repair-refresh-objective-0015.json'
    _, sha = _write(path, {'new_target_sha256': None,
        'receipt': {'status': 'disabled', 'guidance_weight': 0., 'attempts': [{'native_attempt': True}]}})
    case['parent_artifacts_sha256'][str(path.absolute())] = sha
    with pytest.raises(ValueError, match='native repair attempt'):
        disabled_control_target(case, config, gate)
    path.unlink()
    with pytest.raises(ValueError, match='Missing declared'):
        disabled_control_target(case, config, gate)


def _receipt_fixture(tmp_path):
    decoder, point, branch, config, target = _problem(tmp_path)
    case = {"parent_run": str(tmp_path), "arm": "fixed_initial_mask", "step": 20, "parent_artifacts_sha256": {}}
    folder = tmp_path / "fixed_initial_mask/repair-refresh-00020"
    response_path = folder / "attempt-000/response.json"
    source = "c" * 64
    response = {"status": "accepted", "request_sha256": "r" * 64, "candidate_sha256": "a" * 64,
        "occupied_slots": branch.prepared.active.nonzero().flatten().tolist(),
        "native_order_preserved": True, "slot_alignment": "native_order_preserved",
        "gates": {k: True for k in ("chemical_validity", "fixed_atom_count", "native_label_roundtrip", "neutral_charge_subset", "slot_alignment")},
        "provenance": {"adapter_source_sha256": source, "source_commit": config.repair_guidance.provider.get("source", {}).get("commit"),
            "checkpoint_sha256": config.repair_guidance.provider.get("checkpoint", {}).get("sha256"),
            "frozen_state_unchanged": True, "frozen_state_sha256_before": "s" * 64, "frozen_state_sha256_after": "s" * 64},
        "target": {k: getattr(target.graph, k).tolist() for k in ("atoms", "charges", "bonds")}}
    config = replace(config, repair_guidance=replace(config.repair_guidance, provider={
        "source": {"commit": "d" * 40}, "checkpoint": {"sha256": "e" * 64}}))
    response["provenance"].update(source_commit="d" * 40, checkpoint_sha256="e" * 64)
    path, sha = _write(response_path, response)
    case["parent_artifacts_sha256"][path] = sha
    provider = config.repair_guidance.record()
    provider["weight"] = config.repair_guidance.provider_activation_weight
    receipt = {"status": "accepted", "step": 20, "selected_attempt": 0, "temperature": config.temperature,
        "source_sha256": source, "occupied_slots": response["occupied_slots"], "latent_sha256": tensor_digest({"z": point}),
        "config": provider, "selected_target_sha256": target.identity,
        "attempts": [{"response_path": path, "response_sha256": sha, "response": response,
            "candidate_json_sha256": "a" * 64, "candidate_tensor_sha256": "b" * 64,
            "status": "accepted", "target_eligible": True, "timeout": False, "process_returncode": 0,
            "request_sha256": "r" * 64}]}
    path, sha = _write(folder / "refresh.json", receipt)
    case["parent_artifacts_sha256"][path] = sha
    gate = {"repair_only_validation": {"target_sha256": target.identity}}
    return case, config, branch.prepared.active, point, target, gate, source, receipt


def test_selected_target_json_identity_is_distinct_from_tensor_identity(tmp_path):
    case, config, active, point, original, gate, source, receipt = _receipt_fixture(tmp_path)
    target, evidence = _target_from_receipts(case, config, SPEC, active, tensor_digest({"z": point}), gate, source_sha256=source)
    assert target.identity == original.identity
    assert evidence["new_worker_calls"] == 0 and evidence["refresh_step"] == 20
    assert receipt["attempts"][0]["candidate_json_sha256"] != receipt["attempts"][0]["candidate_tensor_sha256"]


@pytest.mark.parametrize("bad", ["candidate_json", "target_hash", "source", "weight", "eligible", "timeout", "response"])
def test_stale_or_misaligned_historical_target_fails_closed(tmp_path, bad):
    case, config, active, point, target, gate, source, receipt = _receipt_fixture(tmp_path)
    if bad == "candidate_json":
        receipt["attempts"][0]["candidate_json_sha256"] = "f" * 64
    elif bad == "target_hash":
        gate["repair_only_validation"]["target_sha256"] = "f" * 64
    elif bad == "source":
        source = "f" * 64
    elif bad == "weight":
        receipt["config"]["weight"] = 9.
    elif bad == "eligible":
        receipt["attempts"][0]["target_eligible"] = 1
    elif bad == "timeout":
        receipt["attempts"][0]["timeout"] = True
    else:
        receipt["attempts"][0]["response"]["status"] = "rejected"
    path, sha = _write(tmp_path / "fixed_initial_mask/repair-refresh-00020/refresh.json", receipt)
    case["parent_artifacts_sha256"][path] = sha
    with pytest.raises(ValueError):
        _target_from_receipts(case, config, SPEC, active, tensor_digest({"z": point}), gate, source_sha256=source)


def test_source_manifest_authenticates_old_uncommitted_bytes_and_rejects_escape(tmp_path):
    root = tmp_path / "old"
    root.mkdir()
    archive = Path(str(root) + ".tar.gz")
    archive.write_bytes(b"trusted fixture archive bytes")
    path, sha = _write(root / "src/latentmol/science.py", {"original": True})
    files = {"src/latentmol/science.py": sha}
    record = {"source": {"files": files, "sha256": _digest(files)}, "source_archive_sha256": _sha(archive)}
    plan = {"production_manifest_sha256": record["source"]["sha256"], "production_archive_sha256": _sha(archive)}
    authenticate_source(root, record, plan)
    Path(path).write_text("changed without git commit")
    with pytest.raises(ValueError, match="changed or escaped"):
        authenticate_source(root, record, plan)
    files = {"../outside.py": hashlib.sha256(b"outside").hexdigest()}
    record["source"] = {"files": files, "sha256": _digest(files)}
    plan["production_manifest_sha256"] = record["source"]["sha256"]
    with pytest.raises(ValueError, match="changed or escaped"):
        authenticate_source(root, record, plan)


def test_plan_hash_and_artifact_map_are_required(tmp_path):
    path, sha = _write(tmp_path / "parent/run.json", {"data": True})
    case = {"id": "failed", "parent_run": str((tmp_path / "parent").absolute()), "arm": "fixed_initial_mask",
            "step": 20, "parent_artifacts_sha256": {path: sha}}
    plan = {"version": 1, "name": "attribution", "output_root": str((tmp_path / "new").absolute()),
            "production_release": str((tmp_path / "old").absolute()), "production_manifest_sha256": "a" * 64,
            "production_archive_sha256": "b" * 64, "cases": [case]}
    p, h = _write(tmp_path / "plan.json", plan)
    assert load_plan(p, h) == plan
    assert authenticated_artifact(case, path) == Path(path)
    with pytest.raises(ValueError, match="plan SHA256"):
        load_plan(p, "f" * 64)
    with pytest.raises(ValueError, match="pinned parent"):
        authenticated_artifact(case, tmp_path / "unlisted.json")
    plan["cases"] = [case, case]
    p, h = _write(tmp_path / "plan.json", plan)
    with pytest.raises(ValueError, match="distinct"):
        load_plan(p, h)
