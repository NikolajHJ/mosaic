"""Allocated-HPC contracts for diagnostic adoption and unchanged shared starts."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from latentmol.boltz import embedding_comparison_correction as correction
from latentmol.boltz import embedding_comparison_policy as policy
from latentmol.boltz import embedding_experiment
from latentmol.experiments import file_sha256
from latentmol.representation import RelaxedGraph
from test_embedding_comparison_experiment import SavedStartVAE


def adoption_fixture(tmp_path):
    """Synthetic complete receipts exercise authentication, not native science."""
    parent = dict(version="epoch36-matched-embedding-comparison-experiment-v1",
        cpu_tests=["tests/original-boundary.py"], start_count=3, seed=138036,
        start_selection="first three saved audit means without replacement",
        optimization=dict(steps=50, learning_rate=400., max_gradient_norm=1., trust_radius=3.),
        step_control=dict(version="best_trial_v2", max_trials=16),
        vae_continuation="untouched")
    paths = {name: tmp_path / filename for name, filename in (
        ("parent_plan", "parent-plan.json"), ("parent_starts", "parent-starts.pt"),
        ("diagnostic", "diagnostic.json"), ("diagnostic_result", "diagnostic-result.json"))}
    paths["parent_plan"].write_text(json.dumps(parent))
    paths["parent_starts"].write_bytes(b"authenticated synthetic original starts")
    parent_starts = dict(path=str(paths["parent_starts"]), sha256=file_sha256(paths["parent_starts"]))
    stages = {}
    for name in ("decoder", "graph_policy", "adapter"):
        keys = {"z"} if name == "decoder" else correction.FIELDS
        stages[name] = dict(captured_output_primal_exact=True, primal_exact_between_ad_modes=True,
            captured_input_cotangent_comparison=dict(complete=True, bitwise_exact=True,
                maximum_absolute_difference=0., difference_l2=0.,
                rows={key: dict(bitwise_exact=True, maximum_absolute_difference=0., difference_l2=0.,
                               dtype="torch.float32", shape=[1, 2]) for key in keys}))
    diagnostic = dict(status="completed_diagnostic_no_fix_selected", optimization="not_run",
        case=dict(case_id="case-003"), native_oracle="passed_native_oracle_only",
        frozen_identity_unchanged=True, frozen_versions_unchanged=True,
        full_forward_primal_exact=True, parent_starts=parent_starts,
        parent_failure_binding=dict(scores_exact=True, gradient_exact=True, direction_exact=True),
        cut_primal_exact={stage: {key: True for key in keys} for stage, keys in correction.CUT_FIELDS.items()},
        local_replays={component: deepcopy(stages) for component in correction.COMPONENTS},
        numerical_controls={"final_policy_float64": {component: dict(
            within_inherited_threshold=True, primal_exact_between_ad_modes=True,
            absolute_error=1e-10, threshold_used_for_promotion=False,
            baseline_graph_max_absolute_change=dict(atoms=1e-7, charges=1e-15, bonds=5e-8))
            for component in correction.COMPONENTS}})
    terminal = dict(status=diagnostic["status"], source_sha256=correction.DIAGNOSTIC_SOURCE,
                    plan_sha256=correction.DIAGNOSTIC_PLAN)
    config = deepcopy(parent)
    config.update(version=correction.EXPERIMENT_VERSION,
        comparison_graph_policy=correction.POLICY,
        cpu_tests=parent["cpu_tests"] + sorted(correction.EXTRA_TESTS),
        precision_correction=dict(version=correction.VERSION,
            cpu_kernel_profile="mkl_compatible_four_threads_v1"))

    def seal():
        starts = dict(path=str(paths["parent_starts"]), sha256=file_sha256(paths["parent_starts"]))
        diagnostic["parent_starts"] = starts
        paths["diagnostic"].write_text(json.dumps(diagnostic))
        paths["diagnostic_result"].write_text(json.dumps(terminal))
        config["precision_correction"].update({name: dict(path=str(path), sha256=file_sha256(path))
                                                for name, path in paths.items()})

    seal()
    return SimpleNamespace(config=config, parent=parent, diagnostic=diagnostic,
                           terminal=terminal, paths=paths, seal=seal)


def test_complete_authenticated_diagnostic_selects_only_a_candidate(tmp_path):
    fixture = adoption_fixture(tmp_path)
    fixture.config["_plan_sha256"] = "a" * 64
    receipt = correction.verify_correction(fixture.config)
    assert receipt["status"] == "authenticated_candidate_not_native_pass"
    assert receipt["policy"] == correction.POLICY
    assert receipt["diagnostic"] == fixture.config["precision_correction"]["diagnostic"]
    assert receipt["parent_starts"] == fixture.config["precision_correction"]["parent_starts"]


def test_unknown_cpu_kernel_profile_cannot_silently_fall_back(tmp_path):
    fixture = adoption_fixture(tmp_path)
    fixture.config["precision_correction"]["cpu_kernel_profile"] = "unreviewed-kernel-profile"
    with pytest.raises(ValueError, match="kernel|profile|precision"):
        correction.verify_correction(fixture.config)


@pytest.mark.parametrize("drift", ["learning_rate", "start_count", "step_control", "missing_test", "new_test"])
def test_precision_retry_cannot_change_inherited_experiment_protocol(tmp_path, drift):
    fixture = adoption_fixture(tmp_path)
    if drift == "learning_rate":
        fixture.config["optimization"]["learning_rate"] = 401.
    elif drift == "start_count":
        fixture.config["start_count"] = 2
    elif drift == "step_control":
        fixture.config["step_control"]["max_trials"] = 15
    elif drift == "missing_test":
        fixture.config["cpu_tests"].remove(sorted(correction.EXTRA_TESTS)[0])
    else:
        fixture.config["cpu_tests"].append("tests/unreviewed-extra.py")
    with pytest.raises(ValueError, match="inherited protocol|omitted correction tests"):
        correction.verify_correction(fixture.config)


@pytest.mark.parametrize("asset", ["parent_plan", "diagnostic", "diagnostic_result"])
def test_real_asset_hash_tampering_blocks_adoption(tmp_path, asset):
    fixture = adoption_fixture(tmp_path)
    fixture.paths[asset].write_text(fixture.paths[asset].read_text() + "\n")
    with pytest.raises(ValueError, match="Asset hash mismatch"):
        correction.verify_correction(fixture.config)


@pytest.mark.parametrize("evidence", [
    "frozen_state", "frozen_versions", "parent_gradient", "missing_parent_binding", "empty_parent_binding",
    "cut_primal", "missing_cut_field", "empty_cuts", "missing_component", "missing_stage", "missing_vector_row",
    "nonexact_vector", "hidden_vector_difference", "failed_control", "empty_control", "other_source", "other_plan"])
def test_incomplete_or_changed_diagnostic_receipts_fail_closed(tmp_path, evidence):
    fixture = adoption_fixture(tmp_path)
    diagnostic = fixture.diagnostic
    component = "dbmol_total_loss"
    if evidence == "frozen_state":
        diagnostic["frozen_identity_unchanged"] = False
    elif evidence == "frozen_versions":
        diagnostic["frozen_versions_unchanged"] = False
    elif evidence == "parent_gradient":
        diagnostic["parent_failure_binding"]["gradient_exact"] = False
    elif evidence == "missing_parent_binding":
        diagnostic["parent_failure_binding"].pop("direction_exact")
    elif evidence == "empty_parent_binding":
        diagnostic["parent_failure_binding"] = {}
    elif evidence == "cut_primal":
        diagnostic["cut_primal_exact"]["graph"]["atoms"] = False
    elif evidence == "missing_cut_field":
        diagnostic["cut_primal_exact"]["native"].pop("ref_charge")
    elif evidence == "empty_cuts":
        diagnostic["cut_primal_exact"] = {}
    elif evidence == "missing_component":
        diagnostic["local_replays"].pop(component)
    elif evidence == "missing_stage":
        diagnostic["local_replays"][component].pop("adapter")
    elif evidence == "missing_vector_row":
        diagnostic["local_replays"][component]["adapter"]["captured_input_cotangent_comparison"]["rows"].pop("charges")
    elif evidence == "nonexact_vector":
        diagnostic["local_replays"][component]["decoder"]["captured_input_cotangent_comparison"]["bitwise_exact"] = False
    elif evidence == "hidden_vector_difference":
        diagnostic["local_replays"][component]["decoder"]["captured_input_cotangent_comparison"]["rows"]["z"]["difference_l2"] = 1e-8
    elif evidence == "failed_control":
        diagnostic["numerical_controls"]["final_policy_float64"][component]["within_inherited_threshold"] = False
    elif evidence == "empty_control":
        diagnostic["numerical_controls"]["final_policy_float64"] = {}
    elif evidence == "other_source":
        fixture.terminal["source_sha256"] = "0" * 64
    elif evidence == "other_plan":
        fixture.terminal["plan_sha256"] = "0" * 64
    fixture.seal()
    with pytest.raises(ValueError, match="diagnostic binding|cut replay|control did not resolve"):
        correction.verify_correction(fixture.config)


def retry_starts_fixture(tmp_path, monkeypatch):
    fixture = adoption_fixture(tmp_path)
    atoms = torch.tensor([[1, 0, 3, 0], [1, 0, 2, 0], [2, 0, 2, 0]])
    charges = torch.full((3, 4), 2, dtype=torch.long)
    bonds = torch.zeros(3, 4, 4, dtype=torch.long)
    bonds[:, 0, 2] = bonds[:, 2, 0] = 1
    vae = SavedStartVAE(atoms, charges, bonds)
    payload = dict(graph_spec=vae.spec.to_dict(), cases=[dict(
        record=dict(case_id=f"case-{index + 1:03d}", identity="synthetic"),
        z=torch.tensor([[float(index), -float(index)]]), active=atoms[index] != 0)
        for index in range(3)])
    output = tmp_path / "cpu"
    output.mkdir()
    torch.save(payload, fixture.paths["parent_starts"])
    torch.save(payload, output / "starts.pt")
    fixture.seal()
    monkeypatch.setattr(embedding_experiment, "load_vae", lambda *args: (vae, {}))
    return fixture, vae, payload, output


def test_candidate_keeps_all_original_latents_masks_and_hard_identities(tmp_path, monkeypatch):
    fixture, vae, _, output = retry_starts_fixture(tmp_path, monkeypatch)
    report = correction.verify_retry_starts(fixture.config, output)
    assert report["all_three_parent_starts_exact"] and report["frozen_vae_unchanged"]
    assert all(row["hard_identity_unchanged"] for row in report["cases"])
    assert vae.decoded_rows == [0, 1, 2]
    assert json.loads((output / "precision-correction.json").read_text()) == report
    assert report["status"] == "authenticated_candidate_not_native_pass"


def test_changed_parent_asset_or_fresh_latent_cannot_enter_retry(tmp_path, monkeypatch):
    fixture, _, payload, output = retry_starts_fixture(tmp_path, monkeypatch)
    changed = deepcopy(payload)
    changed["cases"][0]["z"][0, 1] += .01
    torch.save(changed, output / "starts.pt")
    with pytest.raises(ValueError, match="original matched starts"):
        correction.verify_retry_starts(fixture.config, output)
    fixture.paths["parent_starts"].write_bytes(b"changed parent bytes after authentication")
    with pytest.raises(ValueError, match="Asset hash mismatch"):
        correction.verify_retry_starts(fixture.config, output)


@pytest.mark.parametrize("field,new_dtype", [("z", torch.float64), ("active", torch.int64)])
def test_equal_values_with_changed_latent_or_mask_dtype_are_not_identical_starts(
        tmp_path, monkeypatch, field, new_dtype):
    fixture, vae, payload, output = retry_starts_fixture(tmp_path, monkeypatch)
    changed = deepcopy(payload)
    changed["cases"][0][field] = changed["cases"][0][field].to(new_dtype)
    torch.save(changed, output / "starts.pt")
    with pytest.raises(ValueError, match="original matched starts"):
        correction.verify_retry_starts(fixture.config, output)
    assert vae.decoded_rows == []


@pytest.mark.parametrize("change", ["hard_identity", "probability_drift"])
def test_candidate_graph_changes_cannot_bypass_shared_start_readiness(tmp_path, monkeypatch, change):
    fixture, _, _, output = retry_starts_fixture(tmp_path, monkeypatch)
    original = policy.comparison_graph

    def changed(decoded, active, spec, **kwargs):
        graph = original(decoded, active, spec, **kwargs)
        if kwargs["policy"] == correction.POLICY:
            atoms = graph.atoms.clone()
            if change == "hard_identity":
                atoms[0, 0] = 0
                atoms[0, 0, 2] = 1
            else:
                atoms[0, 0, 1] -= 2e-6
                atoms[0, 0, 2] += 2e-6
            return RelaxedGraph(atoms, graph.charges, graph.bonds)
        return graph

    monkeypatch.setattr(policy, "comparison_graph", changed)
    with pytest.raises(ValueError, match="hard starting graph|probabilities materially"):
        correction.verify_retry_starts(fixture.config, output)
    assert not (output / "precision-correction.json").exists()


def test_readiness_rejects_frozen_vae_mutation(tmp_path, monkeypatch):
    fixture, vae, _, output = retry_starts_fixture(tmp_path, monkeypatch)
    original = vae.decode

    def mutate(z):
        result = original(z)
        vae.identity.add_(1)
        return result

    monkeypatch.setattr(vae, "decode", mutate)
    with pytest.raises(ValueError, match="changed original frozen VAE"):
        correction.verify_retry_starts(fixture.config, output)
