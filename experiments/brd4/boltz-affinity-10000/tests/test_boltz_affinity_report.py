"""Allocated saved-artifact report fixtures; no molecular scorer is executed."""

# Working-source layout bootstrap; stdlib only, no model initialization.
import sys as _layout_sys
from pathlib import Path as _LayoutPath
_layout_sys.path.insert(0, str(_LayoutPath(__file__).resolve().parents[4] / "scripts/layout"))
from experiment_layout import activate as _activate_layout, repository_root as _repository_root, resolve as _layout_path
_activate_layout(__file__)

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPTS = _layout_path(_repository_root(__file__) / "scripts")


@pytest.fixture(autouse=True)
def allocated_only():
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Boltz report fixtures require allocated DTU HPC")


def load_renderer(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "boltz_saved_report_fixture", SCRIPTS / "render_boltz_affinity.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derivative_report(*, raw=False):
    fields = ("X", "E_upper") if raw else ("atoms", "bonds")
    rows = []
    for field in fields:
        probes = []
        for epsilon in (0.01, 0.003, 0.001):
            probes.append(
                {
                    "epsilon": epsilon,
                    "analytic": 1.0,
                    "finite_difference": 1.0,
                    "primal_endpoints": [-epsilon, epsilon],
                    "preparation_identity": "saved-fixed-target",
                    "agrees": True,
                }
            )
        rows.append(
            {
                "field": field,
                "row": 0,
                "source": 0,
                "target": 1,
                "destination": 1,
                "analytic": 1.0,
                "probes": probes,
            }
        )
    result = {"status": "passed", "preparation_identity": "saved-fixed-target"}
    result["rows" if raw else "directions"] = rows
    if raw:
        result.update(typed_bond_contrast_nonzero=True, baseline=-0.5)
    return result


def fixture_study(tmp_path, module):
    """Two updates cover the hold/ramp boundary without 10,001 fake snapshots."""
    root = tmp_path / "study"
    case_root = root / "case-0"
    case_root.mkdir(parents=True)
    config = {
        "version": "brd4-boltz-affinity-chemistry-v1",
        "n_atoms": 6,
        "seed": 17,
        "steps": 2,
        "initial_step_size": 0.03,
        "max_backtracks": 10,
        "pilot_steps": 10,
        "allocation_seconds": 82800,
        "element_js_weight": 0.1,
        "loss": {
            "version": "repair-failure-guidance-v1",
            "extra_valid_weight": 1.0,
            "partial_weight": 1.0,
            "local_bond_weight": 0.1,
            "component_cut_weight": 0.1,
            "nonempty_weight": 0.1,
        },
        "search": {
            "max_edits": 3,
            "checks_per_depth": 32,
            "beam_width": 4,
            "probability_weight": 1.0,
            "probability_floor": 1e-12,
        },
        "progress_search": {
            "version": "repair-progress-v1",
            "max_edits": 6,
            "checks_per_depth": 32,
            "beam_width": 4,
            "probability_weight": 1.0,
            "allow_atom_edits": False,
            "atom_edit_penalty": 3.0,
            "max_atom_edits": 1,
        },
        "initialization": {"kind": "chain_uniform", "mixing": 0.5},
        "repair_schedule": {
            "version": "delayed-geometric-v1",
            "initial": 0.001,
            "final": 50.0,
            "hold_updates": 1,
            "total_updates": 2,
        },
        "label": "Synthetic saved-only Boltz fixture",
    }
    config_path = root / "configuration.json"
    put(config_path, config)
    native = {
        "model_digest": module.MODEL_DIGEST,
        "context_digest": module.CONTEXT_DIGEST,
        "context_file_sha256": module.CONTEXT_SHA,
        "checkpoint_sha256": module.CHECKPOINT_SHA,
        "source": module.BOLTZ_COMMIT,
        "optimized_output": "affinity_pred_value",
        "recycling_steps": 0,
        "status": "ready",
        "protocol": "mosaic-boltz-affinity-v1",
        "pose_policy": "fixed native predicted pose; no diffusion gradients",
    }
    objective = {**native, "coefficient": 1.0, "context_sha256": module.CONTEXT_SHA}
    # Invalid fluorine/triple-bond graphs are intentionally retained as saved.
    x = np.full((6, 6), 0.02, dtype=np.float32)
    x[:, 4] = 0.9
    edges = np.full((15, 5), 0.01, dtype=np.float32)
    edges[:, 3] = 0.96
    np.savez_compressed(root / "initialization.npz", X=x, E_upper=edges)
    states = []
    affinities = (-0.5, -0.75, -0.625)
    weights = (0.001, 50.0, 50.0)
    for step, (affinity, weight) in enumerate(zip(affinities, weights)):
        snapshot = case_root / f"snapshots/step-{step:04d}.npz"
        snapshot.parent.mkdir(exist_ok=True)
        surrogate = affinity + weight * 0.1 + 0.1 * 0.01
        np.savez_compressed(
            snapshot,
            X=x,
            E_upper=edges,
            affinity_primal=np.asarray(affinity),
            repair_weight=np.asarray(weight),
            refreshed_surrogate=np.asarray(surrogate),
            autodiff_surrogate=np.asarray(surrogate),
        )
        states.append(
            {
                "step": step,
                "snapshot": snapshot.relative_to(case_root).as_posix(),
                "snapshot_sha256": digest(snapshot),
                "affinity_loss": affinity,
                "refreshed_surrogate": surrogate,
                "autodiff_surrogate": surrogate,
                "repair_weight": weight,
                "preparation_identity": "saved-fixed-target",
                "previous_fixed_surrogate_after": None,
                "components": {
                    "affinity_loss": affinity,
                    "affinity_autodiff_loss": affinity,
                    "affinity_pred_value": affinity,
                    "affinity_probability_binary": 0.25,
                    "affinity_mean_binary_logit": -1.0,
                    "chemistry_loss": 0.1,
                    "element_js": 0.01,
                    "valid_full_nll": 0.0,
                    "extra_valid_nll": 0.0,
                    "partial_edited_nll": 0.1,
                    "local_bond_risk": 0.1,
                    "component_cut_risk": 0.0,
                    "nonempty_risk": 0.0,
                },
                "readout": {
                    "sanitized": False,
                    "canonical_smiles": None,
                    "element_indices": [4] * 6,
                    "bond_upper_indices": [3] * 15,
                    "elements": ["F"] * 6,
                    "disconnected": False,
                    "policy": "raw argmax; no repair, no fragment removal",
                    "conversion_error": "saved invalid valence; never repaired by renderer",
                },
                "chemistry_preparation": [
                    {
                        "raw_native_valid": False,
                        "original_status": "censored",
                        "route": "partial",
                        "extra_status": "partial",
                    }
                ],
            }
        )
    updates = []
    for step in range(2):
        weight = weights[step]
        before = affinities[step] + weight * 0.1 + 0.001
        after = affinities[step + 1] + weight * 0.1 + 0.001
        updates.append(
            {
                "step": step,
                "repair_weight": weight,
                "fixed_before": before,
                "fixed_after": after,
                "fallback_used": step == 1,
                "fixed_increased": step == 1,
                "no_op": False,
                "selected_trial": 0,
                "accepted_step_size": 0.03,
                "elapsed_seconds": 1.0,
                "preparation_identity": states[step]["preparation_identity"],
                "affinity_before": affinities[step],
                "affinity_after": affinities[step + 1],
                "attempts": [
                    {
                        "trial": 0,
                        "rate": 0.03,
                        "repair_weight": weight,
                        "strict": step == 0,
                        "autodiff_surrogate": after,
                        "independent_primal_surrogate": after,
                    }
                ],
            }
        )
        states[step + 1]["previous_fixed_surrogate_after"] = after
    combined = derivative_report()
    combined["affinity_objective_identity"] = objective
    put(case_root / "combined_initial_checks/report.json", combined)
    case = {
        "status": "passed",
        "requested_updates": 2,
        "parameterization": "categorical_logits_v1",
        "acceptance_policy": "best_primal",
        "repair_weight_schedule": config["repair_schedule"],
        "affinity_objective_identity": objective,
        "states": states,
        "updates": updates,
        "counts": {
            "applied": 2,
            "strict_improving": 1,
            "forced_fallback": 1,
            "no_op": 0,
        },
        "initial_checks_file": "combined_initial_checks/report.json",
        "initial_checks": combined["directions"],
        "best_observed_affinity": {
            "step": 1,
            "loss": affinities[1],
            "snapshot": states[1]["snapshot"],
            "snapshot_sha256": states[1]["snapshot_sha256"],
            "objective_identity": objective,
        },
    }
    raw = derivative_report(raw=True)
    for label in ("native_gradient_checks", "terminal_gradient_checks"):
        put(root / label / "report.json", raw)
    summary = {
        "status": "passed",
        "mode": "run",
        "job_id": "29759212",
        "config": config,
        "config_sha256": digest(config_path),
        "context_sha256": module.CONTEXT_SHA,
        "reference_sha256": module.REFERENCE_SHA,
        "initialization_sha256": digest(root / "initialization.npz"),
        "source": {"execution_content_sha256": module.SOURCE_EXECUTION_SHA},
        "native_identity": native,
        "frozen_identity_after": {
            "status": "passed",
            "model_digest": module.MODEL_DIGEST,
            "context_digest": module.CONTEXT_DIGEST,
        },
        "native_hard_parity": {
            "status": "passed",
            "atol": 2e-5,
            "native": [-0.5, 0.25, -1.0],
            "attached": [-0.5, 0.25, -1.0],
        },
        "native_gradient_checks": copy.deepcopy(raw),
        "terminal_gradient_checks": copy.deepcopy(raw),
        "cases": [
            {
                "label": config["label"],
                "initialization": config["initialization"],
                **case,
            }
        ],
    }
    save_fixture(root, summary, case)
    return root, config_path, summary, case


def save_fixture(root, summary, case):
    summary["cases"] = [
        {
            "label": summary["config"]["label"],
            "initialization": summary["config"]["initialization"],
            **case,
        }
    ]
    put(root / "summary.json", summary)
    put(root / "case-0/report.json", case)
    for name, rows in (("states", case["states"]), ("updates", case["updates"])):
        (root / f"case-0/{name}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )


def load_fixture(module, root, config_path, **overrides):
    options = dict(
        expected_job="29759212",
        expected_summary_sha256=digest(root / "summary.json"),
        expected_config_sha256=digest(config_path),
        expected_budget=2,
    )
    options.update(overrides)
    return module.load_study(root, config_path, **options)


def test_saved_affinity_preserves_invalid_endpoint_and_binder_separately(
    tmp_path, monkeypatch
):
    module = load_renderer(monkeypatch)
    root, config_path, summary, case = fixture_study(tmp_path, module)
    study = load_fixture(module, root, config_path)
    assert study["complete"] and len(study["frames"]) == 3
    final = study["frames"][-1]
    assert final["affinity"] == -0.625 and "contact" not in final
    assert not final["readout"]["sanitized"]
    assert final["readout"]["canonical_smiles"] is None
    assert final["readout"] == case["states"][-1]["readout"]
    assert np.argmax(final["X"], -1).tolist() == [4] * 6
    assert np.argmax(final["E"], -1).tolist() == [3] * 15
    assert final["components"]["affinity_probability_binary"] == 0.25
    assert final["repair_weight"] == 50.0
    assert study["updates"][-1]["fixed_after"] > study["updates"][-1]["fixed_before"]
    assert study["counts"]["forced_fallback"] == 1
    # A run-mode summary does not contain readiness-only runtime/assay fields.
    assert "runtime_gate" not in summary and "chemistry_assay" not in summary


@pytest.mark.parametrize(
    "tamper",
    [
        "native_failed",
        "terminal_failed",
        "case_failed",
        "job",
        "config",
        "model",
        "context",
        "source",
        "missing_state",
        "state_order",
        "missing_update",
        "update_order",
        "budget",
        "state_ramp",
        "trial_ramp",
        "snapshot_hash",
        "readout",
        "objective",
    ],
)
def test_loader_rejects_semantic_failures_even_when_summary_resealed(
    tmp_path, monkeypatch, tamper
):
    module = load_renderer(monkeypatch)
    root, config_path, summary, case = fixture_study(tmp_path, module)
    if tamper == "native_failed":
        summary["native_gradient_checks"]["status"] = "failed"
        put(
            root / "native_gradient_checks/report.json",
            summary["native_gradient_checks"],
        )
    elif tamper == "terminal_failed":
        summary["terminal_gradient_checks"]["status"] = "failed"
        put(
            root / "terminal_gradient_checks/report.json",
            summary["terminal_gradient_checks"],
        )
    elif tamper == "case_failed":
        case["status"] = "failed"
    elif tamper == "job":
        summary["job_id"] = "29753667"
    elif tamper == "config":
        summary["config"]["initial_step_size"] = 0.3
    elif tamper == "model":
        summary["frozen_identity_after"]["model_digest"] = "changed"
    elif tamper == "context":
        summary["context_sha256"] = "changed"
    elif tamper == "source":
        summary["source"]["execution_content_sha256"] = "changed"
    elif tamper == "missing_state":
        case["states"].pop()
    elif tamper == "state_order":
        case["states"][1]["step"] = 2
    elif tamper == "missing_update":
        case["updates"].pop()
    elif tamper == "update_order":
        case["updates"][0]["step"] = 1
    elif tamper == "budget":
        case["counts"]["applied"] = 1
    elif tamper == "state_ramp":
        case["states"][-1]["repair_weight"] = 0.001
    elif tamper == "trial_ramp":
        case["updates"][-1]["attempts"][0]["repair_weight"] = 0.001
    elif tamper == "snapshot_hash":
        case["states"][-1]["snapshot_sha256"] = "0" * 64
    elif tamper == "readout":
        case["states"][-1]["readout"]["element_indices"][0] = 0
    else:
        case["affinity_objective_identity"]["coefficient"] = 2.0
    save_fixture(root, summary, case)
    with pytest.raises(ValueError):
        load_fixture(module, root, config_path)


@pytest.mark.parametrize("pin", ["expected_summary_sha256", "expected_config_sha256"])
def test_loader_rejects_wrong_external_byte_pin(tmp_path, monkeypatch, pin):
    module = load_renderer(monkeypatch)
    root, config_path, _, _ = fixture_study(tmp_path, module)
    with pytest.raises(ValueError):
        load_fixture(module, root, config_path, **{pin: "0" * 64})


def test_missing_saved_snapshot_is_never_reconstructed(tmp_path, monkeypatch):
    module = load_renderer(monkeypatch)
    root, config_path, _, case = fixture_study(tmp_path, module)
    (root / "case-0" / case["states"][-1]["snapshot"]).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        load_fixture(module, root, config_path)


def test_passed_native_label_cannot_override_disagreeing_raw_derivatives(
    tmp_path, monkeypatch
):
    module = load_renderer(monkeypatch)
    root, config_path, summary, case = fixture_study(tmp_path, module)
    report = summary["terminal_gradient_checks"]
    for probe in report["rows"][1]["probes"]:
        probe["finite_difference"] = 5.0
        probe["primal_endpoints"] = [-5 * probe["epsilon"], 5 * probe["epsilon"]]
        probe["agrees"] = True  # A forged status/boolean cannot make this a pass.
    put(root / "terminal_gradient_checks/report.json", report)
    save_fixture(root, summary, case)
    with pytest.raises(ValueError):
        load_fixture(module, root, config_path)


def test_disconnected_saved_endpoint_is_not_joined_or_discarded(tmp_path, monkeypatch):
    module = load_renderer(monkeypatch)
    root, config_path, summary, case = fixture_study(tmp_path, module)
    state = case["states"][-1]
    path = root / "case-0" / state["snapshot"]
    with np.load(path, allow_pickle=False) as saved:
        arrays = dict(saved)
    edges = np.full((15, 5), 0.01, dtype=np.float32)
    edges[:, 0] = 0.96
    np.savez_compressed(path, **{**arrays, "E_upper": edges})
    state["snapshot_sha256"] = digest(path)
    state["readout"].update(
        bond_upper_indices=[0] * 15, disconnected=True, fragment_count=6
    )
    save_fixture(root, summary, case)
    final = load_fixture(module, root, config_path)["frames"][-1]
    assert np.argmax(final["E"], -1).tolist() == [0] * 15
    assert final["readout"]["disconnected"] and final["readout"]["fragment_count"] == 6
    assert len(final["X"]) == 6 and final["readout"]["canonical_smiles"] is None


def test_saved_affinity_html_labels_and_contact_default_are_distinct(
    tmp_path, monkeypatch
):
    module = load_renderer(monkeypatch)
    root, config_path, _, case = fixture_study(tmp_path, module)
    study = load_fixture(module, root, config_path)
    output = tmp_path / "report"
    module.render_report([study], output)
    html = (output / "report.html").read_text(encoding="utf-8")
    data = json.loads(
        html.split('<script id="saved-data" type="application/json">')[1].split(
            "</script>"
        )[0]
    )
    assert data["primary_metric"] == "affinity"
    assert "Boltz" in data["title"] and "affinity" in data["title"].lower()
    assert data.get("historical_contact") is None
    assert "__LOSS_PLOT_BASE64__" not in html
    assert "fixed_contact_loss:frame.contact" not in html
    assert "affinity_probability_binary" in html
    assert data["studies"][0]["frames"][-1]["readout"] == case["states"][-1]["readout"]
    # Exercise the shared loader's old default: contact frames keep contact keys.
    from render_chemistry_trajectory import saved_frames

    contact_state = copy.deepcopy(case["states"][0])
    contact_state["contact_loss"] = -0.125
    frames = saved_frames(root / "case-0", [contact_state])
    assert frames[0]["contact"] == -0.125 and "affinity" not in frames[0]


def test_default_budget_and_cli_cannot_publish_small_fixture_as_real_run(
    tmp_path, monkeypatch
):
    module = load_renderer(monkeypatch)
    root, config_path, _, _ = fixture_study(tmp_path, module)
    with pytest.raises(ValueError):
        module.load_study(
            root,
            config_path,
            expected_job="29759212",
            expected_summary_sha256=digest(root / "summary.json"),
            expected_config_sha256=digest(config_path),
        )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render",
            "--study",
            str(root),
            "--config",
            str(config_path),
            "--job",
            "29759212",
            "--summary-sha256",
            digest(root / "summary.json"),
            "--config-sha256",
            digest(config_path),
            "--readiness",
            str(root / "missing-readiness.json"),
            "--readiness-sha256",
            "0" * 64,
            "--output",
            str(tmp_path / "rejected-report"),
        ],
    )
    with pytest.raises((ValueError, FileNotFoundError)):
        module.main()
    assert not (tmp_path / "rejected-report/report.html").exists()
