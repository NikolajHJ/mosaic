"""Validation-only checkpoint sealing; execute these tests on DTU HPC."""

import json
from pathlib import Path

import pytest
import torch

from latentmol.candidate_panel import seal_panel
from latentmol.experiments import file_sha256, write_json
from latentmol.representation import GraphSpec


def _inputs(root: Path):
    runs = root / "runs"
    runs.mkdir()
    spec = GraphSpec(max_nodes=2, atomic_numbers=(0, 6),
                     formal_charges=(0,), bond_names=("NONE", "SINGLE"))
    data = {"spec": spec.to_dict(), "spec_fingerprint": spec.fingerprint(),
            "dataset_sha256": "a" * 64, "split_sha256": "b" * 64,
            "counts": {"train": 20, "val": 10, "test": 10}}
    data_path = root / "manifest.json"
    write_json(data_path, data)
    requests = []
    for mode in ("vae", "ae"):
        parent = runs / f"20261001T000000Z_{mode}-full-workflow_12345678"
        training = runs / f"20261001T000000Z_{mode}-train_12345678"
        parent.mkdir()
        training.mkdir()
        checkpoint = training / "best_validity.pt"
        train_config = {"mode": mode, "model": {"decoder_variant": "wide_global_bond"},
                        "epochs": 10}
        torch.save({"format_version": 1, "epoch": 7,
                    "train_config": train_config, "run_id": training.name,
                    "model_config": {"decoder_variant": "wide_global_bond", "latent": 256},
                    "dataset_sha256": data["dataset_sha256"], "graph_spec": spec.to_dict(),
                    "continuation": {"split_sha256": data["split_sha256"],
                                     "algorithm": {"sha256": "c" * 64}},
                    "source_archive_sha256": "d" * 64}, checkpoint)
        write_json(parent / "run.json", {"name": f"{mode}-full-workflow",
                                          "run_id": parent.name, "status": "completed",
                                          "config": {"train": train_config},
                                          "stages": [{"training_run": str(training)}],
                                          "comparison_checkpoint": str(checkpoint),
                                          "source_archive_sha256": "d" * 64})
        selection = parent / "selection_before_test.json"
        write_json(selection, {"test_was_used_for_selection": False,
                               "checkpoints": [{"epoch": 7, "path": str(checkpoint),
                                                "roles": ["best_validation_validity"],
                                                "sha256": file_sha256(checkpoint)}]})
        requests.append({"id": f"{mode}-candidate", "mode": mode,
                         "variant": "wide_global_bond", "role": "best_validation_validity",
                         "selection_path": str(selection),
                         "selection_sha256": file_sha256(selection),
                         "checkpoint_sha256": file_sha256(checkpoint)})
    plan_path = root / "plan.json"
    write_json(plan_path, {"format_version": 1, "primary_candidate_id": "vae-candidate",
                           "selection_rationale": "VAE-first method screen; AE is a comparator.",
                           "historical_test_exposure": True,
                           "candidates": requests,
                           "cohort": {"split": "val", "seed": 17, "molecules": 5}})
    return runs, data_path, plan_path


def test_seals_matched_vae_primary_panel_without_test_data(tmp_path):
    runs, data, plan = _inputs(tmp_path)
    result = seal_panel(plan, data, runs)
    panel = json.loads((result / "panel.json").read_text())
    assert panel["primary_candidate_id"] == "vae-candidate"
    assert [item["mode"] for item in panel["candidates"]] == ["vae", "ae"]
    assert len(panel["cohort"]["indices"]) == 5
    assert len(set(panel["cohort"]["indices"])) == 5
    assert all(0 <= index < 10 for index in panel["cohort"]["indices"])
    assert all(item["latent_dimension"] == 256 for item in panel["candidates"])


@pytest.mark.parametrize("mutation", ["test_selection", "last_role", "checkpoint_hash",
                                      "parent_lineage"])
def test_rejects_leakage_or_unsealed_identity(tmp_path, mutation):
    runs, data, plan_path = _inputs(tmp_path)
    plan = json.loads(plan_path.read_text())
    first = plan["candidates"][0]
    if mutation == "test_selection":
        selection = Path(first["selection_path"])
        value = json.loads(selection.read_text())
        value["test_was_used_for_selection"] = True
        write_json(selection, value)
        first["selection_sha256"] = file_sha256(selection)
    elif mutation == "last_role":
        first["role"] = "last"
    else:
        if mutation == "checkpoint_hash":
            first["checkpoint_sha256"] = "0" * 64
        else:
            parent_path = Path(first["selection_path"]).parent / "run.json"
            parent = json.loads(parent_path.read_text())
            parent["stages"] = [{"training_run": str(runs / "unrelated-train")}]
            write_json(parent_path, parent)
    write_json(plan_path, plan)
    with pytest.raises(ValueError):
        seal_panel(plan_path, data, runs)
