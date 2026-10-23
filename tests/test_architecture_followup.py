"""Exact parent and sealed-validation checkpoint selection, on DTU HPC only."""

import json

import pytest
import torch

from latentmol.architecture_followup import selected_checkpoint
from latentmol.experiments import file_sha256


@pytest.mark.parametrize("variant", ["residual", "pair_refine", "deep_pair_refine",
                                      "global_bond", "wide_global_bond"])
def test_followup_selects_only_exact_completed_parent_and_sealed_checkpoint(tmp_path, variant):
    root = tmp_path / "runs"
    parent = root / "20260928T000000Z_ae-full-workflow_01234567"
    checkpoint = root / "20260928T000001Z_ae-train_01234567" / "best.pt"
    parent.mkdir(parents=True)
    checkpoint.parent.mkdir()
    train_config = {"mode": "ae", "model": {"decoder_variant": variant},
                    "lr_milestones": ((61, 0.0003),)}
    torch.save({"source_archive_sha256": "abc", "train_config": train_config,
                "model_config": train_config["model"], "run_id": checkpoint.parent.name,
                "epoch": 80}, checkpoint)
    (parent / "run.json").write_text(json.dumps({
        "name": "ae-full-workflow", "status": "completed",
        "environment": {"job_id": "12345"}, "source_archive_sha256": "abc",
        "config": {"train": {**train_config, "lr_milestones": [[61, 0.0003]]}},
        "stages": [{"training_run": str(checkpoint.parent)}],
    }))
    (parent / "selection_before_test.json").write_text(json.dumps({
        "test_was_used_for_selection": False,
        "checkpoints": [{"path": str(checkpoint), "sha256": file_sha256(checkpoint),
                         "epoch": 80,
                         "roles": ["best_validation_reconstruction"]}],
    }))
    kwargs = {"parent_job_id": "12345", "parent_source_sha256": "abc",
              "mode": "ae", "variant": variant}
    assert selected_checkpoint(root, **kwargs) == checkpoint.resolve()
    with pytest.raises(RuntimeError, match="one exact parent"):
        selected_checkpoint(root, **{**kwargs, "parent_job_id": "12346"})
    parent_record = json.loads((parent / "run.json").read_text())
    parent_record["stages"][-1]["training_run"] = str(root / "unrelated")
    (parent / "run.json").write_text(json.dumps(parent_record))
    with pytest.raises(RuntimeError, match="final recorded stage"):
        selected_checkpoint(root, **kwargs)
    parent_record["stages"][-1]["training_run"] = str(checkpoint.parent)
    (parent / "run.json").write_text(json.dumps(parent_record))
    earlier_stage = root / "20260928T000000Z_ae-train_abcdef00"
    earlier_stage.mkdir()
    parent_record["stages"].insert(0, {"training_run": str(earlier_stage)})
    (parent / "run.json").write_text(json.dumps(parent_record))
    torch.save({"source_archive_sha256": "abc", "train_config": train_config,
                "model_config": train_config["model"], "run_id": earlier_stage.name,
                "epoch": 60}, checkpoint)
    sealed = json.loads((parent / "selection_before_test.json").read_text())
    sealed["checkpoints"][0].update(sha256=file_sha256(checkpoint), epoch=60)
    (parent / "selection_before_test.json").write_text(json.dumps(sealed))
    assert selected_checkpoint(root, **kwargs) == checkpoint.resolve()
    torch.save({"source_archive_sha256": "wrong", "train_config": train_config,
                "model_config": train_config["model"], "run_id": checkpoint.parent.name,
                "epoch": 60}, checkpoint)
    sealed["checkpoints"][0]["sha256"] = file_sha256(checkpoint)
    (parent / "selection_before_test.json").write_text(json.dumps(sealed))
    with pytest.raises(RuntimeError, match="provenance differs"):
        selected_checkpoint(root, **kwargs)
    parent_record["status"] = "failed"
    (parent / "run.json").write_text(json.dumps(parent_record))
    with pytest.raises(RuntimeError, match="did not complete"):
        selected_checkpoint(root, **kwargs)
    parent_record["status"] = "completed"
    (parent / "run.json").write_text(json.dumps(parent_record))
    sealed["test_was_used_for_selection"] = True
    (parent / "selection_before_test.json").write_text(json.dumps(sealed))
    with pytest.raises(RuntimeError, match="unsealed"):
        selected_checkpoint(root, **kwargs)
    sealed["test_was_used_for_selection"] = False
    (parent / "selection_before_test.json").write_text(json.dumps(sealed))
    checkpoint.write_bytes(b"changed checkpoint")
    with pytest.raises(RuntimeError, match="hash changed"):
        selected_checkpoint(root, **kwargs)
