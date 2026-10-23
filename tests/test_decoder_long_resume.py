"""HPC-only epoch-horizon extension and repeated exact continuation fixtures.

These small stable-decoder checks validate engineering on one device/runtime.
They do not establish cross-GPU bitwise reproducibility or chemical benefit.
"""
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest
import torch

from latentmol.decoder_continuation import merge_epoch_history
from latentmol.decoder_finetune import FineTuneConfig, fit_case, make_panel
from latentmol.experiments import file_sha256
from latentmol.losses import LossConfig
from latentmol.models import GraphAutoencoder, ModelConfig
from latentmol.representation import GraphSpec
from latentmol.train import state_digest
from test_decoder_resume import Molecules


SOURCE_SHA = "f" * 64


def make_case(tmp_path, monkeypatch, device):
    """Same frozen initializer, data and independent streams in every branch."""
    monkeypatch.setenv("LATENTMOL_SOURCE_SHA256", SOURCE_SHA)
    spec = GraphSpec(max_nodes=3)
    architecture = ModelConfig(hidden=8, latent=4, layers=1,
        decoder_variant="stable_deep_pair_refine", global_hidden=8, trunk_hidden=12,
        pair_hidden=4, attention_heads=2, refinement_rounds=1)
    model = GraphAutoencoder(spec, architecture)
    initial = tmp_path / "original.pt"
    torch.save({"format_version": 1, "graph_spec": spec.to_dict(),
        "model_config": asdict(architecture), "model_state": model.state_dict(),
        "loss_config": asdict(LossConfig()), "dataset_sha256": "a" * 64,
        "train_config": {"mode": "vae", "beta": .001}, "epoch": 89}, initial)
    data = Molecules(spec)
    config = FineTuneConfig(epochs=5, batch_size=3, prior_batch_size=None,
                          chemistry_reconstruction_batch=3)
    arguments = dict(initializer=initial, train_data=data, val_data=data,
        output_root=tmp_path / "runs", config=config, seed=17, coefficient=1.0,
        train_limit=4, val_limit=4, validation_panel=make_panel(2, 4, 81000),
        device=device, source_root=tmp_path, budget_seconds=300)
    return arguments, {"encoder": state_digest(model.encoder.state_dict()),
                       "reference": state_digest(model.state_dict()),
                       "initializer": file_sha256(initial)}


def envelope(result, metrics):
    record = Path(result["run_dir"]) / "run.json"
    return {"checkpoint": result["checkpoint"], "checkpoint_sha256": result["sha256"],
            "parent_record": str(record), "parent_record_sha256": file_sha256(record),
            "parent_metrics": str(metrics), "parent_metrics_sha256": file_sha256(metrics),
            "parent_archive_sha256": SOURCE_SHA}


def assert_optimizer_equal(left, right):
    """AdamW moments and step counters matter as much as decoder parameters."""
    assert left["param_groups"] == right["param_groups"]
    assert left["state"].keys() == right["state"].keys()
    for param_id, state in left["state"].items():
        other = right["state"][param_id]
        assert state.keys() == other.keys()
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, other[key], rtol=0, atol=0)
            else:
                assert value == other[key]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_authorized_extension_and_second_resume_match_uninterrupted_run(tmp_path, monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Allocated CUDA required")
    arguments, original = make_case(tmp_path, monkeypatch, device)
    full = fit_case(**arguments)
    prefix = fit_case(**{**arguments, "config": replace(arguments["config"], epochs=2)})
    prefix_metrics = Path(prefix["run_dir"]) / "metrics.jsonl"
    continued = fit_case(**arguments, resume=envelope(prefix, prefix_metrics),
                         epoch_extension={"from_epochs": 2, "to_epochs": 5}, stop_after_epoch=4)
    assert continued["completed_epochs"] == 4
    assert continued["requested_epochs"] == 5
    assert continued["endpoint_completed"] is False
    intermediate_record = json.loads((Path(continued["run_dir"]) / "run.json").read_text())
    assert intermediate_record["comparison_status"] == "complete_epoch_segment"
    assert intermediate_record["completed_epochs"] == 4
    assert intermediate_record["config"]["config"]["epochs"] == 5
    current_metrics = Path(continued["run_dir"]) / "metrics.jsonl"
    assert [json.loads(line)["epoch"] for line in current_metrics.read_text().splitlines()] == [3, 4]
    history = tmp_path / "completed-history-004.jsonl"
    merge_epoch_history(prefix_metrics, current_metrics, history, expected_epoch=4)
    assert history.read_bytes() == prefix_metrics.read_bytes() + current_metrics.read_bytes()
    # The second segment keeps the scientific horizon5: no second exception
    # to config identity and no reset/replay-by-skipping of any RNG stream.
    final = fit_case(**arguments, resume=envelope(continued, history))
    assert final["completed_epochs"] == final["requested_epochs"] == 5
    assert final["endpoint_completed"] is True
    final_record = json.loads((Path(final["run_dir"]) / "run.json").read_text())
    assert final_record["comparison_status"] == "complete_fixed_endpoint"
    final_history = tmp_path / "completed-history-005.jsonl"
    merge_epoch_history(history, Path(final["run_dir"]) / "metrics.jsonl", final_history, expected_epoch=5)
    rows = [json.loads(line) for line in final_history.read_text().splitlines()]
    assert [row["epoch"] for row in rows] == [1, 2, 3, 4, 5]
    assert all(row["examples"] == 4 and row["updates"] == 2 for row in rows)
    assert all(row["training_point_counts"] == {"reconstruction": 4, "posterior_chemistry": 4,
        "fresh_prior": 4, "connectivity_graph_draws": 32} for row in rows)
    assert rows[-1]["rng_sha256"] == final["terminal_rng"]
    assert full["terminal_rng"] == final["terminal_rng"]
    assert set(final["terminal_rng"]) == {"loader", "posterior", "prior", "connectivity"}
    assert prefix["encoder_sha256"] == continued["encoder_sha256"] == final["encoder_sha256"] == original["encoder"]
    left, right = [torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
                   for result in (full, final)]
    assert left["epoch"] == right["epoch"] == 5
    assert state_digest(left["model_state"]) == state_digest(right["model_state"])
    assert_optimizer_equal(left["decoder_finetune"]["optimizer_state"], right["decoder_finetune"]["optimizer_state"])
    for result in (prefix, continued, final):
        record = json.loads((Path(result["run_dir"]) / "run.json").read_text())
        assert record["frozen_reference_state_sha256"] == original["reference"]
        assert record["config"]["initializer_sha256"] == original["initializer"]
    assert file_sha256(arguments["initializer"]) == original["initializer"]


@pytest.mark.parametrize("change", ["undeclared", "learning_rate", "from_horizon", "to_horizon",
    "wrong_arm", "wrong_seed", "wrong_source", "wrong_split", "stale_history",
    "incomplete_updates", "incomplete_examples", "wrong_latest_rng"])
def test_extension_rejects_drift_or_unauthenticated_completed_history(tmp_path, monkeypatch, change):
    arguments, _ = make_case(tmp_path, monkeypatch, "cpu")
    prefix = fit_case(**{**arguments, "config": replace(arguments["config"], epochs=2)})
    metrics = Path(prefix["run_dir"]) / "metrics.jsonl"
    resume = envelope(prefix, metrics)
    extension = {"from_epochs": 2, "to_epochs": 5}
    if change == "undeclared": extension = None
    elif change == "learning_rate": arguments["config"] = replace(arguments["config"], learning_rate=4e-5)
    elif change == "from_horizon": extension["from_epochs"] = 1
    elif change == "to_horizon": extension["to_epochs"] = 4
    elif change == "wrong_arm": arguments["coefficient"] = 0.0
    elif change == "wrong_seed": arguments["seed"] = 23
    elif change == "wrong_source": resume["parent_archive_sha256"] = "e" * 64
    elif change == "wrong_split": arguments["train_data"].manifest["split_sha256"] = "c" * 64
    else:
        rows = [json.loads(line) for line in metrics.read_text().splitlines()]
        if change == "stale_history":
            metrics.write_text(metrics.read_text() + "\n")  # Keep old digest intentionally.
        else:
            if change == "incomplete_updates": rows[0]["updates"] -= 1
            elif change == "incomplete_examples": rows[0]["examples"] -= 1
            else: rows[-1]["rng_sha256"]["prior"] = "0" * 64
            metrics.write_text("".join(json.dumps(row) + "\n" for row in rows))
            resume["parent_metrics_sha256"] = file_sha256(metrics)
    with pytest.raises(ValueError):
        fit_case(**arguments, resume=resume, epoch_extension=extension, stop_after_epoch=4)


def write_history(path, epochs):
    rows = [{"epoch": epoch, "examples": 4, "updates": 2,
             "rng_sha256": {name: "f" * 64 for name in ("loader", "posterior", "prior", "connectivity")}}
            for epoch in epochs]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_history_materializer_preserves_bytes_in_fresh_immutable_file(tmp_path):
    initial, current, merged = [tmp_path / name for name in ("initial.jsonl", "current.jsonl", "merged.jsonl")]
    write_history(current, [1, 2])
    merge_epoch_history(None, current, initial, expected_epoch=2)
    assert initial.read_bytes() == current.read_bytes()
    write_history(current, [3, 4])
    merge_epoch_history(initial, current, merged, expected_epoch=4)
    expected = initial.read_bytes() + current.read_bytes()
    assert merged.read_bytes() == expected
    with pytest.raises(FileExistsError):
        merge_epoch_history(initial, current, merged, expected_epoch=4)
    assert merged.read_bytes() == expected


@pytest.mark.parametrize("parent_epochs,current_epochs,target", [([1], [3], 3),
    ([1, 2], [2, 3], 3), ([1, 2], [3], 4), ([], [2, 3], 3)])
def test_history_materializer_rejects_gaps_overlaps_and_missing_terminal_epoch(tmp_path, parent_epochs, current_epochs, target):
    parent, current, destination = [tmp_path / name for name in ("parent.jsonl", "current.jsonl", "bad.jsonl")]
    write_history(parent, parent_epochs)
    write_history(current, current_epochs)
    with pytest.raises(ValueError):
        merge_epoch_history(parent if parent_epochs else None, current, destination, expected_epoch=target)
    assert not destination.exists()
