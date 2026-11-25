"""Real optimizer/stream continuation checks, executed only on DTU HPC.

CPU fixtures establish engineering behavior. The same small stable-decoder
comparison also runs on actual CUDA before recovery training begins.
"""

from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from latentmol.chemistry import encode_smiles
from latentmol.decoder_finetune import FineTuneConfig, fit_case, make_panel
from latentmol.decoder_budget_recovery import read_recovery
from latentmol.experiments import file_sha256
from latentmol.losses import LossConfig
from latentmol.models import GraphAutoencoder, ModelConfig
from latentmol.representation import GraphSpec
from latentmol.train import state_digest


class Molecules:
    def __init__(self, spec):
        self.spec = spec
        rows = [encode_smiles(s, spec) for s in ("C", "CC", "CO", "CN")]
        self.smiles = [row["smiles"] for row in rows]
        self.rows = [{key: torch.tensor(row[source], dtype=torch.long) for key, source in
                     (("atoms", "atom_types"), ("charges", "charges"), ("bonds", "bond_types"))} for row in rows]
        self.manifest = {"dataset_sha256": "a" * 64, "split_sha256": "b" * 64}
    def __len__(self): return len(self.rows)
    def __getitem__(self, index): return self.rows[index]


def setup_case(tmp_path, monkeypatch, device):
    monkeypatch.setenv("LATENTMOL_SOURCE_SHA256", "f" * 64)
    spec = GraphSpec(max_nodes=3)
    config = ModelConfig(hidden=8, latent=4, layers=1, decoder_variant="stable_deep_pair_refine",
                         global_hidden=8, trunk_hidden=12, pair_hidden=4, attention_heads=2, refinement_rounds=1)
    model = GraphAutoencoder(spec, config)
    initial = tmp_path / "original.pt"
    torch.save({"format_version": 1, "graph_spec": spec.to_dict(), "model_config": asdict(config),
        "model_state": model.state_dict(), "loss_config": asdict(LossConfig()), "dataset_sha256": "a" * 64,
        "train_config": {"mode": "vae", "beta": .001}, "epoch": 89}, initial)
    data = Molecules(spec)
    fine = FineTuneConfig(epochs=3, batch_size=2, prior_batch_size=2, chemistry_reconstruction_batch=2)
    arguments = dict(initializer=initial, train_data=data, val_data=data, output_root=tmp_path / "runs",
        config=fine, seed=17, coefficient=.1, train_limit=4, val_limit=4,
        validation_panel=make_panel(2, 4, 81000), device=device, source_root=tmp_path, budget_seconds=180)
    full = fit_case(**arguments)
    parent = Path(full["run_dir"])
    # Keep the completed first epoch and its receipt, simulating cancellation
    # later in epoch2. Do not redraw or reconstruct RNG by skipping samples.
    receipt = tmp_path / "parent-metrics.jsonl"
    receipt.write_text((parent / "metrics.jsonl").read_text().splitlines()[0] + "\n")
    checkpoint = parent / "epoch-001.pt"
    resume = {"checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
        "parent_record": str(parent / "run.json"), "parent_record_sha256": file_sha256(parent / "run.json"),
        "parent_metrics": str(receipt), "parent_metrics_sha256": file_sha256(receipt), "parent_archive_sha256": "f" * 64}
    return arguments, full, resume


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_resume_matches_uninterrupted_optimizer_parameters_and_streams(tmp_path, monkeypatch, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA allocation required")
    arguments, full, resume = setup_case(tmp_path, monkeypatch, device)
    continued = fit_case(**arguments, resume=resume)
    assert full["terminal_rng"] == continued["terminal_rng"]
    assert full["encoder_sha256"] == continued["encoder_sha256"]
    left, right = [torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
                   for result in (full, continued)]
    assert left["epoch"] == right["epoch"] == 3
    assert state_digest(left["model_state"]) == state_digest(right["model_state"])
    for param_id, state in left["decoder_finetune"]["optimizer_state"]["state"].items():
        other = right["decoder_finetune"]["optimizer_state"]["state"][param_id]
        for key, value in state.items():
            if isinstance(value, torch.Tensor): torch.testing.assert_close(value, other[key], rtol=0, atol=0)
            else: assert value == other[key]
    assert right["decoder_finetune"]["resume"]["epoch"] == 1
    row = json.loads((Path(continued["run_dir"]) / "run.json").read_text())
    assert row["config"]["initializer_sha256"] == file_sha256(arguments["initializer"])
    assert row["comparison_status"] == "complete_fixed_endpoint"


def test_resume_rejects_another_arm_and_incomplete_epoch_receipt(tmp_path, monkeypatch):
    arguments, _, resume = setup_case(tmp_path, monkeypatch, "cpu")
    wrong = {**arguments, "coefficient": 1.}
    with pytest.raises(ValueError, match="identity differs"):
        fit_case(**wrong, resume=resume)
    receipt = Path(resume["parent_metrics"])
    row = json.loads(receipt.read_text());row["updates"] -= 1
    receipt.write_text(json.dumps(row) + "\n")
    altered = {**resume, "parent_metrics_sha256": file_sha256(receipt)}
    with pytest.raises(ValueError, match="data budget differs"):
        fit_case(**arguments, resume=altered)
    with pytest.raises(ValueError, match="identity changed"):
        fit_case(**arguments, resume=resume)


def test_resume_from_checkpoint_before_epoch48_model_fields(tmp_path, monkeypatch):
    arguments, full, resume = setup_case(tmp_path, monkeypatch, "cpu")
    checkpoint = torch.load(resume["checkpoint"], map_location="cpu", weights_only=False)
    for field in ("atom_refinement_dropout", "extra_bond_rounds", "encoder_graph_attention_layers"):
        del checkpoint["model_config"][field]
    path = tmp_path / "legacy.pt"
    torch.save(checkpoint, path)
    resumed = fit_case(**arguments, resume={**resume, "checkpoint": str(path),
                                           "checkpoint_sha256": file_sha256(path)})
    assert resumed["terminal_rng"] == full["terminal_rng"]


def test_segment_budget_reserves_three_complete_cases_and_evaluation(tmp_path):
    path = Path(__file__).parents[1] / "configs/decoder_budget_recovery.json"
    current = read_recovery(path, file_sha256(path))
    current["segment_budget_seconds"] = 3 * current["case_budget_seconds"]
    candidate = tmp_path / "short.json";candidate.write_text(json.dumps(current))
    with pytest.raises(ValueError, match="reserve all three"):
        read_recovery(candidate, file_sha256(candidate))
