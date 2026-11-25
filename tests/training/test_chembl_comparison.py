import csv
import gzip
import json
from dataclasses import asdict
from pathlib import Path

import torch

from latentmol.datasets.data import prepare_dataset
from latentmol.experiments import file_sha256
from latentmol.models import GraphAutoencoder, ModelConfig
from latentmol.representation import GraphSpec
from latentmol.training.chembl_comparison import _holdout_scaffold, replacement_slots, run_pilot, select_chembl
from latentmol.training.losses import LossConfig


def test_replacement_budget_is_exact_seeded_and_balanced():
    positions = replacement_slots(400, 32, 15, 21)
    assert len(positions) == len(set(positions)) == 1920
    assert all(0 <= position < 12800 for position in positions)
    assert {sum(position // 32 == step for position in positions) for step in range(400)} == {4, 5}
    assert positions == replacement_slots(400, 32, 15, 21)
    assert positions != replacement_slots(400, 32, 15, 22)


def _audit_csv(path):
    fields = ("chembl_id", "smiles", "scaffold", "scaffold_in_moses_train", "scaffold_in_moses_any", "heavy_atoms")
    # One acyclic row, one MOSES-scaffold overlap and two members of an unseen
    # scaffold family. The whole unseen family belongs to the holdout pool.
    rows = [
        ("CHEMBL1", "CCCCCCC", "", 0, 0, 7),
        ("CHEMBL2", "c1ccccc1", "c1ccccc1", 0, 1, 6),
        ("CHEMBL3", "C1CCCCC1", "C1CCCCC1", 0, 0, 6),
        ("CHEMBL4", "CC1CCCCC1", "C1CCCCC1", 0, 0, 7),
    ]
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)
    return next(seed for seed in range(100) if _holdout_scaffold("C1CCCCC1", seed, 20))


def test_chembl_scaffold_split_keeps_unseen_family_out_of_training(tmp_path):
    path = tmp_path / "eligible.csv.gz"
    seed = _audit_csv(path)
    args = dict(train_count=2, holdout_count=2, seed=42, split_seed=seed, holdout_percent=20)
    panels, seen = select_chembl(path, **args)
    assert (panels, seen) == select_chembl(path, **args)
    assert seen == {"train": 2, "holdout": 2}
    assert {row["scaffold"] for row in panels["holdout"]} == {"C1CCCCC1"}
    assert {row["scaffold"] for row in panels["train"]} == {"", "c1ccccc1"}


def test_short_matched_pilot_trains_and_evaluates_both_arms(tmp_path):
    spec = GraphSpec(max_nodes=8)
    source = tmp_path / "molecules.smi"
    source.write_text("\n".join(["CCO", "CCN", "CCS", "CCC", "CCF", "CCCl", "CCBr", "COC",
                                 "CNC", "CC=O", "CC#N", "CC=C", "CC#C", "C=CO", "C=CN"]))
    data = tmp_path / "cache"
    prepare_dataset(source, data, spec=spec, train_count=7, val_count=7, seed=31)
    manifest = json.loads((data / "manifest.json").read_text())
    torch.manual_seed(19)
    model = GraphAutoencoder(spec, ModelConfig(hidden=8, latent=4, layers=1))
    checkpoint = tmp_path / "epoch45.pt"
    torch.save(dict(format_version=1, epoch=45, graph_spec=spec.to_dict(), model_config=asdict(model.config),
        model_state=model.state_dict(), dataset_sha256=manifest["dataset_sha256"],
        train_config=dict(mode="vae"), loss_config=asdict(LossConfig())), checkpoint)
    audit = tmp_path / "audit"
    audit.mkdir()
    eligible = audit / "eligible.csv.gz"
    split_seed = _audit_csv(eligible)
    audit_summary = audit / "summary.json"
    audit_summary.write_text(json.dumps(dict(version="chembl-moses-audit-v1", source=dict(complete=True),
        moses=dict(dataset_sha256=manifest["dataset_sha256"], manifest_sha256=file_sha256(data / "manifest.json")),
        eligible=dict(path=eligible.name, sha256=file_sha256(eligible), count=4))))
    config = json.loads((Path(__file__).resolve().parents[2] /
                         "experiments/vae/epoch45-chembl-comparison.json").read_text())
    config.update(checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint), data=str(data),
                  data_manifest_sha256=file_sha256(data / "manifest.json"), dataset_sha256=manifest["dataset_sha256"],
                  updates=1, batch_size=2, prior_anchor_count=2, chembl_percent=50,
                  chembl_scaffold_seed=split_seed, chembl_holdout_count=2)
    config["validation_pairs"].update(pool_count=7, pair_count=2, min_atoms=2, max_atoms=8,
                                      min_similarity=.01, min_core_fraction=.5, max_candidate_pairs=30)
    config["path_evaluation"].update(batch_size=4, categorical_draws=1, probability_snapshot_pairs=1)
    config["comparison"].update(reconstruction_count=4, prior_count=4, prior_seeds=[22],
                                anchor_count=2, directions=1, exclude_reconstruction_panels=[],
                                atom_order=dict(count=2, permutations=2, seed=23))
    output = tmp_path / "run"
    result = run_pilot(config, audit, file_sha256(audit_summary), output, device="cpu")
    assert result["status"] == "completed" and result["chembl_replacements"] == 1
    assert result["chembl_holdout_count"] == 2 and result["chembl_holdout_scaffolds"] == 1
    assert set(result["comparison"]) == set(result["chembl_holdout_reconstruction"]) == {
        "epoch45", "moses", "mixture"}
    assert result["paired_chembl_holdout_exact"]["attempts"] == 2
    assert len(result["paired_path_argmax_validity"]) == 9
    assert file_sha256(checkpoint) == config["checkpoint_sha256"]
    for name in ("moses", "mixture"):
        training = json.loads((output / name / "training.jsonl").read_text())
        assert training["step"] == 1
        assert abs(training["prior_decoder_kl"]) < 1e-6
