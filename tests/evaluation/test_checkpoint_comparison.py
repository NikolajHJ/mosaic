"""Small CPU fixtures for paired evaluation; no native models or training."""
import copy
import gzip
import json
from pathlib import Path

import pytest
import torch

from latentmol.chemistry import encode_smiles
from latentmol.datasets.data import prepare_dataset
from latentmol.evaluation.checkpoint_comparison import (
    _validated_config, compare_neighbors, graph_record, paired_difference,
    run_comparison, select_indices, summarize_rows, unit_rms_directions,
)
from latentmol.experiments import file_sha256
from latentmol.models import GraphAutoencoder, ModelConfig
from latentmol.representation import GraphSpec, HardGraphBatch
from latentmol.training.losses import LossConfig
from dataclasses import asdict


def test_panels_are_paired_without_replacement_and_honor_exclusions():
    a = select_indices(30, 10, 913, [1, 4, 7])
    assert a == select_indices(30, 10, 913, [7, 4, 1])
    assert len(set(a)) == 10 and not set(a) & {1, 4, 7}
    with pytest.raises(ValueError, match="Insufficient"):
        select_indices(5, 4, 913, [0, 1])


def test_unit_rms_directions_and_paired_denominators():
    a = unit_rms_directions(4, 8, 9)
    torch.testing.assert_close(a.square().mean(-1), torch.ones(4))
    assert torch.equal(a, unit_rms_directions(4, 8, 9))
    result = paired_difference([False, False, True, True], [True, False, False, True])
    assert result["gained"] == result["lost"] == 1 and result["difference"] == 0
    with pytest.raises(ValueError):
        paired_difference([True], [True, False])


def _hard(smiles, spec):
    row = encode_smiles(smiles, spec)
    return HardGraphBatch(*(torch.tensor(row[k], dtype=torch.long)[None]
                            for k in ("atom_types", "charges", "bond_types")))


def test_invalid_disconnected_and_representation_counts_keep_raw_graphs():
    spec = GraphSpec(max_nodes=8)
    good = graph_record(_hard("CCS", spec), 0, spec)
    disconnected = _hard("CCC", spec)
    disconnected.bonds.zero_()
    bad = graph_record(disconnected, 0, spec)
    assert not bad["valid"] and bad["components"] == 3
    assert bad["classification"]["failure_kind"] == "chemical"
    metrics = summarize_rows([good, bad])
    assert metrics["attempts"] == 2 and metrics["validity"] == .5
    assert metrics["sulfur_atom_fraction"] == pytest.approx(1 / 6)
    assert len(bad["graph"]["atoms"]) == 8
    changed = graph_record(_hard("CCO", spec), 0, spec)
    neighbors = compare_neighbors([good, changed, bad], [good, good, good])
    assert neighbors["valid_changed_count"] == 1
    assert neighbors["fingerprint_eligible"] == 2


def test_config_guards_unsafe_names_and_final_test_split():
    config = dict(version="paired-vae-checkpoints-v1", checkpoints=[dict(id="../oops")])
    with pytest.raises(ValueError, match="Unsafe"):
        _validated_config(config)


def test_small_end_to_end_freezes_checkpoints_and_preserves_all_draws(tmp_path):
    spec = GraphSpec(max_nodes=8)
    source = tmp_path / "molecules.smi"
    source.write_text("\n".join(["C", "CC", "CCC", "CCCC", "CCO", "CCN", "CO", "CN", "CCS", "CCF", "CCl", "CBr"]))
    data = tmp_path / "data"
    prepare_dataset(source, data, spec=spec, train_count=4, val_count=4, seed=19)
    model = GraphAutoencoder(spec, ModelConfig(hidden=8, latent=4, layers=1))
    manifest = json.loads((data / "manifest.json").read_text())
    candidates = []
    for epoch in (43, 44):
        path = tmp_path / f"{epoch}.pt"
        torch.save(dict(format_version=1, epoch=epoch, graph_spec=spec.to_dict(),
            model_config=asdict(model.config), model_state=model.state_dict(),
            dataset_sha256=manifest["dataset_sha256"], train_config=dict(mode="vae"),
            loss_config=asdict(LossConfig())), path)
        candidates.append(dict(id=f"epoch{epoch}", epoch=epoch, path=str(path), sha256=file_sha256(path)))
    config = dict(version="paired-vae-checkpoints-v1", data=str(data),
        data_manifest_sha256=file_sha256(data / "manifest.json"), dataset_sha256=manifest["dataset_sha256"],
        checkpoints=candidates, split="val", prior_count=4, prior_seeds=[9, 10],
        categorical_seed=11, temperature=1., reconstruction_count=4, row_seed=12,
        exclude_reconstruction_panels=[], anchor_count=2, directions=2, direction_seed=13,
        scales=[.05, .1], batch_size=2)
    output = tmp_path / "results"
    result = run_comparison(config, output, device="cpu")
    assert result["status"] == "completed" and result["test_split_untouched"]
    assert result["paired"]["epoch43_to_epoch44"]["prior_valid"]["difference"] == 0
    for candidate in candidates:
        assert file_sha256(Path(candidate["path"])) == candidate["sha256"]
        folder = output / candidate["id"]
        with gzip.open(folder / "prior.jsonl.gz", "rt") as stream:
            rows = [json.loads(line) for line in stream]
        assert len(rows) == 8 and all("graph" in row for row in rows)
        value = result["candidates"][candidate["id"]]
        assert value["unchanged"] and len(value["neighborhoods"]) == 4
        assert all(n["argmax"]["attempts"] == 4 for n in value["neighborhoods"])
    assert (output / "summary.md").is_file() and (output / "comparison.csv").is_file()
    final_test = copy.deepcopy(config)
    final_test["split"] = "test"
    with pytest.raises(ValueError, match="reserves the final test"):
        _validated_config(final_test)
    with pytest.raises(FileExistsError):
        run_comparison(config, output, device="cpu")
    changed = copy.deepcopy(config)
    changed["checkpoints"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Checkpoint bytes"):
        run_comparison(changed, tmp_path / "bad", device="cpu")
    assert json.loads((tmp_path / "bad/failure.json").read_text())["training_started"] is False
