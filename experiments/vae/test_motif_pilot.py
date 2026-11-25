"""The parent cannot use CUDA concurrently with an exclusive-process training child."""

import json
import sys
from pathlib import Path

import torch

from experiments.vae import motif_pilot as pilot


def test_train_both_arms_before_parent_touches_gpu(tmp_path, monkeypatch):
    recipe = Path(__file__).with_name("epoch48-paired-leg1.json")
    policy = pilot.read(Path(__file__).with_name("epoch48-motif-pilot.json"))
    protocol, parent, output = tmp_path / "pilot.json", tmp_path / "parent.json", tmp_path / "runs"
    pilot.write_json(protocol, dict(policy, updates_per_arm=8))
    pilot.write_json(parent, {"checkpoint": "parent.pt"})
    completed = []

    def train(argv, check):
        assert check is True
        directory = Path(argv[-1])
        directory.mkdir()
        weak = directory.name == "weak"
        assert ("functional_groups" in pilot.read(Path(argv[argv.index("--config") + 1]))) == weak
        path = directory / "saved.pt"
        path.write_bytes(b"fixture")
        latest = {"checkpoint": str(path), "sha256": pilot.file_sha256(path), "updates": 108}
        pilot.write_json(directory / "summary.json", dict(segment_updates=8, latest=latest, elapsed_seconds=10))
        start = dict(updates=100, seed=7, prior_points=512, validation_rows=[10], dataset_sha256="abc")
        (directory / "quality-panel.jsonl").write_text(
            json.dumps(start) + "\n" + json.dumps(dict(start, updates=108)) + "\n")
        with (directory / "training.jsonl").open("w") as stream:
            for i in range(1, 9):
                event = {"seconds": 0.1}
                if weak and i % 4 == 0:
                    event["functional_groups"] = dict(draws=48, valid=10, aromatic_hits=0, amide_hits=1,
                        aromatic_informative=False, amide_informative=True, weighted_loss=0.0001)
                stream.write(json.dumps(event) + "\n")
        completed.append(directory.name)

    def require_training_done(*args, **kwargs):
        assert completed == ["control", "weak"]
        return "test GPU"

    def score(path, out, policy):
        require_training_done()
        assert path.is_file()
        rows = [{"failure": "sanitization", "smiles": None, **pilot.chemistry(None)}]
        return {"argmax": rows, "sample": rows}

    monkeypatch.setenv("LSB_JOBID", "1")
    monkeypatch.setattr(sys, "argv", ["pilot", "--config", str(recipe), "--pilot", str(protocol),
                                      "--parent-receipt", str(parent), "--output", str(output)])
    monkeypatch.setattr(pilot.subprocess, "run", train)
    for name in ("get_device_name", "is_available", "_lazy_init"):
        monkeypatch.setattr(torch.cuda, name, require_training_done)
    monkeypatch.setattr(pilot, "evaluate", score)
    pilot.main()
    report = pilot.read(output / "comparison.json")
    assert report["identical_initial_fixed_panel_results"]
    assert report["arms"]["weak"]["functional_group_activity"]["draws"] == 96
    assert report["arms"]["control"]["functional_group_activity"]["draws"] == 0
