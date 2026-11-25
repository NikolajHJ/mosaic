"""Evaluate frozen VAE checkpoints on the combined corpus and held-out paths."""

import argparse
import json
import os
from pathlib import Path
import time

import torch

from latentmol.datasets.combined import CombinedCorpus
from latentmol.evaluation.checkpoint_comparison import decode_panel, paired_difference, reconstruct
from latentmol.evaluation.evaluate import load_model
from latentmol.evaluation.interpolation import _encode, evaluate_arm, load_pair_graphs
from latentmol.evaluation.molecular_pairs import select_pairs
from latentmol.experiments import file_sha256, write_json
from latentmol.train import state_digest
from latentmol.training.losses import LossConfig


def evaluate(config, output, *, checkpoint, checkpoint_sha256, split, device):
    if split not in ("val", "test"):
        raise ValueError("Evaluate only an explicit held-out split")
    if split == "test" and config["seed_by_split"]["val"] == config["seed_by_split"]["test"]:
        raise ValueError("Final test prior draws must differ from validation")
    corpus = CombinedCorpus(config["data"])
    candidates = {"epoch45": config["checkpoint"], "candidate": dict(path=str(checkpoint), sha256=checkpoint_sha256)}
    datasets = {source: corpus.split(split, source=source) for source in ("moses", "chembl")}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    pair_panels = {}
    for source, dataset in datasets.items():
        settings = dict(config["pairs"], pool_count=min(config["pairs"]["pool_count"], len(dataset)))
        settings["pair_seed"] = config["seed_by_split"][split] + (1 if source == "moses" else 2)
        pairs, selection = select_pairs(dataset.smiles, settings)
        if not pairs:
            raise ValueError(f"No usable interpolation pairs in the {source} {split} panel")
        write_json(output / f"{source}-pairs.json", dict(selection=selection, pairs=pairs))
        pair_panels[source] = (pairs, load_pair_graphs(dataset, pairs, dataset.spec))

    scores, exact, paths, prior_valid = {}, {}, {}, {}
    for name, candidate in candidates.items():
        path = Path(candidate["path"])
        if file_sha256(path) != candidate["sha256"]:
            raise ValueError(f"Checkpoint SHA256 differs for {name}")
        model, payload = load_model(path, device, datasets["moses"].spec)
        if name == "candidate" and payload["dataset_sha256"] != corpus.manifest["dataset_sha256"]:
            raise ValueError("Candidate checkpoint belongs to a different combined dataset")
        model.requires_grad_(False)
        before = state_digest(model.state_dict())
        folder = output / name
        folder.mkdir()
        seed = config["seed_by_split"][split]
        latents = torch.randn(config["prior_count"], model.config.latent, generator=torch.Generator().manual_seed(seed))
        if os.environ.get("DBMOL_LAUNCH_READY") and not Path(os.environ["DBMOL_LAUNCH_READY"]).exists():
            write_json(Path(os.environ["DBMOL_LAUNCH_READY"]), dict(status="models_and_inputs_ready",
                nonce=os.environ["DBMOL_LAUNCH_NONCE"], job_id=os.environ.get("LSB_JOBID")))
        prior, prior_rows = decode_panel(model, latents, folder / "prior.jsonl.gz",
            batch_size=config["batch_size"], categorical_seed=seed+3)
        prior_valid[name] = {method: [row["valid"] for row in rows] for method, rows in prior_rows.items()}
        scores[name] = dict(prior=prior, reconstruction={}, interpolation={})
        exact[name], paths[name] = {}, {}
        for source, dataset in datasets.items():
            print(json.dumps(dict(stage="evaluating", split=split, candidate=name, source=source)), flush=True)
            rec, rows, _, _ = reconstruct(model, dataset, range(len(dataset)), folder / f"{source}-reconstruction.jsonl.gz",
                batch_size=config["batch_size"], loss_config=LossConfig(**payload["loss_config"]), anchor_count=1)
            scores[name]["reconstruction"][source] = rec
            exact[name][source] = [r["exact"] for r in rows]
            pairs, (graph_a, graph_b, mapped_b) = pair_panels[source]
            panel = dict(batch_size=config["batch_size"], fractions=config["fractions"],
                         categorical_draws=config["categorical_draws"], categorical_seed=seed+4,
                         probability_snapshot_pairs=config["probability_snapshot_pairs"])
            z_a, z_b = (_encode(model, graph, config["batch_size"]) for graph in (graph_a, graph_b))
            result, raw = evaluate_arm(model, pairs, graph_a, graph_b, z_a, z_b, panel,
                                       folder / f"{source}-interpolation", mapped_b=mapped_b)
            scores[name]["interpolation"][source] = result
            paths[name][source] = {fraction: [r["valid"] for r in raw[(fraction, "argmax")]]
                                   for fraction in config["fractions"]}
        if state_digest(model.state_dict()) != before or file_sha256(path) != candidate["sha256"]:
            raise RuntimeError(f"Evaluation changed frozen checkpoint {name}")
        del model, payload
    result = dict(status="completed", split=split, candidates=candidates, scores=scores,
        counts={source: len(dataset) for source, dataset in datasets.items()},
        paired_prior={method: paired_difference(prior_valid["epoch45"][method], prior_valid["candidate"][method])
                      for method in prior_valid["epoch45"]},
        paired_exact={source: paired_difference(exact["epoch45"][source], exact["candidate"][source])
                      for source in datasets},
        paired_path={source: {str(t): paired_difference(paths["epoch45"][source][t], paths["candidate"][source][t])
                              for t in config["fractions"]} for source in datasets},
        seconds=time.monotonic()-started)
    write_json(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(json.loads(args.config.read_text(encoding="utf-8")), args.output,
             checkpoint=args.checkpoint, checkpoint_sha256=args.checkpoint_sha256,
             split=args.split, device=args.device)


if __name__ == "__main__":
    main()
