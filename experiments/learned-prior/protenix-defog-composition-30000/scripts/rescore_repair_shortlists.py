"""Rescore fixed candidate shortlists from the matched BRD4 repair comparison."""
import argparse
import csv
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from mosaic.small_molecule.core.state import RelaxedGraph, validate_graph
from stage3_campaign import ROOT, _finite, labels_identity, make_evaluators, utc, write_json
from stage3_native import file_sha, open_native

FRESH_SEEDS = (7001, 7002, 7003, 7004, 7005)
PROJECTED = "20261118-defog-vs-structural-repair-01"
UNPROJECTED = "20261118-structural-repair-no-projection-01"
CONFIRMATION = "20261119-defog-vs-repair-seeds71-101"


def matched_runs(campaign):
    for seed in (17, 23, 42, 71, 101):
        groups = [(PROJECTED, "defog-w0p5"), (UNPROJECTED, "repair-w1-no-projection")]
        if seed in (17, 23, 42):
            groups += [(PROJECTED, "repair-w1"), (campaign.name, "repair-w10-no-projection")]
        else:
            groups = [(CONFIRMATION, condition) for _, condition in groups]
        for group, condition in groups:
            run_id = f"{condition}__fragmented__seed-{seed:05}"
            yield seed, condition, campaign.parent / group / "campaign" / "runs" / run_id


def collect(campaign, assets):
    runs, candidates, graphs = [], [], {}
    matched_starts = {}
    for seed, condition, path in matched_runs(campaign):
        def read(name):
            return json.loads((path / name).read_text())
        status, config, archive = (read(n) for n in ("status.json", "config.json", "top_valid.json"))
        if (status["status"] != "complete" or not status["frozen_checks_passed"]
                or status["updates_completed"] != 500 or config["seed"] != seed
                or config["condition_id"] != condition or config["learning_rate"] != 60):
            raise ValueError(f"Incomplete or incompatible input: {path}")
        for field in ("model_digest", "context_digest"):
            if config["scorer_identity"][field] != assets[field]:
                raise ValueError(f"Different {field} in {path}")
        with np.load(path / "initial.npz", allow_pickle=False) as initial:
            current = tuple(initial[k].tobytes() for k in ("X", "E_upper", "model_key"))
            if matched_starts.setdefault(seed, current) != current:
                raise ValueError(f"Unmatched initial graph or model key: {path}")
        shortlist = archive["molecules"][:5]
        if len(shortlist) != min(5, archive["unique_scored_valid_molecules"]):
            raise ValueError(f"Shortlist count mismatch in {path}")
        ids = [r["smiles"] for r in shortlist]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Repeated identity in {path}")
        runs.append({"seed": seed, "condition": condition, "path": str(path),
                     "original_key": config["model_key"], "key_impl": config["model_key_impl"],
                     "archive_sha256": file_sha(path / "top_valid.npz"), "eligible": len(shortlist)})
        with np.load(path / "top_valid.npz", allow_pickle=False) as tensors:
            if tensors["model_key"].tolist() != config["model_key"]:
                raise ValueError(f"Original key mismatch in {path}")
            for i, row in enumerate(shortlist):
                if not (row["valid"] and row["connected"] and row["sanitized"]
                        and row["rank"] == int(tensors["ranks"][i]) == i + 1
                        and row["artifact_index"] == i
                        and row["smiles"] == tensors["smiles"][i]
                        and row["hard_score"] == tensors["hard_scores"][i]
                        and row["hard_id"] == tensors["hard_ids"][i]):
                    raise ValueError(f"Rank or graph mismatch in {path}")
                graph = RelaxedGraph(jnp.asarray(tensors["hard_X"][i]), jnp.asarray(tensors["hard_E_upper"][i]))
                validate_graph(graph)
                if labels_identity(graph) != row["hard_id"] or any(
                    not np.all((np.asarray(head) == 0) | (np.asarray(head) == 1))
                    for head in (graph.X, graph.E_upper)
                ):
                    raise ValueError(f"Expected original ordered one-hot graph in {path}")
                graphs[row["hard_id"]] = graph
                candidates.append({"seed": seed, "condition": condition, "rank": i + 1,
                                   "hard_id": row["hard_id"], "smiles": row["smiles"],
                                   "original_score": row["hard_score"], "update": row["source_update"]})
    return runs, candidates, graphs


def rescore(campaign, assets_path, output):
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Rescoring requires an allocated HPC GPU")
    assets = json.loads(assets_path.read_text())
    output.mkdir(exist_ok=False)
    runs, candidates, graphs = collect(campaign, assets)
    fresh = {s: jax.random.key(s) for s in FRESH_SEEDS}
    bits = {s: np.asarray(jax.random.key_data(key)).tolist() for s, key in fresh.items()}
    if any(run["original_key"] in bits.values() for run in runs):
        raise ValueError("A new key overlaps an optimization key")
    selection = {"frozen_utc": utc(), "fresh_keys": bits, "runs": runs, "candidates": candidates,
                 "primary": "Original rank-1 candidate, arithmetic mean of five fresh-key hard scores",
                 "secondary": "All shortlisted candidates equally weighted within run, then across paired seeds"}
    write_json(output / "selection.json", selection)
    native, identity, finish = open_native(assets, ROOT)
    forward = make_evaluators(native)[1]
    original_keys = {r["seed"]: jax.random.wrap_key_data(jnp.asarray(r["original_key"], dtype=jnp.uint32),
                                                        impl=r["key_impl"]) for r in runs}

    def score(graph, key):
        value, aux = forward(graph, key)
        _finite(value, aux)
        return -float(value)

    fields = ["seed", "condition", "rank", "hard_id", "smiles", "original_score", "update",
              "original_key_replay", *[f"key_{k}" for k in fresh], "fresh_mean"]
    with (output / "scores.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for c in candidates:
            graph = graphs[c["hard_id"]]
            replay = score(graph, original_keys[c["seed"]])
            estimates = {f"key_{s}": score(graph, key) for s, key in fresh.items()}
            writer.writerow(dict(c, original_key_replay=replay, **estimates,
                                 fresh_mean=float(np.mean(list(estimates.values())))))
            stream.flush()
    write_json(output / "completion.json", {"completed_utc": utc(), "job": os.environ["LSB_JOBID"],
               "candidates": len(candidates), "fresh_evaluations": len(candidates) * len(fresh),
               "original_key_replays": len(candidates), "selection_sha256": file_sha(output / "selection.json"),
               "scores_sha256": file_sha(output / "scores.csv"), "scorer": identity,
               "frozen": finish(), "device": jax.devices()[0].device_kind})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("assets", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rescore(args.campaign, args.assets, args.output)
