"""Sequential four-round control and weak functional-group pilot using the epoch-48 trainer."""

import argparse
from collections import Counter
from functools import lru_cache
import gzip
import json
import math
import os
from pathlib import Path
from statistics import mean, median
import subprocess
import sys

import numpy as np
import torch
from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

from experiments.vae.training_data_eda.chemistry_profile import GROUP_DEFINITIONS
from latentmol.chemistry import inspect_graph
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256, write_json
from latentmol.sampling import sample_hard_graph


AMIDE = Chem.MolFromSmarts(GROUP_DEFINITIONS["amide"][1])
ENDPOINTS = ("valid", "aromatic", "amide")


@lru_cache(maxsize=16384)
def chemistry(smiles):
    if smiles is None:
        return {"valid": False, "aromatic": False, "amide": False, "scaffold": None}
    mol = Chem.MolFromSmiles(smiles)
    return {"valid": True, "aromatic": rdMolDescriptors.CalcNumAromaticRings(mol) > 0,
            "amide": mol.HasSubstructMatch(AMIDE), "scaffold": MurckoScaffoldSmiles(mol=mol)}


def read(path):
    return json.loads(path.read_bytes())


def describe(records):
    result = {"attempts": len(records)}
    for label in ENDPOINTS:
        selected = [r for r in records if r[label]]
        result[label] = {"count": len(selected), "unique_smiles": len({r["smiles"] for r in selected}),
                         "distinct_nonempty_scaffolds": len({r["scaffold"] for r in selected if r["scaffold"]})}
    result["failures"] = dict(Counter(r["failure"] for r in records if not r["valid"]))
    return result


def training_activity(log, policy, arm):
    """Catch an unwired treatment while retaining truthful zero-signal batches."""
    config = policy["functional_groups"]
    rows = []
    for step, event in enumerate(log, start=1):
        scheduled = arm == "weak" and step % config["every_updates"] == 0
        if scheduled != ("functional_groups" in event):
            raise RuntimeError(f"Unexpected motif measurement schedule in {arm} at step {step}")
        if scheduled:
            obs = event["functional_groups"]
            if obs["draws"] != config["draws"] or not all(math.isfinite(x) for x in obs.values()):
                raise RuntimeError(f"Invalid motif activity in {arm} at step {step}")
            rows.append(obs)
    totals = {key: sum(r[key] for r in rows) for key in
              ("draws", "valid", "aromatic_hits", "amide_hits", "aromatic_informative", "amide_informative")}
    return {"scheduled_updates": len(rows), **totals,
            "mean_weighted_loss_when_scheduled": mean(r["weighted_loss"] for r in rows) if rows else None}


@torch.inference_mode()
def evaluate(checkpoint, out, policy, device="cuda"):
    """One raw hard draw and one argmax per identical latent for each arm; keep every attempted graph."""
    model, _ = load_model(checkpoint, device)
    seed = policy["evaluation_seed"]
    codes = torch.randn(policy["evaluation_draws"], model.config.latent, generator=torch.Generator().manual_seed(seed))
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    records = {"argmax": [], "sample": []}
    with gzip.open(out, "xt", encoding="utf-8") as stream, rdBase.BlockLogs():
        for batch in codes.split(policy["evaluation_batch"]):
            pred = model.decode(batch.to(device))
            for mode, hard in (("argmax", pred.harden(model.spec)),
                               ("sample", sample_hard_graph(pred, model.spec, generator=generator))):
                cpu = hard.to("cpu")
                for a, c, b in zip(cpu.atoms.numpy(), cpu.charges.numpy(), cpu.bonds.numpy(), strict=True):
                    check = inspect_graph(a, c, b, model.spec)
                    row = {"draw": len(records[mode]), "smiles": check.smiles,
                           "failure": check.failure_reason, **chemistry(check.smiles)}
                    records[mode].append(row)
                    active = np.flatnonzero(a)
                    i, j = np.nonzero(np.triu(b, 1))
                    stream.write(json.dumps({"mode": mode, **row,
                        "nodes": np.column_stack([active, a[active], c[active]]).tolist(),
                        "edges": np.column_stack([i, j, b[i, j]]).tolist()}) + "\n")
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True, help="Sealed common epoch-48 configuration")
    p.add_argument("--pilot", type=Path, required=True)
    p.add_argument("--parent-receipt", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not os.environ.get("LSB_JOBID", "").isdigit():
        raise RuntimeError("Use an allocated GPU")
    shared, policy = read(args.config), read(args.pilot)
    if file_sha256(args.config) != policy["base_config_sha256"]:
        raise ValueError("Use the sealed epoch-48 V100 pair recipe, including the original parent contract")
    if "functional_groups" in shared:
        raise ValueError("Expected a motif-free reference configuration")
    shared.update(leg_updates=policy["updates_per_arm"], wall_seconds=policy["arm_wall_seconds"])
    torch.set_num_threads(shared["execution"]["torch_threads"])
    args.output.mkdir(parents=True, exist_ok=False)
    parent = read(args.parent_receipt)
    result = {"parent": parent, "seed": policy["evaluation_seed"], "draws": policy["evaluation_draws"],
              "pilot": policy, "arms": {}}
    # LSF grants one process the CUDA context at a time. The coordinator stays CPU-only
    # while each standard trainer owns the device. Both processes exit before scoring.
    for arm in ("control", "weak"):
        config = dict(shared)
        if arm == "weak":
            config["functional_groups"] = policy["functional_groups"]
        cfg = args.output / f"{arm}.json"
        write_json(cfg, config)
        directory = args.output / arm
        subprocess.run([sys.executable, "-m", "latentmol.training.projected_continuation",
                        "--config", str(cfg), "--parent-receipt", str(args.parent_receipt),
                        "--extra-rounds", "0", "--output", str(directory)], check=True)
        summary = read(directory / "summary.json")
        if summary["segment_updates"] != policy["updates_per_arm"]:
            raise RuntimeError(f"{arm} ended before the matched update budget: {summary['stop_reason']}")
        latest = summary["latest"]
        path = Path(latest["checkpoint"])
        if path.parent.resolve() != directory.resolve() or file_sha256(path) != latest["sha256"]:
            raise ValueError("Endpoint checkpoint path or digest differs from the completed arm")
        log = [json.loads(line) for line in (directory / "training.jsonl").read_text().splitlines()]
        fixed = [json.loads(line) for line in (directory / "quality-panel.jsonl").read_text().splitlines()]
        if len(log) != policy["updates_per_arm"] or fixed[-1]["updates"] != latest["updates"]:
            raise RuntimeError("Incomplete timing log or fixed reconstruction panel")
        activity = training_activity(log, policy, arm)
        # Ignore the first few cuBLAS/allocator warm-up steps in each arm's update-time estimate.
        seconds = [s["seconds"] for s in log[4:]]
        result["arms"][arm] = {
            "latest": latest, "segment_seconds": summary["elapsed_seconds"],
            "median_update_seconds": median(seconds), "mean_update_seconds": mean(seconds),
            "timed_updates": len(seconds), "functional_group_activity": activity, "fixed_panel": [fixed[0], fixed[-1]]}
        write_json(args.output / "comparison.json", result)
    panels = {}
    result["gpu"] = torch.cuda.get_device_name()
    for arm, measurements in result["arms"].items():
        path = Path(measurements["latest"]["checkpoint"])
        panels[arm] = evaluate(path, args.output / arm / "motif-draws.jsonl.gz", policy)
        measurements["outcomes"] = {mode: describe(rows) for mode, rows in panels[arm].items()}
        write_json(args.output / "comparison.json", result)
    c, w = (result["arms"][s] for s in ("control", "weak"))
    start_c, start_w = c["fixed_panel"][0], w["fixed_panel"][0]
    for key in ("updates", "seed", "dataset_sha256", "prior_points", "validation_rows"):
        if start_c[key] != start_w[key]:
            raise ValueError("Different starting fixed panels in matched arms")
    result["identical_initial_fixed_panel_results"] = start_c == start_w
    result["mean_update_overhead_fraction"] = w["mean_update_seconds"] / c["mean_update_seconds"] - 1
    # Paired gains and losses expose changes hidden in equal marginal rates.
    result["paired_changes"] = {}
    for mode, controls in panels["control"].items():
        pairs = list(zip(controls, panels["weak"][mode], strict=True))
        result["paired_changes"][mode] = {
            key: {"gained": sum(not a[key] and b[key] for a,b in pairs),
                  "lost": sum(a[key] and not b[key] for a,b in pairs)} for key in ENDPOINTS}
    write_json(args.output / "comparison.json", result)
    print(json.dumps({"path": str(args.output / "comparison.json"), "arms": list(result["arms"]),
                      "mean_update_overhead_fraction": result["mean_update_overhead_fraction"]}), flush=True)


if __name__ == "__main__":
    main()
