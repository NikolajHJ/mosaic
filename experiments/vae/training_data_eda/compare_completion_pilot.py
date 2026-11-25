"""Compare the two saved motif-completion pilot arms on identical held-out draws."""

import argparse
import gzip
import json
from pathlib import Path
from statistics import median


def draw_panels(path):
    panels = {"prior": [], "reconstruction": []}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            group = panels[row["panel"]]
            if row["index"] != len(group):
                raise ValueError(f"Non-sequential draws in {path}")
            group.append(row)
    return panels


def paired_results(a, b, labels):
    pairs = list(zip(a, b, strict=True))
    if any(x.get("target") != y.get("target") for x, y in pairs):
        raise ValueError("Reconstruction endpoints refer to different inputs")
    result = {}
    for name in labels:
        gained = sum(y[name] and not x[name] for x, y in pairs)
        lost = sum(x[name] and not y[name] for x, y in pairs)
        result[name] = {
            "control": sum(x[name] for x in a), "treatment": sum(y[name] for y in b),
            "gained": gained, "lost": lost, "net": gained - lost}
        if name in ("aromatic", "amide"):
            # Test the direct mechanism on the actual output. Atom identity, charge,
            # occupied slots and adjacency must all agree; only bond order can differ.
            matched_topology = [(x, y) for x, y in pairs if x["nodes"] == y["nodes"]
                                and [e[:2] for e in x["edges"]] == [e[:2] for e in y["edges"]]]
            result[name]["gained_without_topology_or_atom_change"] = sum(
                y[name] and not x[name] for x, y in matched_topology)
            result[name]["lost_without_topology_or_atom_change"] = sum(
                x[name] and not y[name] for x, y in matched_topology)
    return {
        "attempts": len(pairs), "chemistry": result,
        "different_hard_graphs": sum(x["nodes"] != y["nodes"] or x["edges"] != y["edges"] for x, y in pairs),
        "distinct_valid_scaffolds": [len({r["scaffold"] for r in rows if r["valid"] and r["scaffold"]})
                                     for rows in (a, b)],
        "distinct_valid_smiles": [len({r["smiles"] for r in rows if r["valid"]}) for rows in (a, b)],
        # An aromatic class on one raw edge does not guarantee a perceived aromatic ring.
        # In the sealed training graph schema AROMATIC is bond index 4.
        "raw_aromatic_bond_cases": [
            {"any_explicit_aromatic_bond": sum(any(e[2] == 4 for e in r["edges"]) for r in rows),
             "invalid_with_explicit_aromatic_bond": sum(
                 not r["valid"] and any(e[2] == 4 for e in r["edges"]) for r in rows),
             "perceived_ring_without_explicit_aromatic_bond": sum(
                 r["aromatic"] and all(e[2] != 4 for e in r["edges"]) for r in rows)}
            for rows in (a, b)],
    }


def compare(root, control_dir=None):
    summaries = {}
    records = {}
    fresh = {}
    sampled = {}
    timing = {}
    for arm in ("control", "treatment"):
        folder = control_dir if arm == "control" and control_dir is not None else root / arm
        summaries[arm] = json.loads((folder / "summary.json").read_bytes())
        records[arm] = draw_panels(folder / f"{arm}.jsonl.gz")
        fresh[arm] = draw_panels(folder / f"{arm}-fresh.jsonl.gz")["prior"]
        sampled[arm] = draw_panels(folder / f"{arm}-sampled.jsonl.gz")["prior"]
        log = [json.loads(line) for line in (folder / f"{arm}-training.jsonl").read_text().splitlines()]
        stats = [row["functional_groups"] for row in log if "functional_groups" in row]
        timing[arm] = {
            "updates": len(log), "scheduled": len(stats),
            "seconds": summaries[arm]["arms"][arm]["train_seconds"],
            "aromatic_completion_rows": sum(row.get("completion_aromatic_rows", 0) for row in stats),
            "amide_completion_rows": sum(row.get("completion_amide_rows", 0) for row in stats),
            "completion_draws": sum(row["draws"] for row in stats),
            "positive_completion_batches": sum(row.get("completion_ce", 0) > 0 for row in stats),
            "median_completion_ce": median(row.get("completion_ce", 0) for row in stats)}
        pattern_keys = sorted({k for row in stats for k in row if k.startswith("pattern_")})
        if pattern_keys:
            pattern_losses = [row["completion_pattern_loss"] for row in stats]
            timing[arm]["positive_pattern_batches"] = sum(loss > 0 for loss in pattern_losses)
            timing[arm]["median_pattern_loss"] = median(pattern_losses)
            timing[arm]["pattern_components"] = {
                k: {"mean_per_scheduled_batch": sum(row.get(k, 0) for row in stats) / len(stats),
                    "sum_across_scheduled_batches": sum(row.get(k, 0) for row in stats)}
                for k in pattern_keys}
        neighbor_keys = sorted({k for row in stats for k in row if k.startswith("neighbor_")})
        if neighbor_keys:
            timing[arm]["neighbor_components"] = {
                k: {"mean_per_scheduled_batch": sum(row.get(k, 0) for row in stats) / len(stats),
                    "sum_across_scheduled_batches": sum(row.get(k, 0) for row in stats)}
                for k in neighbor_keys}
    c, t = summaries["control"], summaries["treatment"]
    keys = ("checkpoint", "reference", "train_panel", "heldout_panel", "train_indices_sha256", "seed",
            "updates", "real_per_update", "priors_per_update", "microbatch", "fresh_evaluation_seed",
            "categorical_seed")
    same_platform_pair = (c.get("new_platform_control") and
        all(c[k] == t[k] for k in ("device", "torch", "rdkit")) and
        all(c["arms"]["control"][k] == t["arms"]["treatment"][k] for k in ("rng_start", "rng_end")))
    if (any(c[k] != t[k] for k in keys) or not t["matched_control_rng"]
            or not (c["matched_control_rng"] or same_platform_pair)):
        raise ValueError("The endpoint comparisons do not share inputs or training random streams")
    c_cfg, t_cfg = c["config"], t["config"]
    neighbor = t_cfg["functional_groups"].get("neighbor_weight")
    extra = {"completion_weight": 1.} if neighbor is None else {"neighbor_weight": neighbor}
    expected_cfg = dict(c_cfg, functional_groups=dict(c_cfg["functional_groups"], **extra))
    if t_cfg["functional_groups"].get("ring_encoding") == "kekule":
        expected_cfg["functional_groups"]["ring_encoding"] = "kekule"
    if t_cfg["functional_groups"].get("completion_mode") == "pattern":
        expected_cfg["functional_groups"].update(completion_mode="pattern", amide_relative_weight=.10)
    if t_cfg != expected_cfg:
        raise ValueError("Unexpected objective or recipe change between arms")
    labels = ("valid", "aromatic", "amide", "ring3", "ring8plus", "ring12plus")
    return {
        "historical_control_weights_reproduced": c["reproduced_control_weights"],
        "source_checkpoint_sha256": c["checkpoint"], "timing": timing,
        "primary_prior": paired_results(records["control"]["prior"], records["treatment"]["prior"], labels),
        "fresh_prior": paired_results(fresh["control"], fresh["treatment"], labels),
        "fresh_categorical": paired_results(sampled["control"], sampled["treatment"], labels),
        "reconstruction": paired_results(records["control"]["reconstruction"],
            records["treatment"]["reconstruction"], (*labels, "exact")),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pilot", type=Path)
    p.add_argument("--control-dir", type=Path, help="Reuse saved outputs from a matched control")
    args = p.parse_args()
    path = args.pilot
    report = compare(path, args.control_dir)
    with (path / "paired-summary.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, indent=2))
