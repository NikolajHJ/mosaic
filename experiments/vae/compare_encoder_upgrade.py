"""Compare the production and expanded encoders at shared updates and at their endpoints."""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
from statistics import median

from rdkit import rdBase
import torch
from torch.nn.functional import normalize

from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256, write_json
from latentmol.training.generated_roundtrip import canonical_generated_parents
from latentmol.training.occupancy_monitor import _inspect


def read(path):
    return json.loads(path.read_bytes())


def read_arm(directory):
    fork = read(directory / "fork.json")
    summary = read(directory / "summary.json")
    latest = read(directory / "latest.json")
    if (latest != summary["latest"] or latest["updates"] != summary["updates"]
            or summary["updates"] - fork["resume"]["updates"] != summary["segment_updates"]
            or summary["segment_updates"] <= 0
            or Path(latest["checkpoint"]).resolve().parent != directory.resolve()):
        raise ValueError(f"Inconsistent endpoint in {directory}")
    return fork, summary


def checkpoint(receipt, protocol, spec=None):
    path = Path(receipt["checkpoint"])
    if path.name != f"checkpoint-{receipt['updates']:09d}.pt" or file_sha256(path) != receipt["sha256"]:
        raise ValueError(f"Checkpoint filename or SHA-256 differs at {path}")
    model, state = load_model(path, "cpu", spec)
    if (state.get("diagnostic_only") or state["config_sha256"] != protocol
            or any(state["continuation"][k] != receipt[k] for k in ("updates", "row", "real_points", "epoch"))):
        raise ValueError(f"Unmatched checkpoint protocol or cursor at {path}")
    return model


def quality_rows(directory, start, end):
    rows = {}
    with (directory / "quality-panel.jsonl").open(encoding="utf8") as f:
        for line in f:
            row = json.loads(line)
            if row["updates"] in rows or row["generated_roundtrip"]["leg_start_update"] != start:
                raise ValueError(f"Duplicated or mismatched panel in {directory}")
            rows[row["updates"]] = row
    if min(rows) != start or max(rows) != end:
        raise ValueError(f"Missing start or endpoint panel in {directory}")
    return rows


def counts(row):
    fixed = row["generated_roundtrip"]["fixed_at_leg_start"]
    return dict(heldout_exact=row["reconstruction_exact"], heldout_valid=row["reconstruction_valid"],
                heldout_n=row["validation_points"], prior_valid=row["prior_valid"], prior_n=row["prior_points"],
                prior_unique_valid=row["prior_unique_valid"], generated_parents=fixed["eligible_parents"],
                reencoded_exact=fixed["mean_exact"], reencoded_valid=fixed["mean_valid"],
                extra_bonds=fixed["extra_canonical_slot_edges"], missing_bonds=fixed["missing_canonical_slot_edges"])


def training_times(directory, updates, summary, start, margin_outer_weight=None):
    """Training update wall times recorded around batch updates; exclude monitoring and setup."""
    elapsed = 0.0
    times = []
    result = {start: dict(seconds=0.0, real_molecules=0)}
    previous = start
    size = summary["batch_size"]
    exposure = Counter(scheduled_batches=0, eval_parents_selected=0, exact_pairs=0,
                       batches_with_positive_loss=0, total_effective_loss=0.0)
    with (directory / "training.jsonl").open(encoding="utf8") as f:
        for line in f:
            row = json.loads(line)
            sec, update = row["seconds"], row["updates"]
            if update != previous + 1 or not math.isfinite(sec) or sec <= 0 or row["fresh_prior_points"] != size:
                raise ValueError(f"Invalid update timing, prior budget or cursor in {directory}")
            elapsed += sec
            times.append(sec)
            previous = update
            cycle = row.get("generated_roundtrip")
            if cycle is not None and margin_outer_weight is not None:
                m = cycle["excess_confidence"]
                effective = m["weighted_loss"] * margin_outer_weight
                if not math.isfinite(effective) or effective < 0 or not 0 <= m["exact_pairs"] <= m["selected"]:
                    raise ValueError("Invalid margin loss or eligibility counters")
                exposure.update(scheduled_batches=1, eval_parents_selected=m["selected"],
                                exact_pairs=m["exact_pairs"], batches_with_positive_loss=int(m["loss"] > 0),
                                total_effective_loss=effective)
            if update in updates:
                result[update] = dict(seconds=elapsed, real_molecules=(update-start)*size)
    if previous != summary["updates"] or not updates <= result.keys():
        raise ValueError(f"Missing updates in {directory}")
    exposure["mean_effective_loss_per_training_update"] = exposure["total_effective_loss"]/len(times)
    return result, median(times), dict(exposure) if margin_outer_weight is not None else None


def compare_curves(directories, summaries, start, candidate_outer_weight):
    arms = list(directories)
    qualities = {a: quality_rows(directories[a], start, summaries[a]["updates"]) for a in arms}
    updates = set.intersection(*(set(rows) for rows in qualities.values()))
    if len(updates) < 2:
        raise ValueError("No matched post-start panel to compare")
    times, medians, exposure = {}, {}, {}
    for arm in arms:
        weight = candidate_outer_weight if arm == "candidate" else None
        times[arm], medians[arm], exposure[arm] = training_times(directories[arm], updates, summaries[arm], start, weight)
    primary = ("heldout_exact", "prior_valid", "reencoded_exact", "reencoded_valid")
    baseline = {a: counts(qualities[a][start]) for a in arms}
    rows = []
    original = qualities["control"][start]
    for u in sorted(updates):
        point = dict(updates=u, updates_since_fork=u-start)
        for arm in arms:
            panel = qualities[arm][u]
            if (any(panel[k] != original[k] for k in ("seed", "dataset_sha256", "validation_rows",
                    "validation_points", "prior_points"))
                    or counts(panel)["generated_parents"] != baseline["control"]["generated_parents"]
                    or panel["generated_roundtrip"]["fixed_at_leg_start"]["prior_attempts"]
                    != original["generated_roundtrip"]["fixed_at_leg_start"]["prior_attempts"]):
                raise ValueError("Fixed validation, prior or generated-parent cohort changed")
            metric = counts(panel)
            t = times[arm][u]
            sec = t["seconds"]
            point[arm] = dict(counts=metric, **t, real_per_update_second=t["real_molecules"]/sec if sec else None,
                net_change={k: metric[k] - baseline[arm][k] for k in primary},
                net_change_per_update_hour={k: (metric[k] - baseline[arm][k])*3600/sec if sec else None for k in primary})
        rows.append(point)
    if baseline["candidate"] != baseline["control"]:
        # Both runs use the same rows/seeds and zero residual initialization.
        # Keep any startup numerical difference visible in the descriptive curves.
        rows[0]["warning"] = "Baseline count differences; gains are relative to each arm's own start"
    return dict(matched=rows, median_training_update_seconds=medians, candidate_margin_exposure=exposure["candidate"],
                seed=original["seed"], endpoint_counts={a: counts(qualities[a][summaries[a]["updates"]]) for a in arms})


def decoded_graphs(model, vectors):
    decoded = model.decode(vectors)
    if not all(t.isfinite().all() for t in (decoded.atoms, decoded.charges, decoded.bonds)):
        raise FloatingPointError("Nonfinite neighborhood graph logits")
    return decoded.harden(model.spec)


def score_edits(rows, arm):
    """Unrecovered parents have zero successful edits and remain in the denominator."""
    recovered = edited = unique_per_parent = 0
    distinct = set()
    for row in rows:
        obs = row[arm]
        if obs["center"] != row["parent"]:
            continue
        recovered += 1
        edits = {s for s in obs["neighbors"] if s is not None and s != row["parent"]}
        edited += bool(edits)
        unique_per_parent += len(edits)
        distinct.update(edits)
    count = len(rows)
    return dict(parents=count, exact_centers=recovered, parents_with_valid_edit=edited,
                fraction_parents_with_edit=edited/count if count else None,
                distinct_edits_summed_by_parent=unique_per_parent,
                distinct_edits_per_attempted_parent=unique_per_parent/count if count else None,
                distinct_edit_identities_overall=len(distinct))


@torch.inference_mode()
def neighborhood(models, seed, output, *, draws=128, batch=8):
    """Use the predecessor to generate parents once. Evaluate the same four directions for each arm."""
    base = models["predecessor"]
    priors = torch.randn(draws, base.config.latent, generator=torch.Generator().manual_seed(seed))
    directions = normalize(torch.randn(draws, 4, base.config.latent,
                           generator=torch.Generator().manual_seed(seed+1)), dim=-1)
    rows, exclusions = [], Counter()
    for offset in range(0, draws, batch):
        raw = decoded_graphs(base, priors[offset:offset+batch])
        parents, info = canonical_generated_parents(raw, base.spec, include_smiles=True)
        exclusions.update(info["rejections"])
        if parents is None:
            continue
        ids = [offset+i for i in info["rows"]]
        group = [dict(prior_row=i, parent=s) for i,s in zip(ids, info["smiles"], strict=True)]
        for arm, model in models.items():
            mu, _ = model.encoder(parents, from_canonical_cache=True)
            center = _inspect(decoded_graphs(model, mu), model.spec)
            for j, row in enumerate(group):
                smiles = center[j].smiles
                # A perturbation reached from an incorrect center is not an edit of the intended parent.
                seen = []
                if smiles == row["parent"]:
                    queries = mu[j:j+1] + directions[row["prior_row"]]
                    seen = [v.smiles for v in _inspect(decoded_graphs(model, queries), model.spec)]
                row[arm] = dict(center=smiles, neighbors=seen)
        rows.extend(group)
    if len(rows) + sum(exclusions.values()) != draws:
        raise RuntimeError("Generated parent screening lost rows")
    common = [r for r in rows if all(r[a]["center"] == r["parent"] for a in ("control", "candidate"))]
    stats = {a: score_edits(rows, a) for a in models}
    for arm in stats:
        stats[arm]["parents_with_edit_per_original_draw"] = stats[arm]["parents_with_valid_edit"]/draws
    with (output / "editability-parents.jsonl").open("x", encoding="utf8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    return dict(seed=seed, draws=draws, eligible=len(rows), rejection_reasons=dict(exclusions),
                unique_eligible_identities=len({r["parent"] for r in rows}), radius_l2=1.0, directions_per_parent=4,
                all_eligible_parents=stats, on_common_exact_centers={a: score_edits(common,a) for a in models})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("control", "candidate", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    dirs = {"control": args.control.resolve(), "candidate": args.candidate.resolve()}
    out = args.output.resolve()
    if (dirs["control"] == dirs["candidate"]
            or any(out.is_relative_to(p) or p.is_relative_to(out) for p in dirs.values())):
        raise ValueError("Require separate endpoint and report directories")
    pairs = {arm: read_arm(d) for arm,d in dirs.items()}
    control, candidate = (pairs[a][0] for a in ("control", "candidate"))
    predecessor = control["resume"]
    if (candidate["resume"] != predecessor or candidate["parent"] != control["parent"]
            or Path(candidate["production_control"]).resolve() != dirs["control"]
            or candidate["source_config_sha256"] != control["config_sha256"]):
        raise ValueError("The two arms do not share the designated source checkpoint and production recipe")
    source_protocol = control["config_sha256"]
    summaries = {a: p[1] for a,p in pairs.items()}
    if any(summaries["control"][k] != summaries["candidate"][k]
           for k in ("batch_size", "microbatch_size", "training_molecules", "dataset_sha256")):
        raise ValueError("Training cohorts or batch budgets differ")
    torch.set_num_threads(max(1, min(4, int(os.environ.get("LSB_DJOB_NUMPROC", "2")))))
    source = checkpoint(predecessor, source_protocol)
    models = dict(predecessor=source)
    for arm, (fork, summary) in pairs.items():
        models[arm] = checkpoint(summary["latest"], fork["config_sha256"], source.spec)
    start = predecessor["updates"]
    curve = compare_curves(dirs, summaries, start, candidate["generated_roundtrip"]["weight"])
    out.mkdir(parents=True, exist_ok=False)
    if marker := os.environ.get("DBMOL_LAUNCH_READY"):
        write_json(Path(marker), dict(status="models_and_inputs_ready", job_id=os.environ["LSB_JOBID"],
                                     nonce=os.environ["DBMOL_LAUNCH_NONCE"]))
    with rdBase.BlockLogs():
        edits = neighborhood(models, curve["seed"] + 202, out)
    endpoints = {a: dict(checkpoint=s["latest"], segment_updates=s["segment_updates"],
                    segment_seconds=s["elapsed_seconds"], stop_reason=s["stop_reason"]) for a,s in summaries.items()}
    result = dict(status="completed", parent=predecessor, endpoints=endpoints,
        endpoint_updates_matched=summaries["control"]["updates"] == summaries["candidate"]["updates"],
        curves=curve, geometric_edits=edits,
        interpretation="Geometric valid edits at terminal checkpoints using four shared random directions. "
                       "Endpoint training budgets can differ. Binding scores were not measured.")
    write_json(out / "comparison.json", result)


if __name__ == "__main__":
    main()
