"""Summarize saved local pilot graphs/metrics, without rerunning the VAE."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
from latentmol.chemistry_epoch45_generation import structural_diversity


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pairs(left, right):
    assert len(left) == len(right)
    out = {}
    for field in ("raw_chemical_valid", "strict_teacher_valid"):
        a, b = [np.asarray([r[field] is True for r in rows]) for rows in (left, right)]
        out[field] = dict(valid_to_invalid=int((a & ~b).sum()), invalid_to_valid=int((~a & b).sum()),
                          unchanged_valid=int((a & b).sum()), unchanged_invalid=int((~a & ~b).sum()))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = read(args.run / "result.json")
    if result["status"] != "completed_local_comparison_not_production_adoption":
        raise ValueError("A complete matched comparison is required")
    source = read(args.run / "manifest.json")
    inputs = read(args.inputs / "inputs.json")
    sulfur_index = inputs["manifest"]["spec"]["atomic_numbers"][1:].index(16)
    composition = inputs["files"]["composition"]["value"]["elements_distribution"]
    target_sulfur = composition[sulfur_index] / sum(composition)
    analysis = dict(scope="Bounded local directional pilot; not production adoption, a full epoch or full-test evaluation",
                    source_manifest_sha256=hashlib.sha256((args.run / "manifest.json").read_bytes()).hexdigest(),
                    matched_streams=result["matched_real_rows_and_noise_streams"],
                    training_population_sulfur_atom_fraction=target_sulfur, arms={})
    for name, arm in result["arms"].items():
        folder = args.run / name
        updates = [json.loads(line) for line in (folder / "updates.jsonl").read_text().splitlines()]
        initial, final = arm["initial"], arm["final"]
        n = final["update"]
        first = read(folder / "eval-0000-fixed-argmax-classifications.json")
        last = read(folder / (f"eval-{n:04d}-fixed-argmax-classifications.json"))
        with np.load(folder / "eval-0000-fixed-argmax.npz") as a, np.load(folder / (f"eval-{n:04d}-fixed-argmax.npz")) as b:
            changed = np.zeros(len(first), dtype=bool)
            for key in ("atoms", "charges", "bonds"):
                changed |= (a[key] != b[key]).reshape(len(first), -1).any(-1)
        diversity = structural_diversity([dict(valid=r["raw_chemical_valid"] is True, smiles=r["smiles"]) for r in last])
        a = dict(completed_updates=n, initial=initial, final=final,
                 paired_fixed_argmax_transitions=pairs(first, last), changed_hard_graphs=int(changed.sum()),
                 endpoint_fixed_diversity=diversity, mean_update_seconds=arm["mean_update_seconds"],
                 median_update_seconds_after_first_two=(statistics.median(r["update_seconds"] for r in updates[2:]) if len(updates)>2 else None),
                 mean_search_seconds=statistics.mean(r["metrics"]["search_seconds"] for r in updates),
                 peak_reserved_GiB=arm["peak_reserved_bytes"] / 2**30,
                 all_recorded_losses_finite=all(np.isfinite(r["metrics"]["total"]) for r in updates),
                 frozen_anchors_unchanged=arm["frozen_anchors_unchanged"])
        a["endpoint_sulfur_deviation_from_population_prior_pp"] = 100 * (final["panels"][0]["expected_sulfur_fraction"] - target_sulfur)
        a["fixed_panel_curve"] = [dict(update=v["update"], raw_valid=v["panels"][0]["argmax"]["raw_valid"],
            strict_valid=v["panels"][0]["argmax"]["strict_valid"],
            expected_sulfur_fraction=v["panels"][0]["expected_sulfur_fraction"],
            exact_reconstruction=v["reconstruction"]["exact_reconstruction_count"])
            for v in [read(path) for path in sorted(folder.glob("eval-[0-9][0-9][0-9][0-9].json"))]]
        if name == "neighbor":
            added = [r["metrics"]["additive_guidance"] for r in updates]
            a["added_loss"] = dict(first=added[0]["weighted_loss"], last=added[-1]["weighted_loss"],
                first_ten_mean=statistics.mean(r["weighted_loss"] for r in added[:10]),
                last_ten_mean=statistics.mean(r["weighted_loss"] for r in added[-10:]),
                maximum_unassessed_centers=(max(sum(r["unassessed_centers"].values()) for r in added)
                    if all("unassessed_centers" in r for r in added) else None))
        analysis["arms"][name] = a
    left = read(args.run / "control" / f'eval-{analysis["arms"]["control"]["completed_updates"]:04d}-fixed-argmax-classifications.json')
    right = read(args.run / "neighbor" / f'eval-{analysis["arms"]["neighbor"]["completed_updates"]:04d}-fixed-argmax-classifications.json')
    analysis["endpoint_control_to_neighbor_transitions"] = pairs(left, right)
    (args.output / "analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False), encoding="utf-8")
    def value(arm, key):
        a = analysis["arms"][arm]
        e = a["final"]
        panel = e["panels"][0]
        if key == "valid":return f'{panel["argmax"]["raw_valid"]}/{panel["argmax"]["attempts"]} ({100*panel["argmax"]["raw_valid"]/panel["argmax"]["attempts"]:.2f}%)'
        if key == "strict":return str(panel["argmax"]["strict_valid"])
        if key == "recon":return f'{e["reconstruction"]["exact_reconstruction_count"]}/{e["reconstruction"]["chemistry_samples"]}'
        if key == "valid_recon":return str(e["reconstruction"]["valid_chemistry_count"])
        if key == "nll":return f'{e["reconstruction"]["reconstruction"]:.8f}'
        if key == "sulfur":return f'{100*panel["expected_sulfur_fraction"]:.3f}%'
        if key == "hard_sulfur":return f'{100*panel["argmax"]["hard_sulfur_fraction"]:.3f}%'
        if key == "disconnected":return str(panel["argmax"]["failure_reasons"].get("disconnected", 0))
        if key == "size":return f'{panel["argmax"]["mean_hard_size"]:.2f}'
        if key == "unique":return str(panel["argmax"]["unique_valid"])
        if key == "diversity":return f'{a["endpoint_fixed_diversity"]["internal_diversity"]:.4f}'
        if key == "seconds":return (f'{a["median_update_seconds_after_first_two"]:.2f}' if a["median_update_seconds_after_first_two"] is not None else 'Not measured')
        if key == "vram":return f'{a["peak_reserved_GiB"]:.2f}'
        raise KeyError(key)
    baseline = analysis["arms"]["control"]["initial"]
    budget=analysis["arms"]["control"]["completed_updates"]
    lines = ["# Local neighbor-aware valence-loss pilot", "", f"Matched checkpoint42 comparison,{budget} local updates per arm. Existing HPC continuation unchanged.", "",
             "The new term uses weight0.1 with real/prior weights0.25/0.75 and globally normalized occupied-center means. Both arms retain inherited AdamW moments, quarter-displacement steps, repair/reference guidance and1:2 real/prior sampling. Local real batch16 and microbatch8 differ from production; this is not an exact HPC continuation.", "",
             f'Initial fixed-prior raw validity: {baseline["panels"][0]["argmax"]["raw_valid"]}/{baseline["panels"][0]["argmax"]["attempts"]}; exact reconstruction {baseline["reconstruction"]["exact_reconstruction_count"]}/{baseline["reconstruction"]["chemistry_samples"]}. Expected sulfur fraction: {100*baseline["panels"][0]["expected_sulfur_fraction"]:.3f}%; training-population reference: {100*target_sulfur:.3f}%. A lower sulfur fraction is not automatically better.', '',
             "| Endpoint metric | Existing loss | Existing + neighbor term |", "| --- | ---: | ---: |"]
    for label, key in [("Fixed-prior raw validity", "valid"), ("Strict-valid count", "strict"), ("Exact reconstruction", "recon"),
                       ("Chemically valid reconstruction", "valid_recon"), ("Reconstruction loss", "nll"),
                       ("Expected sulfur atom fraction", "sulfur"), ("Hard sulfur atom fraction", "hard_sulfur"),
                       ("Disconnected prior graphs", "disconnected"), ("Mean hard atom count", "size"),
                       ("Unique valid identities", "unique"), ("Bounded fingerprint diversity", "diversity"),
                       ("Median update seconds after first two", "seconds"), ("Peak reserved GPU GiB", "vram")]:
        lines.append(f"| {label} | {value('control',key)} | {value('neighbor',key)} |")
    lines += ["", "## Paired fixed-panel changes", ""]
    for name,a in analysis["arms"].items():
        t=a["paired_fixed_argmax_transitions"]["raw_chemical_valid"]
        lines.append(f'- {name}: {t["invalid_to_valid"]} invalid-to-valid rescues and {t["valid_to_invalid"]} newly invalid graphs versus the start; {a["changed_hard_graphs"]} changed hard graphs.')
    lines += ["", "## Additional paired generation panels", "", "| Arm | Seed | Argmax raw valid | Categorical raw / strict valid |", "| --- | --- | ---: | ---: |"]
    for name, a in analysis["arms"].items():
        for panel in a["final"]["panels"]:
            c = panel["categorical"]
            lines.append(f'| {name} | {panel["seed"]} | {panel["argmax"]["raw_valid"]}/{panel["argmax"]["attempts"]} | {c["raw_valid"]}/{c["attempts"]}; strict{c["strict_valid"]} |')
    lines += ["", "## Interpretation limits", "", f"One checkpoint, one local training seed and{budget} small-batch updates establish a directional pilot only. Fixed panels are repeated, four categorical draws per latent are correlated, and sulfur frequency is a population diagnostic rather than a per-molecule quota. Local end-to-end timing includes changed repair-search workloads and is not an A100 throughput estimate. Structural diversity uses at most256 unique fingerprints. Loss decrease alone cannot establish global chemistry, synthesis or binding.", "",
              "Every attempted graph remains in the NPZ/classification outputs under the local run folder. `analysis.json` records paired rescue/damage counts, fresh end-panel results, exact input/source identities and full reconstruction statistics. Diagnostic checkpoints are explicitly nonpromotable.", ""]
    (args.output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(dict(report=str(args.output / "report.md"), analysis=str(args.output / "analysis.json")), indent=2))


if __name__ == "__main__":
    main()
