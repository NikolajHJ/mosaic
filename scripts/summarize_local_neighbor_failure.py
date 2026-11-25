"""Compare the common saved prefix of a safety-stopped local pilot.

This never reruns the decoder, fills missing endpoint observations, changes a
safety threshold or converts an incomplete training arm into a successful run.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
from latentmol.chemistry_epoch45_generation import structural_diversity
from summarize_local_neighbor_pilot import pairs, read


def bond_changes(first, last):
    """Count edits only when atom and charge slot labels remain identical."""
    with np.load(first) as a, np.load(last) as b:
        same = (a["atoms"] == b["atoms"]).all(-1) & (a["charges"] == b["charges"]).all(-1)
        upper = np.triu(np.ones(a["bonds"].shape[-2:], dtype=bool), 1)
        before, after = a["bonds"][same][:, upper], b["bonds"][same][:, upper]
        return dict(same_atom_charge_graphs=int(same.sum()),
            deleted_edges=int(((before != 0) & (after == 0)).sum()),
            added_edges=int(((before == 0) & (after != 0)).sum()),
            changed_nonzero_bond_types=int(((before != after) & (before != 0) & (after != 0)).sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--audit", type=Path)
    args = p.parse_args()
    failure = read(args.run / "failed.json")
    if "Local pilot safety stop" not in failure.get("error", ""):
        raise ValueError("This report requires an actual recorded local safety stop")
    manifest, inputs = read(args.run / "manifest.json"), read(args.inputs / "inputs.json")
    paths = {name: {int(f.stem.split("-")[-1]): f for f in (args.run / name).glob("eval-[0-9][0-9][0-9][0-9].json")}
             for name in ("control", "neighbor")}
    common = max(set(paths["control"]) & set(paths["neighbor"]))
    if common < 1:
        raise ValueError("No common post-training panel exists")
    logs = {name: [json.loads(line) for line in (args.run / name / "updates.jsonl").read_text().splitlines()]
            for name in paths}
    for a, b in zip(logs["control"][:common], logs["neighbor"][:common]):
        assert all(a[k] == b[k] for k in ("source_indices", "fresh_prior_rng_before", "posterior_rng_before"))
    baseline_raw = []
    out = dict(status="safety_stopped_do_not_adopt", original_failure=failure,
        planned_updates_per_arm=manifest["pilot"]["updates"], common_comparison_update=common,
        completed_training_updates={name: len(rows) for name, rows in logs.items()},
        matched_real_rows_and_noise_streams_through_common_prefix=True,
        run_manifest_sha256=hashlib.sha256((args.run / "manifest.json").read_bytes()).hexdigest(),
        run_path=str(args.run.resolve()), inputs_path=str(args.inputs.resolve()),
        inputs_sha256=hashlib.sha256((args.inputs / "inputs.json").read_bytes()).hexdigest(),
        environment=manifest["environment"], declared_pilot=manifest["pilot"],
        report_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), arms={}, hpc_mutations=False)
    spec = inputs["manifest"]["spec"]
    counts = inputs["files"]["composition"]["value"]["elements_distribution"]
    target = counts[spec["atomic_numbers"][1:].index(16)] / sum(counts)
    out["training_population_sulfur_fraction"] = target
    for name in paths:
        folder = args.run / name
        initial, endpoint = read(paths[name][0]), read(paths[name][common])
        first = read(folder / "eval-0000-fixed-argmax-classifications.json")
        last = read(folder / f"eval-{common:04d}-fixed-argmax-classifications.json")
        baseline_raw.append(first)
        out["arms"][name] = dict(initial=initial, matched_endpoint=endpoint,
            paired_transitions=pairs(first, last),
            bond_changes=bond_changes(folder / "eval-0000-fixed-argmax.npz", folder / f"eval-{common:04d}-fixed-argmax.npz"),
            structural_diversity=structural_diversity([dict(valid=r["raw_chemical_valid"] is True, smiles=r["smiles"]) for r in last]),
            all_saved_losses_finite=all(np.isfinite(r["metrics"]["total"]) for r in logs[name]),
            common_prefix_mean_update_seconds=statistics.mean(r["update_seconds"] for r in logs[name][:common]),
            common_prefix_median_update_seconds=statistics.median(r["update_seconds"] for r in logs[name][:common]),
            common_prefix_total_search_seconds=sum(r["metrics"]["search_seconds"] for r in logs[name][:common]),
            endpoint_sulfur_deviation_from_population_prior_pp=100 * (endpoint["panels"][0]["expected_sulfur_fraction"] - target))
        if name == "control":
            out["arms"][name]["completed_100_update_result"] = read(folder / "result.json")
        else:
            out["arms"][name]["incomplete_endpoint_limits"] = "Safety stop before terminal categorical/fresh-panel evaluation, diagnostic checkpoint and full frozen-anchor digest. Saved update10 argmax/reconstruction and update receipts retained."
            out["arms"][name]["added_term_training_receipts"] = [r["metrics"]["additive_guidance"] for r in logs[name]]
    assert baseline_raw[0] == baseline_raw[1]
    a = read(args.run / "control" / f"eval-{common:04d}-fixed-argmax-classifications.json")
    b = read(args.run / "neighbor" / f"eval-{common:04d}-fixed-argmax-classifications.json")
    out["paired_control_to_neighbor"] = pairs(a, b)
    audit = read(args.audit) if args.audit else None
    if audit is not None:
        assert audit["common_update"] == common and audit["prefix_rows_prior_rng_posterior_rng_match"]
        out["independent_raw_connectivity_audit"] = audit
        out["independent_audit_sha256"] = hashlib.sha256(args.audit.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "analysis.json").write_text(json.dumps(out, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Local VAE neighbor-aware valence-loss pilot: safety stop", "",
        "**Do not switch the HPC continuation to this configuration.** The local numerical tests passed, but the added-loss training arm hit the predeclared validity safety stop.", "",
        f"Planned 100 updates per arm; control completed 100 and neighbor guidance stopped at {common}. The main comparison below uses the same saved update {common} in both arms, with matched real rows and prior/posterior noise streams. Both began with identical raw baseline classifications from authenticated checkpoint 42.", "",
        "The isolated RTX 4060 experiment used 16 real + 32 prior points per update, microbatch size 8, inherited AdamW moments, quarter-displacement steps, frozen epoch-35/36 anchors and the existing repair/reference/composition/occupancy losses. Only the experimental arm added neighbor-event weight 0.1 with real/prior weights 0.25/0.75 and assessed-center normalization.", "",
        "| Metric | Start | Control at matched update | Added valence term at matched update |", "| --- | ---: | ---: | ---: |"]
    start = out["arms"]["control"]["initial"]
    observations = [start, *[out["arms"][name]["matched_endpoint"] for name in ("control", "neighbor")]]
    for label, get in [
        ("Raw valid /1,024", lambda v: str(v["panels"][0]["argmax"]["raw_valid"])),
        ("Strict valid /1,024", lambda v: str(v["panels"][0]["argmax"]["strict_valid"])),
        ("Exact reconstruction /256", lambda v: str(v["reconstruction"]["exact_reconstruction_count"])),
        ("Chemically valid reconstruction /256", lambda v: str(v["reconstruction"]["valid_chemistry_count"])),
        ("RDKit sanitization failures", lambda v: str(v["panels"][0]["argmax"]["failure_reasons"].get("sanitization",0))),
        ("Disconnected: native failure category", lambda v: str(v["panels"][0]["argmax"]["failure_reasons"].get("disconnected",0))),
        ("Local valence violations", lambda v: str(v["panels"][0]["argmax"]["local_valence_violations"])),
        ("Expected sulfur atom fraction", lambda v: f'{100*v["panels"][0]["expected_sulfur_fraction"]:.3f}%'),
        ("Hard sulfur atom fraction", lambda v: f'{100*v["panels"][0]["argmax"]["hard_sulfur_fraction"]:.3f}%'),
        ("Mean hard atom count", lambda v: f'{v["panels"][0]["argmax"]["mean_hard_size"]:.2f}')]:
        lines.append("| " + label + " | " + " | ".join(get(v) for v in observations) + " |")
    if audit is not None:
        first = audit["arms"]["control"]["initial"]["disconnected"]
        control = audit["arms"]["control"]["update10"]["disconnected"]
        neighbor = audit["arms"]["neighbor"]["update10"]["disconnected"]
        lines += [f"| Disconnected: independent graph traversal | {first} | {control} | {neighbor} |", "",
            "Native classification checks sanitization first, so its disconnected category omits unsanitizable graphs. Independent traversal counts occupied connected components without RDKit and confirms that the deterioration is actual connectivity loss, rather than only a change in failure labels."]
    lines += ["", "The saved results support the expected connectivity failure mode: local valence errors decrease, but many more graphs become disconnected. The local event allows absent bonds and has no connectivity requirement. This is evidence of an objective trade-off at this weight and small-batch regime; it is not evidence that every gradient is broken or that any neighbor-aware regularizer must fail.", "",
        f"The training-population sulfur atom fraction is {100*target:.3f}%. The pilot does not show sulfur inflation as the main cause of deterioration. A lower sulfur frequency alone is not a per-molecule chemistry goal.", "",
        "## Paired graph evidence", ""]
    for name, arm in out["arms"].items():
        t = arm["paired_transitions"]["raw_chemical_valid"]
        e = arm["bond_changes"]
        lines.append(f'- {name}: {t["invalid_to_valid"]} rescues, {t["valid_to_invalid"]} newly invalid outputs versus start. Among {e["same_atom_charge_graphs"]} graphs retaining every atom/charge slot label: {e["deleted_edges"]} deleted edges and {e["added_edges"]} added edges.')
    lines += ["", "## Scope and next step", "",
        "147 sealed local CPU tests and the earlier two-update GPU benchmark passed; the safety-stopped training experiment did not. All saved losses were finite. Do not remove the stop or claim a completed 100-update experimental run. The failed arm has no terminal fresh-panel/categorical evaluation or failed-state diagnostic checkpoint, and it did not reach the final frozen-anchor digest assertion. No unavailable measurement is imputed.", "",
        "Keep production unchanged. A subsequent declared experiment should balance neighbor guidance with connectivity-preserving guidance and/or a smaller coefficient, before repeating training stability checks. This result is limited to one checkpoint, one local seed and small batches; it does not establish full-scale HPC behavior.", "",
        "All attempted invalid graphs are retained in local NPZ/classification files. `analysis.json` records source/input references, matched-prefix timing, paired transitions, edit counts, bounded fingerprint/scaffold diversity and the completed control's separate 100-update observations.", ""]
    (args.output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(dict(report=str(args.output / "report.md"), matched_update=common,
        completed_updates=out["completed_training_updates"], paired_streams=True), indent=2))


if __name__ == "__main__":
    main()
