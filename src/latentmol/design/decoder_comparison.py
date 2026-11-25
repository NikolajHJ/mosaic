"""Compare soft bond-message and triangle VAEs on the same BRD4 contact-edit task."""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path

import torch

from latentmol.datasets.bulk_view import BalancedSparseCorpus
from latentmol.datasets.data import collate_graphs
from latentmol.design.latent_boundaries import sweep
from latentmol.design.latent_landscape import graph_observation, unit
from latentmol.design.latent_protenix import _json, _model_identity, choose_active, hard_from_graph
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256
from latentmol.graph_policy import graph_from_decoder
from latentmol.guidance.mosaic_protenix import MosaicProtenixScorer
from latentmol.representation import RelaxedGraph


ARCHITECTURES = ("triangle_refine", "bond_message")
RAYS = ("descent", "random_plus", "random_minus")
POLICIES = ("fixed", "raw")


def panel_from_validation(data, spec, count, atom_count, seed):
    """Select true inputs before loading either model. Keep exactly the same graph rows."""
    panel = []
    seen = set()
    for row in torch.randperm(len(data), generator=torch.Generator().manual_seed(seed)).tolist():
        hard = collate_graphs([data[row]])
        if int(hard.atoms.count_nonzero()) != atom_count:
            continue
        observed = graph_observation(hard, spec)
        smiles = observed["smiles"]
        if smiles is None or smiles in seen:
            continue
        seen.add(smiles)
        panel.append((f"val-{row}", hard, smiles, str(data.smiles[row])))
        if len(panel) == count:
            return panel
    raise ValueError("Insufficient distinct valid validation molecules with the native context size")


def permute_for_native(graph, order):
    """Relabel the full graph bijectively so its selected slots occupy the native prefix."""
    return RelaxedGraph(graph.atoms[:, order], graph.charges[:, order], graph.bonds[:, order][:, :, order])


def label_rows(hard):
    return {field: getattr(hard, field)[0].tolist() for field in ("atoms", "charges", "bonds")}


def probe(record_id, hard, source, cache_smiles, model, scorer, config, random_direction):
    # The sampled hard row is canonical, but still use the checkpoint's standard encoding entry point.
    with torch.no_grad():
        center, _ = model.encode(hard.to_relaxed(model.spec))
        decoded = model.decode(center)
        active = choose_active(decoded, model.spec, config["active_count"])
        graph = graph_from_decoder(decoded, model.spec, temperature=config["temperature"], fixed_active=active)
        baseline_labels = dict(raw=label_rows(decoded.harden(model.spec)), fixed=label_rows(hard_from_graph(graph)))
    order = torch.cat([active.nonzero().flatten(), (~active).nonzero().flatten()])
    result = sweep(model, center, lambda g: scorer(permute_for_native(g, order)),
                   active, config["temperature"], config["radii"], random_direction)
    return dict(start=record_id, source_smiles=source, validation_cache_smiles=cache_smiles,
                center=center[0].tolist(), baseline_labels=baseline_labels, native_slots=order[:int(active.sum())].tolist(),
                random_direction=random_direction[0].tolist(), **result)


def candidate_rows(records, edit_cap):
    """Predeclare first distinct valid changes in radius order for each ray and readout.

    Input, decoded center and moved graph have separate identities. Even a change back
    to the true input is recorded in the rays but cannot claim a new design. Invalid
    decoded baselines remain eligible for validity-rescue reporting.
    """
    records_seen = set()
    candidates = []
    for record in records:
        start, arch, source = record["start"], record["architecture"], record["source_smiles"]
        if start not in records_seen:
            candidates.append(dict(start=start, arm="source", role="initial", smiles=source))
            records_seen.add(start)
        for ray in RAYS:
            for policy in POLICIES:
                arm = f"{arch}/{ray}/{policy}"
                base = record["baseline"][policy]["smiles"]
                if base:
                    candidates.append(dict(start=start, arm=arm, role="initial", smiles=base))
                seen = {source, base, None}
                accepted = 0
                for sample in record["trials"][ray]["samples"]:
                    smiles = sample["graphs"][policy]["smiles"]
                    if smiles in seen:
                        continue
                    seen.add(smiles)
                    candidates.append(dict(start=start, arm=arm, role="edit", smiles=smiles, radius=sample["radius"]))
                    accepted += 1
                    if accepted == edit_cap:
                        break
    return candidates


def summarize(records, scores, edit_cap):
    """Count each attempted start in the primary denominator, including unavailable comparisons."""
    expected = candidate_rows(records, edit_cap)
    def identity(row):
        return row["start"], row["arm"], row["role"], row["smiles"], row.get("radius")

    if [identity(row) for row in scores] != [identity(row) for row in expected]:
        raise ValueError("Fresh score identities differ from predeclared candidates")
    by_arm = {}
    for row in scores:
        score = row["loss"]
        if score is not None and not math.isfinite(score):
            raise ValueError("Fresh score must be finite or unavailable")
        by_arm.setdefault((row["start"], row["arm"]), []).append(row)
    results = []
    totals = {}
    for record in records:
        start, arch, source = record["start"], record["architecture"], record["source_smiles"]
        source_loss = by_arm[start, "source"][0]["loss"]
        for ray in RAYS:
            for policy in POLICIES:
                key = f"{arch}/{ray}/{policy}"
                base = record["baseline"][policy]
                samples = record["trials"][ray]["samples"]
                selected = by_arm.get((start, key), [])
                base_score = next((r["loss"] for r in selected if r["role"] == "initial"), None)
                edits = [r for r in selected if r["role"] == "edit"]
                scored = [r["loss"] for r in edits if r["loss"] is not None]
                valid_smiles = {s["graphs"][policy]["smiles"] for s in samples} - {None}
                valid_novel = valid_smiles - {source, base["smiles"]}
                compared = base_score is not None and bool(scored)
                improved = compared and min(scored) < base_score
                if base["smiles"] is None:
                    outcome = "invalid_center"
                elif base_score is None:
                    outcome = "unscored_center"
                elif not samples:
                    outcome = "no_direction"
                elif not valid_smiles:
                    outcome = "all_observations_invalid"
                elif all(s["graphs"][policy]["smiles"] == base["smiles"] for s in samples):
                    outcome = "unchanged"
                elif not valid_novel:
                    outcome = "no_distinct_valid_edit"
                elif not scored:
                    outcome = "unscored_edits"
                else:
                    outcome = "improved" if improved else "scored_without_improvement"
                invalid = sum(s["graphs"][policy]["smiles"] is None for s in samples)
                result = dict(start=start, arm=key, source_smiles=source, center_smiles=base["smiles"],
                              exact_start=base["smiles"] == source, valid_center=base["smiles"] is not None,
                              center_failure=base["failure"], observations=len(samples),
                              invalid_observations=invalid, unchanged_observations=sum(
                                  s["graphs"][policy]["signature"] == base["signature"] if base["smiles"] is None
                                  else s["graphs"][policy]["smiles"] == base["smiles"] for s in samples),
                              source_return_observations=sum(s["graphs"][policy]["smiles"] == source
                                  and base["smiles"] != source for s in samples),
                              distinct_valid_edits=len(valid_novel), fresh_selected=len(edits), fresh_scored=len(scored),
                              best_delta_from_center=min(scored)-base_score if compared else None,
                              best_delta_from_input=min(scored)-source_loss if scored and source_loss is not None else None,
                              rescued_invalid_center=base["smiles"] is None and bool(valid_novel),
                              outcome=outcome, improved=improved)
                results.append(result)
                acc = totals.setdefault(key, Counter())
                acc.update(dict(starts=1, exact_start=int(result["exact_start"]), comparable=int(compared),
                                improved=int(improved), rescued=int(result["rescued_invalid_center"])))
                acc["outcome_" + outcome] += 1
    for acc in totals.values():
        acc["improved_fraction_all_starts"] = acc["improved"] / acc["starts"]
        acc["improved_fraction_comparable"] = acc["improved"] / acc["comparable"] if acc["comparable"] else None
    return dict(primary_measure="Fraction of all encoded starts with a distinct valid edit that beats its decoded center on fresh rescoring, using the fixed slots and contact descent",
                by_arm=totals, starts=results)


def run_sweep(args):
    job = os.environ.get("LSB_JOBID", "")
    if not job.isdigit():
        raise RuntimeError("Native edit comparison requires allocated GPU")
    torch.set_num_threads(4)
    config = json.loads(args.config.read_bytes())
    radii = config["radii"]
    if (not radii or radii != sorted(set(radii)) or
            any(not math.isfinite(number) or number <= 0 for number in [*radii, config["temperature"]]) or
            min(config["centers"], config["max_edit_candidates_per_arm"], config["candidate_limit"]) < 1):
        raise ValueError("Require ordered positive radii, temperature and positive candidate budgets")
    training = json.loads((args.training / "comparison.json").read_bytes())
    if training["status"] != "completed" or training["dataset_sha256"] != config["dataset_sha256"]:
        raise ValueError("Require a completed matched training comparison on the configured data")
    corpus = BalancedSparseCorpus(config["data"], expected_sha256=config["dataset_sha256"])
    panel = panel_from_validation(corpus.split("val"), corpus.spec, config["centers"], config["active_count"], config["seed"])
    models, hashes = {}, {}
    for arch in ARCHITECTURES:
        entry = training["arms"][arch]
        expected = f"{arch}/final.pt"
        path = args.training / expected
        if entry["checkpoint"] != expected or entry["checkpoint_sha256"] != file_sha256(path):
            raise ValueError(f"Checkpoint identity mismatch for {arch}")
        model, payload = load_model(path, "cpu", corpus.spec)
        if model.config.decoder_variant != arch or payload["dataset_sha256"] != config["dataset_sha256"]:
            raise ValueError("Checkpoint architecture or dataset mismatch")
        model.eval().requires_grad_(False)
        models[arch], hashes[arch] = model, _model_identity(model)
    if len({m.config.latent for m in models.values()}) != 1:
        raise ValueError("Matched directions require equal latent dimensions")
    initial = torch.arange(corpus.spec.max_nodes) < config["active_count"]
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    with MosaicProtenixScorer(corpus.spec, initial,
            command=[args.mosaic_python, "-m", "latentmol.guidance.mosaic_protenix_worker"],
            config=json.loads(args.mosaic_config.read_bytes()), stderr_path=args.output/"native.stderr.log",
            timeout=360, startup_timeout=900) as scorer:
        if marker := os.environ.get("DBMOL_LAUNCH_READY"):
            _json(marker, dict(status="models_and_inputs_ready", nonce=os.environ["DBMOL_LAUNCH_NONCE"], job_id=job))
        for index, (start, hard, source, cache_smiles) in enumerate(panel):
            rng = torch.Generator().manual_seed(config["seed"] + index + 1)
            direction = unit(torch.randn(1, next(iter(models.values())).config.latent, generator=rng))
            for arch, model in models.items():
                record = probe(start, hard, source, cache_smiles, model, scorer, config, direction)
                record["architecture"] = arch
                _json(args.output / f"{start}-{arch}.json", record)
                records.append(record)
    if any(_model_identity(model) != hashes[arch] or any(p.grad is not None for p in model.parameters())
           for arch, model in models.items()):
        raise RuntimeError("Frozen decoder changed during contact evaluation")
    candidates = candidate_rows(records, config["max_edit_candidates_per_arm"])
    if len(candidates) > config["candidate_limit"]:
        raise ValueError("Fresh scoring candidates exceed the shared budget")
    _json(args.output / "fresh-candidates.json", dict(candidates=candidates))
    _json(args.output / "sweep.json", dict(status="completed", job_id=job, config=config, native=scorer.identity,
          checkpoints={k: training["arms"][k]["checkpoint_sha256"] for k in ARCHITECTURES},
          training_updates=training["updates_per_arm"],
          observations=[f"{r['start']}-{r['architecture']}.json" for r in records],
          relaxed_calls=sum(1+sum(len(trial["samples"]) for trial in r["trials"].values()) for r in records)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    scan = sub.add_parser("sweep")
    for flag in ("training", "config", "mosaic-config", "mosaic-python"):
        scan.add_argument("--"+flag, type=Path if flag != "mosaic-python" else str, required=True)
    report = sub.add_parser("report")
    for cmd in (scan, report):
        cmd.add_argument("--output", type=Path, required=True)
        cmd.add_argument("--plan")
        cmd.add_argument("--plan-sha256")
    args = parser.parse_args()
    if args.action == "sweep":
        run_sweep(args)
        return
    path = args.output / "comparison.json"
    if path.exists():
        raise FileExistsError(path)
    sweep_report = json.loads((args.output / "sweep.json").read_bytes())
    fresh = json.loads((args.output / "fresh/summary.json").read_bytes())
    if sweep_report["status"] != "completed" or fresh["status"] != "completed":
        raise ValueError("Require completed relaxed exploration and fresh scores")
    records = [json.loads((args.output / filename).read_bytes()) for filename in sweep_report["observations"]]
    result = summarize(records, fresh["candidates"], sweep_report["config"]["max_edit_candidates_per_arm"])
    _json(path, dict(status="completed", fresh_unique_calls=fresh["unique_candidates"],
          relaxed_calls=sweep_report["relaxed_calls"], checkpoints=sweep_report["checkpoints"],
          objective=fresh["objective"], **result))


if __name__ == "__main__":
    main()
