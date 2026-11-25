"""Rescore a frozen Protenix candidate panel under shared ligand geometry seeds."""

from collections import Counter
from copy import copy
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from statistics import mean, median, pstdev
from time import monotonic

from rdkit import Chem

from latentmol.design.latent_protenix import _json
from latentmol.design.protenix_restart_selection import max_pairwise_similarity
from latentmol.design.protenix_restarts import fresh_score_records, restart_args, write_json


def _sealed_json(record):
    raw = Path(record["path"]).read_bytes()
    if sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError(f"Source scoring artifact changed: {record['path']}")
    return json.loads(raw)


def verify_panel(config):
    if config.get("panel_kind") == "synthesis_saved":
        return verify_synthesis_panel(config)
    panel, seeds, historical = config["panel"], config["geometry_seeds"], config["historical_seed"]
    anchors = config["anchors"]
    panel_ids = {row["id"] for row in panel}
    groups = config["selection"]["source_waves"]
    if (config["study"] != "conformer_rescore" or not 2 <= len(panel) <= 24
            or not 2 <= len(seeds) <= 8 or len(set(seeds)) != len(seeds) or seeds[0] != historical
            or any(type(seed) is not int or not 0 < seed < 2**31 for seed in seeds)
            or len(panel_ids) != len(panel) or len({row["smiles"] for row in panel}) != len(panel)
            or set(anchors) != {"incumbent", "challenger"} or len(set(anchors.values())) != 2
            or not set(anchors.values()) <= panel_ids
            or Counter(row["group"] for row in panel) != dict.fromkeys(groups, config["selection"]["count_per_group"])
            or not 0 <= config["max_pairwise_similarity"] <= 1):
        raise ValueError("Invalid conformer seed or candidate panel")
    source = _sealed_json(config["source_result"])
    if source["status"] != "completed":
        raise ValueError("Require a completed archive source")
    wave_scores = {}
    for wave in {row["wave"] for row in panel}:
        receipt = _sealed_json(config["sources"][str(wave)])
        if (receipt["status"] != "completed" or receipt["geometry_seed"] != historical
                or receipt["job_id"] != config["source_job"]):
            raise ValueError(f"Wrong original fresh scoring receipt at wave {wave}")
        wave_scores[wave] = {(row["arm"], row["step"], row["smiles"], row["loss"])
                             for row in receipt["candidates"] if row["status"] == "scored"}
    for row in panel:
        molecule = Chem.MolFromSmiles(row["smiles"])
        first, last = groups[row["group"]]
        if (molecule is None or len(Chem.GetMolFrags(molecule)) != 1
                or Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False) != row["smiles"]
                or not math.isfinite(row["score"]) or not first <= row["wave"] <= last
                or (row["branch"], row["step"], row["smiles"], row["score"]) not in wave_scores[row["wave"]]):
            raise ValueError(f"Candidate differs from the sealed fresh score: {row['id']}")
    if max_pairwise_similarity([row["smiles"] for row in panel]) > config["max_pairwise_similarity"]:
        raise ValueError("Frozen candidate panel exceeds its pairwise similarity limit")
    return panel, seeds


def _finite_score(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def verify_synthesis_panel(config):
    """Authenticate saved intermediates/endpoints without inventing a score.

    A source batch is read only after its checksum is verified. Full branch
    paths match exactly; a short branch name is accepted only when unambiguous
    within its explicitly selected batch. Endpoint losses may genuinely be
    absent, while a historical candidate loss must match the saved record.
    """
    panel, seeds = config["panel"], config["geometry_seeds"]
    if (config.get("study") != "conformer_rescore" or not 2 <= len(panel) <= 24
            or not 2 <= len(seeds) <= 8 or len(set(seeds)) != len(seeds)
            or seeds[0] != config["historical_seed"]
            or any(type(seed) is not int or not 0 < seed < 2**31 for seed in seeds)
            or any(not isinstance(row.get("id"), str) or not row["id"] for row in panel)
            or len({row["id"] for row in panel}) != len(panel)
            or len({row.get("smiles") for row in panel}) != len(panel)):
        raise ValueError("Invalid saved synthesis seed or candidate panel")
    if _sealed_json(config["source_result"]).get("status") != "completed":
        raise ValueError("Require a completed synthesis source")
    batches, verified = {}, []
    for row in panel:
        if (row.get("location") not in {"candidate", "endpoint"}
                or not isinstance(row.get("source_key"), str)
                or row["source_key"] not in config.get("sources", {})
                or not isinstance(row.get("branch"), str) or not row["branch"]
                or type(row.get("step")) is not int or row["step"] < 0
                or "score" not in row or (row["score"] is not None and not _finite_score(row["score"]))):
            raise ValueError("Invalid saved synthesis candidate provenance")
        molecule = Chem.MolFromSmiles(row.get("smiles", ""))
        if (molecule is None or not molecule.GetNumAtoms() or len(Chem.GetMolFrags(molecule)) != 1
                or Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False) != row["smiles"]):
            raise ValueError(f"Invalid saved synthesis molecule: {row['id']}")
        key = row["source_key"]
        if key not in batches:
            batches[key] = _sealed_json(config["sources"][key])
        matches = []
        for branch in batches[key].get("branches", []):
            items = ([branch.get("endpoint")] if row["location"] == "endpoint" else
                     [candidate for group in branch.get("candidates", {}).values() for candidate in group])
            for candidate in items:
                if not isinstance(candidate, dict):
                    continue
                path = candidate.get("branch", branch.get("name"))
                branch_matches = (path == row["branch"] if "/" in row["branch"] else
                                  row["branch"] in {branch.get("name"), str(path).rsplit("/", 1)[-1]})
                if (branch_matches and candidate.get("step") == row["step"]
                        and candidate.get("smiles") == row["smiles"]
                        and candidate.get("loss") == row["score"]):
                    matches.append(branch)
                    break
        if len(matches) != 1:
            raise ValueError(f"Candidate differs from the sealed synthesis source: {row['id']}")
        branch = matches[0]
        verified.append(row | dict(
            source_parent=dict(smiles=branch.get("parent_smiles"),
                               fresh_loss=branch.get("parent_fresh_loss"),
                               source=branch.get("parent_source")),
            provenance=row.get("provenance", row.get("description",
                f"Saved {row['location']} from {key}, {row['branch']}, update {row['step']}"))))
    return verified, seeds


def summarize_scores(panel, seeds, scored, anchors):
    per_seed = []
    for seed in seeds:
        successful = {key: row["loss"] for key, row in scored[seed].items() if row["loss"] is not None}
        ranked = sorted(successful, key=lambda key: (successful[key], key))
        per_seed.append(dict(geometry_seed=seed, attempts=len(panel), successful=len(successful),
                             ranking=ranked, best_loss=successful[ranked[0]] if ranked else None))
    candidates = []
    for candidate in panel:
        key = candidate["id"]
        rows = {str(seed): scored[seed][key] for seed in seeds}
        held_out = [rows[str(seed)]["loss"] for seed in seeds[1:]
                    if rows[str(seed)]["loss"] is not None]
        complete = len(held_out) == len(seeds) - 1
        historical = rows[str(seeds[0])]["loss"]
        candidates.append(candidate | dict(
            scores=rows, historical_repeat_delta=None if historical is None else historical - candidate["score"],
            held_out_successful=len(held_out), held_out_mean=mean(held_out) if complete else None,
            held_out_sd=pstdev(held_out) if complete else None,
            held_out_range=[min(held_out), max(held_out)] if held_out else None))
    eligible = [row for row in candidates if row["held_out_mean"] is not None]
    order = [row["id"] for row in sorted(eligible, key=lambda row: (row["held_out_mean"], row["id"]))]
    original_winner = min(panel, key=lambda row: (row["score"], row["id"]))["id"]
    original = {row["id"]: row["score"] for row in panel}
    incumbent, challenger = anchors["incumbent"], anchors["challenger"]
    anchor_deltas = []
    for seed in seeds:
        previous, improved = (scored[seed][key]["loss"] for key in (incumbent, challenger))
        anchor_deltas.append(dict(geometry_seed=seed, incumbent_loss=previous, challenger_loss=improved,
                                  gain=None if previous is None or improved is None else previous - improved))
    held_out_gains = [row["gain"] for row in anchor_deltas[1:] if row["gain"] is not None]
    anchor_comparison = anchors | dict(
        original_gain=original[incumbent] - original[challenger], by_seed=anchor_deltas,
        held_out_pairs=len(held_out_gains), held_out_challenger_wins=sum(gain > 0 for gain in held_out_gains),
        held_out_mean_gain=mean(held_out_gains) if held_out_gains else None)
    groups = {}
    for group in dict.fromkeys(row["group"] for row in panel):
        complete = [row for row in eligible if row["group"] == group]
        groups[group] = dict(complete_candidates=len(complete),
                             mean_of_held_out_means=mean(row["held_out_mean"] for row in complete) if complete else None,
                             best_mean=min((row["held_out_mean"] for row in complete), default=None))
    return dict(candidates=candidates, by_seed=per_seed, by_group=groups, anchor_comparison=anchor_comparison,
                original_winner=original_winner, held_out_mean_ranking=order,
                original_winner_held_out_rank=1 + order.index(original_winner) if original_winner in order else None)


def summarize_synthesis_scores(panel, seeds, scored):
    """Report geometry sensitivity and hard chemistry of authenticated points."""
    from latentmol.design.protenix_synthesis_study import molecule_readout

    candidates, per_seed = [], []
    for seed in seeds:
        successful = {row["id"]: scored[seed][row["id"]]["loss"] for row in panel
                      if scored[seed][row["id"]]["loss"] is not None}
        ranking = sorted(successful, key=lambda key: (successful[key], key))
        per_seed.append(dict(geometry_seed=seed, attempts=len(panel), successful=len(successful),
                             ranking=ranking, best_loss=successful[ranking[0]] if ranking else None))
    for point in panel:
        scores = {str(seed): scored[seed][point["id"]] for seed in seeds}
        losses = [row["loss"] for row in scores.values() if row["loss"] is not None]
        held_out = [scores[str(seed)]["loss"] for seed in seeds[1:]
                    if scores[str(seed)]["loss"] is not None]
        repeated = scores[str(seeds[0])]["loss"]
        candidates.append(point | dict(
            chemistry=molecule_readout(point["smiles"]), scores=scores,
            successful=len(losses), complete=len(losses) == len(seeds),
            mean_loss=mean(losses) if losses else None,
            median_loss=median(losses) if losses else None,
            loss_sd=pstdev(losses) if losses else None,
            loss_range=[min(losses), max(losses)] if losses else None,
            historical_repeat_delta=(repeated - point["score"]
                                     if repeated is not None and point["score"] is not None else None),
            held_out_successful=len(held_out),
            held_out_mean=mean(held_out) if len(held_out) == len(seeds) - 1 else None,
            held_out_median=median(held_out) if len(held_out) == len(seeds) - 1 else None))
    eligible = [row for row in candidates if row["held_out_mean"] is not None]
    return dict(panel_kind="synthesis_saved", candidates=candidates, by_seed=per_seed,
                held_out_mean_ranking=[row["id"] for row in sorted(
                    eligible, key=lambda row: (row["held_out_mean"], row["id"]))],
                interpretation="Read-only saved-molecule rescoring; hard SA/QED are proxy measurements. "
                               "Historical endpoint scores may be absent; no missing score is inferred.")


def score_geometry_seed(args, root, fresh, assets, seed, records, fresh_score):
    """Score fixed hard identities under one geometry seed using the discovery pocket."""
    root.mkdir()
    seed_args = copy(args)
    seed_args.fresh_config = root / "fresh-config.json"
    write_json(seed_args.fresh_config, fresh | dict(geometry_seed=seed, assets_config=assets,
                                                   candidate_limit=len(records)))
    rows = fresh_score(seed_args, root, records)
    expected = {(row["start"], row["smiles"]) for row in records}
    if len(rows) != len(records) or {(row["start"], row["smiles"]) for row in rows} != expected:
        raise ValueError(f"Fresh scoring changed the frozen panel at geometry seed {seed}")
    for row in rows:
        loss = row["loss"]
        if ((row["status"] == "scored") != (loss is not None)
                or (loss is not None and not _finite_score(loss))):
            raise ValueError(f"Invalid fresh score at geometry seed {seed}")
    return {row["start"]: dict(status=row["status"], loss=row["loss"], reason=row.get("reason"))
            for row in rows}


def run_rescore(args, config, fresh_score=fresh_score_records):
    started = monotonic()
    if args.arm not in config["tasks"]:
        raise ValueError(f"Unknown rescore task: {args.arm}")
    panel, seeds = verify_panel(config)
    fresh = json.loads(args.fresh_config.read_text(encoding="utf-8"))
    if fresh["geometry_seed"] != seeds[0]:
        raise ValueError("Repeat seed differs from the original fresh protocol")
    assets = str((args.fresh_config.parent / fresh["assets_config"]).resolve(strict=True))
    args.output.mkdir(parents=True)
    write_json(args.output / "run.json", dict(job_id=os.environ.get("LSB_JOBID"), task=args.arm,
        source_job=config.get("source_job"), source_result=config["source_result"], panel=panel, geometry_seeds=seeds,
        panel_kind=config.get("panel_kind", "archive"),
        planned_attempts=len(panel) * len(seeds), optimization_updates=0))
    saved = config.get("panel_kind") == "synthesis_saved"
    records = [dict(smiles=row["smiles"], start=row["id"], arm="conformer-rescore",
                    role="saved_synthesis_point" if saved else "edit",
                    step=row["step"], source_wave=row.get("wave"), source_branch=row["branch"],
                    **({"provenance": row["provenance"], "source_parent": row["source_parent"]} if saved else {}))
               for row in panel]
    scored = {}
    for seed in seeds:
        root = args.output / f"seed-{seed}"
        scored[seed] = score_geometry_seed(args, root, fresh, assets, seed, records, fresh_score)
        if seed == seeds[0] and (ready := os.environ.get("DBMOL_LAUNCH_READY")):
            _json(ready, dict(status="models_and_inputs_ready", nonce=os.environ.get("DBMOL_LAUNCH_NONCE"),
                              job_id=os.environ.get("LSB_JOBID")))
    summary = (summarize_synthesis_scores(panel, seeds, scored) if saved else
               summarize_scores(panel, seeds, scored, config["anchors"]))
    result = summary | dict(
        status="completed", source_job=config.get("source_job"),
        fresh_attempts=len(panel) * len(seeds), optimization_updates=0, elapsed_seconds=monotonic() - started)
    write_json(args.output / "result.json", result)
    return result


def main(argv=None):
    args = restart_args(__doc__, argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    return run_rescore(args, config)


if __name__ == "__main__":
    main()
