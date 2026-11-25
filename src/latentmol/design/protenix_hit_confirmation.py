"""Held-out BRD4 geometry checks on strict hard-molecule hits, after parent selection ends."""

import json
from pathlib import Path
from statistics import mean

from latentmol.design.protenix_conformer_rescore import score_geometry_seed
from latentmol.design.protenix_restarts import fresh_score_records, write_json


DISCOVERY_SEED = 1201
HELD_OUT_SEEDS = (2203, 3209)


def confirm_search_hits(args, config, search_result, fresh_score=fresh_score_records):
    """Freeze the best novel identities per arm before either held-out seed is evaluated."""
    fresh = json.loads(args.fresh_config.read_text(encoding="utf-8"))
    if (fresh["geometry_seed"] != DISCOVERY_SEED or config["sa_mode"] != "pre_smoothing"
            or search_result["status"] != "completed"):
        raise ValueError("Confirmation requires the completed audited search and discovery geometry seed 1201")
    assets = str((args.fresh_config.parent / fresh["assets_config"]).resolve(strict=True))
    cap = config["confirm_top_hits"]
    if type(cap) is not int or not 1 <= cap <= 24:
        raise ValueError("Confirmation requires a positive per-arm cap up to 24")
    panel = json.loads(Path(config["parent_panel"]["path"]).read_text(encoding="utf-8"))
    excluded = set(panel["historical_valid_smiles"]) | {r["smiles"] for r in panel["points"]}
    by_arm = {arm: {} for arm in search_result["arms"]}
    for wave in search_result["cycles"]:
        batch = args.output / f"cycle-{wave['cycle']:02d}" / "batch.json"
        for branch in json.loads(batch.read_text(encoding="utf-8"))["branches"]:
            candidates = by_arm[branch["arm"]]
            for row in branch["candidates"]["mixed"]:
                smi, chemistry = row["smiles"], row["chemistry"]
                if (row["is_founder"] or smi in excluded or row["status"] != "scored"
                        or not chemistry["valid"] or chemistry["sa_mode"] != "pre_smoothing"
                        or not (chemistry["sa_score"] <= 5 and row["loss"] <= 2.3)):
                    continue
                point = dict(smiles=smi, audited_sa=chemistry["sa_score"], discovery_loss=row["loss"],
                             wave=wave["cycle"], branch=branch["name"], step=row["step"])
                if smi not in candidates or row["loss"] < candidates[smi]["discovery_loss"]:
                    candidates[smi] = point
    selected = {}
    for arm, candidates in by_arm.items():
        ranked = sorted(candidates.values(), key=lambda r: (r["discovery_loss"], r["smiles"]))
        for point in ranked[:cap]:
            selected.setdefault(point["smiles"], {})[arm] = point
    root = args.output / "confirmation"
    root.mkdir()
    records = [dict(start=f"hit-{i:02d}", arm="confirmation", role="hard_hit", smiles=smi,
                    step=next(iter(sources.values()))["step"], discovery_sources=sources)
               for i, (smi, sources) in enumerate(sorted(selected.items()), 1)]
    write_json(root / "panel.json", dict(candidates=records, top_per_arm=cap, discovery_seed=DISCOVERY_SEED,
               held_out_seeds=HELD_OUT_SEEDS, eligible_counts={a: len(v) for a, v in by_arm.items()},
               fresh_attempt_allowance=cap * len(by_arm) * len(HELD_OUT_SEEDS)))
    scores = {seed: score_geometry_seed(args, root / f"seed-{seed}", fresh, assets, seed, records, fresh_score)
              for seed in HELD_OUT_SEEDS} if records else {}
    hits = []
    for record in records:
        repeats = {seed: scores[seed][record["start"]] for seed in HELD_OUT_SEEDS}
        losses = [r["loss"] for r in repeats.values() if r["loss"] is not None]
        complete = len(losses) == len(HELD_OUT_SEEDS)
        hits.append(record | dict(by_seed=repeats, complete=complete,
                    held_out_mean=mean(losses) if complete else None,
                    held_out_range=[min(losses), max(losses)] if complete else None,
                    confirmed=(max(losses) <= 2.3) if complete else None))
    summary = dict(status="completed", candidates=hits, fresh_attempts=len(hits) * len(HELD_OUT_SEEDS),
                   fresh_successes=sum(r["loss"] is not None for group in scores.values() for r in group.values()),
                   confirmed_by_arm={arm: sum(h["confirmed"] is True and arm in h["discovery_sources"]
                                         for h in hits) for arm in by_arm})
    write_json(root / "result.json", summary)
    return summary
