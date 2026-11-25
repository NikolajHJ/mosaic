"""Prepare or fresh-score paired archived low-SA panels against the historical BRD4 pocket."""

import argparse
from collections import Counter
from copy import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
from statistics import mean, median

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)


def stratum(row):
    return row["founder"], math.floor(2 * row["audited_sa"])


def make_panel(pool, near, seed=1711021):
    """Choose diverse relaxed-loss leaders with disjoint random founder/SA-matched controls."""
    if len(pool) != 445 or len({r["smiles"] for r in pool}) != len(pool):
        raise ValueError("Expected the 445 distinct, unscored audited archive molecules")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = {}
    for row in pool:
        mol = Chem.MolFromSmiles(row["smiles"])
        if (mol is None or len(Chem.GetMolFrags(mol)) != 1 or not 1 <= row["audited_sa"] <= 5
                or not math.isfinite(row["relaxed_loss"])):
            raise ValueError("Malformed archive candidate")
        fps[row["smiles"]] = generator.GetFingerprint(mol)
    ordered = sorted(pool, key=lambda row: (row["relaxed_loss"], row["smiles"]))
    available = Counter(map(stratum, pool))
    picked, used, totals, strata = [], set(), Counter(), Counter()
    quota = {0: 3, 1: 18, 2: 3}
    for ceiling in (.65, .75, .85, 1.):
        for row in ordered:
            smi, founder, group = row["smiles"], row["founder"], stratum(row)
            if smi in used or totals[founder] >= quota[founder] or 2*(strata[group] + 1) > available[group]:
                continue
            similarity = max(DataStructs.BulkTanimotoSimilarity(fps[smi], [fps[r["smiles"]] for r in picked]), default=0.)
            if similarity > ceiling:
                continue
            picked.append(dict(row, max_similarity_to_previous=similarity, selection_ceiling=ceiling))
            used.add(smi)
            totals[founder] += 1
            strata[group] += 1
    if dict(totals) != quota:
        raise ValueError("Insufficient paired founder/SA support for ranked panel")
    rng = random.Random(seed)
    result = []
    for idx, row in enumerate(picked, 1):
        choices = sorted((r for r in pool if stratum(r)==stratum(row) and r["smiles"] not in used),
                         key=lambda r:r["smiles"])
        control = rng.choice(choices)
        used.add(control["smiles"])
        for arm, point in (("ranked",row),("sampled",control)):
            result.append(dict(point, id=f"{arm}-{idx:02d}", group=arm, pair=idx))
    result.append(dict(near, id="near-miss", group="near-miss"))
    return dict(target="human BRD4", geometry_seeds=[1201,2203,3209], source_pool_size=len(pool),
                selection_seed=seed, panel=result, max_attempts=51, updates=0,
                policy="Minimum recorded relaxed contact loss per identity; founder quotas 3/18/3; "
                       "ranked diversity cutoffs 0.65/0.75/0.85/1.0; control sampled without replacement "
                       "within the same founder and half-point audited SA bin as its paired leader.")


def score_panel(args, cfg, scorer=None):
    from latentmol.design.protenix_restarts import fresh_score_records

    scorer = scorer or fresh_score_records
    source = cfg["panel"]
    counts = Counter(p["group"] for p in source)
    seeds = cfg["geometry_seeds"]
    if (counts != {"ranked":24, "sampled":24, "near-miss":1}
            or len({p["smiles"] for p in source}) != 49 or len({p["id"] for p in source}) != 49
            or seeds != [1201,2203,3209] or cfg["max_attempts"] != 51):
        raise ValueError("Invalid archive readout panel or budget")
    fresh = read(args.fresh_config)
    if fresh["geometry_seed"] != seeds[0]:
        raise ValueError("Historical conformer seed differs from archive")
    assets = (args.fresh_config.parent / fresh["assets_config"]).resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/"run.json", cfg)
    outcomes = {}
    for seed in (seeds[1],seeds[0],seeds[2]):
        # Start the one-point held-out readout first to establish scorer readiness quickly.
        points = source if seed == seeds[0] else [p for p in source if p["group"]=="near-miss"]
        root = args.output / f"seed-{seed}"
        root.mkdir()
        seed_args = copy(args)
        seed_args.fresh_config = root/"config.json"
        write(seed_args.fresh_config, fresh | dict(geometry_seed=seed,assets_config=str(assets),candidate_limit=len(points)))
        records = [dict(smiles=p["smiles"],start=p["id"],arm=p["group"],role="archive",step=p["step"],
                        source_job=p["job"],source_branch=p["branch"],audited_sa=p["audited_sa"])
                   for p in points]
        rows = scorer(seed_args, root, records)
        if (len(rows) != len(points)
                or {(r["start"],r["smiles"]) for r in rows} != {(p["id"],p["smiles"]) for p in points}):
            raise ValueError("Fresh scoring changed archive panel identities")
        for r in rows:
            value = r["loss"]
            if (r["status"]=="scored") != (value is not None) or value is not None and not math.isfinite(value):
                raise ValueError("Inconsistent fresh status or nonfinite loss")
        outcomes[seed] = rows
        if seed == seeds[1] and os.environ.get("DBMOL_LAUNCH_READY"):
            ready = Path(os.environ["DBMOL_LAUNCH_READY"])
            temporary = ready.with_suffix(".tmp")
            write(temporary, dict(job_id=os.environ["LSB_JOBID"],nonce=os.environ["DBMOL_LAUNCH_NONCE"],
                              status="models_and_inputs_ready"))
            temporary.replace(ready)
    groups = {}
    for group in ("ranked","sampled"):
        records = [r for r in outcomes[seeds[0]] if r["arm"]==group]
        losses = [r["loss"] for r in records if r["loss"] is not None]
        groups[group] = dict(attempts=len(records),successful=len(losses),failed=len(records)-len(losses),
                             at_2p3=sum(v<=2.3 for v in losses),at_2p4=sum(v<=2.4 for v in losses),
                             best=min(losses,default=None),median=median(losses) if losses else None)
    near = next(p for p in source if p["group"]=="near-miss")
    repeats = {s:next(r for r in outcomes[s] if r["start"]=="near-miss")["loss"] for s in seeds}
    holdout = [repeats[s] for s in seeds[1:]]
    summary = dict(status="completed", job_id=os.environ["LSB_JOBID"], groups=groups, near_miss=dict(
        historical=near["historical_loss"], by_seed=repeats, held_out_mean=mean(holdout) if None not in holdout else None),
        fresh_attempts=sum(map(len,outcomes.values())), optimization_updates=0)
    write(args.output/"result.json",summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    cmds=parser.add_subparsers(dest="action",required=True)
    prep=cmds.add_parser("prepare")
    for name in ("pool","audit","output"):
        prep.add_argument("--"+name,required=True,type=Path)
    score=cmds.add_parser("score")
    for name in ("config","fresh-config","fresh-data","fresh-deps","output","plan"):
        score.add_argument("--"+name,type=Path,required=True)
    score.add_argument("--plan-sha256",required=True)
    args=parser.parse_args()
    if args.action=="prepare":
        pool=list(read(args.pool).values())
        near=next(r for r in read(args.audit)["distinct_fresh_easy"] if r["pre"]<=5)
        point=dict(smiles=near["smiles"],audited_sa=near["pre"],job=near["job"],
                   branch=near["cycle"]+"/"+near["branch"],step=near["step"],historical_loss=near["loss"])
        write(args.output, make_panel(pool,point))
    else:
        if not os.environ.get("LSB_JOBID","").isdigit():
            raise RuntimeError("Scoring requires an allocated GPU")
        plan=read(args.plan)
        if (hashlib.sha256(args.config.read_bytes()).hexdigest()!=plan["config_sha256"]
                or plan["planned_fresh_attempts"]!={"panel":51} or plan["arms"]!={"panel":str(args.output)}):
            raise ValueError("Allocated archive panel differs from its plan")
        print(json.dumps(score_panel(args,read(args.config)),allow_nan=False),flush=True)


if __name__ == "__main__":
    main()
