"""Build a matched blind synthesis panel or run a fixed-budget AiZynthFinder shard."""
import argparse
import gzip
import json
import math
import random
import sys
import time
from pathlib import Path


def save(path, item):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(item, stream, indent=2)


def prepare(audit, output):
    panels = json.loads(gzip.decompress((audit/"followup-batches.json.gz").read_bytes()))
    scores = json.loads(gzip.decompress((audit/"followup-trajectory-scores.json.gz").read_bytes()))
    best = {}
    for panel in panels:
        for branch in panel["branches"]:
            for candidate in branch["candidates"]:
                if candidate["status"] != "scored" or candidate["is_founder"]:
                    continue
                smiles, loss = candidate["smiles"], candidate["loss"]
                if smiles in best and best[smiles]["fresh_loss"] <= loss:
                    continue
                native, audited = scores[smiles]
                if audited <= 5:
                    stratum = "SA <=5"
                elif audited <= 6:
                    stratum = "SA 5-6"
                elif audited <= 8:
                    stratum = "SA 6-8"
                elif native <= 5:
                    stratum = "high-side artifact"
                else:
                    continue
                best[smiles] = dict(smiles=smiles, fresh_loss=loss, sa_native=native,
                    sa_before_smoothing=audited, stratum=stratum, job=panel["job"], arm=branch["arm"],
                    cycle=branch["cycle"], branch=branch["name"], step=candidate["step"])
    panel = []
    used = set()
    centers = [2.3025, 2.34, 2.36, 2.38, 2.40, 2.44]
    for block, center in enumerate(centers, start=1):
        for stratum in ["SA <=5", "SA 5-6", "SA 6-8", "high-side artifact"]:
            choices = [v for v in best.values() if v["stratum"] == stratum and
                       v["smiles"] not in used and abs(v["fresh_loss"]-center) <= .015]
            if not choices:
                raise ValueError(f"Empty matched cell {center} {stratum}")
            chosen = min(choices, key=lambda c: (abs(c["fresh_loss"]-center), c["smiles"]))
            panel.append(dict(chosen, block=block, center=center))
            used.add(chosen["smiles"])
    for name, smiles in [("aspirin","CC(=O)Oc1ccccc1C(=O)O"),("paracetamol","CC(=O)Nc1ccc(O)cc1")]:
        panel.append(dict(smiles=smiles,label=name,stratum="external control",block=0))
    random.Random(17112026).shuffle(panel)
    for i,row in enumerate(panel,1):
        row["id"] = f"M{i:02d}"
    if len({c["smiles"] for c in panel}) != len(panel):
        raise ValueError("Duplicate chemical identity")
    output.mkdir(parents=True, exist_ok=True)
    save(output/"key.json",panel)
    save(output/"blind-panel.json",[{k:r[k] for k in ["id","smiles"]} for r in panel])


def search(panel, assets, output, shard, shards, *, save_routes=False):
    import numpy as np
    from aizynthfinder.aizynthfinder import AiZynthFinder
    data=json.loads(panel.read_text())
    config = dict(
        expansion={"uspto":[str(assets/"uspto_model.onnx"),str(assets/"uspto_unique_templates.csv.gz")],
                   "ringbreaker":[str(assets/"uspto_ringbreaker_model.onnx"),str(assets/"uspto_ringbreaker_unique_templates.csv.gz")]},
        filter={"uspto":str(assets/"uspto_filter_model.onnx")},stock={"zinc":str(assets/"zinc_stock.hdf5")},
        search={"iteration_limit":100,"time_limit":75,"max_transforms":6,
                "return_first":True,"exclude_target_from_stock":True})
    finder=AiZynthFinder(configdict=config)
    finder.expansion_policy.select(["uspto","ringbreaker"])
    finder.filter_policy.select("uspto")
    finder.stock.select("zinc")
    assert finder.config.search.exclude_target_from_stock
    output.mkdir(parents=True,exist_ok=True)
    save(output/f"settings-{shard}.json",config)
    for row in data[shard::shards]:
        seed = 881200 + int(row["id"][1:])
        random.seed(seed)
        np.random.seed(seed)
        begin=time.monotonic()
        try:
            finder.target_smiles=row["smiles"]
            finder.prepare_tree()
            finder.tree_search()
            finder.build_routes()
            routes=finder.routes.reaction_trees
            solved_routes = [tree for tree in routes if tree.is_solved]
            solved=bool(solved_routes) or bool(finder.search_stats.get("returned_first"))
            readout=dict(solved=solved,iterations=finder.search_stats["iterations"],
                         search_seconds=finder.search_stats["time"],
                         min_steps=min((len(list(t.reactions())) for t in routes if t.is_solved),default=None),
                         first_solution_iteration=finder.search_stats.get("first_solution_iteration"))
            if save_routes and solved_routes:
                tree = min(solved_routes, key=lambda t: len(list(t.reactions())))
                path = output / (row["id"] + "-route.json")
                save(path, tree.to_dict())
                readout["route_file"] = path.name
        except Exception as exc:
            readout=dict(solved=None,error=f"{type(exc).__name__}: {exc}")
        result=dict(id=row["id"],seed=seed,elapsed_seconds=time.monotonic()-begin,**readout)
        save(output/(row["id"]+".json"),result)
        print(json.dumps(result),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="action",required=True)
    build=sub.add_parser("prepare")
    build.add_argument("audit",type=Path)
    build.add_argument("output",type=Path)
    plan=sub.add_parser("search")
    plan.add_argument("panel",type=Path)
    plan.add_argument("assets",type=Path)
    plan.add_argument("output",type=Path)
    plan.add_argument("--shard",type=int,default=0)
    plan.add_argument("--shards",type=int,default=1)
    args=parser.parse_args()
    if args.action=="prepare":
        prepare(args.audit,args.output)
    else:
        if not 0 <= args.shard < args.shards:
            parser.error("Expected 0 <= shard < shards")
        search(args.panel,args.assets,args.output,args.shard,args.shards)
