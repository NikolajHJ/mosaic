"""Confirm a frozen archive panel on two held-out geometries, then route confirmed identities."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import mean


def read(path):
    return path.read_text(encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def confirm(args, panel, scorer=None):
    from rdkit import Chem
    from latentmol.design.protenix_conformer_rescore import score_geometry_seed
    from latentmol.design.protenix_hit_confirmation import DISCOVERY_SEED, HELD_OUT_SEEDS
    from latentmol.design.protenix_restarts import fresh_score_records
    from latentmol.evaluation.synthesis_accessibility import audited_sa_score

    fresh = json.loads(read(args.fresh_config))
    assets = (args.fresh_config.parent / fresh["assets_config"]).resolve(strict=True)
    if (fresh["geometry_seed"] != DISCOVERY_SEED or panel["geometry_seeds"] != list(HELD_OUT_SEEDS)
            or panel["fresh_config_sha256"] != digest(args.fresh_config) or panel["assets_sha256"] != digest(assets)):
        raise ValueError("Frozen discovery protocol and held-out configuration differ")
    records = []
    for row in panel["hits"]:
        mol = Chem.MolFromSmiles(row["smiles"])
        source = row["discovery"]
        receipt = Path(source["receipt"])
        if (digest(receipt) != source["sha256"] or mol is None or len(Chem.GetMolFrags(mol)) != 1
                or abs(audited_sa_score(mol) - row["audited_sa"]) > 1e-6 or row["audited_sa"] > 5):
            raise ValueError(f"Source receipt or hard-molecule SA differs for {row['id']}")
        matched = []
        for branch in json.loads(read(receipt))["branches"]:
            if branch["name"] == source["branch"] and branch["arm"] == source["arm"]:
                candidates = branch["candidates"]
                groups = candidates.values() if isinstance(candidates, dict) else [candidates]
                matched = [r for group in groups for r in group if r["smiles"] == row["smiles"]
                           and r["step"] == source["step"] and r["status"] == "scored"]
        if (not matched or any(r["loss"] != source["fresh_loss"] for r in matched)
                or source["fresh_loss"] > 2.3):
            raise ValueError(f"Discovery hit could not be verified for {row['id']}")
        records.append(dict(start=row["id"], arm="held-out", role="archive", smiles=row["smiles"],
                            step=source["step"], discovery=source, audited_sa=row["audited_sa"]))
    if (not records or len({r["smiles"] for r in records}) != len(records)
            or len({r["start"] for r in records}) != len(records)):
        raise ValueError("Expected distinct frozen identities and IDs")
    plan = json.loads(read(args.plan))
    if (digest(args.plan) != args.plan_sha256 or plan["config_sha256"] != digest(args.panel)
            or plan["planned_fresh_attempts"] != {"panel": 2*len(records)}
            or plan["arms"] != {"panel": str(args.output)}):
        raise ValueError("Archive input, output or scoring budget differs from the launch plan")
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "panel.json", panel)
    scores = {}
    for seed in HELD_OUT_SEEDS:
        scores[seed] = score_geometry_seed(args, args.output / f"seed-{seed}", fresh, str(assets), seed,
                                            records, scorer or fresh_score_records)
        if seed == HELD_OUT_SEEDS[0] and (ready := os.environ.get("DBMOL_LAUNCH_READY")):
            from latentmol.design.latent_protenix import _json
            _json(Path(ready), dict(status="models_and_inputs_ready", job_id=os.environ["LSB_JOBID"],
                                    nonce=os.environ["DBMOL_LAUNCH_NONCE"]))
    rows = []
    for row in panel["hits"]:
        by_seed = {seed: items[row["id"]] for seed, items in scores.items()}
        losses = [r["loss"] for r in by_seed.values() if r["loss"] is not None]
        complete = len(losses) == len(HELD_OUT_SEEDS)
        rows.append(row | dict(by_seed=by_seed, held_out_mean=mean(losses) if complete else None,
                    held_out_range=[min(losses), max(losses)] if complete else None,
                    confirmed=(max(losses) <= 2.3) if complete else None))
    result = dict(status="completed", job_id=os.environ["LSB_JOBID"], panel_sha256=digest(args.panel),
                  fresh_attempts=len(records)*2, fresh_successes=sum(r["status"] == "scored"
                  for items in scores.values() for r in items.values()), optimization_updates=0, hits=rows)
    save(args.output / "result.json", result)
    return result


def routes(args, panel):
    from synthesis_route_readout import search

    previous = json.loads(read(args.confirmation / "result.json"))
    if (previous["status"] != "completed" or previous["panel_sha256"] != digest(args.panel)
            or previous["job_id"] != args.gpu_job
            or [r["smiles"] for r in previous["hits"]] != [r["smiles"] for r in panel["hits"]]):
        raise ValueError("Geometry receipt does not match the archive panel and expected job")
    selected = [r for r in previous["hits"] if r["confirmed"] is True]
    args.output.mkdir(parents=True, exist_ok=False)
    key = [{k: r[k] for k in ("id", "smiles", "discovery", "audited_sa")} for r in selected]
    key += [dict(id="M06", label="aspirin control", smiles="CC(=O)Oc1ccccc1C(=O)O"),
            dict(id="M24", label="paracetamol control", smiles="CC(=O)Nc1ccc(O)cc1")] if selected else []
    save(args.output / "key.json", key)
    blind = args.output / "blind.json"
    save(blind, [{k: r[k] for k in ("id", "smiles")} for r in key])
    if key:
        search(blind, args.assets, args.output / "readout", 0, 1, save_routes=True)
    scores = [r | json.loads(read(args.output / "readout" / (r["id"] + ".json"))) for r in key]
    result = dict(status="completed" if key else "no_confirmed_candidates",
                  job_id=os.environ["LSB_JOBID"], confirmation_job=args.gpu_job,
                  candidates_and_controls=scores)
    save(args.output / "result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    conf = commands.add_parser("confirm")
    route = commands.add_parser("routes")
    for cmd in (conf, route):
        for name in ("panel", "output"):
            cmd.add_argument("--"+name, type=Path, required=True)
    for name in ("fresh-config", "fresh-deps", "fresh-data", "plan"):
        conf.add_argument("--"+name, type=Path, required=True)
    conf.add_argument("--plan-sha256", required=True)
    route.add_argument("--confirmation", type=Path, required=True)
    route.add_argument("--assets", type=Path, required=True)
    route.add_argument("--gpu-job", required=True)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID", "").isdigit():
        raise RuntimeError("Use an allocated worker")
    panel = json.loads(read(args.panel))
    print(json.dumps((confirm if args.action == "confirm" else routes)(args, panel),
                     indent=2, allow_nan=False), flush=True)
