"""Read the three completed paired BRD4 runs and apply the prospective slot decision."""
import argparse
import hashlib
import json
from pathlib import Path

from rdkit import Chem, DataStructs

from latentmol.design.protenix_restart_selection import _morgan_fingerprints


ARMS = ("standard", "independent-slot")
STRICT = "sa<=5,loss<=2.3"
LOOSE = "sa<=6,loss<=2.3"
MOTIF = Chem.MolFromSmarts("[CX4]([O;H1;+0])([O;H1;+0])([#7])[#7]")


def read(path):
    return json.loads(path.read_bytes())


def unique_to_slot(candidates, reference):
    """Keep candidates diverse from every strict reference hit and earlier accepted candidates."""
    fingerprints = dict(_morgan_fingerprints(sorted(reference | set(candidates))))
    seen = [fingerprints[s] for s in reference]
    novel = []
    for smi in sorted(candidates, key=lambda s: (candidates[s], s)):
        fp = fingerprints[smi]
        if max(DataStructs.BulkTanimotoSimilarity(fp, seen), default=0.) <= .6:
            seen.append(fp)
            novel.append(smi)
    return novel


def seed_result(root, excluded):
    run = read(root / "run.json")
    result = read(root / "result.json")
    confirmation = read(root / "confirmation/result.json")
    if (result["status"] != "completed" or confirmation["status"] != "completed" or
            set(result["arms"]) != set(ARMS) or len(result["cycles"]) != 3):
        raise ValueError(f"Incomplete paired result in {root}")
    strict = {arm: {} for arm in ARMS}
    loose = {arm: set() for arm in ARMS}
    slot = {arm: {} for arm in ARMS}
    for wave in range(1, 4):
        for branch in read(root / f"cycle-{wave:02d}/batch.json")["branches"]:
            arm = branch["arm"]
            for row in branch["candidates"]["mixed"]:
                smi, loss, chem = row["smiles"], row["loss"], row["chemistry"]
                if not (row["status"] == "scored" and not row["is_founder"] and smi not in excluded
                        and chem["valid"] and chem["sa_mode"] == "pre_smoothing" and loss <= 2.3):
                    continue
                if chem["sa_score"] <= 6:
                    loose[arm].add(smi)
                if chem["sa_score"] <= 5:
                    strict[arm][smi] = min(strict[arm].get(smi, loss), loss)
                    if wave == 3 and branch["parent_index"] == 2:
                        slot[arm][smi] = loss
    arms = {}
    for arm in ARMS:
        saved = result["arms"][arm]
        quality = saved["joint_quality_new_to_history"]
        slot_quality = saved["final_slot"]["joint_quality_new_to_history"]
        if (quality[STRICT]["distinct"] != len(strict[arm]) or quality[LOOSE]["distinct"] != len(loose[arm])
                or slot_quality[STRICT]["distinct"] != len(slot[arm])):
            raise ValueError(f"Hard hit sets differ from the frozen runner in {root}/{arm}")
        tested = [r for r in confirmation["candidates"] if arm in r["discovery_sources"]]
        arms[arm] = dict(strict=quality[STRICT], sa6=quality[LOOSE], final_slot=saved["final_slot"],
            fresh_attempts=saved["fresh_attempts"], fresh_successes=saved["fresh_successes"],
            trajectory_valid_fraction=saved["trajectory_valid_fraction"],
            strict_motif=sum(Chem.MolFromSmiles(s).HasSubstructMatch(MOTIF) for s in strict[arm]),
            sa6_motif=sum(Chem.MolFromSmiles(s).HasSubstructMatch(MOTIF) for s in loose[arm]),
            held_out=[dict(smiles=r["smiles"], confirmed=r["confirmed"], by_seed=r["by_seed"]) for r in tested])
    reference = set(strict["standard"]) | (set(strict["independent-slot"]) - set(slot["independent-slot"]))
    return dict(seed=run["seed"], job_id=run["job_id"], arms=arms,
                new_diverse_slot_hits=unique_to_slot(slot["independent-slot"], reference))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs", nargs=3, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = read(args.config)
    panel = Path(cfg["parent_panel"]["path"]).read_bytes()
    if hashlib.sha256(panel).hexdigest() != cfg["parent_panel"]["sha256"]:
        raise ValueError("Changed historical identity reference")
    data = json.loads(panel)
    excluded = set(data["historical_valid_smiles"]) | {r["smiles"] for r in data["points"]}
    seeds = [seed_result(root, excluded) for root in args.runs]
    if sorted(s["seed"] for s in seeds) != sorted(t["seed"] for t in cfg["tasks"].values()):
        raise ValueError("Expected three distinct configured seeds")
    pooled = {a: sorted({r["smiles"] for s in seeds for r in s["arms"][a]["held_out"]
                         if r["confirmed"] is True}) for a in ARMS}
    wins = sum(bool(s["new_diverse_slot_hits"]) for s in seeds)
    advance = wins >= 2 and bool(pooled["independent-slot"]) and len(pooled["independent-slot"]) >= len(pooled["standard"])
    output = dict(status="completed", reference_sha256=cfg["parent_panel"]["sha256"], paired_seeds=seeds,
                  seeds_with_added_diverse_hit=wins, pooled_confirmed=pooled, advance_to_larger_test=advance)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(output, out, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
