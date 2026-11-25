"""Reclassify completed hard-molecule discoveries against two frozen histories, without rescoring."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


MOTIF = "[CX4]([O;H1;+0])([O;H1;+0])([#7])[#7]"


def read(path):
    return json.loads(path.read_bytes())


def summarize(rows, excluded, ceiling):
    hits = sorted((r for s, r in rows.items() if s not in excluded and r["sa"] <= ceiling and r["loss"] <= 2.3),
                  key=lambda r: (r["loss"], r["smiles"]))
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    pattern = Chem.MolFromSmarts(MOTIF)
    fps, diverse, motifs = [], [], []
    for row in hits:
        smiles = row["smiles"]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or len(Chem.GetMolFrags(mol)) != 1:
            raise ValueError("Invalid hard identity in fresh score")
        fp = generator.GetFingerprint(mol)
        if all(s <= .60 for s in DataStructs.BulkTanimotoSimilarity(fp, fps)):
            fps.append(fp)
            diverse.append(smiles)
        if mol.HasSubstructMatch(pattern):
            motifs.append(smiles)
    return dict(distinct=len(hits), diverse=len(diverse), with_motif=len(motifs),
                hits=hits, diverse_smiles=diverse, motif_smiles=motifs)


def report(root, original_path, expanded_path):
    run, result = read(root / "run.json"), read(root / "result.json")
    if result["status"] != "completed" or run["runner"] != "synthesis_waves" or run["sa_mode"] != "pre_smoothing":
        raise ValueError("Readout requires a completed synthesis wave run with audited hard SA")
    founders = {p["smiles"] for p in run["parents"]}
    sets, references = {}, {}
    for key, path in (("original", original_path), ("expanded", expanded_path)):
        raw = path.read_bytes()
        panel = json.loads(raw)
        if any(panel[field] != run[field] for field in ("checkpoint_sha256", "decoder_sha256", "graph_spec")):
            raise ValueError("Incompatible panel and search model")
        if any(str(s["job_id"]) == str(run["job_id"]) for s in panel.get("history_sources", [])):
            raise ValueError("History contains the query run itself")
        sets[key] = set(panel["historical_valid_smiles"]) | founders
        references[key] = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                                identities_including_founders=len(sets[key]), pocket=panel["pocket_sha256"])
    if (not sets["original"] <= sets["expanded"] or len(sets["original"]) != run["historical_identity_exclusions"]
            or references["original"]["pocket"] != references["expanded"]["pocket"]):
        raise ValueError("Expected the original exclusion count and a same-pocket superset")
    unique = {arm: {} for arm in result["arms"]}
    for wave in result["cycles"]:
        cycle = wave["cycle"]
        batch = read(root / f"cycle-{cycle:02d}" / "batch.json")
        if batch["native_identity"]["pocket_sha256"] != references["expanded"]["pocket"]:
            raise ValueError("Search and reference pockets differ")
        for branch in batch["branches"]:
            arm = branch["arm"]
            for row in branch["candidates"]["mixed"]:
                if row["status"] != "scored" or row["is_founder"]:
                    continue
                smi, loss, chemistry = row["smiles"], row["loss"], row["chemistry"]
                sa = chemistry["sa_score"]
                if (not chemistry["valid"] or chemistry["sa_mode"] != "pre_smoothing"
                        or not all(map(math.isfinite, (sa, loss)))):
                    raise ValueError("Invalid fresh score or audited SA")
                if smi not in unique[arm] or loss < unique[arm][smi]["loss"]:
                    unique[arm][smi] = dict(smiles=smi, sa=sa, loss=loss, wave=cycle, branch=branch["name"],
                                            step=row["step"])
    output = {}
    for arm, rows in unique.items():
        output[arm] = {}
        for ceiling in (5, 6):
            label = f"sa<={ceiling},loss<=2.3"
            comparisons = {key: summarize(rows, excluded, ceiling) for key, excluded in sets.items()}
            recorded = result["arms"][arm]["joint_quality_new_to_history"][label]
            if any(comparisons["original"][k] != recorded[k] for k in ("distinct", "diverse")):
                raise ValueError(f"Original novelty counts differ from sealed run for {arm}, {label}")
            output[arm][label] = comparisons
    confirmation = None
    if (confirmation_path := root / "confirmation/result.json").exists():
        held_out = read(confirmation_path)
        if held_out["status"] != "completed":
            raise ValueError("Incomplete held-out scoring receipt")
        confirmation = {}
        for arm in unique:
            passed = {h["smiles"] for h in held_out["candidates"]
                      if arm in h["discovery_sources"] and h["confirmed"] is True}
            strict = {r["smiles"] for r in output[arm]["sa<=5,loss<=2.3"]["original"]["hits"]}
            if len(passed) != held_out["confirmed_by_arm"][arm] or not passed <= strict:
                raise ValueError("Confirmation and discovery identities differ")
            confirmation[arm] = {key: sorted(passed - excluded) for key, excluded in sets.items()}
    return dict(job_id=run["job_id"], run=str(root), references=references, motif_smarts=MOTIF,
                diversity="Morgan radius 2, 2048 bits, Tanimoto <=0.60, lowest fresh loss first",
                arms=output, held_out_confirmed_smiles=confirmation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "original-panel", "expanded-panel", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.run.resolve()):
        parser.error("Choose an output outside the original search directory")
    result = report(args.run, args.original_panel, args.expanded_panel)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
