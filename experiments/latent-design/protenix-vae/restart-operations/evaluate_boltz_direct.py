"""Redock one direct-graph pilot seed's founder and DBMol-ranked descendants on BRD4."""
import argparse
from functools import lru_cache
import json
import math
from pathlib import Path
from statistics import median

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from latentmol.design.protenix_vina_parents import VinaParentPanel
from latentmol.evaluation.synthesis_accessibility import audited_sa_score
from latentmol.experiments import file_sha256


_CARBONYL = Chem.MolFromSmarts("[CX3]=[OX1]")
_DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "restart-boltz-brd4-direct.json"


@lru_cache(maxsize=65536)
def chemistry(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or not molecule.GetNumAtoms() or len(Chem.GetMolFrags(molecule)) != 1:
        raise ValueError(f"Unexpected molecule in the valid archive: {smiles}")
    return dict(smiles=Chem.MolToSmiles(molecule), carbonyl=molecule.HasSubstructMatch(_CARBONYL),
                aromatic_rings=rdMolDescriptors.CalcNumAromaticRings(molecule),
                triangles=sum(len(ring) == 3 for ring in molecule.GetRingInfo().AtomRings()),
                audited_sa=audited_sa_score(molecule))


def valid_chemistry(folder, top, founder):
    path = folder / "metrics.jsonl"
    if path.is_file():
        with path.open(encoding="utf-8") as stream:
            names = {smiles for line in stream if (smiles := json.loads(line)["smiles"])}
        scope = "all_valid_steps"
    else:
        names = {row["smiles"] for row in top}
        scope = "reported_top20_only"
    records = {chemistry(name)["smiles"]: chemistry(name) for name in names}
    records.pop(founder, None)
    rows = list(records.values())
    return dict(scope=scope, distinct_new=len(rows),
                with_carbonyl=sum(r["carbonyl"] for r in rows),
                with_aromatic_ring=sum(r["aromatic_rings"] > 0 for r in rows),
                median_triangles=median(r["triangles"] for r in rows) if rows else None,
                median_audited_sa=median(r["audited_sa"] for r in rows) if rows else None)


def evaluate(run, source, docking, *, per_arm=2):
    if source["status"] != "completed" or source["optimization_space"] != "relaxed_graph_logits":
        raise ValueError("Expected a completed direct-graph pilot")
    founder = chemistry(source["founder_smiles"])
    arms = {}
    identities = {founder["smiles"]}
    logical = 1
    for name, variant in source["variants"].items():
        if (not name.isidentifier() or variant["status"] != "completed" or
                variant["seed"] != source["seed"] or chemistry(variant["founder_smiles"])["smiles"] != founder["smiles"]):
            raise ValueError("Expected completed variants from the same founder and seed")
        top = variant["top20"]
        selected = {}
        scored = [r for r in top if r["loss"] is not None]
        if any(not math.isfinite(r["loss"]) for r in scored):
            raise ValueError("Received a nonfinite hard DBMol loss")
        for row in sorted(scored, key=lambda r: (r["loss"], r["smiles"], r["step"])):
            props = chemistry(row["smiles"])
            smi = props["smiles"]
            if smi != founder["smiles"] and smi not in selected:
                selected[smi] = dict(smiles=smi, hard_dbmol_loss=row["loss"], step=row["step"], chemistry=props)
                if "scoring_key" in row:
                    selected[smi]["scoring_key"] = row["scoring_key"]
                if len(selected) == per_arm:
                    break
        arms[name] = dict(selected=list(selected.values()), chemistry=valid_chemistry(run / name, top, founder["smiles"]),
                          fresh_attempts=variant["fresh_attempts"], fresh_successes=variant["fresh_successes"])
        identities.update(selected)
        logical += len(selected)
    # A single external worker handles all arms. Cache both successful and failed preparations by canonical SMILES.
    scores = docking.score_many(sorted(identities))
    founder_vina = scores[founder["smiles"]]["affinity_kcal_mol"]
    for arm in arms.values():
        for row in arm["selected"]:
            row["vina"] = scores[row["smiles"]]
        successful = [r for r in arm["selected"] if r["vina"]["status"] == "ok"]
        best = min(successful, key=lambda r: r["vina"]["affinity_kcal_mol"], default=None)
        arm["docked"] = len(successful)
        arm["best_vina_smiles"] = best["smiles"] if best else None
        arm["best_vina"] = best["vina"]["affinity_kcal_mol"] if best else None
        arm["best_minus_founder_kcal_mol"] = arm["best_vina"] - founder_vina if best and founder_vina is not None else None
    return dict(status="completed", job_id=source.get("job_id"), seed=source["seed"],
                selection=f"Up to {per_arm} lowest hard DBMol losses per arm among distinct newly scored molecules",
                founder=dict(chemistry=founder, vina=scores[founder["smiles"]]), variants=arms,
                slots_requested=logical, repeated_slots=logical-len(identities), docking=docking.accounting())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vina-config", type=Path, default=_DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    source_path = args.run / "result.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    cfg = json.loads(args.vina_config.read_text(encoding="utf-8"))
    if (source["prepared_config"]["sha256"] != cfg["prepared_config"]["sha256"] or
            cfg["scorer"]["pocket_pdb_id"] != "3MXF" or cfg["vina_parent_panel"]["exhaustiveness"] != 4):
        raise ValueError("This readout requires matched BRD4 discovery and docking contexts and exhaustiveness 4")
    hotspot_readout = "pharmaconet_map" in cfg
    if hotspot_readout != ("pharmaconet_map_sha256" in source):
        raise ValueError("Mismatched guidance experiment and docking configuration")
    per_arm = cfg["fresh_budget"] if hotspot_readout else 2
    if hotspot_readout and any(v["fresh_budget"] != per_arm or len(v["top20"]) != v["fresh_successes"]
                               for v in source["variants"].values()):
        raise ValueError("Expected all successfully fresh-scored candidates within the declared budget")
    panel_cfg = dict(cfg["vina_parent_panel"], panel_size=1 + per_arm*len(source["variants"]))
    if hotspot_readout:
        panel_cfg["pose_output_dir"] = str((args.output / "poses").resolve())
    panel = VinaParentPanel(panel_cfg)
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = dict(source=str(source_path), source_sha256=file_sha256(source_path),
                   vina_config=str(args.vina_config), vina_config_sha256=file_sha256(args.vina_config))
    try:
        if hotspot_readout:
            (args.output / "poses").mkdir()
        receipt.update(evaluate(args.run, source, panel, per_arm=per_arm))
        if hotspot_readout:
            from latentmol.evaluation.pharmaconet_redock import add_hotspot_readout
            add_hotspot_readout(receipt, source, cfg, args.run, args.output)
    except Exception as exc:
        receipt.update(status="failed", error=str(exc))
        raise
    finally:
        with (args.output / "result.json").open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")


if __name__ == "__main__":
    main()
