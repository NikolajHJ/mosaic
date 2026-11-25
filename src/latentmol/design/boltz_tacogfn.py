"""Run a founder-matched TacoGFN comparison through the full Boltz2 DBMol objective."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import torch

from latentmol.design.protenix_restarts import write_json
from latentmol.design.protenix_tacogfn_study import resolve_pair, run_pair
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256
from latentmol.guidance.boltz_search import BoltzSearchScorer


def run(args, cfg, design, budget):
    if not os.environ.get("LSB_JOBID", "").isdigit():
        raise RuntimeError("The Boltz2 guidance experiment requires an allocated GPU")
    if args.output.exists():
        raise FileExistsError(args.output)
    if file_sha256(args.plan) != args.plan_sha256 or file_sha256(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("GPU plan or frozen decoder checksum differs")
    plan = json.loads(args.plan.read_text())
    if (plan["config_sha256"] != file_sha256(args.config)
            or plan["arms"][args.arm] != str(args.output)
            or plan["planned_updates"][args.arm] != budget["updates"]
            or plan["planned_fresh_attempts"][args.arm] != budget["fresh_attempts"]
            or plan["planned_native_vina_attempts"][args.arm] != budget["native_vina_attempts"]):
        raise ValueError("Paired budgets, config or output differ from the sealed GPU plan")
    prepared = cfg["prepared_config"]
    prepared_path = Path(prepared["path"])
    if file_sha256(prepared_path) != prepared["sha256"]:
        raise ValueError("Prepared Boltz config differs from the archived BRD4 setup")
    assets = json.loads(args.boltz_assets.read_text())
    native = json.loads(prepared_path.read_text())
    settings = cfg["scorer"]
    if (native["checkpoint"] != assets["checkpoint"] or native["checkpoint_sha256"] != assets["checkpoint_sha256"]
            or native["dbmol"]["confidence_checkpoint"] != assets["confidence_checkpoint"]
            or native["recycling_steps"] != assets["recycling_steps"]
            or native.get("allow_lossy_charge", True) != assets["allow_lossy_charge"]
            or settings["target_id"] != "brd4-jq1" or settings["pocket_pdb_id"] != "3MXF"):
        raise ValueError("Scoring target or Boltz weights differ from the prepared campaign")
    native["execution"] = settings["execution"]
    if native["execution"]["tf32"] or native["execution"]["autocast"] != "none":
        raise ValueError("Use the same full FP32 Boltz2 objective for every update and hard score")
    torch.set_num_threads(4)
    torch.manual_seed(cfg["tasks"][args.arm]["seed"])
    decoder, _ = load_model(args.checkpoint, "cpu")
    scorer = BoltzSearchScorer.load(native, decoder.spec, device="cuda",
                                    source_sha256=os.environ["LATENTMOL_SOURCE_SHA256"])
    try:
        metadata = scorer.prepared.metadata
        pocket = metadata["target"]["pocket"]
        if (metadata["target_id"] != settings["target_id"] or pocket["pdb_id"] != settings["pocket_pdb_id"]
                or pocket["protein_chain"] != "A" or int(scorer.prepared.active.sum()) != design.occupied_atoms):
            raise ValueError("Boltz target, pocket or ligand pose slots do not match this paired study")
        result = run_pair(decoder, lambda *_: scorer, scorer.score_valid_records, cfg, args.arm,
                          args.output, args.checkpoint_sha256)
    finally:
        shutdown = scorer.close()
    write_json(args.output / "boltz-context.json", dict(scorer=scorer.identity, frozen=shutdown,
               pocket=pocket, prepared_config=prepared, optimized_output="dbmol_total_loss",
               trajectory_graph="shared decoder probabilities", hard_scoring="full FP32 fixed-pose DBMol"))
    write_json(args.output / "result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--plan-only", action="store_true")
    paths = ("plan", "output", "checkpoint", "boltz-assets")
    hashes = ("checkpoint-sha256", "plan-sha256")
    for key in paths:
        parser.add_argument("--" + key, type=Path)
    for key in hashes:
        parser.add_argument("--" + key)
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    if cfg.get("study") != "boltz_tacogfn":
        parser.error("Expected a Boltz2 TacoGFN paired configuration")
    design, budget = resolve_pair(cfg, args.arm)
    if args.plan_only:
        print(json.dumps(dict(task=args.arm, design=asdict(design), budget=budget,
                              variants=cfg["variants"], scorer=cfg["scorer"]), indent=2))
        return
    for key in (*paths, *hashes):
        if getattr(args, key.replace("-", "_")) is None:
            parser.error(f"--{key} is required for an allocated run")
    return run(args, cfg, design, budget)


if __name__ == "__main__":
    main()
