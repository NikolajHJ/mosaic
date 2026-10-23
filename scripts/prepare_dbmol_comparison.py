"""Materialize immutable, matched binary/full-loss configs inside a DTU job.

Use the previously authenticated start/target/context protocol; no new z draws.
This script performs numerical pocket preprocessing and must not run locally.
The new head path is unvalidated until CPU and native profile gates pass.
"""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path

import yaml

from latentmol.boltz.active_set import ActiveSetConfig
from latentmol.boltz.contracts import PreparedComplex
from latentmol.boltz.dbmol_objective import DBMolObjectiveConfig
from latentmol.boltz.dbmol_prepare import initial_pose_pocket
from latentmol.boltz.stability_study import build_config, load_protocol, verify_comparison
from latentmol.experiments import ExperimentRun, file_sha256, write_json
from latentmol.multimodel.study import verified_file


def prepare(protocol_path, expected_sha, output, *, steps=3, starts=("carbon-c13", "prior-0")):
    if not os.environ.get("LSB_JOBID") or not os.environ.get("LATENTMOL_SOURCE_SHA256"):
        raise ValueError("DBMol pocket preprocessing requires an immutable DTU HPC allocation")
    protocol_path = verified_file(protocol_path, expected_sha)
    protocol = load_protocol(protocol_path)
    verify_comparison(protocol["step_comparison"])
    if type(steps) is not int or steps < 1 or len(set(starts)) != len(starts) or not starts:
        raise ValueError("Require positive steps and unique prescribed starts")
    confidence = protocol["confidence_checkpoint"]
    verified_file(confidence["path"], confidence["sha256"])
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    config = {"protocol": str(protocol_path), "protocol_sha256": expected_sha,
              "starts": list(starts), "steps": steps, "output": str(output.resolve())}
    with ExperimentRun(output / "preparation", "dbmol-initial-pose-pocket", config, Path.cwd()) as run:
        all_starts = {s["id"]: s for s in protocol["starts"]}
        rows, pockets = [], {}
        for start_id in starts:
            start = all_starts[start_id]
            base = build_config(protocol, start, "backtracking", "fixed_initial_mask", output / "placeholder", smoke=True)
            # Resolve the historical inputs with the same parser used by jobs.
            temp = output / f"{start_id}-inputs.yaml"
            temp.write_text(yaml.safe_dump(base))
            parsed = ActiveSetConfig.load(temp)
            anchor_asset = parsed.contexts[parsed.anchor_count]
            anchor = PreparedComplex.load(verified_file(anchor_asset["path"], anchor_asset["sha256"]), "cpu")
            pocket = initial_pose_pocket(anchor)
            # Reuse an identical biological pocket if two starts share this
            # reference context. Never derive it from the prior-decoded graph.
            key = pocket["reference_context_sha256"]
            pockets.setdefault(key, {**pocket, "reference_prepared_sha256": anchor_asset["sha256"]})
            profile_raw = {"confidence_checkpoint": deepcopy(confidence),
                           "pocket_residues": pockets[key]["pocket_residues"],
                           "pocket_origin": "initial_pose_heavy_atom_8A",
                           "pocket_reference": {"prepared_sha256": anchor_asset["sha256"],
                                                "context_sha256": key}}
            profile = DBMolObjectiveConfig.from_mapping(profile_raw, output).record()
            for loss in ("binary", "dbmol"):
                for arm in ("fixed_initial_mask", "active_set"):
                    raw = build_config(protocol, start, "backtracking", arm, output / "runs" / f"{start_id}-{loss}-{arm}", smoke=True)
                    raw["name"] = f"dbmol-comparison-{start_id}-{loss}-{arm}"
                    raw["optimizer"]["steps"] = steps
                    raw["gradient_audit_interval"] = 1 if steps <= 3 else 50
                    raw["diagnostics"]["hard_samples"] = 1 if steps <= 3 else 8
                    if loss == "dbmol":
                        raw["objective"] = {"output": "dbmol_total_loss", "direction": "minimize"}
                        raw["dbmol"] = deepcopy(profile)
                    path = output / f"{start_id}-{loss}-{arm}.yaml"
                    path.write_text(yaml.safe_dump(raw, sort_keys=False))
                    resolved = ActiveSetConfig.load(path)
                    rows.append({"id": f"{start_id}-{loss}-{arm}", "start": start_id,
                                 "loss": loss, "arm": arm, "path": str(path.resolve()),
                                 "sha256": file_sha256(path), "initial_latent": resolved.initial_latent,
                                 "pocket_context_sha256": key, "steps": steps})
        write_json(output / "pockets.json", pockets)
        write_json(output / "comparison.json", {"config": config, "cases": rows,
            "pockets_sha256": file_sha256(output / "pockets.json"),
            "claim_limit": "Prepared configurations only; no passed native gradient/chemistry result"})
        run.annotate(prepared_cases=len(rows), pockets_sha256=file_sha256(output / "pockets.json"))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--starts", nargs="+", default=["carbon-c13", "prior-0"])
    a = p.parse_args()
    print(prepare(a.protocol, a.sha256, a.output, steps=a.steps, starts=tuple(a.starts)))


if __name__ == "__main__":
    main()
