"""UNTESTED independent fresh-pose Boltz follow-up for active-set endpoints.

Endpoint selection is frozen before any native prediction. Each valid selected
hard molecule is predicted with the same target/MSA, settings and two seeds.
Invalid endpoints remain in the report and are never silently repaired.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import yaml

from latentmol.experiments import ExperimentRun, file_sha256, write_json
from latentmol.config_identity import verify_config_identity
from .contracts import AFFINITY_SHA256
from .native_rescore import POSE_SEEDS, _native_prediction
from .prepare import installed_boltz_provenance
from .trajectory_terminal import read_arm_terminal, terminal_evidence_paths


ARMS = ("active_set", "fixed_initial_mask")


def _read_candidates(path: Path, spec_sha256: str, steps: int) -> list[dict]:
    """Recompute saved chemistry from labels, including deliberately invalid draws."""
    import torch
    from latentmol.chemistry import inspect_graph
    from latentmol.multimodel.study import hard_digest
    from latentmol.representation import GraphSpec, HardGraphBatch

    rows, seen = [], set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        spec = GraphSpec.from_dict(row["graph_spec"])
        if spec.fingerprint() != spec_sha256:
            raise ValueError("Candidate schema differs from the generating run")
        labels = {key: torch.as_tensor(row["labels"][key])
                  for key in ("atoms", "charges", "bonds")}
        shapes = {"atoms": (spec.max_nodes,), "charges": (spec.max_nodes,),
                  "bonds": (spec.max_nodes, spec.max_nodes)}
        if any(value.dtype != torch.int64 or tuple(value.shape) != shapes[key]
               for key, value in labels.items()):
            raise ValueError("Candidate categorical labels changed type/shape")
        hard = HardGraphBatch(**{key: value[None] for key, value in labels.items()})
        if hard_digest(hard) != row["graph_sha256"]:
            raise ValueError("Candidate graph hash differs from saved labels")
        if (type(row["step"]) is not int or not 0 <= row["step"] <= steps
                or type(row["draw"]) is not int or row["draw"] < 0):
            raise ValueError("Candidate step/draw is outside the declared budget")
        expected = f"{path.parent.name}-s{row['step']:04d}-d{row['draw']:03d}"
        if (row["candidate_id"] != expected or row["arm"] != path.parent.name
                or expected in seen
                or row["kind"] != ("argmax" if row["draw"] == 0 else "categorical")):
            raise ValueError("Candidate id/arm/kind is inconsistent or duplicated")
        seen.add(expected)
        inspection = inspect_graph(labels["atoms"], labels["charges"], labels["bonds"], spec)
        if (inspection.smiles != row["smiles"] or inspection.failure_reason != row["failure"]
                or int((labels["atoms"] != 0).sum()) != row["heavy_atoms"]):
            raise ValueError("Candidate SMILES/diagnostic differs from the raw graph")
        rows.append(row)
    return rows


def select_endpoints(pilot: Path) -> dict:
    """Use only the predeclared argmax endpoint per arm, with no score ranking."""
    run = json.loads((pilot / "run.json").read_text())
    if run.get("status") != "completed":
        raise ValueError("Parent active-set run is not complete")
    config = run["config"]
    arms = tuple(config.get("arms", ARMS))
    if not arms or len(set(arms)) != len(arms) or any(arm not in ARMS for arm in arms):
        raise ValueError("Parent declares unsupported arms")
    config_identity = verify_config_identity(run, allow_legacy_active_set_contexts=True)
    steps = config["optimizer"]["steps"]
    if type(steps) is not int or steps < 1:
        raise ValueError("Parent optimization budget is invalid")
    summary = json.loads((pilot / "summary.json").read_text())
    if summary.get("frozen_state_unchanged") is not True or set(summary.get("arms", {})) != set(arms):
        raise ValueError("Parent lacks complete matched arms or frozen-state verification")
    hashes = {arm: file_sha256(pilot / arm / "candidates.jsonl") for arm in arms}
    if summary.get("candidate_file_sha256") != hashes:
        raise ValueError("Parent candidate files changed after the completed pilot")
    selection = {"pilot_run_id": run["run_id"], "steps": steps, "arms": {},
                  "configuration_identity": config_identity}
    terminal_hashes = {
        str(path.relative_to(pilot)): file_sha256(path)
        for arm in arms for path in terminal_evidence_paths(pilot / arm, config)}
    for arm in arms:
        arm_summary = json.loads((pilot / arm / "summary.json").read_text())
        gates = json.loads((pilot / arm / "gates.json").read_text())
        if (arm_summary != summary["arms"][arm] or arm_summary.get("arm") != arm
                or arm_summary.get("gated_contexts") != len(gates) or not gates
                or any(gate.get("passed") is not True or gate.get("native_seed_parity") is not True
                       for gate in gates.values())):
            raise ValueError(f"{arm}: parent arm completion or gradient/native-parity gate failed")
        # A recorded Armijo stall is a terminal result at its true saved step,
        # while the original legacy path still requires the full update budget.
        # This never selects an intermediate checkpoint by its score.
        termination = read_arm_terminal(pilot / arm, config, arm_summary)
        endpoint_step = termination["endpoint_step"]
        if ("step-0000" not in gates
                or any(not re.fullmatch(r"step-[0-9]{4,}", key)
                        or int(key[5:]) > endpoint_step for key in gates)
                or any(not re.fullmatch(r"[0-9a-f]{64}", gate.get("context_sha256", ""))
                       for gate in gates.values())):
            raise ValueError(f"{arm}: malformed native context gate history")
        latest_gate = gates[max(gates, key=lambda key: int(key[5:]))]
        if arm_summary["endpoint"].get("context_sha256") != latest_gate["context_sha256"]:
            raise ValueError(f"{arm}: endpoint native context has no matching passed gate")
        rows = _read_candidates(pilot / arm / "candidates.jsonl", run["graph_spec_sha256"], endpoint_step)
        draws = config["diagnostics"]["hard_samples"] + 1
        if {(row["step"], row["draw"]) for row in rows} != {
                (step, draw) for step in range(endpoint_step + 1) for draw in range(draws)}:
            raise ValueError(f"{arm}: incomplete candidate step/draw budget")
        first = [row for row in rows if row["step"] == 0 and row["draw"] == 0]
        final = [row for row in rows if row["step"] == endpoint_step and row["draw"] == 0]
        if len(first) != 1 or len(final) != 1:
            raise ValueError(f"{arm}: missing unique initial or final argmax")
        if arm_summary["endpoint"].get("argmax_smiles") != final[0]["smiles"]:
            raise ValueError(f"{arm}: endpoint summary disagrees with raw graph identity")
        selection["arms"][arm] = {"start": first[0], "endpoint": final[0],
                                  "termination": termination,
                                  "endpoint_valid": bool(final[0]["smiles"]),
                                  "identity_comparable": bool(final[0]["smiles"] and first[0]["smiles"]),
                                  "endpoint_changed": (final[0]["smiles"] != first[0]["smiles"])
                                  if final[0]["smiles"] and first[0]["smiles"] else None,
                                  "hard_graph_changed": final[0]["graph_sha256"] != first[0]["graph_sha256"]}
    if len({selection["arms"][arm]["start"]["graph_sha256"] for arm in arms}) != 1:
        raise ValueError("Matched arms do not share an identical hard start")
    if any(file_sha256(pilot / arm / "candidates.jsonl") != hashes[arm] for arm in arms):
        raise ValueError("Parent candidate files changed during endpoint selection")
    selection["parent_artifact_sha256"] = {
        "run.json": file_sha256(pilot / "run.json"),
        "summary.json": file_sha256(pilot / "summary.json"),
        **{f"{arm}/{name}": file_sha256(pilot / arm / name)
            for arm in arms for name in ("summary.json", "gates.json", "candidates.jsonl")}}
    current_terminal_paths = {str(path.relative_to(pilot))
        for arm in arms for path in terminal_evidence_paths(pilot / arm, config)}
    if (current_terminal_paths != set(terminal_hashes)
            or any(file_sha256(pilot / name) != digest for name, digest in terminal_hashes.items())):
        raise ValueError("Parent terminal evidence changed during endpoint selection")
    selection["parent_artifact_sha256"].update(terminal_hashes)
    return selection


def _parent_target_identity(parent: dict) -> tuple[str, str]:
    """Use identities established from actual prepared metadata during optimization."""
    target_hash, msa_hash = parent.get("target_sequence_sha256"), parent.get("msa_sha256")
    if not all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
               for value in (target_hash, msa_hash)):
        raise ValueError("Parent lacks run-verified target/MSA identities")
    asset = parent["config"].get("context_manifest")
    if asset:
        path = Path(asset["path"])
        if file_sha256(path) != asset["sha256"]:
            raise ValueError("Parent native context catalog changed")
        catalog = json.loads(path.read_text())
        if (catalog.get("target_sequence_sha256"), catalog.get("msa_sha256")) != (target_hash, msa_hash):
            raise ValueError("Catalog header differs from the run-verified target/MSA")
    return target_hash, msa_hash


def _target_identity(template_path: Path, expected_sequence: str, expected_msa: str) -> dict:
    template = yaml.safe_load(template_path.read_text())
    if (not isinstance(template, dict) or set(template) != {"version", "sequences", "properties"}
            or template["version"] != 1):
        raise ValueError("Native template requires plain version-1 protein/ligand affinity input")
    sequences = template.get("sequences", [])
    if (len(sequences) != 2 or any(not isinstance(entry, dict) for entry in sequences)
            or sorted(tuple(entry) for entry in sequences) != [("ligand",), ("protein",)]):
        raise ValueError("Native template requires one protein and one ligand")
    proteins = [entry["protein"] for entry in sequences if "protein" in entry]
    ligands = [entry["ligand"] for entry in sequences if "ligand" in entry]
    # Extra target modifications or a simultaneous CCD declaration would evade
    # sequence-only identity or survive replacement of the endpoint SMILES.
    if (set(proteins[0]) != {"id", "sequence", "msa"}
            or set(ligands[0]) != {"id", "smiles"}
            or any(not isinstance(item["id"], str) or not item["id"] for item in (proteins[0], ligands[0]))
            or proteins[0]["id"] == ligands[0]["id"]):
        raise ValueError("Native template requires unmodified protein and unambiguous SMILES ligand")
    if template.get("properties") != [{"affinity": {"binder": ligands[0]["id"]}}]:
        raise ValueError("Native template must request affinity for the sole ligand")
    sequence_hash = hashlib.sha256(proteins[0]["sequence"].encode()).hexdigest()
    msa_path = Path(proteins[0]["msa"])
    if not msa_path.is_absolute():
        msa_path = template_path.resolve().parent / msa_path
    if sequence_hash != expected_sequence or file_sha256(msa_path) != expected_msa:
        raise ValueError("Fresh-pose template changes the prepared protein/MSA")
    proteins[0]["msa"] = str(msa_path.resolve())
    return {"template": template, "template_sha256": file_sha256(template_path),
            "msa_sha256": expected_msa, "target_sequence_sha256": sequence_hash}


def _retention_result(result: dict, identity: dict) -> dict:
    """A native scalar on standardized, changed chemistry is not retained gain."""
    same = (identity.get("parse_failure") is False
            and identity.get("standardization_failure") is None
            and identity.get("standardization_changed_identity") is False
            and bool(identity.get("native_standardized_smiles")))
    return {**result, "native_identity": identity,
            "retention_eligible": result.get("status") == "completed" and same,
            "retention_failure": None if same else "Native standardization changes or cannot verify molecular identity"}


def _paired_results(selection: dict, rows: list[dict], objective: dict) -> dict:
    """Both affinity heads and the configured signed objective use eligible pairs."""
    from latentmol.multimodel.config import Objective
    configured = Objective(**objective)
    by_key = {(row["smiles"], row["seed"]): row for row in rows}
    comparisons = {}
    for arm in selection['arms']:
        chosen = selection["arms"][arm]
        start, endpoint = chosen["start"]["smiles"], chosen["endpoint"]["smiles"]
        paired = []
        for seed in POSE_SEEDS:
            a, b = by_key.get((start, seed)), by_key.get((endpoint, seed))
            initial = a.get("affinity") if a and a.get("retention_eligible") else None
            final = b.get("affinity") if b and b.get("retention_eligible") else None
            eligible = initial is not None and final is not None
            sign = -1.0 if configured.direction == "maximize" else 1.0
            paired.append({"seed": seed, "paired_retention_eligible": eligible,
                           "start": initial, "endpoint": final,
                           "endpoint_minus_start_binary": final["affinity_probability_binary"] - initial["affinity_probability_binary"] if eligible else None,
                           "endpoint_minus_start_continuous": final["affinity_pred_value"] - initial["affinity_pred_value"] if eligible else None,
                           "endpoint_minus_start_signed_objective": sign * configured.weight * (final[configured.output] - initial[configured.output]) / configured.scale if eligible and configured.output in initial and configured.output in final else None,
                           **({"configured_objective_available": False,
                               "configured_objective_unavailable_reason": "Fresh native affinity JSON does not reproduce DBMol logits/contact/pocket/confidence reductions; report affinity metrics only."}
                              if configured.output not in ("affinity_probability_binary", "affinity_pred_value") else {})})
        comparisons[arm] = {"endpoint_valid": chosen["endpoint_valid"],
                            "endpoint_changed": chosen["endpoint_changed"],
                            "identity_comparable": bool(start and endpoint),
                            "objective": objective, "signed_objective_improvement_direction": "negative",
                            "paired_seeds": paired}
    return comparisons


def rescore(pilot: Path, template_path: Path, cache: Path, run_root: Path) -> Path:
    import os
    if not os.environ.get("LSB_JOBID") or not os.environ.get("LATENTMOL_SOURCE_SHA256"):
        raise ValueError("Native follow-up requires an HPC allocation and immutable source identity")
    selection = select_endpoints(pilot)
    candidate_hashes = {arm: file_sha256(pilot / arm / "candidates.jsonl") for arm in selection['arms']}
    parent = json.loads((pilot / "run.json").read_text())
    parent_config = parent["config"]
    target_hash, msa_hash = _parent_target_identity(parent)
    identity = _target_identity(template_path, target_hash, msa_hash)
    template = identity.pop("template")
    native = installed_boltz_provenance()
    affinity_hash = file_sha256(cache / "boltz2_aff.ckpt")
    if affinity_hash != AFFINITY_SHA256 or affinity_hash != parent_config["affinity_checkpoint"]["sha256"]:
        raise ValueError("Native rescore affinity checkpoint differs from the optimization checkpoint")
    confidence_hash = file_sha256(cache / "boltz2_conf.ckpt")
    if parent_config.get("dbmol") is not None and confidence_hash != parent_config["dbmol"]["confidence_checkpoint"]["sha256"]:
        raise ValueError("Native rescore confidence checkpoint differs from the DBMol profile")
    molecules = sorted({row[part]["smiles"]
                        for row in selection["arms"].values()
                        for part in ("start", "endpoint") if row[part]["smiles"]})
    config = {"parent_run_id": parent["run_id"],
              "parent_source_archive_sha256": parent.get("source_archive_sha256"),
              "parent_config_sha256": parent["config_sha256"],
              "selection_policy": "initial and final argmax only, no native-score ranking",
              "candidate_file_sha256": candidate_hashes,
              "selection": selection, "molecules": molecules,
              "pose_seeds": list(POSE_SEEDS), "template_identity": identity,
              "affinity_checkpoint_sha256": affinity_hash,
              "confidence_checkpoint_sha256": confidence_hash,
              "native_provenance": native,
              "scope": "fresh native Boltz predictions, not experimental binding"}
    for relative, expected in selection["parent_artifact_sha256"].items():
        if file_sha256(pilot / relative) != expected:
            raise ValueError("Parent artifacts changed after endpoint selection")
    with ExperimentRun(run_root, "boltz-active-set-fresh-pose", config, Path.cwd()) as run:
        write_json(run.path / "frozen_selection.json", selection)
        # Import the pinned native standardizer only in the HPC follow-up. The
        # same audit caught charge-driven identity collapse in the prior pilot.
        from .chemical_audit import _catalog, describe
        alert_catalog = _catalog()
        identities = {smiles: describe(smiles, alert_catalog) for smiles in molecules}
        write_json(run.path / "identity_audit.json", identities)
        rows = []
        for index, smiles in enumerate(molecules):
            for seed in POSE_SEEDS:
                def attempted(row):
                    rows.append(row.copy())
                    write_json(run.path / "native_results.json", rows)
                result = _native_prediction(smiles, index, seed, template, run.path,
                                            cache, attempted)
                result = _retention_result(result, identities[smiles])
                rows[-1] = result
                write_json(run.path / "native_results.json", rows)
                run.log({"stage": "native_prediction", "smiles": smiles,
                         "seed": seed, "status": result["status"],
                         "failure": result["failure"]})
        comparisons = _paired_results(selection, rows, parent_config["objective"])
        write_json(run.path / "summary.json", {
            "selection": selection, "comparisons": comparisons,
            "prediction_attempts": len(rows),
            "failed_predictions": sum(row["status"] != "completed" for row in rows),
            "identity_ineligible_predictions": sum(row["retention_failure"] is not None for row in rows),
            "retention_eligible_predictions": sum(row["retention_eligible"] for row in rows),
            "identity_audit_scope": "Pinned native SMILES standardization preflight; not an independent processed-tensor audit.",
            "claim_limit": "Native fresh-pose model predictions only; no measured binding or selectivity."})
        return run.path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    print(rescore(args.pilot, args.template, args.cache, args.run_root), flush=True)


if __name__ == "__main__":
    main()
