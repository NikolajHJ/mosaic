"""HPC report of initialization sensitivity, retaining failures and missing finals.

The latent width never changes. Count changes refer to selected molecular atom
slots. Final chemical diversity uses only completed, valid argmax endpoints;
invalid/unfinished outcomes stay in the planned-seed denominator and do not get
a zero similarity or an invented final molecule.
"""

from collections import Counter
import csv
from itertools import combinations
import json
from pathlib import Path

from latentmol.experiments import file_sha256, write_json


def read_json(path):
    return json.loads(Path(path).read_text())


def _warning(warnings, path, exc, *, line=None):
    item = {"path": str(path), "error_type": type(exc).__name__, "error": str(exc)}
    if line is not None:
        item["line"] = line
    warnings.append(item)


def _observed_json(path, *, strict, warnings):
    """Keep a failed trial's readable evidence without forgiving completed data."""
    try:
        value = read_json(path)
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON mapping")
        return value
    except (OSError, ValueError, TypeError) as exc:
        if strict:
            raise
        _warning(warnings, path, exc)
        return None


def hard_composition(candidate: dict | None) -> dict | None:
    """Count saved categorical atom labels, including chemically invalid graphs.

    Atomic numbers come from this candidate's declared vocabulary, never from
    an assumed label order. Absent slots are excluded from every denominator.
    """
    if candidate is None:
        return None
    vocabulary = candidate["graph_spec"]["atomic_numbers"]
    labels = candidate["labels"]["atoms"]
    if not vocabulary or vocabulary[0] != 0 or any(
            type(label) is not int or not 0 <= label < len(vocabulary) for label in labels):
        raise ValueError("Candidate atom labels disagree with their vocabulary")
    counts = Counter(vocabulary[label] for label in labels if label != 0)
    total = sum(counts.values())
    return {"heavy_atoms": total, "carbon_atoms": counts[6],
            "carbon_fraction": counts[6] / total if total else None,
            "heteroatoms": total - counts[6],
            "atomic_number_counts": {str(number): count for number, count in sorted(counts.items())}}


def pair_tanimoto(initial: str | None, final: str | None) -> float | None:
    """No invented similarity when either endpoint has no valid identity."""
    if not initial or not final:
        return None
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    molecules = [Chem.MolFromSmiles(smiles) for smiles in (initial, final)]
    if any(molecule is None for molecule in molecules):
        raise ValueError("Reported valid start/endpoint cannot be parsed")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048,
                                                          includeChirality=False)
    return DataStructs.TanimotoSimilarity(*(generator.GetFingerprint(mol) for mol in molecules))


def arm_report(pilot: Path, arm: str) -> dict:
    folder = pilot / arm
    if not folder.exists():
        return {"status": "not_attempted", "final_smiles": None}
    parent = read_json(pilot / "run.json")
    config = parent["config"]
    strict = parent.get("status") == "completed"
    warnings = []
    metric_paths = sorted(folder.glob("metrics-*.json"))
    rows = []
    for path in metric_paths:
        row = _observed_json(path, strict=strict, warnings=warnings)
        if row is None:
            continue
        try:
            # Validate the fields consumed below before accepting a partial
            # run's observation. Malformed evidence is listed, never filled in.
            if type(row["step"]) is not int:
                raise ValueError("Metric step must be an integer")
            for selection in (row["selection"], row["natural_selection"]):
                if type(selection["count"]) is not int or not isinstance(selection["active_slots"], list):
                    raise ValueError("Malformed observed occupancy")
            for key in ("raw", "loss", "gradient_norm"):
                row["score"][key]
            for key in ("latent_displacement", "argmax_smiles", "context_sha256"):
                row[key]
        except (KeyError, TypeError, ValueError) as exc:
            if strict:
                raise
            _warning(warnings, path, exc)
            continue
        rows.append(row)
    trajectory_completed = (folder / "summary.json").exists()
    summary = None
    if trajectory_completed:
        summary = _observed_json(folder / "summary.json", strict=strict, warnings=warnings)
        try:
            if summary is None or [row["step"] for row in rows] != list(range(config["optimizer"]["steps"] + 1)):
                raise ValueError("Arm summary lacks its exact readable step budget")
            if summary["endpoint"] != rows[-1]:
                raise ValueError("Completed arm endpoint does not match its saved metrics")
        except (KeyError, TypeError, ValueError) as exc:
            if strict:
                raise
            _warning(warnings, folder / "summary.json", exc)
            trajectory_completed = False
    parent_summary = (_observed_json(pilot / "summary.json", strict=strict, warnings=warnings)
                      if (pilot / "summary.json").exists() else None) or {}
    # run_arm writes its summary before run() verifies frozen model state. A
    # later failed invariant or incomplete paired run must not promote this
    # observed endpoint into the verified final-molecule diversity sample.
    completed = bool(trajectory_completed and parent.get("status") == "completed"
                     and parent_summary.get("frozen_state_unchanged") is True
                     and parent_summary.get("arms", {}).get(arm) == summary)
    first = next((row for row in rows if row["step"] == 0), None)
    last = rows[-1] if rows else None
    count_changes = sum(a["step"] + 1 == b["step"] and a["selection"]["count"] != b["selection"]["count"]
                        for a, b in zip(rows, rows[1:]))
    same_count_slot_changes = sum(
        a["step"] + 1 == b["step"] and a["selection"]["count"] == b["selection"]["count"]
        and a["selection"]["active_slots"] != b["selection"]["active_slots"]
        for a, b in zip(rows, rows[1:]))
    candidates = folder / "candidates.jsonl"
    hard = {kind: {"attempted": 0, "valid": 0} for kind in ("argmax", "categorical")}
    argmax = {}
    if candidates.exists():
        with candidates.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    item = json.loads(line)
                    if item["kind"] not in hard or type(item["step"]) is not int:
                        raise ValueError("Malformed candidate kind/step")
                    if item["smiles"] is not None and not isinstance(item["smiles"], str):
                        raise ValueError("Malformed candidate molecular identity")
                    hard_composition(item)
                except (KeyError, TypeError, ValueError) as exc:
                    if strict:
                        raise
                    # candidates.jsonl is append-written. A timeout can leave
                    # one partial row; it must not prevent the next seed from
                    # running or turn the surviving rows into a complete budget.
                    _warning(warnings, candidates, exc, line=line_number)
                    continue
                hard[item["kind"]]["attempted"] += 1
                hard[item["kind"]]["valid"] += int(item["smiles"] is not None)
                if item["kind"] == "argmax":
                    if item["step"] in argmax:
                        raise ValueError("Duplicate argmax candidate at a saved step")
                    argmax[item["step"]] = item
    if completed:
        expected_hash = parent_summary.get("candidate_file_sha256", {}).get(arm)
        if expected_hash is None or file_sha256(candidates) != expected_hash:
            raise ValueError("Completed arm candidate file differs from its frozen parent summary")
    initial_composition = hard_composition(argmax.get(first["step"])) if first else None
    last_composition = hard_composition(argmax.get(last["step"])) if last else None
    counts = [row["selection"]["count"] for row in rows]
    natural_counts = [row["natural_selection"]["count"] for row in rows]
    clipped = sum(row["score"]["gradient_norm"] is not None
                  and row["score"]["gradient_norm"] > config["optimizer"]["max_gradient_norm"]
                  for row in rows)
    trace = [{"step": row["step"], "raw_score": row["score"]["raw"],
              "signed_loss": row["score"]["loss"], "count": row["selection"]["count"],
              "natural_count": row["natural_selection"]["count"],
              "latent_displacement": row["latent_displacement"],
              "gradient_norm": row["score"]["gradient_norm"],
              "argmax_smiles": row["argmax_smiles"],
              "context_sha256": row["context_sha256"]} for row in rows]
    rejected = [value for path in sorted(folder.glob("rejected-transition-*.json"))
                if (value := _observed_json(path, strict=strict, warnings=warnings)) is not None]
    return {
        "status": "completed" if completed else (
            "trajectory_complete_unverified" if trajectory_completed else "failed_or_partial"),
        "trajectory_completed": trajectory_completed, "verified_completed": completed,
        "unreadable_artifacts": warnings,
        "metrics_states": len(rows), "initial_count": first["selection"]["count"] if first else None,
        "last_observed_count": last["selection"]["count"] if last else None,
        "final_count": last["selection"]["count"] if completed else None,
        "initial_smiles": first["argmax_smiles"] if first else None,
        "last_observed_smiles": last["argmax_smiles"] if last else None,
        "final_smiles": last["argmax_smiles"] if completed else None,
        "initial_composition": initial_composition,
        "last_observed_composition": last_composition,
        "final_composition": last_composition if completed else None,
        "initial_final_tanimoto": pair_tanimoto(first["argmax_smiles"], last["argmax_smiles"])
                                  if completed else None,
        "observed_count_range": [min(counts), max(counts)] if counts else None,
        "maximum_absolute_count_deviation_from_start": max(
            abs(count - first["selection"]["count"]) for count in counts) if first else None,
        "final_count_minus_initial": counts[-1] - counts[0] if completed else None,
        "observed_natural_count_range": [min(natural_counts), max(natural_counts)] if natural_counts else None,
        "maximum_absolute_natural_count_deviation_from_start": max(
            abs(count - first["natural_selection"]["count"]) for count in natural_counts) if first else None,
        "accepted_count_change_events": count_changes,
        "same_count_slot_change_events": same_count_slot_changes,
        "rejected_transitions": rejected,
        "gradient_clipped_updates": clipped,
        "trust_boundary_states": sum(row["latent_displacement"] >= config["optimizer"]["trust_radius"] - 1e-5
                                     for row in rows),
        "hard_graph_diagnostics": hard,
        "best_proxy_step_descriptive_only": min(rows, key=lambda row: row["score"]["loss"])["step"] if rows else None,
        "trace": trace,
        "artifacts": {str(path.relative_to(pilot)): file_sha256(path)
                      for path in (pilot / "run.json", pilot / "summary.json",
                                   folder / "summary.json", candidates, *metric_paths) if path.exists()},
    }


def molecular_diversity(entries: list[dict]) -> dict:
    """Morgan/Tanimoto comparisons among valid chemical identities only."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048,
                                                          includeChirality=False)
    fingerprints = []
    for item in entries:
        molecule = Chem.MolFromSmiles(item["smiles"])
        if molecule is None:
            raise ValueError("Reported valid endpoint cannot be parsed")
        fingerprints.append(generator.GetFingerprint(molecule))
    pairs = [{"seed_a": entries[a]["seed"], "seed_b": entries[b]["seed"],
              "tanimoto": DataStructs.TanimotoSimilarity(fingerprints[a], fingerprints[b]),
              "identical_smiles": entries[a]["smiles"] == entries[b]["smiles"]}
             for a, b in combinations(range(len(entries)), 2)]
    return {"valid_endpoints": entries, "unique_smiles": len({row["smiles"] for row in entries}),
            "valid_endpoint_count": len(entries), "pair_count": len(pairs), "pairs": pairs,
            "mean_pairwise_tanimoto": sum(row["tanimoto"] for row in pairs) / len(pairs) if pairs else None,
            "fingerprint": "RDKit Morgan radius2, 2048 bits; stereochemistry excluded by graph schema"}


def write_report(study_path: Path, trials: list[dict], *, reference: Path | None = None) -> dict:
    """Rebuild a partial/final report without loading or evaluating any model.

    Pass the entire declared trial plan, including pending/not-attempted seeds,
    on every call. Its denominator must not shrink to successful attempts.
    """
    seeds = [trial["seed"] for trial in trials]
    if any(type(seed) is not int for seed in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("Trial plan requires distinct integer seeds")
    rows = []
    trajectories = []
    for trial in trials:
        row = dict(trial)
        pilot = Path(trial["pilot"]) if trial.get("pilot") else None
        record = None
        row["arms"] = {}
        if pilot and (pilot / "run.json").exists():
            record = read_json(pilot / "run.json")
            declared_start = record["config"].get("start", {})
            if declared_start.get("kind") != "standard_normal" or declared_start.get("seed") != trial["seed"]:
                raise ValueError("Trial seed differs from its declared prior initialization")
            row["pilot_status"] = record["status"]
            row["pilot_error"] = record.get("error")
            warnings = []
            audit = pilot / "start-audit.json"
            row["start_audit"] = _observed_json(audit, strict=record["status"] == "completed", warnings=warnings) if audit.exists() else None
            row["start_file_sha256"] = file_sha256(pilot / "start.pt") if (pilot / "start.pt").exists() else None
            row["pilot_run_json_sha256"] = file_sha256(pilot / "run.json")
            for arm in ("active_set", "fixed_initial_mask"):
                row["arms"][arm] = arm_report(pilot, arm)
                for event in row["arms"][arm].get("trace", []):
                    trajectories.append({"seed": trial["seed"], "arm": arm, **event})
            costs = pilot / "oracle_cost.json"
            row["oracle_cost"] = _observed_json(costs, strict=record["status"] == "completed", warnings=warnings) if costs.exists() else None
            row["unreadable_artifacts"] = warnings
        rescore = Path(trial["rescore"]) if trial.get("rescore") else None
        if rescore and (rescore / "summary.json").exists():
            rescore_record = read_json(rescore / "run.json")
            if record is None or rescore_record["config"].get("parent_run_id") != record["run_id"]:
                raise ValueError("Native confirmation refers to a different pilot")
            row["native_confirmation_status"] = rescore_record.get("status")
            row["native_summary_sha256"] = file_sha256(rescore / "summary.json")
            if rescore_record.get("status") == "completed":
                if record.get("status") != "completed":
                    raise ValueError("Completed native confirmation has an incomplete parent")
                value = read_json(rescore / "summary.json")
                row["native_confirmation"] = {key: value[key] for key in (
                    "comparisons", "prediction_attempts", "failed_predictions", "retention_eligible_predictions")}
        rows.append(row)
    arms = {}
    for arm in ("active_set", "fixed_initial_mask"):
        records = [(row["seed"], row["arms"].get(arm, {"status": "not_attempted", "final_smiles": None}))
                   for row in rows]
        valid = [{"seed": seed, "smiles": item["final_smiles"]} for seed, item in records
                 if item["status"] == "completed" and item.get("final_smiles")]
        arms[arm] = {
            "planned_seeds": len(trials),
            "status_counts": dict(Counter(item["status"] for _, item in records)),
            "completed_valid_endpoints_per_planned_seed": len(valid) / len(trials) if trials else None,
            "seeds_with_observed_count_changes": [seed for seed, item in records if item.get("accepted_count_change_events", 0)],
            "diversity_among_completed_valid_endpoints": molecular_diversity(valid),
        }
    reference_record = None
    if reference and (reference / "run.json").exists():
        original = read_json(reference / "run.json")
        reference_record = {"path": str(reference), "status": original["status"],
                            "source_archive_sha256": original.get("source_archive_sha256"),
                            "run_json_sha256": file_sha256(reference / "run.json"), "arms": {}}
        if original["status"] == "completed":
            for arm in arms:
                reference_arm = arm_report(reference, arm)
                reference_record["arms"][arm] = reference_arm
                if reference_arm["final_smiles"]:
                    items = arms[arm]["diversity_among_completed_valid_endpoints"]["valid_endpoints"]
                    reference_record["arms"][arm]["comparison_to_random_endpoints"] = molecular_diversity(
                        [{"seed": "encoded_C13_reference", "smiles": reference_arm["final_smiles"]}, *items])
    result = {"planned_seed_count": len(trials), "trial_status_counts": dict(Counter(row["status"] for row in rows)),
              "trials": rows, "arms": arms, "encoded_reference": reference_record,
              "claim_limit": "Exploratory initialization study; validity/similarity are conditional diagnostics, not binding. Carbon reference poses remain. Latent width stays fixed."}
    write_json(study_path / "initialization_report.json", result)
    if trajectories:
        with (study_path / "trajectories.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(trajectories[0]))
            writer.writeheader()
            writer.writerows(trajectories)
    return result
