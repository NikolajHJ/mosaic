"""Balanced fresh-pose Boltz follow-up to the fixed-context latent pilot.

Candidate selection is frozen before the first native prediction. Each unique
hard molecule receives the same CLI settings and two pose seeds. These remain
model predictions, not experimental binding observations.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import time

import yaml

from latentmol.experiments import ExperimentRun, file_sha256, write_json
from .prepare import installed_boltz_provenance


ARMS = ("target", "trust", "joint", "random_matched_displacement")
POSE_SEEDS = (17, 23)


def select_candidates(baseline: dict, attempts: list[dict], limit_per_arm: int = 2) -> dict:
    """Rank valid changed hard endpoints in each arm by frozen-context score."""
    if len(attempts) != 64 or limit_per_arm < 1:
        raise ValueError("Require the complete 16-start, four-arm pilot")
    if any(sum(row["arm"] == arm for row in attempts) != 16 for arm in ARMS):
        raise ValueError("Pilot arms do not have their declared 16 attempts")
    baseline_smiles = baseline["hard"]["smiles"]
    if not baseline_smiles:
        raise ValueError("Encoded baseline must be a valid hard molecule")
    selected = {}
    for arm in ARMS:
        eligible = []
        for row in attempts:
            if row["arm"] != arm:
                continue
            smiles = row["hard"]["smiles"]
            start = row["start_hard"]["smiles"]
            score = row["hard_projection_score"]["fixed_context_discrete_affinity_probability_binary"]
            if smiles and start and smiles not in (start, baseline_smiles) and score is not None:
                eligible.append(row)
        eligible.sort(key=lambda row: (-row["hard_projection_score"][
            "fixed_context_discrete_affinity_probability_binary"],
            row["repeat"], row["start_index"], row["hard"]["smiles"]))
        chosen, seen = [], set()
        for row in eligible:
            smiles = row["hard"]["smiles"]
            start = row["start_hard"]["smiles"]
            if smiles in seen:
                continue
            seen.add(smiles)
            chosen.append({"smiles": smiles, "start_smiles": start,
                           "repeat": row["repeat"],
                           "start_index": row["start_index"],
                           "fixed_context_binary": row["hard_projection_score"][
                               "fixed_context_discrete_affinity_probability_binary"],
                           "start_fixed_context_binary": row["start_hard_projection_score"][
                               "fixed_context_discrete_affinity_probability_binary"]})
            if len(chosen) == limit_per_arm:
                break
        selected[arm] = {"eligible_attempts": len(eligible),
                         "unique_eligible": len({row["hard"]["smiles"] for row in eligible}),
                         "chosen": chosen}
    chosen_rows = [row for group in selected.values() for row in group["chosen"]]
    return {"baseline_smiles": baseline_smiles, "by_arm": selected,
            "unique_smiles": sorted({baseline_smiles} |
                {row["smiles"] for row in chosen_rows} |
                {row["start_smiles"] for row in chosen_rows})}


def _native_prediction(smiles: str, index: int, seed: int, template: dict,
                       run_path: Path, cache: Path, on_attempt) -> dict:
    stem = f"candidate_{index:03d}_seed_{seed}"
    inp = run_path / "inputs" / f"{stem}.yaml"
    inp.parent.mkdir(exist_ok=True)
    value = json.loads(json.dumps(template))
    ligands = [entry["ligand"] for entry in value["sequences"] if "ligand" in entry]
    if len(ligands) != 1:
        raise ValueError("Template must contain exactly one ligand")
    ligands[0]["smiles"] = smiles
    inp.write_text(yaml.safe_dump(value, sort_keys=False))
    output = run_path / "predictions" / stem
    command = ["boltz", "predict", str(inp), "--cache", str(cache),
               "--out_dir", str(output), "--recycling_steps", "0",
               "--sampling_steps", "100", "--diffusion_samples", "1",
               "--sampling_steps_affinity", "100", "--diffusion_samples_affinity", "1",
               "--num_workers", "0", "--preprocessing-threads", "1",
               "--no_kernels", "--seed", str(seed),
               "--subsample_msa", "--num_subsampled_msa", "128"]
    log = run_path / "logs" / f"{stem}.log"
    log.parent.mkdir(exist_ok=True)
    affinity = output / f"boltz_results_{stem}" / "predictions" / stem / f"affinity_{stem}.json"
    record = {"smiles": smiles, "seed": seed, "stem": stem,
              "input_sha256": file_sha256(inp), "command": command,
              "status": "attempted", "exit_code": None, "log": str(log),
              "affinity_path": str(affinity), "affinity": None, "failure": None}
    on_attempt(record)
    started = time.monotonic()
    try:
        with log.open("w") as handle:
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                    check=False, text=True, timeout=1800)
        record["exit_code"] = result.returncode
        if result.returncode:
            record["failure"] = f"Boltz CLI exit {result.returncode}"
    except subprocess.TimeoutExpired:
        record["failure"] = "Boltz CLI exceeded 1800-second per-attempt limit"
    except OSError as exc:
        record["failure"] = f"Boltz CLI could not launch or write output: {exc}"
    record["elapsed_seconds"] = time.monotonic() - started
    if record["failure"]:
        record["status"] = "failed"
    elif not affinity.is_file():
        record["failure"] = "native affinity JSON missing"
        record["status"] = "failed"
    else:
        try:
            values = json.loads(affinity.read_text())
            if any(not math.isfinite(float(values[key])) for key in
                   ("affinity_probability_binary", "affinity_pred_value")):
                raise ValueError("nonfinite native affinity output")
            record["affinity"] = values
            record["affinity_sha256"] = file_sha256(affinity)
            record["status"] = "completed"
        except (KeyError, TypeError, ValueError, OSError) as exc:
            record["failure"] = f"affinity JSON unreadable: {exc}"
            record["status"] = "failed"
    return record


def _paired_native_results(selection: dict, results: list[dict]) -> dict:
    """Compare endpoints with both controls at identical pose seeds."""
    by_key = {(row["smiles"], row["seed"]): row for row in results}
    baseline = selection["baseline_smiles"]
    report = {}
    for arm in ARMS:
        report[arm] = []
        for chosen in selection["by_arm"][arm]["chosen"]:
            smiles, start = chosen["smiles"], chosen["start_smiles"]
            paired = []
            for seed in POSE_SEEDS:
                candidate = by_key[(smiles, seed)]["affinity"]
                base = by_key[(baseline, seed)]["affinity"]
                initial = by_key[(start, seed)]["affinity"]
                if candidate is None or base is None or initial is None:
                    paired.append({"seed": seed, "success": False,
                                   "endpoint_success": candidate is not None,
                                   "baseline_success": base is not None,
                                   "start_success": initial is not None})
                    continue
                paired.append({"seed": seed, "success": True,
                               "endpoint_binary": candidate["affinity_probability_binary"],
                               "baseline_binary": base["affinity_probability_binary"],
                               "start_binary": initial["affinity_probability_binary"],
                               "endpoint_minus_baseline": candidate["affinity_probability_binary"]
                                                          - base["affinity_probability_binary"],
                               "endpoint_minus_start": candidate["affinity_probability_binary"]
                                                       - initial["affinity_probability_binary"],
                               "endpoint_continuous": candidate["affinity_pred_value"],
                               "baseline_continuous": base["affinity_pred_value"],
                               "start_continuous": initial["affinity_pred_value"]})
            available = [row for row in paired if row["success"]]
            report[arm].append({"smiles": smiles, "start_smiles": start,
                                "repeat": chosen["repeat"], "start_index": chosen["start_index"],
                                "paired_seeds": paired, "successful_triples": len(available),
                                "mean_endpoint_minus_baseline":
                                sum(row["endpoint_minus_baseline"] for row in available) / len(available)
                                if available else None,
                                "mean_endpoint_minus_start":
                                sum(row["endpoint_minus_start"] for row in available) / len(available)
                                if available else None,
                                "positive_vs_both_on_both_seeds":
                                len(available) == len(POSE_SEEDS) and all(
                                    row["endpoint_minus_baseline"] > 0 and row["endpoint_minus_start"] > 0
                                    for row in available)})
    return report


def run_native_rescore(pilot_path: Path, template_path: Path, cache: Path,
                       run_root: Path, *, mode: str) -> Path:
    parent = json.loads((pilot_path / "run.json").read_text())
    if parent.get("status") != "completed" or parent.get("name") != f"{mode}-boltz-latent-pilot":
        raise ValueError("Parent pilot is not completed or mode does not match")
    summary = json.loads((pilot_path / "summary.json").read_text())
    if summary["mode"] != mode or not summary["binary_gradient_gate"]["passed"]:
        raise ValueError("Parent pilot summary or binary gradient gate failed")
    attempts_path = pilot_path / "final_attempts.json"
    baseline_path = pilot_path / "baseline.json"
    attempts = json.loads(attempts_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    selection = select_candidates(baseline, attempts)
    if not (selection["by_arm"]["target"]["chosen"] or
            selection["by_arm"]["joint"]["chosen"]):
        raise ValueError("No target-guided valid changed hard candidate to rescore")
    template = yaml.safe_load(template_path.read_text())
    msa_paths = [Path(entry["protein"]["msa"]) for entry in template["sequences"]
                 if "protein" in entry]
    if len(msa_paths) != 1 or not msa_paths[0].is_file():
        raise ValueError("Template must reference one available protein MSA")
    native_provenance = installed_boltz_provenance()
    config = {"mode": mode, "parent_run": parent["run_id"],
              "parent_source_archive_sha256": parent["source_archive_sha256"],
              "parent_checkpoint_sha256": parent["config"]["checkpoint_sha256"],
              "parent_prepared_sha256": parent["config"]["prepared_sha256"],
              "attempts_sha256": file_sha256(attempts_path),
              "baseline_sha256": file_sha256(baseline_path),
              "template_sha256": file_sha256(template_path),
              "msa_sha256": file_sha256(msa_paths[0]),
              "confidence_checkpoint_sha256": file_sha256(cache / "boltz2_conf.ckpt"),
              "affinity_checkpoint_sha256": file_sha256(cache / "boltz2_aff.ckpt"),
              "boltz_provenance": native_provenance,
              "pose_seeds": list(POSE_SEEDS), "sampling_steps": 100,
              "sampling_steps_affinity": 100, "diffusion_samples": 1,
              "diffusion_samples_affinity": 1, "structure_recycling_steps": 0,
              "native_affinity_recycling_steps": 5,
              "native_precision": "bf16-mixed",
              "subsample_msa": True, "num_subsampled_msa": 128,
              "selection": "top two unique valid changed endpoints per arm by fixed-context binary score"}
    with ExperimentRun(run_root, f"{mode}-native-fresh-pose", config, Path.cwd()) as run:
        write_json(run.path / "frozen_selection.json", selection)
        results = []
        for index, smiles in enumerate(selection["unique_smiles"]):
            for seed in POSE_SEEDS:
                def record_attempt(attempt):
                    results.append(attempt.copy())
                    write_json(run.path / "native_results.json", results)

                row = _native_prediction(smiles, index, seed, template, run.path,
                                         cache, record_attempt)
                results[-1] = row
                write_json(run.path / "native_results.json", results)
                run.log({"stage": "native_prediction", "index": index,
                         "smiles": smiles, "seed": seed,
                         "exit_code": row["exit_code"], "failure": row["failure"],
                         "affinity": row["affinity"]})
        success = sum(row["affinity"] is not None for row in results)
        comparisons = _paired_native_results(selection, results)
        per_arm = {arm: {"original_attempts": 16,
                         "eligible_changed_attempts": selection["by_arm"][arm]["eligible_attempts"],
                         "unique_eligible": selection["by_arm"][arm]["unique_eligible"],
                         "selected_endpoints": len(selection["by_arm"][arm]["chosen"]),
                         "selected_endpoint_molecules_confirmed_vs_both_seeds": sum(
                             row["positive_vs_both_on_both_seeds"] for row in comparisons[arm]),
                         "selected_endpoint_molecules_with_incomplete_triples": sum(
                             row["successful_triples"] < len(POSE_SEEDS) for row in comparisons[arm]),
                         "not_selected_changed_attempts": max(0,
                             selection["by_arm"][arm]["eligible_attempts"]
                             - len(selection["by_arm"][arm]["chosen"])),
                         "invalid_unchanged_or_invalid_start_attempts":
                         16 - selection["by_arm"][arm]["eligible_attempts"]}
                   for arm in ARMS}
        write_json(run.path / "summary.json", {
            "mode": mode, "unique_molecules": len(selection["unique_smiles"]),
            "prediction_attempts": len(results), "successful_predictions": success,
            "failed_predictions": len(results) - success,
            "per_arm": per_arm, "paired_native_comparison": comparisons,
            "claim_limit": "Exploratory native Boltz predictions with newly generated poses, not observed binding or calibrated selectivity."})
    return run.path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("ae", "vae"), required=True)
    args = parser.parse_args()
    print(run_native_rescore(args.pilot, args.template, args.cache,
                             args.run_root, mode=args.mode))


if __name__ == "__main__":
    main()
