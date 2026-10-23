"""Fresh-pose native Boltz comparison for one completed C13-anchor VAE pilot.

Selection uses the saved fixed-context hard endpoint before any native prediction.
The baseline and selected endpoint each receive the same native pose seeds and
100-step CLI schedule. These are model predictions, not measured binding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from latentmol.experiments import ExperimentRun, file_sha256, write_json
from .chemical_audit import _catalog, describe
from .native_rescore import POSE_SEEDS, _native_prediction
from .prepare import installed_boltz_provenance


def select_carbon_anchor(baseline: dict, attempts: list[dict]) -> dict:
    """Accept only the declared target and matched-random endpoint records."""
    if len(attempts) != 2 or {row.get("arm") for row in attempts} != {
            "target", "random_matched_displacement"}:
        raise ValueError("Require one target and one random-control endpoint")
    if baseline["hard"]["smiles"] != "CCCCCCCCCCCCC":
        raise ValueError("Native C13 baseline is not the declared carbon chain")
    target = next(row for row in attempts if row["arm"] == "target")
    random = next(row for row in attempts if row["arm"] == "random_matched_displacement")
    if target["start_hard"]["smiles"] != baseline["hard"]["smiles"]:
        raise ValueError("Target trajectory did not begin at the C13 hard graph")
    endpoint = target["hard"]["smiles"]
    if not endpoint or endpoint == baseline["hard"]["smiles"]:
        raise ValueError("No valid changed C13-anchor endpoint to rescore")
    baseline_score = baseline["hard_projection_score"][
        "fixed_context_discrete_affinity_probability_binary"]
    endpoint_score = target["hard_projection_score"][
        "fixed_context_discrete_affinity_probability_binary"]
    if baseline_score is None or endpoint_score is None:
        raise ValueError("Fixed-context hard score missing")
    random_smiles = random["hard"]["smiles"]
    if random["start_hard"]["smiles"] != baseline["hard"]["smiles"]:
        raise ValueError("Random control did not begin at the C13 hard graph")
    unique = {baseline["hard"]["smiles"], endpoint}
    if random_smiles:
        unique.add(random_smiles)
    return {"baseline_smiles": baseline["hard"]["smiles"],
            "endpoint_smiles": endpoint,
            "random_control_smiles": random_smiles,
            "fixed_context_baseline_binary": baseline_score,
            "fixed_context_endpoint_binary": endpoint_score,
            "fixed_context_random_binary": random["hard_projection_score"][
                "fixed_context_discrete_affinity_probability_binary"],
            "unique_smiles": sorted(unique),
            "selection_policy": "sole changed valid target endpoint from exactly one C13-anchor trajectory; no native-score selection"}


def run_rescore(pilot_path: Path, template_path: Path, cache: Path,
                run_root: Path) -> Path:
    parent = json.loads((pilot_path / "run.json").read_text())
    if parent.get("status") != "completed" or parent.get("name") != "vae-boltz-carbon-anchor-pilot":
        raise ValueError("Require completed VAE C13-anchor pilot")
    summary = json.loads((pilot_path / "summary.json").read_text())
    if summary.get("start_policy") != "carbon_anchor" or not summary["binary_gradient_gate"]["passed"]:
        raise ValueError("C13-anchor gradient gate did not pass")
    prepared_path = Path(parent["config"]["prepared"])
    if file_sha256(prepared_path) != parent["config"]["prepared_sha256"]:
        raise ValueError("Prepared C13 context differs from pilot identity")
    prepared_manifest = json.loads(prepared_path.with_suffix(".json").read_text())
    if (prepared_manifest.get("snapshot_sha256") != parent["config"]["prepared_sha256"]
            or prepared_manifest.get("seed_smiles") != "CCCCCCCCCCCCC"
            or not prepared_manifest.get("hard_feature_parity")):
        raise ValueError("Prepared context lacks exact native C13 parity")
    expected_template = prepared_path.parent / "input" / "carbon_chain_smoke.yaml"
    input_manifest = json.loads((prepared_path.parent / "input" / "input_manifest.json").read_text())
    if (template_path.resolve() != expected_template.resolve()
            or input_manifest["derived_yaml_sha256"] != file_sha256(template_path)):
        raise ValueError("Native template differs from prepared C13 protein/MSA input")
    if file_sha256(cache / "boltz2_aff.ckpt") != parent["config"]["affinity_checkpoint_sha256"]:
        raise ValueError("Native affinity checkpoint differs from pilot scorer")
    baseline_path = pilot_path / "baseline.json"
    attempts_path = pilot_path / "final_attempts.json"
    selection = select_carbon_anchor(json.loads(baseline_path.read_text()),
                                     json.loads(attempts_path.read_text()))
    template = yaml.safe_load(template_path.read_text())
    msa_paths = [Path(entry["protein"]["msa"]) for entry in template["sequences"]
                 if "protein" in entry]
    if len(msa_paths) != 1 or not msa_paths[0].is_file():
        raise ValueError("Native template must reference one available protein MSA")
    if file_sha256(msa_paths[0]) != input_manifest["assets"]["seq1.a3m"]["sha256"]:
        raise ValueError("Protein MSA differs from prepared C13 input")
    ligands = [entry["ligand"] for entry in template["sequences"] if "ligand" in entry]
    if len(ligands) != 1 or ligands[0]["smiles"] != selection["baseline_smiles"]:
        raise ValueError("Native template is not the C13 baseline")
    config = {"parent_run": parent["run_id"],
              "parent_source_archive_sha256": parent["source_archive_sha256"],
              "parent_checkpoint_sha256": parent["config"]["checkpoint_sha256"],
              "parent_prepared_sha256": parent["config"]["prepared_sha256"],
              "attempts_sha256": file_sha256(attempts_path),
              "baseline_sha256": file_sha256(baseline_path),
              "template_sha256": file_sha256(template_path),
              "msa_sha256": file_sha256(msa_paths[0]),
              "confidence_checkpoint_sha256": file_sha256(cache / "boltz2_conf.ckpt"),
              "affinity_checkpoint_sha256": file_sha256(cache / "boltz2_aff.ckpt"),
              "boltz_provenance": installed_boltz_provenance(),
              "pose_seeds": list(POSE_SEEDS), "sampling_steps": 100,
              "sampling_steps_affinity": 100, "diffusion_samples": 1,
              "diffusion_samples_affinity": 1, "structure_recycling_steps": 0,
              "native_affinity_recycling_steps": 5,
              "native_precision": "bf16-mixed", "subsample_msa": True,
              "num_subsampled_msa": 128,
              "selection_policy": selection["selection_policy"]}
    with ExperimentRun(run_root, "vae-carbon-anchor-native-rescore", config, Path.cwd()) as run:
        write_json(run.path / "frozen_selection.json", selection)
        catalog = _catalog()
        identities = [describe(smiles, catalog) for smiles in selection["unique_smiles"]]
        write_json(run.path / "identity_audit.json", identities)
        identity_by_input = {row["input_smiles"]: row for row in identities}
        effective = {row["input_smiles"]: row.get("native_standardized_smiles")
                     for row in identities}
        baseline_effective = effective[selection["baseline_smiles"]]
        endpoint_effective = effective[selection["endpoint_smiles"]]
        random_smiles = selection["random_control_smiles"]
        random_effective = effective.get(random_smiles) if random_smiles else None
        effective_changed_from_baseline = (baseline_effective is not None
                                           and endpoint_effective is not None
                                           and endpoint_effective != baseline_effective)
        effective_changed = (effective_changed_from_baseline and random_effective is not None
                             and endpoint_effective != random_effective)
        effective_sizes_preserved = all(row.get("native_standardized_heavy_atoms") == 13
                                        for row in identity_by_input.values())
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
                run.log({"stage": "native_prediction", "smiles": smiles,
                         "seed": seed, "failure": row["failure"],
                         "exit_code": row["exit_code"], "affinity": row["affinity"]})
        lookup = {(row["smiles"], row["seed"]): row for row in results}
        paired = []
        for seed in POSE_SEEDS:
            base = lookup[(selection["baseline_smiles"], seed)]["affinity"]
            endpoint = lookup[(selection["endpoint_smiles"], seed)]["affinity"]
            random_smiles = selection["random_control_smiles"]
            random = lookup[(random_smiles, seed)]["affinity"] if random_smiles else None
            if base is None or endpoint is None or random is None:
                paired.append({"seed": seed, "paired_success": False,
                               "baseline_success": base is not None,
                               "endpoint_success": endpoint is not None,
                               "random_success": random is not None})
            else:
                paired.append({"seed": seed, "paired_success": True,
                               "baseline_binary": base["affinity_probability_binary"],
                               "endpoint_binary": endpoint["affinity_probability_binary"],
                               "random_binary": random["affinity_probability_binary"],
                               "endpoint_minus_baseline_binary": endpoint[
                                   "affinity_probability_binary"] - base[
                                   "affinity_probability_binary"],
                               "endpoint_minus_random_binary": endpoint[
                                   "affinity_probability_binary"] - random[
                                   "affinity_probability_binary"],
                               "baseline_continuous": base["affinity_pred_value"],
                               "endpoint_continuous": endpoint["affinity_pred_value"],
                               "random_continuous": random["affinity_pred_value"],
                               "endpoint_minus_baseline_continuous": endpoint[
                                   "affinity_pred_value"] - base["affinity_pred_value"],
                               "endpoint_minus_random_continuous": endpoint[
                                   "affinity_pred_value"] - random["affinity_pred_value"]})
        write_json(run.path / "summary.json", {
            "selection": selection, "paired_seeds": paired,
            "effective_native_baseline": baseline_effective,
            "effective_native_endpoint": endpoint_effective,
            "effective_native_random": random_effective,
            "effective_changed_from_baseline": effective_changed_from_baseline,
            "effective_changed_from_baseline_and_random": effective_changed,
            "effective_native_heavy_atoms": {smiles: row.get("native_standardized_heavy_atoms")
                                             for smiles, row in identity_by_input.items()},
            "effective_sizes_preserved_at_13": effective_sizes_preserved,
            "native_standardization_failures": sum(bool(row.get("standardization_failure"))
                                                   for row in identities),
            "native_prediction_attempts": len(results),
            "successful_native_predictions": sum(row["affinity"] is not None for row in results),
            "positive_binary_on_both_seeds": effective_sizes_preserved and effective_changed_from_baseline and all(row["paired_success"] and row[
                "endpoint_minus_baseline_binary"] > 0 for row in paired),
            "positive_vs_baseline_and_random_on_both_seeds": effective_sizes_preserved and effective_changed and all(
                row["paired_success"] and row["endpoint_minus_baseline_binary"] > 0
                and row["endpoint_minus_random_binary"] > 0 for row in paired),
            "random_control_same_as_baseline": selection["random_control_smiles"] ==
                selection["baseline_smiles"],
            "claim_limit": "Two-seed fresh-pose Boltz model comparison for one selected C13 endpoint, not observed binding, synthesis or selectivity."})
    return run.path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    print(run_rescore(args.pilot, args.template, args.cache, args.run_root))


if __name__ == "__main__":
    main()
