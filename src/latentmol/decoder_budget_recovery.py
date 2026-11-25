"""Serial, two-allocation decoder comparison with explicit evaluation reserves.

Only runtime/orchestration changes: the authenticated source070 scientific
protocol, ten-epoch endpoints, original anchors and evaluation stay unchanged.
Segment0 resumes the seven completed control epochs; other cases start fresh.
Ordinary failures are retained. No partial checkpoint is ranked as an endpoint.
"""

import argparse
import json
import os
from pathlib import Path
import time

import torch

from latentmol.data import GraphDataset
from latentmol.decoder_finetune import FineTuneConfig, assess_advantage, fit_case, make_panel, read_protocol, subset_loader
from latentmol.decoder_quality_evaluation import evaluate_random_panel, paired_compare
from latentmol.evaluate import load_model, reconstruction_metrics
from latentmol.experiments import file_sha256, write_json
from latentmol.losses import LossConfig


def read_recovery(path, expected_sha):
    if file_sha256(path) != expected_sha:
        raise ValueError("Recovery config hash mismatch")
    value = json.loads(Path(path).read_text())
    expected = {"version", "base_protocol", "base_protocol_sha256", "base_campaign", "base_original_sha256",
                "resume", "case_budget_seconds", "segment_budget_seconds", "evaluation_reserve_seconds",
                "original_job", "search_pilot_job"}
    if set(value) != expected or value["version"] != 1:
        raise ValueError("Unknown recovery config fields")
    if any(type(value[key]) is not int or value[key] <= 0 for key in
           ("case_budget_seconds", "segment_budget_seconds", "evaluation_reserve_seconds")):
        raise ValueError("Invalid recovery budgets")
    if 3 * (value["case_budget_seconds"] + value["evaluation_reserve_seconds"]) > value["segment_budget_seconds"]:
        raise ValueError("Segment cannot reserve all three cases/evaluations")
    if value["segment_budget_seconds"] > 21 * 3600:
        raise ValueError("Keep two-hour scheduler headroom inside23h allocation")
    return value


def verify_shared_source(parent, current_root):
    """Continuation may change the trainer, never frozen architecture/objectives."""
    prefixes = ("src/latentmol/models/", "src/latentmol/training/",
                "src/latentmol/datasets/", "src/latentmol/evaluation/",
                "src/latentmol/guidance/", "src/latentmol/design/")
    names = {"src/latentmol/losses.py", "src/latentmol/data.py", "src/latentmol/representation.py",
             "src/latentmol/decoder_regularization.py", "src/latentmol/chemistry_regularization.py",
             "src/latentmol/connectivity_regularization.py", "src/latentmol/sampling.py",
             "src/latentmol/evaluate.py", "src/latentmol/decoder_quality_evaluation.py",
             "src/latentmol/train.py", "src/latentmol/chemistry.py"}
    for relative, expected in parent["source"]["files"].items():
        if relative in names or relative.startswith(prefixes):
            if file_sha256(Path(current_root) / relative) != expected:
                raise ValueError("Recovery changed scientific source: " + relative)


def run_segment(recovery, root, source_root, segment, *, device="cuda"):
    """Three cases for one seed; segments are externally scheduled sequentially."""
    if segment not in (0, 1):
        raise ValueError("Only two declared seed segments")
    base = read_protocol(Path(recovery["base_protocol"]), recovery["base_protocol_sha256"])
    if base["seeds"] != [17, 23] or base["coefficients"] != [0., .1, 1.]:
        raise ValueError("Recovery changed matched scientific case grid")
    parent_path = Path(recovery["resume"]["parent_record"])
    if file_sha256(parent_path) != recovery["resume"]["parent_record_sha256"]:
        raise ValueError("Parent record changed")
    parent = json.loads(parent_path.read_text())
    verify_shared_source(parent, source_root)
    data_path = Path(base["data"])
    train, val, test = [GraphDataset(data_path, split) for split in ("train", "val", "test")]
    if train.manifest["dataset_sha256"] != base["dataset_sha256"] or train.manifest["split_sha256"] != base["split_sha256"]:
        raise ValueError("Recovery protected data/split changed")
    initializer = Path(base["initializer"])
    if file_sha256(initializer) != base["initializer_sha256"]:
        raise ValueError("Recovery original initializer changed")
    reference, original = load_model(initializer, device, expected_spec=train.spec)
    if original["train_config"]["mode"] != "vae" or original["dataset_sha256"] != base["dataset_sha256"]:
        raise ValueError("Need original protected VAE")
    if base["train_limit"] != len(train):
        raise ValueError("Recovery must use entire protected training split")
    old_root = Path(recovery["base_campaign"])
    if file_sha256(old_root / "original.json") != recovery["base_original_sha256"]:
        raise ValueError("Untouched reference evaluation changed")
    baseline = json.loads((old_root / "original.json").read_text())
    if baseline["checkpoint_sha256"] != base["initializer_sha256"]:
        raise ValueError("Reference evaluation belongs to different original")
    config = FineTuneConfig(**base["fine_tune"])
    validation = make_panel(base["validation_prior_count"], reference.config.latent, 81000)
    final = make_panel(base["final_prior_count"], reference.config.latent, 91000)
    del reference
    segment_root = root / f"segment-{segment}"
    segment_root.mkdir(parents=True, exist_ok=False)
    write_json(segment_root / "protocol.json", {"recovery": recovery, "base": base, "seed": base["seeds"][segment]})
    torch.save({"validation": validation, "final": final}, segment_root / "latent-panels.pt")
    started = time.monotonic()
    control, cases = None, []
    common = dict(initializer=initializer, train_data=train, val_data=val, device=device, source_root=source_root)
    for coefficient in base["coefficients"]:
        seed = base["seeds"][segment]
        item = {"seed": seed, "coefficient": coefficient}
        remaining = recovery["segment_budget_seconds"] - (time.monotonic() - started)
        if remaining < recovery["case_budget_seconds"] + recovery["evaluation_reserve_seconds"]:
            cases.append({**item, "status": "not_attempted_time_reservation", "remaining_seconds": remaining})
            write_json(segment_root / "cases.json", cases)
            continue
        try:
            resume = recovery["resume"] if segment == 0 and coefficient == 0 else None
            fit = fit_case(**common, output_root=segment_root / "continuations", config=config,
                seed=seed, coefficient=coefficient, train_limit=base["train_limit"], val_limit=base["validation_limit"],
                validation_panel=validation, budget_seconds=recovery["case_budget_seconds"], resume=resume)
            model, endpoint = load_model(Path(fit["checkpoint"]), device, expected_spec=train.spec)
            if endpoint["epoch"] != config.epochs:
                raise ValueError("Partial training endpoint cannot be evaluated as complete")
            rec, _ = reconstruction_metrics(model, subset_loader(test, len(test), 17, config.batch_size),
                LossConfig(**original["loss_config"]), device, chemistry_limit=len(test), training_mode="vae",
                graded=False, training_smiles=set(train.smiles))
            evaluation_dir = Path(fit["run_dir"]) / "final-prior"
            prior = evaluate_random_panel(model, final, set(train.smiles), evaluation_dir,
                batch_size=config.batch_size, samples_per_latent=4, sample_seed=91001, repair_limit=base["final_repair_limit"])
            item.update(status="completed", fit=fit, reconstruction=rec, prior=prior, evaluation_dir=str(evaluation_dir),
                        restored_epochs=endpoint["decoder_finetune"].get("resume", {}).get("epoch", 0),
                        vs_original=paired_compare(old_root / "original-prior", evaluation_dir, bootstrap_seed=92000))
            if coefficient == 0:
                control = item
            elif control is not None:
                if fit["terminal_rng"] != control["fit"]["terminal_rng"] or fit["encoder_sha256"] != control["fit"]["encoder_sha256"]:
                    raise RuntimeError("Recovery matched final training streams/encoder differ")
                item["vs_control"] = paired_compare(Path(control["evaluation_dir"]), evaluation_dir, bootstrap_seed=92000)
                item["advantage_assessment"] = assess_advantage(baseline["reconstruction"], control["reconstruction"], rec,
                                                               item["vs_original"], item["vs_control"])
            write_json(Path(fit["run_dir"]) / "final_evaluation.json", item)
            del model
        except Exception as error:
            item.update(status="failed_or_censored", error=repr(error), run_dir=getattr(error, "experiment_run_dir", None))
        cases.append(item)
        write_json(segment_root / "cases.json", cases)
    complete = len(cases) == 3 and all(case["status"] == "completed" for case in cases)
    write_json(segment_root / "summary.json", {"cases": cases, "complete": complete,
        "elapsed_seconds": time.monotonic() - started, "original": str(old_root / "original.json"),
        "scope": "Full ten-epoch decoder endpoints; bounded repair not exact distance; no binding/synthesis claim"})
    if segment == 1:
        summaries = [json.loads((root / f"segment-{index}" / "summary.json").read_text())
                     if (root / f"segment-{index}" / "summary.json").exists() else {"complete": False, "cases": []}
                     for index in (0, 1)]
        write_json(root / "summary.json", {"segments": summaries, "complete": all(row["complete"] for row in summaries)})
    if not complete:
        raise RuntimeError("Retained failed/censored/incomplete recovery cases")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--segment", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Numerical recovery requires DTU allocation")
    run_segment(read_recovery(args.config, args.sha256), args.output_root, Path.cwd(), args.segment)


if __name__ == "__main__":
    main()
