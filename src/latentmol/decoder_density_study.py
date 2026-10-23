"""Allocated 64-real/64-prior decoder-only chemistry coverage comparison.

Reuses original reconstruction, probability-aware local chemistry, MC
connectivity, reference anchors, checkpoint/resume and paired evaluation.
Search and DeFoG are separate experiments. All native work requires DTU LSF.
"""
import argparse
import json
import os
from pathlib import Path
import time

import torch

from latentmol.data import GraphDataset
from latentmol.decoder_density_protocol import read_protocol, cases
from latentmol.decoder_finetune import FineTuneConfig, fit_case, make_panel, subset_loader, assess_advantage
from latentmol.decoder_quality_evaluation import evaluate_random_panel, paired_compare
from latentmol.evaluate import load_model, reconstruction_metrics
from latentmol.experiments import file_sha256, write_json
from latentmol.losses import LossConfig


def assets(protocol, device):
    """Reauthenticate protected data/checkpoint before any native operation."""
    initializer, data = Path(protocol["initializer"]), Path(protocol["data"])
    if file_sha256(initializer) != protocol["initializer_sha256"]:
        raise ValueError("Density initializer changed")
    datasets = [GraphDataset(data, part) for part in ("train", "val", "test")]
    if (datasets[0].manifest["dataset_sha256"] != protocol["dataset_sha256"]
            or datasets[0].manifest["split_sha256"] != protocol["split_sha256"]
            or len(datasets[0]) != protocol["train_limit"]):
        raise ValueError("Density protected dataset/split changed")
    reference, checkpoint = load_model(initializer, device, expected_spec=datasets[0].spec)
    if checkpoint["train_config"]["mode"] != "vae" or checkpoint["dataset_sha256"] != protocol["dataset_sha256"]:
        raise ValueError("Density requires protected original VAE")
    return datasets, reference, checkpoint


def evaluate(model, data, checkpoint, protocol, directory, device):
    train, _, test = data
    reconstruction, _ = reconstruction_metrics(model,
        subset_loader(test, len(test), 17, 64), LossConfig(**checkpoint["loss_config"]), device,
        chemistry_limit=len(test), training_mode="vae", graded=False, training_smiles=set(train.smiles))
    prior = evaluate_random_panel(model, make_panel(protocol["final_prior_count"], model.config.latent, 91000),
        set(train.smiles), directory, batch_size=64, samples_per_latent=4, sample_seed=91001,
        repair_limit=protocol["final_repair_limit"])
    return reconstruction, prior


def gate(protocol, root, source_root, device):
    """Actual full-width checkpoint, complete/tail updates, paired RNG and VRAM.

    A gate proves engineering/memory compatibility, never a generation benefit.
    It also writes the original independent endpoint for all trained cases.
    """
    directory = root / "gate"
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "protocol.json", protocol)
    started = time.monotonic()
    properties = torch.cuda.get_device_properties(device)
    data, reference, checkpoint = assets(protocol, device)
    config = FineTuneConfig(**{**protocol["fine_tune"], "epochs": 1})
    validation = make_panel(32, reference.config.latent, 81000)
    del reference
    results = []
    for coefficient in protocol["coefficients"]:
        result = fit_case(Path(protocol["initializer"]), data[0], data[1], directory / "smoke", config,
            seed=17, coefficient=coefficient, train_limit=513, val_limit=64, validation_panel=validation,
            device=device, source_root=source_root, budget_seconds=protocol["gate_budget_seconds"] / 2,
            profile_updates=9)
        profile = result["update_profile"]
        if ([row["real_points"] for row in profile] != [64] * 8 + [1]
                or any(row["real_points"] != row["fresh_prior_points"]
                       or row["real_points"] != row["posterior_chemistry_points"] for row in profile)):
            raise ValueError("Native complete/tail update coverage differs from1:1")
        results.append(result)
    if results[0]["encoder_sha256"] != results[1]["encoder_sha256"] or results[0]["terminal_rng"] != results[1]["terminal_rng"]:
        raise ValueError("Density native control/chemical encoder or RNG budgets differ")
    reference, _ = load_model(Path(protocol["initializer"]), device, expected_spec=data[0].spec)
    rec, prior = evaluate(reference, data, checkpoint, protocol, directory / "original-prior", device)
    write_json(directory / "original.json", {"reconstruction": rec, "prior": prior,
        "initializer_sha256": protocol["initializer_sha256"]})
    write_json(directory / "passed.json", {"status": "passed", "protocol": protocol,
        "source_archive_sha256": os.environ["LATENTMOL_SOURCE_SHA256"], "cases": results,
        "elapsed_seconds": time.monotonic() - started,
        "runtime": {"torch": str(torch.__version__), "torch_cuda": torch.version.cuda,
            "gpu_name": properties.name, "gpu_total_memory_bytes": properties.total_memory,
            "lsf_job_id": os.environ["LSB_JOBID"]},
        "training_only_forecast": [{"coefficient": coefficient,
            "mean_profiled_full_update_seconds": sum(row["seconds"] for row in result["update_profile"][2:8]) / 6,
            "ten_epoch_training_seconds": sum(row["seconds"] for row in result["update_profile"][2:8]) / 6 * 33600,
            "scope": "Six warmed full updates; excludes per-epoch validation/evaluation, not a runtime guarantee"}
            for coefficient, result in zip(protocol["coefficients"], results)],
        "scope": "Actual frozen-encoder full-width GPU compatibility and profile; no efficacy gate"})


def historical(protocol, seed, encoder_sha256):
    """Use only authenticated completed historical16/16 alpha1 endpoints."""
    directory = Path(protocol["historical_root"]) / f"segment-{protocol['seeds'].index(seed)}"
    path = directory / "cases.json"
    if not path.is_file():
        return None
    for row in json.loads(path.read_text()):
        if row.get("seed") != seed or row.get("coefficient") != 1.0 or row.get("status") != "completed":
            continue
        fit = row["fit"]
        record = json.loads((Path(fit["run_dir"]) / "run.json").read_text())
        identity = record["config"]
        if (record.get("source_archive_sha256") != protocol["historical_archive_sha256"]
                or record.get("status") != "completed" or record.get("completed_epochs") != 10
                or identity["initializer_sha256"] != protocol["initializer_sha256"]
                or identity["dataset_sha256"] != protocol["dataset_sha256"]
                or identity["split_sha256"] != protocol["split_sha256"]
                or identity["seed"] != seed or identity["coefficient"] != 1.0
                or identity["train_count"] != protocol["train_limit"]
                or identity["frozen_encoder_sha256"] != encoder_sha256
                or fit["encoder_sha256"] != encoder_sha256
                or identity["config"]["prior_batch_size"] != 16
                or identity["config"]["chemistry_reconstruction_batch"] != 16
                or file_sha256(Path(fit["checkpoint"])) != fit["sha256"]):
            raise ValueError("Historical density reference identity/endpoint differs")
        return {"evaluation_dir": row["evaluation_dir"], "reconstruction": row["reconstruction"],
                "checkpoint_sha256": fit["sha256"], "record_sha256": file_sha256(Path(fit["run_dir"]) / "run.json")}
    return None


def run_case(protocol, root, source_root, index, device):
    declaration = cases(protocol)[index]
    directory = root / declaration["label"]
    directory.mkdir(parents=True, exist_ok=False)
    gate_record = json.loads((root / "gate/passed.json").read_text())
    if (gate_record["status"] != "passed" or gate_record["protocol"] != protocol
            or gate_record["source_archive_sha256"] != os.environ["LATENTMOL_SOURCE_SHA256"]):
        raise ValueError("Density native gate belongs to another immutable study")
    write_json(directory / "protocol.json", protocol)
    started = time.monotonic()
    fit, model, rec, prior = None, None, None, None
    phase = "protected_assets"
    try:
        data, reference, checkpoint = assets(protocol, device)
        latent = reference.config.latent
        del reference
        phase = "training"
        fit = fit_case(Path(protocol["initializer"]), data[0], data[1], directory / "training",
            FineTuneConfig(**protocol["fine_tune"]), seed=declaration["seed"], coefficient=declaration["coefficient"],
            train_limit=protocol["train_limit"], val_limit=protocol["validation_limit"],
            validation_panel=make_panel(protocol["validation_prior_count"], latent, 81000), device=device,
            source_root=source_root, budget_seconds=protocol["case_budget_seconds"], profile_updates=8)
        phase = "final_evaluation"
        model, _ = load_model(Path(fit["checkpoint"]), device, expected_spec=data[0].spec)
        rec, prior = evaluate(model, data, checkpoint, protocol, directory / "final-prior", device)
        phase = "paired_comparison"
        original = json.loads((root / "gate/original.json").read_text())
        versus_original = paired_compare(root / "gate/original-prior", directory / "final-prior", bootstrap_seed=92000)
        summary = {**declaration, "status": "completed", "fit": fit, "reconstruction": rec,
                   "prior": prior, "vs_original": versus_original,
                   "elapsed_seconds": time.monotonic() - started}
        if declaration["coefficient"] != 0:
            control_path = root / f"seed{declaration['seed']}-a0" / "summary.json"
            control = json.loads(control_path.read_text()) if control_path.exists() else {}
            if control.get("status") == "completed":
                if control["fit"]["terminal_rng"] != fit["terminal_rng"] or control["fit"]["encoder_sha256"] != fit["encoder_sha256"]:
                    raise ValueError("Dense matched control/candidate streams or encoder differ")
                summary["vs_dense_control"] = paired_compare(control_path.parent / "final-prior", directory / "final-prior", bootstrap_seed=92000)
                summary["advantage_assessment"] = assess_advantage(original["reconstruction"], control["reconstruction"],
                    rec, versus_original, summary["vs_dense_control"])
            else:
                summary["matched_control_status"] = "unavailable; no matched advantage claim"
            # An optional old endpoint cannot invalidate completed new training
            # or its primary matched-control comparison. Preserve its failure.
            try:
                old = historical(protocol, declaration["seed"], fit["encoder_sha256"])
                if old is not None:
                    summary["historical_reference"] = old
                    summary["vs_historical_16_16"] = paired_compare(Path(old["evaluation_dir"]), directory / "final-prior", bootstrap_seed=92000)
                else:
                    summary["historical_reference_status"] = "not yet completed; density advantage unassessed"
            except Exception as error:
                summary["historical_reference_status"] = "failed_authentication_or_comparison; density advantage unassessed"
                summary["historical_reference_error"] = repr(error)
        phase = "completed_receipt"
        write_json(directory / "summary.json", summary)
    except Exception as error:
        write_json(directory / "summary.json", {**declaration, "status": "failed_or_time_censored",
            "failure_phase": phase, "training_completed": fit is not None,
            "fit": fit, "reconstruction": rec, "prior": prior,
            "error": repr(error), "run_dir": fit["run_dir"] if fit is not None else getattr(error, "experiment_run_dir", None),
            "elapsed_seconds": time.monotonic() - started,
            "scope": "Retain checkpoints; no partial endpoint comparison"})
        raise
    finally:
        if model is not None:
            del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("gate", "case"), required=True)
    parser.add_argument("--case", type=int, choices=range(4))
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID") or not os.environ.get("LATENTMOL_SOURCE_SHA256"):
        raise RuntimeError("Numerical density study requires immutable DTU allocation")
    protocol = read_protocol(args.protocol, args.sha256)
    if args.stage == "gate":
        gate(protocol, args.output_root, Path.cwd(), "cuda")
    else:
        if args.case is None:
            parser.error("Case stage requires its declared index")
        run_case(protocol, args.output_root, Path.cwd(), args.case, "cuda")


if __name__ == "__main__":
    main()
