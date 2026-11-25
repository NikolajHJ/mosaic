"""Paired +128/+512 motif pilot with exact +128 sampling prefix and saved raw panels."""

import argparse
import csv
import gc
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from rdkit import rdBase

from experiments.vae.training_data_eda.teacher_motif_training_pilot import digest_rng, encode, evaluate
from experiments.vae.training_data_eda.compare_completion_pilot import draw_panels, paired_results
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256
from latentmol.train import learning_rate_at_epoch, state_digest
from latentmol.training.combined_continuation import _fresh
from latentmol.training.projected_continuation import projected_update


def save(path, data):
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)


def compare(control_dir, treatment_dir, reference_arm="control"):
    report = {}
    labels = ("valid", "aromatic", "amide", "ring3", "ring8plus", "ring12plus")
    for suffix, panels in (("", ("prior", "reconstruction")), ("-fresh", ("prior",)), ("-sampled", ("prior",))):
        left = draw_panels(control_dir / f"{reference_arm}{suffix}.jsonl.gz")
        right = draw_panels(treatment_dir / f"treatment{suffix}.jsonl.gz")
        for panel in panels:
            key = ("fresh_argmax" if suffix == "-fresh" else "fresh_categorical" if suffix else panel)
            report[key] = paired_results(left[panel], right[panel], (*labels, "exact") if panel == "reconstruction" else labels)
    return report


def rescore_new_priors(control_dir, control_summary, candidate, output, n, saved_pattern=None):
    """Re-evaluate frozen endpoints on new paired priors, independent of all training/prefix panels."""
    z_seed, sample_seed = 20261122471, 20261122472
    z = torch.randn(n, candidate.config.latent, generator=torch.Generator().manual_seed(z_seed))
    if saved_pattern:
        archived = json.loads((saved_pattern / "independent-512.json").read_bytes())
        if (archived["priors"], archived["z_seed"], archived["categorical_seed"]) != (n, z_seed, sample_seed):
            raise ValueError("Archived independent panel uses different priors or sampling")
    else:
        control, payload = load_model(control_dir / "endpoint.pt", "cuda", candidate.spec)
        if (not payload.get("diagnostic_only") or payload["pilot_updates"] != 512 or payload["source_update"] != 580498
                or state_digest(control.state_dict()) != control_summary["parameter_sha"]):
            raise ValueError("Saved evaluation weights differ from the recorded +512 V100 control")
    result = dict(priors=n, z_seed=z_seed, categorical_seed=sample_seed)
    pattern_result = dict(result)
    labels = ("valid", "aromatic", "amide", "ring3", "ring8plus", "ring12plus")
    for label, seed in (("argmax", None), ("categorical", sample_seed)):
        left = (saved_pattern or output) / f"independent-control-{label}.jsonl.gz"
        if not saved_pattern:
            evaluate(control, z, [], left, categorical_seed=seed)
        f = output / f"independent-treatment-{label}.jsonl.gz"
        evaluate(candidate, z, [], f, categorical_seed=seed)
        rows = draw_panels(f)["prior"]
        result[label] = paired_results(draw_panels(left)["prior"], rows, labels)
        if saved_pattern:
            baseline = draw_panels(saved_pattern / f"independent-treatment-{label}.jsonl.gz")["prior"]
            pattern_result[label] = paired_results(baseline, rows, labels)
    save(output / "independent-512.json", result)
    if saved_pattern:
        save(output / "independent-512-vs-pattern.json", pattern_result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint", "reference", "config", "train-panel", "heldout-panel", "output", "old-control"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--arm", choices=("control", "treatment"), required=True)
    p.add_argument("--pattern-control", type=Path,
                   help="Run pattern guidance against saved matched +128/+512 controls with no retraining.")
    p.add_argument("--independent-draws", type=int, default=0)
    p.add_argument("--release-teacher-vs", type=Path,
                   help="Also release the local ring teacher and compare with the saved pattern-only run.")
    args = p.parse_args()
    if (args.pattern_control and args.arm != "treatment") or (args.independent_draws and not args.pattern_control):
        p.error("Additional independent draws require a pattern treatment and its archived controls")
    if args.independent_draws < 0:
        p.error("--independent-draws must be nonnegative")
    if args.release_teacher_vs and not args.pattern_control:
        p.error("--release-teacher-vs requires the pattern treatment and archived baseline")
    wall_start = time.monotonic()
    torch.set_num_threads(3)
    arm = args.arm
    arm_dir = args.output / arm
    arm_dir.mkdir(parents=True, exist_ok=False)
    old = json.loads((args.old_control / "summary.json").read_bytes())
    seed, draws, size, micro = 2026112168, 1024, 64, 16
    cfg = json.loads(args.config.read_bytes())
    cfg["functional_groups"]["draws"] = micro
    if (old["config"] != cfg or old["torch"] != torch.__version__ or old["rdkit"] != rdBase.rdkitVersion
            or old["device"] != torch.cuda.get_device_name() or old["seed"] != seed
            or old["updates"] != 128 or old["gaussian_evaluation"] != draws
            or old["real_per_update"] != size or old["microbatch"] != micro):
        raise ValueError("Previous control does not match the projected recipe, platform or panel")
    for key, file in (("checkpoint", args.checkpoint), ("reference", args.reference),
                      ("train_panel", args.train_panel), ("heldout_panel", args.heldout_panel)):
        if file_sha256(file) != old[key]:
            raise ValueError(f"Prior control input changed: {key}")
    if arm == "treatment":
        cfg["functional_groups"].update(completion_weight=1., amide_relative_weight=.10)
        if args.pattern_control:
            cfg["functional_groups"]["completion_mode"] = "pattern"
        if args.release_teacher_vs:
            cfg["functional_groups"]["release_ring_teacher"] = True

    parent, payload = load_model(args.checkpoint, "cpu")
    reference, _ = load_model(args.reference, "cuda", parent.spec)
    reference.eval().requires_grad_(False)
    parent_digest = state_digest(parent.state_dict())
    frozen_digest = state_digest(reference.state_dict())
    start = payload["continuation"]["updates"]
    if start != 580498:
        raise ValueError("Expected the previous matched snapshot")

    with gzip.open(args.train_panel, "rt") as stream:
        smiles = [row["smiles"] for row in csv.DictReader(stream, delimiter="\t")]
    holdout = [row[2] for row in json.loads(args.heldout_panel.read_bytes())["rows"]]
    if set(smiles).intersection(holdout):
        raise ValueError("Train/held-out panel overlap")
    indices = np.random.default_rng(seed).choice(len(smiles), 128*size, replace=False)
    if hashlib.sha256(indices.tobytes()).hexdigest() != old["train_indices_sha256"]:
        raise ValueError("First 128 updates must use exactly the archived 8192 training rows")
    unused = np.setdiff1d(np.arange(len(smiles)), indices, assume_unique=True)
    extension = np.random.default_rng(seed+512).choice(unused, 384*size, replace=False)
    all_indices = np.concatenate((indices, extension))
    held = np.random.default_rng(seed+1).choice(len(holdout), 128, replace=False)
    validation = [[holdout[i] for i in ids] for ids in np.array_split(held, 16)]
    validation = [(encode(sm, parent.spec), sm) for sm in validation]
    z = torch.randn(draws, parent.config.latent, generator=torch.Generator().manual_seed(seed+2))
    fresh = torch.randn(draws, parent.config.latent, generator=torch.Generator().manual_seed(old["fresh_evaluation_seed"]))

    model, optimizer, training, gens = _fresh(parent, payload, None, size, "cuda", None,
                                              atom_refinement_dropout=cfg["atom_refinement_dropout"])
    if state_digest(model.state_dict()) != parent_digest or training.loss.atom_order != cfg["atom_order"]:
        raise ValueError("Working model or baseline reconstruction differs")
    for group in optimizer.param_groups:
        group["lr"] = learning_rate_at_epoch(training,48)
    torch.manual_seed(seed+3)
    gens["fresh_prior"].manual_seed(seed+4)
    initial_rng = digest_rng(gens)
    if initial_rng != old["arms"]["control"]["rng_start"]:
        raise ValueError("Starting RNG differs on this device")
    training_seconds = 0.
    scheduled = []
    with rdBase.BlockLogs(), (arm_dir/"training.jsonl").open("x") as trace:
        for step, ids in enumerate(all_indices.reshape(-1,size), 1):
            hard = encode([smiles[i] for i in ids],parent.spec)
            t = time.monotonic()
            metrics = projected_update(cfg,model,reference,optimizer,hard,training,None,None,gens,start+step-1,48,micro)
            training_seconds += time.monotonic()-t
            trace.write(json.dumps(dict(step=step, **metrics), allow_nan=False)+"\n")
            if "functional_groups" in metrics:
                scheduled.append(metrics["functional_groups"])
            if step % 32 == 0:
                trace.flush()
                print(arm, step, round(training_seconds,2), flush=True)
            if step not in (128,512):
                continue

            endpoint = arm_dir / str(step)
            endpoint.mkdir()
            if state_digest(reference.state_dict()) != frozen_digest or state_digest(parent.state_dict()) != parent_digest:
                raise RuntimeError("Frozen model changed")
            rng = digest_rng(gens)
            parameter_sha = state_digest(model.state_dict())
            was_training = model.training
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                evaluate(model,z,validation,endpoint/(arm+".jsonl.gz"))
                evaluate(model,fresh,[],endpoint/(arm+"-fresh.jsonl.gz"))
                evaluate(model,fresh,[],endpoint/(arm+"-sampled.jsonl.gz"),categorical_seed=old["categorical_seed"])
            model.train(was_training)
            if rng != digest_rng(gens):
                raise RuntimeError("Panel evaluation changed the training RNG")
            torch.save({key:payload[key] for key in ("format_version","model_config","graph_spec")} |
                       {"model_state":model.state_dict(),"diagnostic_only":True,"source_update":start,
                        "pilot_updates":step},endpoint/"endpoint.pt")
            indices_sha = hashlib.sha256(all_indices[:step*size].tobytes()).hexdigest()
            summary = dict(updates=step,config=cfg,seed=seed,checkpoint=old["checkpoint"],reference=old["reference"],
                           train_indices_sha256=indices_sha,rng_start=initial_rng,rng_end=rng,
                           parameter_sha=parameter_sha,training_seconds=training_seconds,
                           wall_seconds=time.monotonic()-wall_start,scheduled=len(scheduled),
                           ring_targets=sum(r.get("pattern_aromatic_selected",r.get("completion_aromatic_rows",0)) for r in scheduled),
                           amide_targets=sum(r.get("pattern_amide_selected",r.get("completion_amide_rows",0)) for r in scheduled))
            if step == 128:
                summary["prefix_rng_matches_saved_control"] = rng == old["arms"]["control"]["rng_end"]
                if not summary["prefix_rng_matches_saved_control"]:
                    raise ValueError("128-update training stream differs from archived V100 control")
            save(endpoint/"summary.json", summary)
            if arm == "treatment" and step == 128:
                save(args.output/"paired-128-saved-control.json",compare(args.old_control, endpoint))
            if args.pattern_control:
                archived = args.pattern_control / str(step)
                summary_c = json.loads((archived / "summary.json").read_bytes())
                if any(summary_c[k] != summary[k] for k in (
                        "rng_start", "rng_end", "train_indices_sha256", "checkpoint", "reference", "updates", "seed")):
                    raise ValueError("Pattern candidate and archived control have different inputs/RNG")
                if summary_c["config"] != old["config"]:
                    raise ValueError("Archived control objective differs from the pinned original control")
                save(args.output/f"paired-{step}.json",compare(archived, endpoint))
                if args.release_teacher_vs:
                    prev = args.release_teacher_vs / "treatment" / str(step)
                    prev_s = json.loads((prev / "summary.json").read_bytes())
                    expected_cfg = dict(cfg, functional_groups=dict(cfg["functional_groups"]))
                    del expected_cfg["functional_groups"]["release_ring_teacher"]
                    if (prev_s["config"] != expected_cfg or any(prev_s[k] != summary[k] for k in (
                            "rng_start", "rng_end", "train_indices_sha256", "checkpoint", "reference", "updates", "seed"))):
                        raise ValueError("Pattern-only reference differs in training inputs or objective")
                    save(args.output/f"paired-{step}-vs-pattern.json",compare(prev, endpoint, reference_arm="treatment"))
                if step == 512 and args.independent_draws:
                    rescore_new_priors(archived,summary_c,model,args.output,args.independent_draws,args.release_teacher_vs)
            if arm == "control":
                treatment = args.output/"treatment"/str(step)
                other = json.loads((treatment/"summary.json").read_bytes())
                if any(other[k] != summary[k] for k in ("rng_start","rng_end","train_indices_sha256","checkpoint","reference")):
                    raise ValueError("The two arms do not share training inputs and random draws")
                save(args.output/f"paired-{step}.json",compare(endpoint,treatment))
                if step == 128:
                    print("Replayed prefix exactly reproduces saved control weights:",
                          parameter_sha == old["arms"]["control"]["parameter_sha"], flush=True)
            gc.collect()


if __name__ == "__main__":
    main()
