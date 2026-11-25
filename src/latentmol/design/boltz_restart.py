"""Boltz2 DBMol setup for the shared config-driven restart search engine."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics

import torch

from latentmol.design.protenix_restarts import write_json
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256


def run_config(args, native, config, design, options, settings, started):
    from latentmol.design.protenix_search import run_search
    from latentmol.guidance.boltz_search import BoltzSearchScorer

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("checkpoint", "output", "plan", "boltz-assets"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("checkpoint-sha256", "plan-sha256"):
        parser.add_argument("--" + name, required=True)
    runtime = parser.parse_args(native)
    if not os.environ.get("LSB_JOBID", "").isdigit():
        raise RuntimeError("Boltz restart search requires an allocated GPU job")
    if options.hard_scoring != "fixed_pose_dbmol" or options.initialization != "bootstrap":
        raise ValueError("New-target Boltz searches require explicit fixed-pose hard scoring and bootstrap starts")
    parallel_workers=config["scorer"].get("parallel_workers",1)
    pooled=config["scorer"].get("backend")=="vjp_pool" or parallel_workers>1
    if type(parallel_workers) is not int or not 1<=parallel_workers<=8:
        raise ValueError("Boltz parallel worker count must be 1..8")
    if not pooled and (options.batch_size != 1 or options.cpu_workers != 1 or options.overlap):
        raise ValueError("Direct Boltz execution requires batch one; choose the VJP pool for parallel branches")
    if pooled and options.batch_size>parallel_workers:
        raise ValueError("Boltz pipeline batch cannot exceed its native worker count")
    if file_sha256(runtime.checkpoint) != runtime.checkpoint_sha256:
        raise ValueError("Frozen VAE checkpoint changed")
    if file_sha256(runtime.plan) != runtime.plan_sha256:
        raise ValueError("Campaign launch plan changed")
    plan = json.loads(runtime.plan.read_text())
    if (file_sha256(args.config) != plan["config_sha256"]
            or settings["budget"]["updates"] != plan["planned_updates"][args.arm]
            or settings["budget"]["fresh_attempts"] != plan["planned_fresh_attempts"][args.arm]
            or plan["arms"][args.arm] != str(runtime.output)):
        raise ValueError("Boltz config, output or budget differs from the staged plan")
    if bool(args.resume_state) != bool(args.resume_sha256):
        raise ValueError("Resume requires both completed-wave state and its SHA256")
    resume = [str(args.resume_state), args.resume_sha256] if args.resume_state else None
    if plan.get("continuations", {}).get(args.arm) != resume:
        raise ValueError("Resume state differs from the staged plan")
    if args.resume_state and file_sha256(args.resume_state) != args.resume_sha256:
        raise ValueError("Completed-wave checkpoint changed")
    torch.set_num_threads(1 if pooled else 4)
    decoder, _ = load_model(runtime.checkpoint, "cpu")
    if design.occupied_atoms > decoder.spec.max_nodes:
        raise ValueError("Requested atom count exceeds the VAE capacity")
    assets = json.loads(runtime.boltz_assets.read_text())
    target = settings["target"]
    if runtime.output.exists():
        raise FileExistsError(runtime.output)
    inputs = runtime.output.with_name(runtime.output.name + "-inputs")
    prepared_ref = config["arms"][args.arm].get("prepared_config")
    if prepared_ref:
        prepared_path = Path(prepared_ref["path"])
        if file_sha256(prepared_path) != prepared_ref["sha256"]:
            raise ValueError("Target's prepared Boltz scorer config changed")
        scorer_config = json.loads(prepared_path.read_text())
        inputs.mkdir()
        write_json(inputs / "scorer-config.json", scorer_config)
    else:
        if args.resume_state:
            raise ValueError("Resume must reuse its original prepared Boltz context")
        from latentmol.boltz.search_targets import prepare_target
        receipt = prepare_target(target, Path(target["structure_path"]), Path(assets["cache"]), inputs,
            spec=decoder.spec, affinity_checkpoint=Path(assets["checkpoint"]),
            confidence_checkpoint=Path(assets["confidence_checkpoint"]["path"]),
            max_tokens=assets.get("max_tokens", 256), max_atoms=assets.get("max_atoms", 2048))
        scorer_config = json.loads((inputs / "scorer-config.json").read_text())
    if (scorer_config["checkpoint"] != assets["checkpoint"]
            or scorer_config["checkpoint_sha256"] != assets["checkpoint_sha256"]
            or scorer_config["dbmol"]["confidence_checkpoint"] != assets["confidence_checkpoint"]
            or scorer_config.get("recycling_steps", 0) != assets["recycling_steps"]
            or scorer_config.get("allow_lossy_charge", True) != assets["allow_lossy_charge"]):
        raise ValueError("Prepared target model/loss assets differ from the declared campaign")
    settings["scoring_configs"] = dict(backend="boltz2_dbmol_fixed_pose",
        assets=file_sha256(runtime.boltz_assets), prepared=scorer_config["prepared_sha256"],
        scorer=file_sha256(inputs / "scorer-config.json"))
    write_json(inputs / "target.json", target)
    if "execution" in config["scorer"]:
        scorer_config["execution"]=config["scorer"]["execution"]
        write_json(inputs/"execution-config.json",scorer_config)
        settings["scoring_configs"]["execution"]=file_sha256(inputs/"execution-config.json")
    if pooled:
        from latentmol.guidance.boltz_vjp import BoltzVJPScorer
        scorer=BoltzVJPScorer.load(scorer_config,decoder.spec,workers=parallel_workers,
            source_sha256=os.environ.get("LATENTMOL_SOURCE_SHA256","unrecorded-source"),
            output=inputs/"vjp-workers",managed_mps=True,
            worker_cpu_threads=config["scorer"].get("worker_cpu_threads",2))
    else:
        scorer = BoltzSearchScorer.load(scorer_config, decoder.spec, device=assets.get("device", "cuda"),
            source_sha256=os.environ.get("LATENTMOL_SOURCE_SHA256", "unrecorded-source"))
    try:
        metadata = scorer.prepared.metadata
        if (metadata.get("target_id") != target["id"]
                or metadata.get("mmcif_sha256") != target["pocket"]["mmcif_sha256"]
                or metadata.get("target", {}).get("pocket") != target["pocket"]
                or int(scorer.prepared.active.sum()) != design.occupied_atoms):
            raise ValueError("Prepared Boltz target/pocket differs from the requested search")
        if config.get("scorer", {}).get("timing_pilot"):
            if plan.get("reference_forward_attempts", {}).get(args.arm) != 1:
                raise ValueError("Timing pilot's reference forward was not in the staged budget")
            seed = scorer.prepared.seed.to("cpu")
            with torch.no_grad():
                reference_loss, details = scorer.score_at(seed.to_relaxed(decoder.spec), seed.atoms[0] != 0)
            write_json(inputs / "reference-score.json", dict(loss=float(reference_loss), details=details,
                objective="dbmol_total_loss", role="Native CCD preprocessing fixture; excluded from generated archive",
                reference_codec_roundtrip=metadata.get("reference_roundtrip"),
                warnings=metadata.get("warnings", []),
                scope="One hard-label fixed-pose forward; generated molecules use strict independent chemical validation"))
        result = run_search(decoder, lambda *_: scorer, scorer.score_valid_records, design, options,
            settings, runtime.output, runtime.checkpoint_sha256, run_started=started, resume_state=args.resume_state)
        panel = json.loads((runtime.output / "diversity.json").read_text())["score_only"]["molecules"]
        write_json(runtime.output / "selected-molecules.json", dict(target=target,
            requested_primary=10, requested_distinct=20, available=len(panel),
            primary=panel[:10], additional=panel[10:20], search_status=result["status"],
            ranking="Ascending hard fixed-pose Boltz2 DBMol total loss; best scored pose mapping per distinct molecule",
            scope="Frozen reference coordinates; no diffusion or experimental binding claims"))
        if options.profile_steps:
            rows = [json.loads(line) for path in sorted(runtime.output.glob("cycle-*/branch-*/metrics.jsonl"))
                    for line in path.read_text().splitlines()]
            measured = [row["timing"]["iteration_seconds"] for row in rows
                        if 2 <= row["step"] < design.updates]
            worker_peaks={}
            for row in rows:
                native=row["timing"].get("native_worker")
                if native:
                    peak=worker_peaks.setdefault(native["worker"],dict(allocated=0,reserved=0))
                    peak["allocated"]=max(peak["allocated"],native["peak_allocated_bytes"])
                    peak["reserved"]=max(peak["reserved"],native["peak_reserved_bytes"])
            write_json(runtime.output / "timing.json", dict(target_id=target["id"], learning_rate=design.learning_rate,
                measured_updates=len(measured), update_seconds=measured,
                median_update_seconds=statistics.median(measured) if measured else None,
                mean_update_seconds=statistics.mean(measured) if measured else None,
                peak_allocated_bytes=max((row["timing"]["cuda_peak_allocated_bytes"] or
                    row["timing"].get("native_worker",{}).get("peak_allocated_bytes",0) for row in rows), default=0),
                peak_reserved_bytes=max((row["timing"]["cuda_peak_reserved_bytes"] or
                    row["timing"].get("native_worker",{}).get("peak_reserved_bytes",0) for row in rows), default=0),
                native_worker_peaks=worker_peaks,
                sum_worker_peak_reserved_bytes=sum(p["reserved"] for p in worker_peaks.values()) if pooled else None,
                memory_scope="Maximum per-worker peaks; summed worker reserved peaks are an upper envelope, not simultaneous NVML usage" if pooled else "Direct native CUDA process",
                scope="Full lean latent/chemistry/DBMol iteration; first two and terminal observations excluded"))
        return result
    finally:
        try:
            scorer.close()
        finally:
            if hasattr(scorer,"shutdown"):
                stopped=scorer.shutdown()
                if stopped.get("requires_job_stop") or not stopped.get("cooperative_cleanup_complete"):
                    raise RuntimeError("Boltz workers did not shut down cooperatively; stop this MPS allocation")
