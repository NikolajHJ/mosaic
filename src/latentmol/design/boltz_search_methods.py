"""Paired allocation, founder screening and gradient-reuse pilots on a fixed Boltz pocket."""
import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
from time import perf_counter

import torch

from latentmol.chemistry import inspect_graph
from latentmol.design.boltz_allocation_pilot import allocation_budget, run_allocation
from latentmol.design.boltz_speed import _point_digest
from latentmol.design.latent_protenix import _model_identity, hard_from_graph
from latentmol.design.protenix_restarts import Candidate, write_json
from latentmol.design.protenix_search import bootstrap, resolve_config
from latentmol.design.protenix_search_branches import Branch, refill_exact_starts
from latentmol.design.protenix_search_frontier import SearchPoint
from latentmol.design.protenix_search_state import load_search_state
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256
from latentmol.graph_policy import graph_from_decoder
from latentmol.guidance.boltz_search import BoltzSearchScorer, hard_graph_record, hard_pose_key
from latentmol.representation import GraphSpec


GENERATION_STUDIES = {"boltz_generations", "boltz_generation_search", "boltz_native_graph_search"}


def resolve_case(config, arm):
    task = config["tasks"][arm]
    case = dict(config["cases"][task["case"]], seed=config["seeds"][task["panel"]])
    native_founders = config["study"] == "boltz_native_graph_search"
    if (case["start"] == "native_graph") != native_founders or case["start"] not in (
            "archive", "random", "screened", "native_graph", "vae_prior"):
        raise ValueError("Unknown source for Boltz pilot starts")
    space = config.get("optimization_space", "vae_latent")
    if space not in ("vae_latent", "relaxed_graph_logits") or (
            space == "relaxed_graph_logits" and config["study"] not in {"boltz_generation_search", "boltz_native_graph_search"}):
        raise ValueError("Unsupported optimization space for this search")
    if native_founders and (space != "relaxed_graph_logits" or "parent_state" in config
            or config["generation_settings"]["parents"] >= config["generation_settings"]["fresh_per_generation"]):
        raise ValueError("Native graph search needs direct logits, new founders and first-wave scoring capacity")
    if config["study"] in GENERATION_STUDIES:
        from latentmol.design.boltz_generations import generation_budget
        if case["start"] not in ("archive", "native_graph", "vae_prior") or case.get("reuse"):
            raise ValueError("Generation comparison requires ordinary full gradients from archived parents")
        if case["start"] == "vae_prior":
            source, settings = config["prior_founders"], config["generation_settings"]
            if (space != "vae_latent" or "parent_state" in config or not config.get("vae_checkpoint")
                    or not settings["parents"] <= source["count"] < settings["fresh_per_generation"]
                    or source["max_draws"] < source["count"]):
                raise ValueError("Fresh VAE founders require latent optimization and first-wave scoring capacity")
        case.update(generation_budget(case["policy"], **config["generation_settings"]))
        return case
    if type(case["fresh_budget"]) is not int or case["fresh_budget"] < 1:
        raise ValueError("A positive scoring allowance is required")
    case["updates"] = allocation_budget(case["policy"], case["count"])
    return case


@torch.no_grad()
def archive_starts(decoder, design, state, count, seed, *, seen=None, fallback_prior=False, parent_count=3):
    eligible = [p for smi, p in state["archive"].items() if smi not in state["ineligible"]]
    parents = sorted(eligible, key=lambda p: (p.fresh_loss, p.candidate.smiles))[:parent_count]
    if len(parents) != parent_count or any(int(p.active.sum()) != design.occupied_atoms for p in parents):
        raise ValueError("Need the requested number of eligible source parents with matching occupancy")
    points, records = [], []
    if seen is None:
        seen = set()
    for slot in range(count):
        parent = parents[slot % len(parents)]
        name = f"start-{slot + 1:02d}"
        branch = Branch(name, parent, seed + slot)
        accepted, collisions = refill_exact_starts(decoder, design, [branch], seen, attempts=8,
                                                  fallback_policy="latent", repair_limit=0)
        if not accepted:
            if not fallback_prior:
                raise ValueError(f"Could not generate a distinct child from {parent.candidate.branch}")
            prior_seed = seed + 10_000 + slot
            prior = bootstrap(decoder, design, seed=prior_seed, name=name + "-prior")
            prior_branch = Branch(name, prior, prior_seed, policy="bootstrap")
            accepted, _ = refill_exact_starts(decoder, design, [prior_branch], seen, attempts=0,
                                              fallback_policy="bootstrap", repair_limit=0)
            if not accepted:
                raise ValueError("Fresh prior collided with an earlier start after bounded retries")
            points.append(prior)
            records.append(dict(name=name, seed=prior_seed, source_branch=None, source_step=None,
                                source_smiles=None, source_loss=None, fallback="fresh_prior",
                                exhausted_parent=parent.candidate.smiles, collisions=collisions))
            continue
        branch, = accepted
        latent = branch.prepared.latent
        graph = graph_from_decoder(decoder.decode(latent), decoder.spec, fixed_active=parent.active,
                                   temperature=design.temperature_initial)
        proposal = branch.prepared.diagnostics["proposal"]
        candidate = Candidate(proposal.get("smiles"), latent, hard_from_graph(graph), 0., name, 0)
        points.append(SearchPoint(candidate, parent.active, None, parent.ancestry + (name,)))
        records.append(dict(name=name, seed=branch.seed, source_branch=parent.candidate.branch,
                            source_step=parent.candidate.step, source_smiles=parent.candidate.smiles,
                            source_loss=parent.fresh_loss, proposal=proposal, collisions=collisions))
    return points, records


@torch.no_grad()
def parent_state(config, decoder, design=None, prepared_digest=None):
    source = config["parent_state"]
    path = Path(source["path"])
    if file_sha256(path) != source["sha256"]:
        raise ValueError("Frozen parent archive changed")
    if source.get("format") == "boltz-scored-latents-v1":
        panel = json.loads(read(path))
        if (panel["format"] != source["format"] or panel["target"] != config["target_arm"]
                or panel["prepared_sha256"] != prepared_digest):
            raise ValueError("Scored founders belong to a different target or prepared pocket")
        archive = {}
        excluded = set(panel["excluded_smiles"])
        for row in panel["parents"]:
            latent = torch.tensor(row["latent"], dtype=torch.float32).unsqueeze(0)
            slots = row["active_slots"]
            if (latent.ndim != 2 or not latent.isfinite().all() or len(set(slots)) != design.occupied_atoms
                    or len(slots) != design.occupied_atoms or any(type(i) is not int or not 0 <= i < decoder.spec.max_nodes for i in slots)
                    or not all(math.isfinite(row[k]) for k in ("loss", "relaxed_loss"))):
                raise ValueError("Invalid scored founder latent, occupancy or loss")
            active = torch.zeros(decoder.spec.max_nodes, dtype=torch.bool)
            active[slots] = True
            graph = graph_from_decoder(decoder.decode(latent), decoder.spec, fixed_active=active,
                                       temperature=design.temperature_initial)
            hard = hard_from_graph(graph)
            smi = inspect_graph(hard.atoms[0], hard.charges[0], hard.bonds[0], decoder.spec).smiles
            if (smi != row["smiles"] or smi not in excluded or smi in archive
                    or hard_pose_key(hard_graph_record(hard)) != row["pose_key"]):
                raise ValueError("Could not replay a unique founder molecule and its exact scored pose")
            candidate = Candidate(smi, latent, hard, row["relaxed_loss"], row["source"], row["step"])
            archive[smi] = SearchPoint(candidate, active, row["loss"], (row["source"],))
        return dict(archive=archive, ineligible=set(), excluded_smiles=excluded, native_identity=panel["identity"])
    return load_search_state(path, source["identity"], decoder.spec)


def prepare_starts(decoder, design, config, case):
    seed, count = case["seed"], case["count"]
    if case["start"] == "archive":
        source = config["parent_state"]
        state = parent_state(config, decoder)
        points, records = archive_starts(decoder, design, state, count, seed)
        return points, dict(source=source, children=records), set(state["archive"])
    if case["start"] == "screened":
        from latentmol.design.bootstrap_screen import screened_bootstrap
        points, receipt = screened_bootstrap(decoder, design, count=count, seed=seed,
                                              pool_size=config["screen_pool_size"])
        return points, receipt, set()
    points = [bootstrap(decoder, design, seed=seed + i, name=f"bootstrap-{i+1:02d}") for i in range(count)]
    return points, dict(raw_prior_seeds=list(range(seed, seed + count))), set()


def read(path):
    return Path(path).read_text(encoding="utf-8")


def run(args):
    started = perf_counter()
    if not os.environ.get("LSB_JOBID", "").isdigit():
        raise RuntimeError("A GPU allocation is required")
    cfg, plan = (json.loads(read(path)) for path in (args.config, args.plan))
    case = resolve_case(cfg, args.arm)
    if (file_sha256(args.plan) != args.plan_sha256 or file_sha256(args.config) != plan["config_sha256"]
            or str(args.output) != plan["arms"][args.arm]
            or case["updates"] != plan["planned_updates"][args.arm]
            or case["fresh_budget"] != plan["planned_fresh_attempts"][args.arm]):
        raise ValueError("Pilot configuration differs from the staged budget or output")
    native_founders = case["start"] == "native_graph"
    if native_founders:
        if args.checkpoint is not None or args.checkpoint_sha256 is not None:
            raise ValueError("Native graph search does not accept a VAE checkpoint")
    elif args.checkpoint is None or file_sha256(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("Missing or changed frozen VAE checkpoint")
    base = json.loads(read(args.config.parent / cfg["search_config"]))
    design, _, settings = resolve_config(base, cfg["target_arm"])
    design = replace(design, seed=case["seed"])
    prepared = base["arms"][cfg["target_arm"]]["prepared_config"]
    if file_sha256(Path(prepared["path"])) != prepared["sha256"]:
        raise ValueError("Prepared Boltz configuration changed")
    assets = json.loads(read(args.boltz_assets))
    native_config = json.loads(read(prepared["path"]))
    if (native_config["checkpoint_sha256"] != assets["checkpoint_sha256"]
            or native_config["dbmol"]["confidence_checkpoint"] != assets["confidence_checkpoint"]):
        raise ValueError("Pilot does not use the staged affinity and confidence checkpoints")
    native_config["execution"] = (cfg["scorer"].get("execution") or base["scorer"]["execution"])
    if native_config["execution"]["autocast"] != "none" or native_config["execution"]["tf32"]:
        raise ValueError("Paired hard scoring and full gradients require FP32")
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(case["seed"])
    prep_start = perf_counter()
    generations = cfg["study"] in GENERATION_STUDIES
    decoder = None
    fresh_prior = case["start"] == "vae_prior"
    if native_founders or fresh_prior:
        exclusions = cfg["history_exclusions"]
        if file_sha256(Path(exclusions["path"])) != exclusions["sha256"]:
            raise ValueError("Historical molecule exclusion list changed")
        history = json.loads(read(exclusions["path"]))
        if not isinstance(history, list) or not all(isinstance(s, str) and s for s in history):
            raise ValueError("Expected a list of historical molecule identities")
    if native_founders:
        spec = GraphSpec.from_dict(cfg["graph_spec"])
    else:
        decoder, checkpoint = load_model(args.checkpoint, "cpu")
        pinned = cfg.get("vae_checkpoint")
        if pinned and (pinned != plan.get("vae_checkpoint") or str(args.checkpoint) != pinned["path"]
                       or args.checkpoint_sha256 != pinned["sha256"]
                       or checkpoint["continuation"]["updates"] != pinned["updates"]):
            raise ValueError("Loaded VAE differs from the pinned search checkpoint")
        del checkpoint
        decoder.eval().requires_grad_(False)
        decoder_identity = _model_identity(decoder)
        spec = decoder.spec
    if fresh_prior:
        from latentmol.design.boltz_latent_founders import prior_founder_panel
        points, founder_records = prior_founder_panel(decoder, design, seed=case["seed"], history=history,
                                                      **cfg["prior_founders"])
    elif generations and not native_founders:
        state = parent_state(cfg, decoder, design, prepared["sha256"])
    elif not generations:
        starts, receipt, excluded = prepare_starts(decoder, design, cfg, case)
        hashes = [_point_digest(p.candidate.latent, p.active) for p in starts]
        write_json(args.output / "starts.json", dict(case=case, strategy=case["start"], starts=receipt,
                   latent_mask_hashes=hashes, excluded_source_molecules=sorted(excluded),
                   preparation_seconds=perf_counter()-prep_start))
    preparation_seconds = perf_counter() - prep_start
    native = BoltzSearchScorer.load(native_config, spec, device="cuda",
                                    source_sha256=os.environ["LATENTMOL_SOURCE_SHA256"])
    try:
        if (native.prepared.metadata.get("target", {}).get("pocket") != settings["target"]["pocket"]
                or int(native.prepared.active.sum()) != design.occupied_atoms):
            raise ValueError("Boltz context differs from requested pocket or occupancy")
        if fresh_prior:
            from latentmol.design.boltz_native_graph_founders import score_founder_panel
            from latentmol.design.protenix_search_state import _pack
            tick = perf_counter()
            state = score_founder_panel(native, args.output, points, founder_records, _pack, "vae_latent",
                                        dict(vae_checkpoint=pinned, seed=case["seed"], source="native_size_prior"))
            state["excluded_smiles"].update(history)
            preparation_seconds += perf_counter() - tick
        elif native_founders:
            from latentmol.design.boltz_native_graph_founders import native_graph_founders
            tick = perf_counter()
            state = native_graph_founders(native, design, args.output, seed=case["seed"],
                                          count=cfg["generation_settings"]["parents"])
            state["excluded_smiles"].update(history)
            preparation_seconds += perf_counter() - tick
        elif generations and "native_identity" in state and any(
                native.identity[k] != state["native_identity"][k] for k in ("context_digest", "model_digest")):
            raise ValueError("Founder scores used a different Boltz model or pose context")
        scorer = native
        if case.get("reuse"):
            from latentmol.guidance.boltz_gradient_reuse import LocalBoltzGradient
            scorer = LocalBoltzGradient(native, **cfg["gradient_reuse"])
        search_start = perf_counter()
        if generations:
            from latentmol.design.boltz_generations import run_generations
            result = run_generations(decoder, scorer, design, state, args.output,
                                     policy=case["policy"], seed=case["seed"],
                                     optimization_space=cfg.get("optimization_space", "vae_latent"),
                                     reject_unstable_identity=cfg.get("reject_unstable_identity", False),
                                     ready=os.getenv("DBMOL_LAUNCH_READY"), **cfg["generation_settings"])
            hashes = result["start_hashes"]
        else:
            result = run_allocation(decoder, scorer, design, starts, args.output, policy=case["policy"],
                                    fresh_budget=case["fresh_budget"], seed=case["seed"],
                                    ready=os.getenv("DBMOL_LAUNCH_READY"), excluded_smiles=excluded)
        if decoder is not None and (_model_identity(decoder) != decoder_identity
                                    or any(p.grad is not None for p in decoder.parameters())):
            raise ValueError("Frozen VAE changed during optimization")
        diagnostics = scorer.diagnostics() if case.get("reuse") else dict(
            full_gradient_calls=result["relaxed_observations"], proxy_calls=0,
            hard_fp32_forwards=result["fresh_attempts"])
        result.update(job_id=os.environ["LSB_JOBID"], case=case, case_name=cfg["tasks"][args.arm]["case"],
                      start_hashes=hashes,
                      source_sha256=os.environ["LATENTMOL_SOURCE_SHA256"],
                      model_device=torch.cuda.get_device_name(), execution=native_config["execution"],
                      full_objective="fixed_pose_dbmol", call_counts=diagnostics,
                      start_preparation_seconds=preparation_seconds, setup_seconds=search_start-started,
                      total_seconds=perf_counter()-started)
        if cfg.get("vae_checkpoint"):
            result["vae_checkpoint"] = cfg["vae_checkpoint"]
    finally:
        native.close()
    result["total_seconds"] = perf_counter() - started
    write_json(args.output / "result.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "plan", "output", "boltz-assets"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    for key in ("arm", "plan-sha256"):
        parser.add_argument("--" + key, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
