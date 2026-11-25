"""Matched per-update docking guidance and hard redocking of one archived founder pair."""
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import re
import sys

import torch

from latentmol.design.latent_protenix import LatentDesignConfig, _model_identity
from latentmol.design.protenix_restart_study import score_shortlists
from latentmol.design.protenix_restarts import write_json
from latentmol.design.protenix_search_branches import Branch, execute_branches
from latentmol.design.protenix_synthesis_study import (
    close_models, guidance_factory, joint_quality_metrics, load_pilot_parents,
    molecule_readout, pilot_variants, synthesis_shortlist,
)
from latentmol.design.protenix_vina_parents import VinaParentPanel
from latentmol.experiments import file_sha256
from latentmol.guidance.synthesis import load_models


def resolve_pair(config, task):
    design = LatentDesignConfig.for_restart(config["design"])
    boltz = config["study"] == "boltz_tacogfn"
    ratio_key = "taco_ratio" if boltz else "taco_cap_ratio"
    if boltz and (config.get("norm_mode") != "match" or config["scorer"]["kind"] != "boltz2_dbmol"
                  or design.occupied_atoms != 31):
        raise ValueError("The BRD4 Boltz pair requires fixed-pose 31-slot DBMol and norm matching")
    variants = pilot_variants(config, task)
    rows = list(variants.values())
    cap = config["fresh_per_parent"]
    tasks = config["tasks"]
    if (not (config["parent_count"] == len(tasks) == 3)
            or {row["founder_index"] for row in tasks.values()} != {0, 1, 2}
            or len({row["founder_smiles"] for row in tasks.values()}) != 3
            or len(rows) != 2 or sorted(row[ratio_key] for row in rows) != [0, .5 if boltz else .05]
            or {k:v for k,v in rows[0].items() if k != ratio_key}
               != {k:v for k,v in rows[1].items() if k != ratio_key}
            or config["restart_policy"] != "continuous" or design.noise_initial != 0
            or design.updates != design.schedule_horizon or config["waves"] != 1
            or config["sa_mode"] != "pre_smoothing"
            or config["founder_policy"] != ("boltz-archive-fresh" if boltz else "audited-near-miss")
            or config["fresh_scoring_policy"] != "matched_minimum"
            or type(cap) is not int or not 1 <= cap <= 16 or config["vina_parent_panel"]["panel_size"] != cap):
        raise ValueError("Expected two matched Taco arms, identical base protocol and equal scoring budgets")
    for name in ("checkpoint", "pocket", "topologies"):
        asset = config["tacogfn"][name]
        if not asset["path"] or not re.fullmatch(r"[0-9a-f]{64}", asset["sha256"]):
            raise ValueError(f"Missing path or digest for Taco {name}")
    return design, dict(branches=2, updates=2*design.updates, relaxed_calls=2*(design.updates+1),
                        updates_per_arm=design.updates, fresh_attempts=2*cap, fresh_attempts_per_arm=cap,
                        native_vina_attempts=2*cap, native_vina_attempts_per_arm=cap)


def matched_candidates(results, founder_smiles, limit):
    """Use hard valid nonfounders, falling back to any SA and historical identity."""
    pools = {}
    for arm, result in results.items():
        unique = {}
        for candidate in sorted(result.valid, key=lambda c: (c.relaxed_loss, c.step, c.smiles)):
            if candidate.smiles and candidate.smiles not in founder_smiles:
                unique.setdefault(candidate.smiles, candidate)
        pools[arm] = list(unique.values())
    count = min(limit, *(len(pool) for pool in pools.values()))
    selected = {}
    for arm, pool in pools.items():
        preferred = synthesis_shortlist(pool, count, "audited-binding-diverse", "pre_smoothing") if count else []
        chosen = {c.smiles for c in preferred}
        selected[arm] = (preferred + [c for c in pool if c.smiles not in chosen])[:count]
    return selected, {arm: len(pool) for arm, pool in pools.items()}


def _guidance_usage(metrics, updates, cap, norm_mode="cap"):
    counts = dict(updates=updates, active_steps=0, nonzero_steps=0, max_applied_ratio=0., skipped={})
    with metrics.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["step"] == updates:
                continue  # Endpoint is scored but takes no update.
            metadata = row["synthesis"].get("tacogfn")
            if cap == 0:
                if metadata is not None:
                    raise ValueError("Unexpected Taco guidance in the off arm")
                continue
            ratio = metadata["actual_ratio"]
            if ratio is None and metadata["binding_gradient_norm"] == 0:
                ratio = 0.
            if not math.isfinite(ratio) or not 0 <= ratio <= cap + 1e-6:
                raise ValueError("Taco contribution exceeded the configured gradient cap")
            counts["max_applied_ratio"] = max(counts["max_applied_ratio"], ratio)
            if metadata["active"]:
                if (norm_mode == "match" and metadata["raw_gradient_norm"] > 0
                        and metadata["binding_gradient_norm"] > 0
                        and not math.isclose(ratio, cap, abs_tol=1.e-6, rel_tol=0)):
                    raise ValueError("Taco gradient did not reach the requested Boltz norm ratio")
                counts["active_steps"] += 1
                counts["nonzero_steps"] += metadata["applied_gradient_norm"] > 0
            else:
                reason = metadata["skip_reason"]
                counts["skipped"][reason] = counts["skipped"].get(reason, 0) + 1
    if cap and counts["active_steps"] + sum(counts["skipped"].values()) != updates:
        raise ValueError("Incomplete per-update Taco guidance records")
    return counts


def _boltz_hard_scores(root, selected, fresh_score):
    """Pass the original decoded hard graph in its exact native pose-slot order."""
    from latentmol.guidance.boltz_search import hard_graph_record, hard_pose_key

    inputs = []
    for candidate in selected:
        graph = hard_graph_record(candidate.hard)
        inputs.append(dict(smiles=candidate.smiles, hard_graph=graph, scoring_key=hard_pose_key(graph),
                           step=candidate.step, arm=candidate.branch, role="child_trajectory"))
    results = fresh_score(root, inputs) if inputs else []
    expected = {row["smiles"]: row["scoring_key"] for row in inputs}
    if (len(results) != len(inputs) or {r["smiles"]: r["scoring_key"] for r in results} != expected
            or any(row["loss"] is not None and not math.isfinite(row["loss"]) for row in results)):
        raise ValueError("Invalid Boltz FP32 score or different hard molecules/pose slots returned")
    return {r["smiles"]: {k:r[k] for k in ("status","loss","scoring_key")} for r in results}


def _evaluate_arm(root, selected, fresh_score, docking_config, history, *, boltz=False):
    """Submit exactly the same identities to both scorers regardless of individual failures."""
    records = [dict(smiles=c.smiles, step=c.step, relaxed_loss=c.relaxed_loss) for c in selected]
    names = [row["smiles"] for row in records]
    chemistry = {name: molecule_readout(name, "pre_smoothing") for name in names}
    if len(names) != len(set(names)) or any(not r["valid"] for r in chemistry.values()):
        raise ValueError("Only distinct connected valid argmax identities can enter the paired panel")
    root.mkdir()
    if boltz:
        cache = _boltz_hard_scores(root, selected, fresh_score)
        attempts = len(cache)
    else:
        cache = {}
        attempts = score_shortlists(fresh_score, root, {root.name: {"mixed": records}}, cache)
    docking = VinaParentPanel(docking_config)
    vina = docking.score_many(names)
    if attempts != len(names) or docking.accounting()["unique_attempts"] != attempts:
        raise ValueError("Fresh and native Vina attempt counts differ")
    rows = []
    for row in records:
        name = row["smiles"]
        if (cache[name]["status"] == "scored") != (cache[name]["loss"] is not None):
            raise ValueError("Inconsistent fresh scoring outcome")
        rows.append(row | cache[name] | dict(chemistry=chemistry[name], vina=vina[name],
                    new_to_history=name not in history))
    write_json(root / "hard-evaluations.json", rows)
    scored = [r for r in rows if r["vina"]["status"] == "ok"]
    jointly_scored = [r for r in scored if r["loss"] is not None]
    outcome = dict(candidates=rows, fresh_attempts=attempts, fresh_scored=sum(r["loss"] is not None for r in rows),
                native_vina=docking.accounting(), jointly_scored=len(jointly_scored),
                best_vina=min((r["vina"]["affinity_kcal_mol"] for r in scored), default=None))
    if boltz:
        return outcome | dict(fresh_objective="fixed_pose_boltz2_dbmol_total_loss",
            best_hard_dbmol_loss=min((r["loss"] for r in rows if r["loss"] is not None), default=None),
            best_vina_at_sa5=min((r["vina"]["affinity_kcal_mol"] for r in scored if r["chemistry"]["sa_score"] <= 5), default=None))
    return outcome | dict(best_vina_at_sa5_loss23=min((r["vina"]["affinity_kcal_mol"] for r in jointly_scored
                    if r["loss"] <= 2.3 and r["chemistry"]["sa_score"] <= 5), default=None),
                joint_quality=joint_quality_metrics(rows), new_to_history_quality=joint_quality_metrics(rows, history))


@torch.no_grad()
def _boltz_parents(decoder, config, design):
    from latentmol.chemistry import inspect_graph
    from latentmol.design.boltz_search_methods import parent_state
    from latentmol.design.latent_protenix import hard_from_graph
    from latentmol.graph_policy import graph_from_decoder
    from latentmol.guidance.boltz_search import hard_graph_record

    state = parent_state(config, decoder)
    eligible = [point for key, point in state["archive"].items() if key not in state["ineligible"]]
    parents = sorted(eligible, key=lambda p: (p.fresh_loss, p.candidate.smiles))[:config["parent_count"]]
    records = []
    for parent in parents:
        c = parent.candidate
        graph = graph_from_decoder(decoder.decode(c.latent), decoder.spec, fixed_active=parent.active,
                                   temperature=design.temperature_initial)
        replay = hard_from_graph(graph)
        labels = hard_graph_record(c.hard)
        if (int(parent.active.sum()) != design.occupied_atoms
                or hard_graph_record(replay) != labels
                or inspect_graph(replay.atoms[0], replay.charges[0], replay.bonds[0], decoder.spec).smiles != c.smiles):
            raise ValueError("Archived Boltz founder does not replay its saved hard graph and slot mask")
        records.append(dict(smiles=c.smiles, fresh_loss=parent.fresh_loss, branch=c.branch, step=c.step,
                            active_slots=parent.active.nonzero().flatten().tolist(), labels=labels))
    return parents, records, set(state["archive"])


def run_pair(decoder, make_scorer, fresh_score, config, task, output, checkpoint_sha256):
    design, budget = resolve_pair(config, task)
    boltz = config["study"] == "boltz_tacogfn"
    norm_mode = config.get("norm_mode", "cap")
    ratio_key = "taco_ratio" if boltz else "taco_cap_ratio"
    variants = pilot_variants(config, task)
    settings = config["tasks"][task]
    decoder.eval().requires_grad_(False)
    if any(p.grad is not None for p in decoder.parameters()):
        raise ValueError("Frozen decoder carries preexisting gradients")
    decoder_identity = _model_identity(decoder)
    if boltz:
        parents, records, history = _boltz_parents(decoder, config, design)
    else:
        parents, records = load_pilot_parents(decoder, config, design, checkpoint_sha256)
        panel = json.loads(Path(config["parent_panel"]["path"]).read_text())
        history = set(panel["historical_valid_smiles"])
    founder_smiles = [row["smiles"] for row in records]
    if founder_smiles != [r["founder_smiles"] for r in sorted(config["tasks"].values(), key=lambda r:r["founder_index"])]:
        raise ValueError("Archived parents do not match the requested founders in order")
    index = settings["founder_index"]
    parent, record = parents[index], records[index]
    assets = config["tacogfn"]
    for name in ("checkpoint", "pocket", "topologies"):
        if file_sha256(Path(assets[name]["path"])) != assets[name]["sha256"]:
            raise ValueError(f"Changed Taco {name} asset")
    if assets.get("openbabel_site"):
        sys.path.insert(0, assets["openbabel_site"])
    from latentmol.guidance.tacogfn import TacoGFNGuidance
    from latentmol.guidance.tacogfn_head import TacoGFNHead

    head = TacoGFNHead.from_checkpoint(assets["checkpoint"]["path"], assets["pocket"]["path"], device="cpu")
    topologies = json.loads(Path(assets["topologies"]["path"]).read_text())
    models = load_models(decoder.spec, config["synthesis_assets"])
    try:
        model_ids = {name: _model_identity(model) for name, model in models.items()}
        head_identity = _model_identity(head)
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        root = output / "cycle-01"
        root.mkdir()
        write_json(output / "run.json", dict(runner="tacogfn_paired", task=task, budget=budget,
            parent=record, founder_smiles=founder_smiles, seed=settings["seed"], design=asdict(design),
            variants=variants, arm_order=list(variants), job_id=os.environ.get("LSB_JOBID"),
            assets=assets, parent_panel=config["parent_state" if boltz else "parent_panel"], checkpoint_sha256=checkpoint_sha256,
            decoder_sha256=decoder_identity, native_vina=config["vina_parent_panel"],
            base_guidance_assets=config["synthesis_assets"], historical_identities=len(history),
            binding_model="Boltz2 DBMol" if boltz else "Protenix", norm_mode=norm_mode))
        scorer = make_scorer(decoder.spec, parent.active, root)
        results = {}
        try:
            for arm, variant in variants.items():
                branch = Branch(arm, parent, settings["seed"], policy="continuous")
                base = guidance_factory(models, variant)(branch)
                guide = (TacoGFNGuidance(head, record, decoder.spec, topologies[record["smiles"]],
                         cap_ratio=variant[ratio_key], base_guidance=base, norm_mode=norm_mode)
                         if variant[ratio_key] else base)
                (row,) = execute_branches(decoder, scorer, design, root, [branch], batch_size=1,
                    cpu_workers=1, overlap=False, repair_limit=0, invalid_limit=0,
                    guidance_factory=lambda _branch: guide,
                    ready=os.environ.get("DBMOL_LAUNCH_READY") if not results else None)
                if row.summary["updates"] != design.updates:
                    raise ValueError("Unexpected completed update count")
                results[arm] = row
        finally:
            if not boltz:
                scorer.close()  # Boltz stays resident for pose-aware hard scoring; its caller owns it.
        candidates, eligible = matched_candidates(results, set(founder_smiles), config["fresh_per_parent"])
        write_json(root / "shortlists.json", {arm:[dict(smiles=c.smiles, step=c.step,
                   relaxed_loss=c.relaxed_loss) for c in rows] for arm, rows in candidates.items()})
        arms = {}
        for arm, rows in candidates.items():
            readout = _evaluate_arm(root / f"fresh-{arm}", rows, fresh_score,
                                    config["vina_parent_panel"], history, boltz=boltz)
            arms[arm] = results[arm].summary | readout | dict(distinct_valid_nonfounders=eligible[arm],
                taco_usage=_guidance_usage(root / arm / "metrics.jsonl", design.updates, variants[arm][ratio_key], norm_mode),
                fresh_attempt_allowance=budget["fresh_attempts_per_arm"],
                vina_attempt_allowance=budget["native_vina_attempts_per_arm"],
                matched_shortfall=config["fresh_per_parent"] - len(rows))
            write_json(root / f"{arm}-outcome.json", arms[arm])
        counts = {row["fresh_attempts"] for row in arms.values()} | {
            row["native_vina"]["unique_attempts"] for row in arms.values()}
        if len(counts) != 1 or max(counts) > config["fresh_per_parent"]:
            raise ValueError("Actual paired scoring budgets differ")
        frozen = [(decoder, decoder_identity), (head, head_identity), *((model, model_ids[key]) for key, model in models.items())]
        if any(_model_identity(model) != digest or any(p.grad is not None for p in model.parameters())
               for model, digest in frozen):
            raise ValueError("Frozen model weights or gradients changed")
        result = dict(status="completed", task=task, founder=record["smiles"],
                      completed_updates=sum(r["updates"] for r in arms.values()),
                      planned=budget, arms=arms, matched_attempts_per_scorer_per_arm=max(counts))
        if not boltz:
            write_json(output / "result.json", result)
        return result
    finally:
        close_models(models)
