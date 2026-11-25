"""Small, equal-allowance synthesis wave comparisons through the shared runner."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
from time import perf_counter

from rdkit import DataStructs
import torch

from latentmol.design.latent_protenix import _model_identity
from latentmol.design.protenix_matched_search import gaussian_start
from latentmol.design.protenix_restart_selection import _morgan_fingerprints
from latentmol.design.protenix_restart_study import score_shortlists
from latentmol.design.protenix_restarts import write_json
from latentmol.design.protenix_search_branches import Branch, execute_branches
from latentmol.design.protenix_search_frontier import SearchPoint
from latentmol.experiments import file_sha256
from latentmol.guidance.synthesis import load_models


def _frontier(entries):
    """Nondominated fresh loss / detached hard SA entries; both are minimized."""
    return [entry for entry in entries if not any(
        other[0].fresh_loss <= entry[0].fresh_loss
        and other[1]["chemistry"]["sa_score"] <= entry[1]["chemistry"]["sa_score"]
        and (other[0].fresh_loss < entry[0].fresh_loss
             or other[1]["chemistry"]["sa_score"] < entry[1]["chemistry"]["sa_score"])
        for other in entries)]


def select_wave_parents(archive, count, policy, max_similarity=.85, *,
                        min_step_gap=0, history=None, binding_ceiling=None,
                        ranking="legacy", apply_binding_ceiling_to_all=False,
                        vina=None, vina_role=None, vina_receipt=None):
    """Reserve the fresh winner, then fill from diverse binding/Pareto points.

    If a strict similarity threshold cannot fill the fixed budget, select the
    least similar remaining point and explicitly record that relaxation.
    ``legacy`` retains the original policy-specific ranking. ``diversity``
    applies the same similarity/fresh-loss ordering to all eligible pools;
    Pareto differs only by frontier filtering. ``sa-slot`` reserves the second
    slot for the lowest-SA eligible point, then uses the common diversity rank.
    The opt-in common ceiling also constrains binding-only nonelite slots.
    """
    if (policy not in {"binding", "pareto", "sa-slot"} or not 0 <= max_similarity <= 1
            or ranking not in {"legacy", "diversity"}
            or type(apply_binding_ceiling_to_all) is not bool
            or type(min_step_gap) is not int or min_step_gap < 0
            or (binding_ceiling is not None and (isinstance(binding_ceiling, bool)
                                                or not math.isfinite(binding_ceiling)))):
        raise ValueError("Invalid synthesis parent policy or similarity threshold")
    history = history or {}
    ranked = sorted(archive.values(), key=lambda row: (row[0].fresh_loss, row[0].candidate.smiles))
    if len(ranked) < count:
        raise ValueError("Not enough distinct archived parents for the fixed wave budget")
    fingerprints = dict(_morgan_fingerprints([row[0].candidate.smiles for row in ranked]))
    chosen, relaxed = ranked[:1], []
    enforce_ceiling = policy in {"pareto", "sa-slot"} or apply_binding_ceiling_to_all

    def too_close(row, others):
        c = row[0].candidate
        steps = [*history.get(c.branch, ()), *(p[0].candidate.step for p in others
                  if p[0].candidate.branch == c.branch)]
        return any(abs(c.step - step) < min_step_gap for step in steps)

    protected_reasons = []
    if too_close(chosen[0], []):
        protected_reasons.append("protected_best_spacing_exemption")
    if enforce_ceiling and binding_ceiling is not None and chosen[0][0].fresh_loss > binding_ceiling:
        protected_reasons.append("protected_best_ceiling_exemption")
    if protected_reasons:
        relaxed.append(dict(smiles=chosen[0][0].candidate.smiles, reasons=protected_reasons))
    while len(chosen) < count:
        remaining = [row for row in ranked if row[0].candidate.smiles not in
                     {item[0].candidate.smiles for item in chosen}]
        similarities = {row[0].candidate.smiles: max(DataStructs.BulkTanimotoSimilarity(
            fingerprints[row[0].candidate.smiles], [fingerprints[item[0].candidate.smiles] for item in chosen]))
                        for row in remaining}
        under_ceiling = [row for row in remaining if not enforce_ceiling or binding_ceiling is None
                         or row[0].fresh_loss <= binding_ceiling]
        spaced = [row for row in under_ceiling if not too_close(row, chosen)]
        eligible = [row for row in spaced if similarities[row[0].candidate.smiles] <= max_similarity]
        pool = eligible or spaced or under_ceiling or remaining
        if policy == "sa-slot" and len(chosen) == 1:
            selected = min(pool, key=lambda row: (row[1]["chemistry"]["sa_score"],
                           similarities[row[0].candidate.smiles], row[0].fresh_loss,
                           row[0].candidate.smiles))
        elif ranking == "diversity" or policy == "sa-slot":
            if policy == "pareto":
                pool = _frontier(pool)
            # Identical rank and ceiling across arms isolates frontier filtering.
            selected = min(pool, key=lambda row: (similarities[row[0].candidate.smiles],
                           row[0].fresh_loss, row[0].candidate.smiles))
        elif policy == "pareto":
            pool = _frontier(pool)
            selected = min(pool, key=lambda row: (similarities[row[0].candidate.smiles],
                           row[1]["chemistry"]["sa_score"], row[0].fresh_loss, row[0].candidate.smiles))
        elif eligible:
            selected = pool[0]
        else:
            selected = min(pool, key=lambda row: (similarities[row[0].candidate.smiles], row[0].fresh_loss))
        if vina_role and len(chosen) == count - 1:
            # Preserve earlier choices. Only the last slot may change, and only
            # when the ordinary policy can fill it without relaxing any gate.
            docking_pool = [row for row in pool if row[1]["chemistry"]["valid"]] if eligible else []
            ordering = sorted(docking_pool, key=lambda row: (
                row[0].candidate.smiles != selected[0].candidate.smiles,
                similarities[row[0].candidate.smiles], row[0].fresh_loss, row[0].candidate.smiles))
            rows = ordering[:vina.panel_size]
            scores = vina.score_many([row[0].candidate.smiles for row in rows])
            successful = [row for row in rows if scores[row[0].candidate.smiles]["status"] == "ok"]
            winner = min(successful, key=lambda row: scores[row[0].candidate.smiles]["affinity_kcal_mol"],
                         default=None)
            usual = selected[0].candidate.smiles
            if vina_role == "select" and winner is not None:
                selected = winner
            vina_receipt.update(slot_index=len(chosen), role=vina_role, eligible=len(ordering), scores=scores,
                standard_parent=usual, vina_parent=winner[0].candidate.smiles if winner else None,
                actual_parent=selected[0].candidate.smiles, changed=selected[0].candidate.smiles != usual)
        reasons = []
        if similarities[selected[0].candidate.smiles] > max_similarity:
            reasons.append("structural_similarity")
        if too_close(selected, chosen):
            reasons.append("source_step_spacing")
        if enforce_ceiling and binding_ceiling is not None and selected[0].fresh_loss > binding_ceiling:
            reasons.append("binding_ceiling")
        if reasons:
            relaxed.append(dict(smiles=selected[0].candidate.smiles, reasons=reasons))
        chosen.append(selected)
    return [row[0] for row in chosen], relaxed


def recovery_parent(row, original, *, strategy="terminal", archive=None, binding_ceiling=2.35):
    """Replay a terminal or scored joint-quality intermediate from its own branch.

    A repair witness is evidence, never a new latent. Joint recovery falls back
    to the original terminal/witness policy when no eligible scored point exists.
    """
    if (strategy not in {"terminal", "joint"}
            or (binding_ceiling is not None and (isinstance(binding_ceiling, bool)
                                                or not math.isfinite(binding_ceiling)))):
        raise ValueError("Invalid recovery strategy or binding ceiling")
    if strategy == "joint":
        branches = {candidate.branch for candidate in row.valid}
        if row.terminal is not None:
            branches.add(row.terminal.branch)
        eligible = [(point, record) for point, record in (archive or {}).values()
                    if point.candidate.branch in branches
                    and point.candidate.step not in row.nonreplayable_steps
                    and point.candidate.smiles is not None
                    and point.fresh_loss is not None and math.isfinite(point.fresh_loss)
                    and record.get("status") == "scored"
                    and record.get("chemistry", {}).get("valid", False)
                    and (binding_ceiling is None or point.fresh_loss <= binding_ceiling)]
        if eligible:
            point, _ = min(eligible, key=lambda item: (item[1]["chemistry"]["sa_score"],
                           item[0].fresh_loss, item[0].candidate.step, item[0].candidate.smiles))
            return point, "joint_archive"
        recovered, kind = recovery_parent(row, original)
        return recovered, "joint_fallback_" + kind
    parent = row.branch.parent
    if row.terminal is not None and row.terminal.smiles and row.terminal.step not in row.nonreplayable_steps:
        return SearchPoint(row.terminal, parent.active, None,
                           parent.ancestry + (row.terminal.branch,)), "valid_terminal"
    invalid = [point for point in row.invalid if point.candidate.smiles is None
               and type(point.repair_edits) is int and 1 <= point.repair_edits <= 2
               and point.repair_smiles and point.repair_target_sha256
               and point.candidate.step not in row.nonreplayable_steps]
    if invalid:
        return min(invalid, key=lambda p: (p.candidate.relaxed_loss, p.repair_edits, -p.candidate.step)), "near_valid_witness"
    valid = [candidate for candidate in row.valid if candidate.step not in row.nonreplayable_steps]
    if valid:
        last = max(valid, key=lambda c: (c.step, -c.relaxed_loss))
        return SearchPoint(last, parent.active, None, parent.ancestry + (last.branch,)), "last_saved_valid"
    return original, "original_parent_fallback"


def _point_state(point):
    c = point.candidate
    return dict(smiles=c.smiles, latent=c.latent.detach().cpu(),
                hard={key: getattr(c.hard, key).detach().cpu() for key in ("atoms", "charges", "bonds")},
                step=c.step, branch=c.branch, relaxed_loss=c.relaxed_loss, active=point.active.cpu(),
                fresh_loss=point.fresh_loss, ancestry=point.ancestry,
                repair_edits=point.repair_edits, repair_smiles=point.repair_smiles,
                repair_target_sha256=point.repair_target_sha256)


def _archive_metrics(archive):
    ranked = sorted(archive.values(), key=lambda row: (row[0].fresh_loss, row[0].candidate.smiles))
    if not ranked:
        return dict(best_fresh_loss=None, mean_top_three_fresh_loss=None,
                    archive_size=0, pareto_size=0, archive_best_sa=None)
    return dict(best_fresh_loss=ranked[0][0].fresh_loss,
                mean_top_three_fresh_loss=sum(row[0].fresh_loss for row in ranked[:3]) / min(3, len(ranked)),
                archive_size=len(ranked), pareto_size=len(_frontier(ranked)),
                archive_best_sa=min(row[1]["chemistry"]["sa_score"] for row in ranked))


def run_waves(decoder, make_scorer, fresh_score, config, task, output, checkpoint_sha256):
    """Run related synthesis schedules without cross-arm scoring starvation."""
    from latentmol.design import protenix_synthesis_study as pilot

    started = perf_counter()
    design, budget = pilot.resolve_pilot(config, task)
    waves, count = config["waves"], config.get("parent_count", 3)
    profile = config.get("chemistry_profile", [design.chemistry_initial] * waves)
    recovery_wave = config.get("recovery_wave", waves)
    if (not isinstance(profile, (list, tuple)) or len(profile) != waves
            or any(isinstance(x, bool) or not isinstance(x, (int, float))
                                     or not math.isfinite(x) or x < 0 for x in profile)
            or type(recovery_wave) is not int or not 2 <= recovery_wave <= waves
            or design.temperature_initial != design.temperature_final or design.max_step_norm_final is not None):
        raise ValueError("Synthesis waves require a finite chemistry profile and fixed replay temperature/cap")
    variants = pilot.pilot_variants(config, task)
    from latentmol.design.protenix_vina_parents import VinaParentPanel
    docking = VinaParentPanel(config["vina_parent_panel"]) if "vina_parent_panel" in config else None
    for variant in variants.values():
        for wave in range(1, waves + 1):
            resolved = pilot.resolved_variant(variant, wave)
            if resolved.get("vina_parent_role") not in {None, "audit", "select"} or (
                    resolved.get("vina_parent_role") is not None and (docking is None or count < 2)):
                raise ValueError("Vina parent roles require an explicit docking panel and a nonelite slot")
            if (resolved.get("parent_selection", "binding") not in {"binding", "pareto", "sa-slot"}
                    or resolved.get("selection_ranking", config.get("selection_ranking", "legacy"))
                        not in {"legacy", "diversity"}
                    or type(resolved.get("apply_binding_ceiling_to_all",
                                        config.get("apply_binding_ceiling_to_all", False))) is not bool
                    or resolved.get("recovery_start", "terminal") not in {"terminal", "joint"}):
                raise ValueError("Invalid synthesis selection ranking, ceiling policy, or recovery strategy")
    seed, limit = config["tasks"][task]["seed"], config.get("fresh_per_parent", 6)
    sa_mode = config.get("sa_mode", "rdkit")
    decoder.eval().requires_grad_(False)
    if any(p.grad is not None for p in decoder.parameters()):
        raise ValueError("Frozen decoder carries preexisting gradients")
    identity = _model_identity(decoder)
    parents, parent_records = pilot.load_pilot_parents(decoder, config, design, checkpoint_sha256)
    if any(not torch.equal(p.active, parents[0].active) for p in parents):
        raise ValueError("Shared synthesis scorer requires identical occupied slots across founders")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    assets = pilot.pilot_assets(config, task)
    models = load_models(decoder.spec, assets)
    try:
        model_ids = {name: _model_identity(model) for name, model in models.items()}
        anchor = pilot.load_anchor_libraries(config, parents, checkpoint_sha256)
        anchor_models = anchor if isinstance(anchor, dict) else ({"low_sa": anchor} if anchor is not None else {})
        anchor_ids = {name: _model_identity(model) for name, model in anchor_models.items()}
        anchor_id = _model_identity(anchor) if anchor is not None and not isinstance(anchor, dict) else None
        groups = config.get("founder_groups")
        fresh_founders = bool(groups) or config.get("founder_policy") == "audited-near-miss"
        all_founders = {p.candidate.smiles for p in parents}
        initial = [pilot._candidate_readout(p.candidate, p.active, models, decoder.spec, sa_mode)
                   | {"loss": p.fresh_loss, "status": "historical_founder"} for p in parents]
        group_indices = groups or {arm: list(range(count)) for arm in variants}
        prior_identities = set()
        founder_scoring_seconds = 0.
        if fresh_founders:
            panel = json.loads(Path(config["parent_panel"]["path"]).read_text(encoding="utf-8"))
            prior_identities = set(panel["historical_valid_smiles"]) | all_founders
            founder_root = output / "founder-fresh"
            founder_root.mkdir()
            founder_owners = groups or {"shared": list(range(count))}
            records = [dict(smiles=parents[i].candidate.smiles, start="founder-panel", role="founder",
                            arm=arm, step=parents[i].candidate.step) for arm, indices in founder_owners.items() for i in indices]
            began = perf_counter()
            scores = fresh_score(founder_root, records)
            founder_scoring_seconds = perf_counter() - began
            by_smiles = {r["smiles"]: r for r in scores}
            if len(scores) != len(records) or set(by_smiles) != all_founders:
                raise ValueError("Fresh founder score identities differ from the input panel")
            for i, p in enumerate(parents):
                score = by_smiles[p.candidate.smiles]
                loss = score["loss"]
                if (score["status"] == "scored") != (loss is not None) or (loss is not None and not math.isfinite(loss)):
                    raise ValueError("Inconsistent or nonfinite fresh founder score")
                parents[i] = replace(p, fresh_loss=loss)
                initial[i].update(loss=loss, status=score["status"])
        parents_by_arm = {arm: [parents[i] for i in indices] for arm, indices in group_indices.items()}
        archives = {arm: {parents[i].candidate.smiles: (parents[i], initial[i]) for i in indices
                          if parents[i].fresh_loss is not None} for arm, indices in group_indices.items()}
        caches = {arm: {} for arm in variants}
        totals = {arm: dict(updates=0, fresh_attempts=0, fresh_successes=0, valid_states=0,
                            observed_states=0, optimization_seconds=0., fresh_seconds=0.,
                            selected_sa_sum=0., selected_valid_count=0) for arm in variants}
        previous, cycles = {}, []
        if fresh_founders:
            for arm, indices in group_indices.items():
                caches[arm] = {parents[i].candidate.smiles: {k: initial[i][k] for k in ("loss", "status")}
                               for i in indices}
                totals[arm].update(fresh_attempts=count, fresh_successes=len(archives[arm]),
                                   founder_fresh_attempts=count, founder_fresh_successes=len(archives[arm]))
        histories = {arm: {} for arm in variants}
        write_json(output / "run.json", dict(runner="synthesis_waves", task=task, seed=seed,
            job_id=os.environ.get("LSB_JOBID"), budget=budget, design=asdict(design), variants=variants,
            chemistry_profile=profile, recovery_wave=recovery_wave, parents=parent_records, sa_mode=sa_mode,
            founder_groups=groups, historical_identity_exclusions=len(prior_identities),
            vina_parent_panel=config.get("vina_parent_panel"),
            shared_founder_scoring_seconds=founder_scoring_seconds,
            initial_parents=initial, graph_spec=decoder.spec.to_dict(), checkpoint_sha256=checkpoint_sha256,
            decoder_sha256=identity, synthesis_model_identities=model_ids,
            fresh_scoring_policy="equal_allowance", synthesis_assets=assets,
            anchor_bank=config.get("anchor_bank"), anchor_identity=anchor_id,
            anchor_libraries=config.get("anchor_libraries"), anchor_identities=anchor_ids,
            anchor_adapters={name: getattr(model, "adapter_metadata", {}) for name, model in anchor_models.items()},
            synthesis_adapters={name: getattr(model, "adapter_metadata", {
                "feature_policy": getattr(model, "feature_policy", "unspecified")}) for name, model in models.items()},
            guidance_scope="Frozen graph or latent objectives; hard chemistry is evaluation-only. "
            "Approximate model extensions do not establish synthesizability."))
        for wave in range(1, waves + 1):
            root = output / f"cycle-{wave:02d}"
            root.mkdir()
            current_design = replace(design, chemistry_initial=profile[wave - 1], chemistry_final=profile[wave - 1])
            wave_records, branch_records, results, start_plan = {}, [], {}, {}
            scorer = make_scorer(decoder.spec, parents[0].active, root)
            native_identity = getattr(scorer, "identity", None)
            try:
                for index, (arm, variant) in enumerate(variants.items()):
                    resolved = pilot.resolved_variant(variant, wave)
                    policy = resolved.get("parent_selection", "binding")
                    ranking = resolved.get("selection_ranking", config.get("selection_ranking", "legacy"))
                    common_ceiling = resolved.get("apply_binding_ceiling_to_all",
                                                 config.get("apply_binding_ceiling_to_all", False))
                    binding_ceiling = resolved.get("parent_binding_ceiling", config.get("parent_binding_ceiling"))
                    recovery = resolved.get("recovery_start", "terminal")
                    vina_receipt = {}
                    if wave == 1:
                        sources = parents_by_arm[arm]
                        kinds, relaxed = ["panel_founder" if groups else "shared_founder"] * count, []
                    elif wave == recovery_wave:
                        recovered = [recovery_parent(row, parents_by_arm[arm][i], strategy=recovery,
                            archive=archives[arm], binding_ceiling=resolved.get(
                                "recovery_binding_ceiling", binding_ceiling if binding_ceiling is not None else 2.35))
                                     for i, row in enumerate(previous[arm])]
                        sources, kinds = map(list, zip(*recovered))
                        relaxed = []
                    elif fresh_founders and len(archives[arm]) < count:
                        sources = [p for p, _ in sorted(archives[arm].values(), key=lambda row: row[0].fresh_loss)]
                        seen_sources = {p.candidate.smiles for p in sources}
                        fillers = [p for p in parents_by_arm[arm] if p.candidate.smiles not in seen_sources]
                        kinds = ["scored_archive"] * len(sources) + ["founder_shortfall"] * (count - len(sources))
                        sources += fillers[:count - len(sources)]
                        relaxed = [{"reason": "fewer scored parents than slots"}]
                    else:
                        sources, relaxed = select_wave_parents(archives[arm], count, policy,
                            resolved.get("parent_max_similarity", config.get("parent_max_similarity", .85)),
                            history=histories[arm], min_step_gap=resolved.get(
                                "min_parent_step_gap", config.get("min_parent_step_gap", 0)),
                            binding_ceiling=binding_ceiling, ranking=ranking,
                            apply_binding_ceiling_to_all=common_ceiling, vina=docking,
                            vina_role=resolved.get("vina_parent_role"), vina_receipt=vina_receipt)
                        kinds = ["archive_" + policy] * count
                    branch_policy = "continuous" if wave == recovery_wave and wave > 1 else config.get("restart_policy", "latent")
                    branches = [Branch(f"parent-{i:02d}-{arm}-00", source,
                                       seed + 1000 * wave + 100 * i, policy=branch_policy)
                                for i, source in enumerate(sources)]
                    independent = wave == waves and resolved.get("independent_final_slot", False)
                    if independent:
                        # The matched independent arm draws once, then uses that latent's
                        # own highest-occupancy slots without screening it for validity.
                        draw_seed = seed + (wave - 1) * count + count - 1
                        start = gaussian_start(decoder, current_design, draw_seed)
                        displaced = dict(source=kinds[-1], branch=sources[-1].candidate.branch,
                                         step=sources[-1].candidate.step, smiles=sources[-1].candidate.smiles)
                        branches[-1] = replace(branches[-1], parent=start, policy="bootstrap")
                        kinds[-1] = "raw_gaussian"
                    for b in branches:
                        source = b.parent
                        histories[arm].setdefault(source.candidate.branch, set()).add(source.candidate.step)
                    start_plan[arm] = [dict(source=kind, policy=b.policy,
                        parent_smiles=b.parent.candidate.smiles, parent_step=b.parent.candidate.step,
                        parent_branch=b.parent.candidate.branch, seed=b.seed,
                        active_slots=b.parent.active.nonzero().flatten().tolist(),
                        parent_fresh_loss=b.parent.fresh_loss,
                        parent_sa=archives[arm].get(b.parent.candidate.smiles, (None, {}))[1]
                            .get("chemistry", {}).get("sa_score"),
                        repair_edits=b.parent.repair_edits, repair_smiles=b.parent.repair_smiles)
                        for b, kind in zip(branches, kinds)]
                    if independent:
                        start_plan[arm][-1].update(draw_seed=draw_seed, displaced_continuation=displaced,
                            initialization="unscreened_standard_normal", initial_valid=start.candidate.smiles is not None)
                    write_json(root / f"parent-plan-{arm}.json", dict(wave=wave,
                        chemistry_weight=profile[wave - 1], arm=arm, starts=start_plan[arm],
                        **({"vina_decision": vina_receipt} if vina_receipt else {})))
                    arm_started = perf_counter()
                    results[arm] = execute_branches(decoder, scorer, current_design, root, branches,
                        batch_size=1, cpu_workers=1, overlap=False, repair_limit=2, invalid_limit=8,
                        guidance_factory=pilot.guidance_factory(models, variant, anchor, wave=wave),
                        ready=os.environ.get("DBMOL_LAUNCH_READY") if wave == 1 and index == 0 else None)
                    if len(results[arm]) != count or any(row.summary["updates"] != design.updates for row in results[arm]):
                        raise ValueError("Synthesis wave returned mismatched optimizer budgets")
                    totals[arm]["optimization_seconds"] += perf_counter() - arm_started
                    wave_records[arm] = dict(parent_selection=policy, parent_selection_relaxations=relaxed,
                        selection_ranking=ranking, apply_binding_ceiling_to_all=common_ceiling,
                        parent_binding_ceiling=binding_ceiling, recovery_start=recovery,
                        phase=resolved.get("phase", "constant"), guidance=resolved,
                        **({"vina_decision": vina_receipt} if vina_receipt else {}))
            finally:
                scorer.close()
            write_json(root / "wave-plan.json", dict(wave=wave, chemistry_weight=profile[wave - 1], arms=start_plan))
            for arm, rows in results.items():
                fresh_root = root / f"fresh-{arm}"
                fresh_root.mkdir()
                seen, selections, selected = set(caches[arm]), {}, {}
                for row in rows:
                    eligible = sorted((c for c in row.valid if c.smiles not in seen and c.step not in row.nonreplayable_steps),
                                      key=lambda c: (c.relaxed_loss, c.step, c.smiles))
                    chosen = pilot.synthesis_shortlist(eligible, limit, pilot.resolved_variant(
                        variants[arm], wave).get("shortlist_policy", config.get("shortlist_policy", "binding")), sa_mode)
                    selected[row.branch.name] = chosen
                    seen.update(c.smiles for c in chosen)
                    selections[row.branch.name] = {"mixed": [dict(smiles=c.smiles, step=c.step,
                                                                 relaxed_loss=c.relaxed_loss) for c in chosen]}
                fresh_started = perf_counter()
                attempts = score_shortlists(fresh_score, fresh_root, selections, caches[arm])
                if attempts != sum(map(len, selected.values())) or attempts > count * limit:
                    raise ValueError("Synthesis wave exceeded its independent fresh allowance")
                successes, readouts = 0, []
                for i, row in enumerate(rows):
                    candidates = []
                    for candidate in selected[row.branch.name]:
                        score = caches[arm][candidate.smiles]
                        if (score["status"] == "scored") != (score["loss"] is not None):
                            raise ValueError("Inconsistent fresh synthesis score/status")
                        record = pilot._candidate_readout(candidate, row.branch.parent.active, models, decoder.spec, sa_mode) | score
                        record["is_founder"] = candidate.smiles in all_founders
                        candidates.append(record)
                        if score["loss"] is not None:
                            successes += 1
                            point = SearchPoint(candidate, row.branch.parent.active, score["loss"],
                                                row.branch.parent.ancestry + (candidate.branch,))
                            old = archives[arm].get(candidate.smiles)
                            if old is None or point.fresh_loss < old[0].fresh_loss:
                                archives[arm][candidate.smiles] = point, record
                    readouts.extend(candidates)
                    branch_records.append(dict(name=row.branch.name, arm=arm, parent_index=i,
                        seed=row.branch.seed, policy=row.branch.policy, parent_source=start_plan[arm][i],
                        parent_smiles=row.branch.parent.candidate.smiles,
                        parent_fresh_loss=row.branch.parent.fresh_loss, summary=row.summary,
                        candidates={"mixed": candidates}, endpoint=pilot._candidate_readout(
                            row.terminal, row.branch.parent.active, models, decoder.spec, sa_mode) if row.terminal else None))
                current = dict(updates=count * design.updates, fresh_attempt_allowance=count * limit,
                    fresh_attempts=attempts, fresh_successes=successes, fresh_failures=attempts - successes,
                    fresh_shortfall=count * limit - attempts, valid_states=sum(r.summary.get("valid_states", 0) for r in rows),
                    observed_states=count * (design.updates + 1), fresh_seconds=perf_counter() - fresh_started)
                for key in current.keys() & totals[arm].keys():
                    totals[arm][key] += current[key]
                chem = [r["chemistry"] for r in readouts if r["chemistry"]["valid"]]
                totals[arm]["selected_sa_sum"] += sum(c["sa_score"] for c in chem)
                totals[arm]["selected_valid_count"] += len(chem)
                wave_records[arm].update(current, **_archive_metrics(archives[arm]),
                    trajectory_valid_fraction=current["valid_states"] / current["observed_states"],
                    selected_mean_sa=sum(c["sa_score"] for c in chem) / len(chem) if chem else None,
                    joint_quality=pilot.joint_quality_metrics([record for _, record in archives[arm].values()],
                        exclude=all_founders),
                    cumulative_updates=totals[arm]["updates"], cumulative_fresh_attempts=totals[arm]["fresh_attempts"])
                if fresh_founders:
                    wave_records[arm]["joint_quality_new_to_history"] = pilot.joint_quality_metrics(
                        [record for _, record in archives[arm].values()], exclude=prior_identities)
            record = dict(cycle=wave, chemistry_weight=profile[wave - 1], fresh_scoring_policy="equal_allowance", arms=wave_records)
            write_json(root / "batch.json", dict(branches=branch_records, native_identity=native_identity,
                       optimization_updates=sum(r["updates"] for r in wave_records.values()),
                       new_fresh_attempts=sum(r["fresh_attempts"] for r in wave_records.values())))
            write_json(root / "wave-result.json", record)
            if config.get("save_wave_state", True):
                next_variants = {arm: pilot.resolved_variant(variant, min(wave + 1, waves))
                                 for arm, variant in variants.items()}
                torch.save(dict(wave=wave, caches=caches, archives={arm: {s: _point_state(p) for s, (p, _) in archive.items()}
                           for arm, archive in archives.items()}, recovery_starts={arm: [_point_state(recovery_parent(
                               r, r.branch.parent if r.branch.policy == "bootstrap" else parents_by_arm[arm][i],
                               strategy=next_variants[arm].get("recovery_start", "terminal"),
                               archive=archives[arm], binding_ceiling=next_variants[arm].get(
                                   "recovery_binding_ceiling", next_variants[arm].get(
                                       "parent_binding_ceiling", config.get("parent_binding_ceiling", 2.35))))[0])
                           for i, r in enumerate(rows)] for arm, rows in results.items()},
                           histories={arm: {branch: sorted(steps) for branch, steps in history.items()}
                                      for arm, history in histories.items()}), root / "archive-state.pt")
            cycles.append(record)
            previous = results
            print(json.dumps(record, allow_nan=False), flush=True)
        if (_model_identity(decoder) != identity or any(p.grad is not None for p in decoder.parameters())
                or any(_model_identity(model) != model_ids[name] or any(p.grad is not None for p in model.parameters())
                       for name, model in models.items())
                or any(_model_identity(model) != anchor_ids[name]
                       or any(p.grad is not None for p in model.parameters())
                       for name, model in anchor_models.items())):
            raise ValueError("Frozen synthesis wave model changed")
        if file_sha256(Path(config["parent_panel"]["path"])) != config["parent_panel"]["sha256"]:
            raise ValueError("Synthesis founder panel changed during waves")
        if any(t["updates"] != budget["updates_per_arm"] or t["fresh_attempts"] > budget["fresh_attempts_per_arm"]
               for t in totals.values()):
            raise ValueError("Completed synthesis wave budgets differ from the launch plan")
        result = dict(status="completed", planned=budget, cycles=cycles, arms={arm: totals[arm] | _archive_metrics(archives[arm])
            | {"trajectory_valid_fraction": totals[arm]["valid_states"] / totals[arm]["observed_states"],
               "fresh_attempt_allowance": budget["fresh_attempts_per_arm"],
               "fresh_shortfall": budget["fresh_attempts_per_arm"] - totals[arm]["fresh_attempts"],
               "relaxed_calls": totals[arm]["updates"] + count * waves,
               "selected_mean_sa": totals[arm]["selected_sa_sum"] / totals[arm]["selected_valid_count"]
                   if totals[arm]["selected_valid_count"] else None,
               "joint_quality": pilot.joint_quality_metrics([record for _, record in archives[arm].values()],
                   exclude=all_founders),
               "valid_endpoints": sum(bool(row.terminal and row.terminal.smiles) for row in previous[arm]),
               "endpoint_count": count} for arm in variants},
            fresh_scoring_policy="equal_allowance",
            physical_discovery_fresh_attempts=sum(t["fresh_attempts"] for t in totals.values())
                - (count * (len(variants) - 1) if fresh_founders and not groups else 0),
            frozen_decoder_unchanged=True, frozen_synthesis_models_unchanged=True,
            interpretation="Equal optimizer budgets and independent maximum fresh allowances; actual scoring can differ. "
            "Recovery replays saved latents, including witnessed near-valid states; no discrete graph edit is used.")
        if fresh_founders:
            for arm in variants:
                result["arms"][arm]["joint_quality_new_to_history"] = cycles[-1]["arms"][arm]["joint_quality_new_to_history"]
        if any(variant.get("independent_final_slot") for variant in variants.values()):
            excluded = prior_identities | all_founders
            for row in branch_records:
                if row["parent_index"] == count - 1:
                    scored = row["candidates"]["mixed"]
                    result["arms"][row["arm"]]["final_slot"] = dict(
                        branch=row["name"], start=row["parent_source"], trajectory=row["summary"],
                        fresh_attempts=len(scored), fresh_successes=sum(r["loss"] is not None for r in scored),
                        joint_quality_new_to_history=pilot.joint_quality_metrics(scored, exclude=excluded))
        if docking:
            excluded = prior_identities | all_founders
            for arm, archive in archives.items():
                # Dock descendants chosen solely by endpoint chemistry and fresh
                # loss, so scoring a parent well is not itself a success.
                eligible = sorted(((p, r) for smiles, (p, r) in archive.items()
                                   if smiles not in excluded
                                   and r["chemistry"]["valid"] and r["chemistry"]["sa_score"] <= 6
                                   and p.fresh_loss <= 2.3),
                                  key=lambda row: (row[0].fresh_loss, row[1]["chemistry"]["sa_score"],
                                                   row[0].candidate.smiles))
                panel = eligible[:docking.panel_size]
                final_scores = docking.score_many([p.candidate.smiles for p, _ in panel])
                result["arms"][arm]["vina_new_hits"] = dict(eligible=len(eligible),
                    candidates=[dict(smiles=p.candidate.smiles, sa=r["chemistry"]["sa_score"],
                                     fresh_loss=p.fresh_loss, branch=p.candidate.branch, step=p.candidate.step,
                                     vina=final_scores[p.candidate.smiles]) for p, r in panel])
            result["vina_accounting"] = docking.accounting()
        result["elapsed_seconds"] = perf_counter() - started
        write_json(output / "result.json", result)
        return result
    finally:
        pilot.close_models(models)
