"""Matched exploratory comparisons of Protenix restart and allocation policies."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import partial
from itertools import zip_longest
import json
import math
import os
from pathlib import Path

from latentmol.design.latent_protenix import LatentDesignConfig, _model_identity
from latentmol.design.protenix_restart_parents import RestartParent, load_parent_panel, make_prior_parent
from latentmol.design.protenix_restart_proposals import propose_start
from latentmol.design.protenix_restart_selection import select_candidates, select_diverse_output
from latentmol.design.protenix_restarts import (
    fresh_score_records, native_factory, restart_args, run_branch, write_json,
)
from latentmol.evaluation.evaluate import load_model


@dataclass(frozen=True)
class Branch:
    arm: str
    parent: RestartParent
    parent_index: int
    repeat: int
    seed: int
    updates: int
    cap: float
    policy: str
    shortlist: int
    strength: float | None = None
    update_noise: float | None = None
    cap_final: float | None = None
    no_bond_weight: float | None = None

    @property
    def name(self):
        return f"parent-{self.parent_index:02d}-{self.arm}-{self.repeat:02d}"


FEEDBACK_ARMS = ("adaptive", "fixed", "independent")


def budgets(comparison, *, feedback=False):
    if feedback and ("variants" in comparison or any(type(comparison[key]) is not int or comparison[key] < 1
                     for key in ("parents", "cycles", "branches_per_parent", "updates", "shortlist"))
                     or not math.isfinite(comparison["high_score_ceiling"])):
        raise ValueError("Invalid lineage feedback budget or high-score ceiling")
    hybrid_priors = comparison.get("hybrid_prior_children", 0)
    if (type(hybrid_priors) is not int or not 0 <= hybrid_priors <= comparison["parents"]
            or (hybrid_priors and (not feedback or comparison["branches_per_parent"] < 2))):
        raise ValueError("Hybrid priors may replace at most one repeat per parent in a lineage feedback study")
    if "variants" in comparison:
        runs = [updates for variant in comparison["variants"].values() for updates in variant["updates"]]
        branches = comparison["parents"] * len(runs)
        updates = comparison["parents"] * sum(runs)
    else:
        arms = len(FEEDBACK_ARMS) if feedback else 2
        branches = arms * comparison["parents"] * comparison["cycles"] * comparison["branches_per_parent"]
        updates = branches * comparison["updates"]
    if branches < 1 or not 1 <= updates <= 10_000:
        raise ValueError(f"Exploratory comparison exceeds its 10,000-update cap: {updates}")
    return dict(branches=branches, optimization_updates=updates, relaxed_calls=updates + branches)


def paired_branches(parents, comparison, design, seed):
    for index, parent in enumerate(parents):
        for arm, variant in comparison["variants"].items():
            for repeat, updates in enumerate(variant["updates"]):
                yield Branch(arm, parent, index, repeat, seed + 100 * index + repeat,
                    updates, variant.get("cap", design.max_step_norm), variant["policy"], variant["shortlist"],
                    variant.get("strength"), variant.get("update_noise"), variant.get("cap_final"),
                    variant.get("no_bond_weight"))


def parent_record(parent):
    return dict(parent.candidate.record(parent.fresh_loss),
                active_slots=parent.active.nonzero().flatten().tolist(), provenance=parent.provenance)


def matched_shortlists(branches, reserves, arm_caches, limit):
    """Give every arm the same number of new SMILES, spreading calls over branches."""
    pools = {}
    for arm, cache in arm_caches.items():
        names = [branch.name for branch in branches if branch.arm == arm]
        pool = {}
        for round_rows in zip_longest(*(reserves[name]["mixed"] for name in names)):
            for name, row in zip(names, round_rows):
                if row is not None and row.smiles not in cache:
                    pool.setdefault(row.smiles, (name, row))
                if len(pool) == limit:
                    break
            if len(pool) == limit:
                break
        pools[arm] = list(pool.values())
    count = min(map(len, pools.values()))
    selections = {branch.name: {"mixed": []} for branch in branches}
    for pool in pools.values():
        for name, row in pool[:count]:
            selections[name]["mixed"].append(row)
    return selections


def matched_parent_shortlists(branches, reserves, arm_caches, limit):
    """Match attempts within each parent and reserve identities across parents."""
    reserved = {arm: set(cache) for arm, cache in arm_caches.items()}
    selections = {}
    for parent in dict.fromkeys(branch.parent_index for branch in branches):
        group = [branch for branch in branches if branch.parent_index == parent]
        selected = matched_shortlists(group, reserves, reserved, limit)
        selections.update(selected)
        for branch in group:
            reserved[branch.arm].update(row.smiles for row in selected[branch.name]["mixed"])
    return selections


def score_shortlists(fresh_score, root, selections, cache):
    pending = {}
    for name, lists in selections.items():
        for rows in lists.values():
            for row in rows:
                if row["smiles"] not in cache:
                    pending.setdefault(row["smiles"], dict(smiles=row["smiles"], start=root.name,
                        arm=name, role="edit", step=row["step"]))
    if pending:
        fresh = fresh_score(root, list(pending.values()))
        if len(fresh) != len(pending) or {row["smiles"] for row in fresh} != pending.keys():
            raise ValueError("Hard scoring returned a different candidate set")
        for row in fresh:
            loss = row["loss"]
            if loss is not None and not math.isfinite(loss):
                raise ValueError("Nonfinite fresh contact loss")
            cache[row["smiles"]] = dict(loss=loss, status=row["status"])
    return len(pending)


def run_batch(decoder, make_scorer, fresh_score, design, root, branches, cache, *,
              compare_shortlists=False, first_batch=True, noise_study=False,
              arm_caches=None, fresh_limit=None, match_by_parent=False):
    if noise_study:
        from latentmol.design.protenix_restart_noise import propose_noise_start

    root.mkdir()
    selections = {}
    scorer = make_scorer(decoder.spec, branches[0].parent.active, root)
    try:
        for index, branch in enumerate(branches):
            parent = branch.parent
            proposal_kwargs = dict(policy=branch.policy, seed=branch.seed,
                                   temperature=design.temperature_initial)
            noise_kwargs = {"no_bond_weight": branch.no_bond_weight} if branch.no_bond_weight is not None else {}
            proposal = (propose_noise_start(decoder, parent.candidate, parent.active,
                                            strength=branch.strength, **proposal_kwargs, **noise_kwargs)
                        if noise_study else
                        propose_start(decoder, parent.candidate, parent.active, **proposal_kwargs))
            config = replace(design, updates=branch.updates, max_step_norm=branch.cap)
            if branch.cap_final is not None:
                config = replace(config, max_step_norm_final=branch.cap_final)
            if branch.update_noise is not None:
                config = replace(config, noise_initial=branch.update_noise,
                                 noise_seed=design.noise_seed + branch.seed)

            def select(ranked, limit):
                if arm_caches is not None:
                    eligible = [row for row in ranked if row.smiles not in arm_caches[branch.arm]]
                    preferred = select_candidates(eligible, limit, False)["mixed"]
                    preferred_ids = {row.smiles for row in preferred}
                    reserve = preferred + [row for row in eligible if row.smiles not in preferred_ids]
                    selections[branch.name] = {"mixed": reserve if match_by_parent else reserve[:fresh_limit]}
                    return reserve[:fresh_limit]
                # Include an independent start in its own fresh budget. Elite
                # starts already have a matched archived score.
                initial = [row for row in ranked if row.smiles == parent.candidate.smiles
                           and parent.fresh_loss is None]
                novel = [row for row in ranked if row.smiles != parent.candidate.smiles]
                lists = {policy: initial + rows for policy, rows in
                         select_candidates(novel, limit - len(initial), compare_shortlists).items()}
                selections[branch.name] = lists
                return list({row.smiles: row for rows in lists.values() for row in rows}.values())

            run_branch(decoder, partial(scorer.score_at, active=parent.active), config,
                root / branch.name, parent.candidate.latent, parent.active, parent=parent.candidate,
                proposal=proposal, perturbation_seed=branch.seed,
                shortlist_limit=branch.shortlist, select_shortlist=select,
                track_bonds=noise_study,
                ready=os.environ.get("DBMOL_LAUNCH_READY") if first_batch and index == 0 else None)
            print(f"{root.name} {branch.name} updates={branch.updates}", flush=True)
    finally:
        scorer.close()
    if arm_caches is not None:
        match = matched_parent_shortlists if match_by_parent else matched_shortlists
        selections = match(branches, selections, arm_caches, fresh_limit)
    planned = {name: {policy: [dict(smiles=r.smiles, step=r.step, relaxed_loss=r.relaxed_loss)
                                for r in rows] for policy, rows in lists.items()}
               for name, lists in selections.items()}
    write_json(root / "shortlists.json", planned)
    if arm_caches is None:
        attempts = score_shortlists(fresh_score, root, planned, cache)
    else:
        by_arm = {}
        for arm, arm_cache in arm_caches.items():
            arm_root = root / f"fresh-{arm}"
            arm_root.mkdir()
            by_arm[arm] = score_shortlists(fresh_score, arm_root,
                {branch.name: planned[branch.name] for branch in branches if branch.arm == arm}, arm_cache)
        if len(set(by_arm.values())) != 1:
            raise ValueError(f"Mismatched fresh scoring attempts: {by_arm}")
        attempts = sum(by_arm.values())
    readouts = []
    for branch in branches:
        branch_cache = cache if arm_caches is None else arm_caches[branch.arm]
        data = json.loads((root / branch.name / "summary.json").read_text())
        noise_metadata = (dict(strength=branch.strength, update_noise=branch.update_noise,
                               proposal=json.loads((root / branch.name / "start.json").read_text())["proposal"])
                          if noise_study else {})
        cap_metadata = (dict(max_step_norm_final=branch.cap_final) if branch.cap_final is not None else {})
        readouts.append(dict(name=branch.name, arm=branch.arm, parent_index=branch.parent_index,
            repeat=branch.repeat,
            parent_smiles=branch.parent.candidate.smiles, parent_fresh_loss=branch.parent.fresh_loss,
            parent_provenance=branch.parent.provenance, active_slots=branch.parent.active.nonzero().flatten().tolist(),
            policy=branch.policy, seed=branch.seed, max_step_norm=branch.cap, summary=data,
            candidates={policy: [row | branch_cache[row["smiles"]] for row in rows]
                        for policy, rows in planned[branch.name].items()}, **noise_metadata, **cap_metadata))
    write_json(root / "batch.json", dict(new_fresh_attempts=attempts, native_identity=scorer.identity,
               optimization_updates=sum(branch.updates for branch in branches), branches=readouts))
    return selections, readouts


def paired_summary(readouts):
    groups = {}
    for row in readouts:
        for policy, candidates in row["candidates"].items():
            key = (row["parent_index"], row["arm"], policy)
            baseline = row["parent_fresh_loss"]
            group = groups.setdefault(key, dict(parent_index=key[0], arm=key[1], shortlist=policy,
                parent_fresh_loss=baseline, requested_candidates=0, distinct_scores={}, best=baseline))
            group["requested_candidates"] += len(candidates)
            for candidate in candidates:
                if candidate["loss"] is not None:
                    group["distinct_scores"][candidate["smiles"]] = candidate["loss"]
                    group["best"] = min(group["best"], candidate["loss"])
    return [row | {"gain": row["parent_fresh_loss"] - row["best"]} for row in groups.values()]


def matched_paired_limits(comparison, *, step_cap=False):
    variants = list(comparison.get("variants", {}).values())
    if len(variants) < 2 or comparison.get("compare_shortlists"):
        raise ValueError("Matched paired budgets require at least two arms and one shortlist policy")
    updates, shortlist = variants[0]["updates"], variants[0]["shortlist"]
    if (not updates or any(type(count) is not int or count < 1 for count in updates)
            or type(shortlist) is not int or shortlist < 1
            or any(variant["updates"] != updates or variant["shortlist"] != shortlist for variant in variants)):
        raise ValueError("Matched arms require the same branch updates and shortlist slots")
    if step_cap and (any(count != 500 for count in updates)
                     or len({(variant["policy"], variant.get("update_noise")) for variant in variants}) != 1):
        raise ValueError("Step cap arms require paired starts, paired noise, and 500 updates per branch")
    return sum(updates), len(updates) * shortlist


def paired_budget_readout(readouts, *, updates_per_parent, attempts_per_parent):
    groups = {}
    for branch in readouts:
        parent = groups.setdefault(branch["parent_index"], {})
        arm = parent.setdefault(branch["arm"], dict(updates=0, fresh_attempts=0, fresh_scored=0))
        candidates = branch["candidates"]["mixed"]
        arm["updates"] += branch["summary"]["updates"]
        arm["fresh_attempts"] += len(candidates)
        arm["fresh_scored"] += sum(row["loss"] is not None for row in candidates)
    totals = {}
    for parent in groups.values():
        if ({row["updates"] for row in parent.values()} != {updates_per_parent}
                or len({row["fresh_attempts"] for row in parent.values()}) != 1
                or any(row["fresh_attempts"] > attempts_per_parent for row in parent.values())):
            raise ValueError(f"Paired arms have inconsistent realized budgets: {parent}")
        for name, row in parent.items():
            total = totals.setdefault(name, dict.fromkeys(row, 0))
            for key, value in row.items():
                total[key] += value
    return dict(updates_per_parent=updates_per_parent, fresh_attempt_limit_per_parent=attempts_per_parent,
                by_parent=groups, by_arm=totals)


def analyze_noise_paths(root, branches, sources):
    from latentmol.design.protenix_restart_path_metrics import PathTrace, read_hard_path, read_path, summarize_paths

    traces = [PathTrace(name=branch.name, arm=branch.arm, parent_id=branch.parent_index,
        parent_smiles=branch.parent.candidate.smiles,
        observations=read_path(root / branch.name / "metrics.jsonl"),
        source_observations=sources[branch.parent_index], source_step=branch.parent.candidate.step,
        expected_updates=branch.updates, hard_observations=read_hard_path(root / branch.name / "raw-blocks"))
        for branch in branches]
    return summarize_paths(traces)


def early_bond_readout(root, branches):
    """Summarize graph density before and during the first 100 updates."""
    readout = []
    for branch in branches:
        with (root / branch.name / "metrics.jsonl").open(encoding="utf-8") as stream:
            rows = []
            for line in stream:
                row = json.loads(line)
                if row["step"] > 100:
                    break
                rows.append(row)
        first = rows[0]
        readout.append(dict(branch=branch.name, arm=branch.arm, parent_index=branch.parent_index,
            initial_hard_bonds=first["hard_bond_count"], initial_soft_bond_mass=first["soft_bond_mass"],
            max_hard_bonds=max(row["hard_bond_count"] for row in rows),
            max_soft_bond_mass=max(row["soft_bond_mass"] for row in rows),
            valid_observations=sum(row["raw_valid"] for row in rows), observations=len(rows)))
    return readout


def draw_prior(decoder, design, used, seed):
    for next_seed in range(seed, seed + 256):
        prior = make_prior_parent(decoder, next_seed, temperature=design.temperature_initial,
                                  occupied_atoms=design.occupied_atoms)
        if prior is not None and prior.candidate.smiles not in used:
            used.add(prior.candidate.smiles)
            return prior, next_seed + 1
    raise ValueError("Could not find a distinct valid independent prior within 256 draws")


def lineage_study(decoder, make_scorer, fresh_score, design, output, parents, comparison, cache, seed, prior_seed):
    archives = {arm: {p.candidate.smiles: p for p in parents} for arm in ("elite", "independent")}
    used_priors = {p.candidate.smiles for p in parents}
    cycles = []
    for cycle in range(comparison["cycles"]):
        branches = []
        ordered = sorted(archives["elite"].values(), key=lambda p: (p.fresh_loss, p.candidate.smiles))
        for index, parent in enumerate(ordered):
            for repeat in range(comparison["branches_per_parent"]):
                prior, prior_seed = draw_prior(decoder, design, used_priors, prior_seed)
                for arm, start, policy in (("elite", parent, "latent"), ("independent", prior, "none")):
                    branches.append(Branch(arm, start, index, repeat, seed + 10000 * cycle + 100 * index + repeat,
                        comparison["updates"], design.max_step_norm, policy, comparison["shortlist"]))
        root = output / f"cycle-{cycle + 1:02d}"
        selections, _ = run_batch(decoder, make_scorer, fresh_score, design, root, branches, cache,
                                  first_batch=cycle == 0)
        for branch in branches:
            archive = archives[branch.arm]
            for row in selections[branch.name]["mixed"]:
                loss = cache[row.smiles]["loss"]
                if loss is not None and row.smiles not in archive:
                    archive[row.smiles] = RestartParent(candidate=row, active=branch.parent.active,
                        fresh_loss=loss, provenance=dict(cycle=cycle + 1, branch=branch.name,
                                                        parent=branch.parent.candidate.smiles))
        for arm, archive in archives.items():
            archives[arm] = dict(sorted(archive.items(), key=lambda item: (item[1].fresh_loss, item[0]))
                                 [:comparison["parents"]])
        write_json(root / "archives.json", {arm: [parent_record(p) for p in archive.values()]
                                            for arm, archive in archives.items()})
        cycles.append(dict(cycle=cycle + 1, updates=sum(b.updates for b in branches),
            best={arm: min(p.fresh_loss for p in archive.values()) for arm, archive in archives.items()}))
    return cycles


def lineage_feedback(decoder, make_scorer, fresh_score, design, output, parents, comparison, cache, seed, prior_seed):
    def ranked(archive):
        return sorted(archive.values(), key=lambda parent: (parent.fresh_loss, parent.candidate.smiles))

    initial = {parent.candidate.smiles: parent for parent in parents}
    fixed = ranked(initial)
    hybrid_priors = comparison.get("hybrid_prior_children", 0)
    middle_arm = "hybrid" if hybrid_priors else "fixed"
    arms = ("adaptive", middle_arm, "independent")
    archives = {arm: dict(initial) for arm in arms}
    caches = {arm: dict(cache) for arm in arms}
    used_priors = set(initial)
    repeats, updates, shortlist = (comparison[key] for key in ("branches_per_parent", "updates", "shortlist"))
    updates_per_arm = len(fixed) * repeats * updates
    limit = len(fixed) * repeats * shortlist
    cycles = []
    for cycle in range(1, comparison["cycles"] + 1):
        adaptive = ranked(archives["adaptive"])[:len(fixed)]
        middle = ranked(archives["hybrid"])[:len(fixed)] if hybrid_priors else fixed
        prior_ranks = {(cycle - 1 + index) % len(fixed) for index in range(hybrid_priors)}
        branches = []
        first_prior_seed = prior_seed
        for index, (elite, middle_parent) in enumerate(zip(adaptive, middle, strict=True)):
            for repeat in range(repeats):
                prior, prior_seed = draw_prior(decoder, design, used_priors, prior_seed)
                slot_seed = seed + 10000 * (cycle - 1) + 100 * index + repeat
                # Keep a child of each elite. Rotate which ranks yield their final repeat
                # to the exact same prior drawn for the independent arm at that slot.
                use_prior = index in prior_ranks and repeat == repeats - 1
                source, policy = (prior, "none") if use_prior else (middle_parent, "latent")
                for arm, parent, policy in (("adaptive", elite, "latent"), (middle_arm, source, policy),
                                            ("independent", prior, "none")):
                    branches.append(Branch(arm, parent, index, repeat, slot_seed, updates,
                                           design.max_step_norm, policy, shortlist))
        root = output / f"cycle-{cycle:02d}"
        selections, readouts = run_batch(decoder, make_scorer, fresh_score, design, root, branches, None,
            first_batch=cycle == 1, arm_caches=caches, fresh_limit=limit)
        fresh_attempts = dict.fromkeys(arms, 0)
        fresh_scored = dict.fromkeys(arms, 0)
        actual_updates = dict.fromkeys(arms, 0)
        for branch, readout in zip(branches, readouts, strict=True):
            arm = branch.arm
            actual_updates[arm] += readout["summary"]["updates"]
            for row in selections[branch.name]["mixed"]:
                fresh_attempts[arm] += 1
                loss = caches[arm][row.smiles]["loss"]
                if loss is not None:
                    fresh_scored[arm] += 1
                    archives[arm][row.smiles] = RestartParent(row, branch.parent.active, loss,
                        dict(cycle=cycle, branch=branch.name, parent=branch.parent.candidate.smiles))
        if any(count != updates_per_arm for count in actual_updates.values()):
            raise ValueError(f"Mismatched lineage update counts: {actual_updates}")
        leaders = {arm: ranked(archive)[:len(fixed)] for arm, archive in archives.items()}
        write_json(root / "archives.json", {arm: [parent_record(parent) for parent in rows]
                                            for arm, rows in leaders.items()})
        high = {}
        for arm, archive in archives.items():
            eligible = {smiles: (parent.candidate, parent.fresh_loss) for smiles, parent in archive.items()
                        if smiles not in initial and parent.fresh_loss <= comparison["high_score_ceiling"]}
            high[arm] = list(select_diverse_output(eligible, len(eligible), max_similarity=0.60)) if eligible else []
        summary = dict(cycle=cycle, updates=sum(actual_updates.values()), updates_by_arm=actual_updates,
            fresh_attempt_limit=limit, fresh_attempt_shortfall=limit - min(fresh_attempts.values()),
            fresh_attempts=fresh_attempts, fresh_scored=fresh_scored, prior_draws=prior_seed - first_prior_seed,
            best={arm: rows[0].fresh_loss for arm, rows in leaders.items()},
            top_three_mean={arm: sum(parent.fresh_loss for parent in rows) / len(rows) for arm, rows in leaders.items()},
            diverse_high_scorers={arm: len(smiles) for arm, smiles in high.items()}, diverse_high_smiles=high)
        if hybrid_priors:
            summary["prior_starts_by_arm"] = {
                arm: sum(branch.arm == arm and branch.policy == "none" for branch in branches) for arm in arms
            }
        cycles.append(summary)
    return cycles


def run_study(decoder, make_scorer, fresh_score, config, task_name, output, checkpoint_sha256):
    task = config["tasks"][task_name]
    comparison = config["comparisons"][task["comparison"]]
    noise_study = config.get("study") == "initialization_noise"
    feedback = config.get("study") == "lineage_feedback"
    step_cap = config.get("study") == "step_cap"
    matched_paired = step_cap or comparison.get("matched_fresh", False)
    paired_limits = matched_paired_limits(comparison, step_cap=step_cap) if matched_paired else None
    budget = budgets(comparison, feedback=feedback)
    design = LatentDesignConfig.for_restart(config["design"])
    expected_lr = 70 if noise_study or feedback or step_cap else 150
    if design.learning_rate != expected_lr or design.chemistry_initial != design.chemistry_final:
        raise ValueError(f"Require LR {expected_lr} and a common constant chemistry weight")
    if step_cap and design.schedule_horizon != 500:
        raise ValueError("Step cap schedules require a 500-update horizon")
    decoder.eval().requires_grad_(False)
    if any(p.grad is not None for p in decoder.parameters()):
        raise ValueError("Frozen decoder carries preexisting gradients")
    identity = _model_identity(decoder)
    output.mkdir(parents=True)
    offset = task["panel_offset"]
    size = comparison["parents"]
    panel = load_parent_panel(decoder, [Path(root) for root in config["parent_cohorts"]],
        count=config["panel_size"], seed=config["panel_seed"], temperature=design.temperature_initial,
        occupied_atoms=design.occupied_atoms, checkpoint_sha256=checkpoint_sha256)
    parents = panel[offset:offset + size]
    if len(parents) != size or any(p.fresh_loss is None or not math.isfinite(p.fresh_loss) for p in parents):
        raise ValueError("Missing scored parents for declared panel slice")
    seed = config["perturbation_seed"] + (100 if noise_study else 100000) * offset
    prior_seed = config["prior_seed"] + 100000 * offset if "prior_seed" in config else None
    write_json(output / "run.json", dict(task=task_name, budget=budget, settings=asdict(design),
        comparison=comparison, panel_offset=offset, decoder_sha256=identity,
        perturbation_seed=seed, prior_seed=prior_seed,
        checkpoint_sha256=checkpoint_sha256, job_id=os.environ.get("LSB_JOBID"),
        graph_spec=decoder.spec.to_dict(), parents=[parent_record(p) for p in parents]))
    cache = {p.candidate.smiles: dict(loss=p.fresh_loss, status="archived_parent") for p in parents}
    if "variants" in comparison:
        branches = list(paired_branches(parents, comparison, design, seed))
        paired = output / "paired"
        if noise_study:
            from latentmol.design.protenix_restart_path_metrics import read_path

            sources = [read_path(Path(p.provenance["trajectory"]) / "arms" / "latent" / "candidates.jsonl")
                       for p in parents]
            if any(source.get(p.candidate.step) != p.candidate.smiles
                   for p, source in zip(parents, sources)):
                raise ValueError("Archived source trajectory does not contain its recorded parent identity")
        arm_caches = {arm: dict(cache) for arm in comparison["variants"]} if matched_paired else None
        _, readouts = run_batch(decoder, make_scorer, fresh_score, design, paired, branches,
                                cache, compare_shortlists=comparison.get("compare_shortlists", False),
                                noise_study=noise_study, arm_caches=arm_caches,
                                fresh_limit=paired_limits[1] if paired_limits else None,
                                match_by_parent=matched_paired)
        result = dict(paired=paired_summary(readouts))
        if paired_limits:
            result["matched_budget"] = paired_budget_readout(
                readouts, updates_per_parent=paired_limits[0], attempts_per_parent=paired_limits[1])
        if noise_study:
            write_json(output / "path-analysis.json", analyze_noise_paths(paired, branches, sources))
            result["path_analysis"] = "path-analysis.json"
            if comparison.get("report_bond_density"):
                write_json(output / "early-bond-density.json", early_bond_readout(paired, branches))
                result["early_bond_density"] = "early-bond-density.json"
        if comparison.get("compare_shortlists", False):
            result["matched_first_branch"] = paired_summary([row for row in readouts if row["repeat"] == 0])
    else:
        run_lineage = lineage_feedback if feedback else lineage_study
        cycles = run_lineage(decoder, make_scorer, fresh_score, design, output, parents, comparison,
                             cache, seed, prior_seed)
        result = dict(cycles=cycles)
    if _model_identity(decoder) != identity or any(p.grad is not None for p in decoder.parameters()):
        raise ValueError("Frozen decoder parameters or gradients changed")
    if paired_limits:
        attempts = sum(row["fresh_attempts"] for row in result["matched_budget"]["by_arm"].values())
    elif feedback:
        attempts = sum(sum(cycle["fresh_attempts"].values()) for cycle in cycles)
    else:
        attempts = len(cache) - len(parents)
    result.update(status="completed", budget=budget, new_fresh_attempts=attempts)
    write_json(output / "result.json", result)
    return result


def main(argv=None):
    args = restart_args(__doc__, argv)
    config = json.loads(args.config.read_text())
    decoder, _ = load_model(args.checkpoint, "cpu")
    return run_study(decoder, native_factory(args), partial(fresh_score_records, args), config,
                     args.arm, args.output, args.checkpoint_sha256)


if __name__ == "__main__":
    main()
