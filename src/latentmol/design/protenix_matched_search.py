"""Compare Protenix search methods and adaptive widths at matched budgets."""
from __future__ import annotations

from dataclasses import asdict
from functools import partial
from itertools import zip_longest
import json
import os
from time import perf_counter

import torch

from latentmol.chemistry import inspect_graph
from latentmol.design.latent_protenix import LatentDesignConfig, _model_identity, choose_active, hard_from_graph
from latentmol.design.protenix_restart_selection import select_candidates, select_diverse_output
from latentmol.design.protenix_restart_study import score_shortlists
from latentmol.design.protenix_restarts import Candidate, fresh_score_records, native_factory, restart_args, write_json
from latentmol.design.protenix_search import SearchOptions, frontier_args, point_record
from latentmol.design.protenix_search_branches import Branch, execute_branches
from latentmol.design.protenix_search_frontier import SearchPoint, select_frontier
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256
from latentmol.graph_policy import graph_from_decoder


ARMS = ("adaptive", "continuous", "independent")
WIDTH_ARMS = ("p6-c2", "p4-c3", "p3-c4")


def fresh_scoring_policy(config):
    default = "equal_allowance" if config["study"] == "matched_search" else "matched_minimum"
    policy = config.get("fresh_scoring_policy", default)
    if policy not in {"equal_allowance", "matched_minimum"}:
        raise ValueError(f"Unknown fresh scoring policy: {policy}")
    return policy


def resolve_config(config, task):
    study = config["study"]
    if study not in {"matched_search", "matched_width"}:
        raise ValueError("Expected a matched-search or matched-width config")
    fresh_scoring_policy(config)
    design = LatentDesignConfig.for_restart(config["design"])
    search = dict(config["search"])
    width = search.pop("branches_per_arm")
    common = {"diversity_limit": 6, "high_score_ceiling": 2.30} | search
    if study == "matched_width":
        allocations = config["allocations"]
        if (set(allocations) != set(WIDTH_ARMS) or {"parents", "children"} & search.keys()
                or any(allocations[arm] != {"parents": parents, "children": children}
                       for arm, parents, children in (("p6-c2", 6, 2), ("p4-c3", 4, 3), ("p3-c4", 3, 4)))):
            raise ValueError("Matched width requires the 6×2, 4×3 and 3×4 allocations")
        arm_options = {arm: SearchOptions(**(common | allocations[arm])) for arm in WIDTH_ARMS}
    else:
        arm_options = dict.fromkeys(ARMS, SearchOptions(**common))
    options = next(iter(arm_options.values()))
    seed = config["tasks"][task]["seed"]
    if (type(seed) is not int or not 0 <= seed < 2**63 - 1_000_000
            or type(width) is not int or any(width != item.parents * item.children for item in arm_options.values())
            or options.waves > 8 or width > 12 or design.updates > 500
            or options.fresh_per_branch > 16 or options.high_score_ceiling is None):
        raise ValueError("Invalid seed or matched search budget")
    if (design.noise_initial != 0 or design.max_step_norm_final is not None
            or design.temperature_initial != design.temperature_final
            or design.chemistry_initial != design.chemistry_final):
        raise ValueError("Continuous replay requires zero update noise and constant chemistry, temperature and cap")
    if (options.parent_policy != "frontier" or options.perturbation != "latent"
            or options.shortlist != "mixed" or options.invalid_children or options.batch_size != 1
            or options.overlap or options.cpu_workers != 1 or options.frontier_keep_best
            or options.duplicate_refill_attempts):
        raise ValueError("Matched search requires scalar execution and the standard valid adaptive frontier")
    return design, arm_options, width, seed


def plan_budget(config, task):
    design, arm_options, width, _ = resolve_config(config, task)
    options = next(iter(arm_options.values()))
    windows = options.waves * width
    return dict(updates=len(arm_options) * windows * design.updates,
                updates_per_arm=windows * design.updates,
                branches=len(arm_options) * windows,
                fresh_attempts=len(arm_options) * windows * options.fresh_per_branch,
                fresh_attempts_per_arm=windows * options.fresh_per_branch,
                relaxed_calls_per_arm=windows * (design.updates + 1))


@torch.no_grad()
def gaussian_start(decoder, design, seed):
    latent = torch.randn((1, decoder.config.latent), dtype=torch.float32,
                         generator=torch.Generator().manual_seed(seed))
    decoded = decoder.decode(latent)
    active = choose_active(decoded, decoder.spec, design.occupied_atoms)
    graph = graph_from_decoder(decoded, decoder.spec, fixed_active=active,
                               temperature=design.temperature_initial)
    if any(not getattr(graph, key).isfinite().all() for key in ("atoms", "charges", "bonds")):
        raise ValueError("Nonfinite raw Gaussian start graph")
    hard = hard_from_graph(graph)
    smiles = inspect_graph(hard.atoms[0], hard.charges[0], hard.bonds[0], decoder.spec).smiles
    name = f"prior-{seed}"
    return SearchPoint(Candidate(smiles, latent, hard, 0., name, 0), active, None, (name,))


def _branch(arm, slot, point, policy, seed, options):
    return Branch(f"parent-{slot:02d}-{arm}-00", point, seed, policy=policy,
                  fade_updates=options.fade_updates)


def _adaptive_starts(decoder, design, options, wave, seed, initial, parents, exact_children):
    if wave == 1:
        return [(point, "bootstrap") for point in initial]
    starts = [(point, "latent") for point in parents for _ in range(options.children)]
    ordered = sorted(range(len(starts)), key=lambda slot: (slot % options.children, slot))
    for slot in ordered[:exact_children]:
        starts[slot] = (starts[slot][0], "none")
    width = len(initial)
    for slot in range(len(starts), width):
        point = gaussian_start(decoder, design, seed + 100_000 + (wave - 1) * width + slot)
        starts.append((point, "bootstrap"))
    return starts


def _round_robin_pool(results, cache, per_branch):
    """Select at most six per window, excluding prior scores and duplicate window picks."""
    seen = set(cache)
    pools = []
    for result in results:
        rows = [row for row in result.valid if row.smiles not in seen]
        chosen = select_candidates(rows, per_branch, False)["mixed"]
        seen.update(row.smiles for row in chosen)
        pools.append([(result, row) for row in chosen])
    return [item for group in zip_longest(*pools) for item in group if item is not None]


def _score_wave(root, fresh_score, results, caches, archives, per_branch, *, policy):
    pools = {arm: _round_robin_pool(rows, caches[arm], per_branch) for arm, rows in results.items()}
    available = {arm: len(rows) for arm, rows in pools.items()}
    if policy == "equal_allowance":
        targets = available
    elif policy == "matched_minimum":
        targets = dict.fromkeys(results, min(available.values()))
    else:
        raise ValueError(f"Unknown fresh scoring policy: {policy}")
    attempts, successes, seconds, candidates = {}, {}, {}, {}
    for arm in results:
        target = targets[arm]
        started = perf_counter()
        selections = {}
        for result, row in pools[arm][:target]:
            selections.setdefault(result.branch.name, {"mixed": []})["mixed"].append(
                dict(smiles=row.smiles, step=row.step, relaxed_loss=row.relaxed_loss))
        arm_root = root / f"fresh-{arm}"
        arm_root.mkdir()
        attempts[arm] = score_shortlists(fresh_score, arm_root, selections, caches[arm])
        successes[arm] = 0
        for result, row in pools[arm][:target]:
            score = caches[arm][row.smiles]
            if (score["status"] == "scored") != (score["loss"] is not None):
                raise ValueError("Fresh scoring returned inconsistent score and status")
            candidates.setdefault(result.branch.name, []).append(
                dict(smiles=row.smiles, step=row.step, relaxed_loss=row.relaxed_loss, **score))
            if score["loss"] is not None:
                successes[arm] += 1
                parent = result.branch.parent
                archives[arm][row.smiles] = SearchPoint(
                    row, parent.active, score["loss"], parent.ancestry + (row.branch,))
        seconds[arm] = perf_counter() - started
    if attempts != targets or any(attempts[arm] > len(rows) * per_branch for arm, rows in results.items()):
        raise ValueError(f"Mismatched or excessive fresh-score attempts: {attempts}")
    return available, attempts, successes, seconds, candidates


def _branch_readout(root, arm, slot, result, candidates, wave):
    branch = result.branch
    with (root / branch.name / "metrics.jsonl").open(encoding="utf-8") as stream:
        first = json.loads(next(stream))
    if first["step"] != 0:
        raise ValueError("Branch is missing its initial observation")
    parent = branch.parent
    if branch.policy == "bootstrap":
        source_kind = "random_fallback" if arm != "independent" and wave > 1 else "raw_gaussian"
    else:
        source_kind = "continuation" if arm == "continuous" else "archive"
    return dict(name=branch.name, arm=arm, parent_index=slot, repeat=0, policy=branch.policy,
        source_kind=source_kind, seed=branch.seed, parent_smiles=parent.candidate.smiles,
        parent_fresh_loss=parent.fresh_loss,
        parent_provenance=dict(branch=parent.candidate.branch, step=parent.candidate.step),
        active_slots=parent.active.nonzero().flatten().tolist(),
        initial_occupied_count=int(parent.active.sum()), initial_raw_valid=first["raw_valid"],
        initial_hard_bond_count=first["hard_bond_count"], summary=result.summary,
        candidates={"mixed": candidates.get(branch.name, [])})


def _archive_readout(root, archives, options):
    leaders, losses, means, diverse_counts, diverse_leaders = {}, {}, {}, {}, {}
    for arm, archive in archives.items():
        ranked = sorted(archive.values(), key=lambda point: (point.fresh_loss, point.candidate.smiles))
        elite = ranked[:3]
        leaders[arm] = [point_record(point) for point in elite]
        losses[arm] = elite[0].fresh_loss if elite else None
        means[arm] = sum(point.fresh_loss for point in elite) / 3 if len(elite) == 3 else None
        high = {point.candidate.smiles: (point.candidate, point.fresh_loss) for point in ranked
                if point.fresh_loss <= options.high_score_ceiling}
        diverse = select_diverse_output(high, len(high), max_similarity=options.diversity_max_similarity) if high else {}
        diverse_counts[arm] = len(diverse)
        diverse_leaders[arm] = [dict(smiles=smiles, fresh_loss=diverse[smiles][1])
                                for smiles in list(diverse)[:options.diversity_limit]]
    write_json(root / "archives.json", leaders)
    return losses, means, diverse_counts, diverse_leaders


def run_matched(decoder, make_scorer, fresh_score, config, task, output, checkpoint_sha256, *, started=None):
    started = perf_counter() if started is None else started
    design, arm_options, width, seed = resolve_config(config, task)
    arms = tuple(arm_options)
    options = arm_options[arms[0]]
    width_study = config["study"] == "matched_width"
    adaptive_arms = arms if width_study else ("adaptive",)
    allocations = {arm: dict(parents=opt.parents, children=opt.children)
                   for arm, opt in arm_options.items()} if width_study else {}
    budget = plan_budget(config, task)
    scoring_policy = fresh_scoring_policy(config)
    decoder.eval().requires_grad_(False)
    identity = _model_identity(decoder)
    if any(parameter.grad is not None for parameter in decoder.parameters()):
        raise ValueError("Frozen decoder carries preexisting gradients")
    output.mkdir(parents=True)
    initial = [gaussian_start(decoder, design, seed + slot) for slot in range(width)]
    write_json(output / "run.json", dict(runner=config["study"], task=task, arms=arms, budget=budget,
        design=asdict(design), search=config["search"], seed=seed, fresh_scoring_policy=scoring_policy,
        job_id=os.environ.get("LSB_JOBID"), checkpoint_sha256=checkpoint_sha256, decoder_sha256=identity,
        graph_spec=decoder.spec.to_dict(), initial_starts=[point_record(point) for point in initial])
        | ({"allocations": allocations} if width_study else {}))
    caches = {arm: {} for arm in arms}
    archives = {arm: {} for arm in arms}
    cumulative_updates = dict.fromkeys(arms, 0)
    cumulative_attempts = dict.fromkeys(arms, 0)
    cumulative_successes = dict.fromkeys(arms, 0)
    cumulative_calls = dict.fromkeys(arms, 0)
    histories = {arm: {} for arm in adaptive_arms}
    cycles = []
    continuation = initial[0]
    stop = None
    for wave in range(1, options.waves + 1):
        wave_started = perf_counter()
        if options.time_budget_seconds is not None and cycles:
            remaining = options.time_budget_seconds - (wave_started - started)
            reserve = 1.20 * max(row["elapsed_seconds"] for row in cycles[-3:])
            if remaining < reserve:
                stop = dict(next_wave=wave, remaining_seconds=remaining, estimated_wave_seconds=reserve)
                break
        root = output / f"cycle-{wave:02d}"
        root.mkdir()
        parents = {arm: select_frontier(archives[arm].values(), max_similarity=opt.max_similarity,
                                       **frontier_args(opt, wave, histories[arm])) if wave > 1 else []
                   for arm in adaptive_arms for opt in (arm_options[arm],)}
        exact = min(options.exact_children, *(len(rows) for rows in parents.values())) if width_study else options.exact_children
        starts = {arm: _adaptive_starts(decoder, design, arm_options[arm], wave, seed, initial, parents[arm], exact)
                  for arm in adaptive_arms}
        for arm, rows in parents.items():
            for point in rows:
                histories[arm].setdefault(point.candidate.branch, set()).add(point.candidate.step)
        plan = dict(wave=wave, requested_windows_per_arm=width)
        fallback_counts = {arm: sum(policy == "bootstrap" for _, policy in starts[arm]) if wave > 1 else 0
                           for arm in adaptive_arms}
        if width_study:
            allocation_readout = {arm: dict(requested_parents=arm_options[arm].parents,
                children_per_parent=arm_options[arm].children, selected_parents=len(parents[arm]),
                fallback_children=fallback_counts[arm], exact_children=sum(policy == "none" for _, policy in starts[arm]))
                for arm in arms}
            plan["allocations"] = {arm: allocation_readout[arm] | {
                "parent_sources": [point_record(point) for point in parents[arm]]} for arm in arms}
        else:
            independent_seeds = [seed + (wave - 1) * width + slot for slot in range(width)]
            independent = initial if wave == 1 else [gaussian_start(decoder, design, i) for i in independent_seeds]
            starts["independent"] = [(point, "bootstrap") for point in independent]
            plan.update(adaptive_parents=[point_record(point) for point in parents["adaptive"]],
                        adaptive_fallbacks=fallback_counts["adaptive"], independent_seeds=independent_seeds)
        write_json(root / "wave-plan.json", plan)
        scorer_started = perf_counter()
        scorer = make_scorer(decoder.spec, continuation.active, root)
        scorer_seconds = perf_counter() - scorer_started
        results, optimization_seconds = {}, {}

        def execute(branches, ready=None):
            return execute_branches(decoder, scorer, design, root, branches, batch_size=1,
                cpu_workers=1, overlap=False, repair_limit=0, invalid_limit=0, ready=ready)

        try:
            for arm in arms:
                arm_started = perf_counter()
                if arm == "continuous":
                    results[arm] = []
                    for slot in range(width):
                        policy = "bootstrap" if wave == 1 and slot == 0 else "continuous"
                        branch = _branch(arm, slot, continuation, policy, seed + wave * 1000 + slot, options)
                        result, = execute([branch])
                        if result.terminal is None or result.terminal.step != design.updates:
                            raise ValueError("Continuous window did not return its terminal state")
                        results[arm].append(result)
                        continuation = SearchPoint(result.terminal, continuation.active, None,
                                                   continuation.ancestry + (result.terminal.branch,))
                else:
                    branches = [_branch(arm, slot, point, policy, seed + wave * 1000 + slot, options)
                                for slot, (point, policy) in enumerate(starts[arm])]
                    results[arm] = execute(branches, os.environ.get("DBMOL_LAUNCH_READY")
                                           if wave == 1 and arm == arms[0] else None)
                optimization_seconds[arm] = perf_counter() - arm_started
        finally:
            scorer.close()
        expected_updates = width * design.updates
        updates = {arm: sum(row.summary["updates"] for row in results[arm]) for arm in arms}
        if (set(updates.values()) != {expected_updates}
                or any(len(rows) != width or any(row.summary["updates"] != design.updates for row in rows)
                       for rows in results.values())):
            raise ValueError(f"Mismatched actual optimizer budgets: {updates}")
        available, attempts, successes, fresh_seconds, candidates = _score_wave(
            root, fresh_score, results, caches, archives, options.fresh_per_branch, policy=scoring_policy)
        best, top_three, diverse, diverse_leaders = _archive_readout(root, archives, options)
        readouts = [_branch_readout(root, arm, slot, row, candidates, wave)
                    for arm in arms for slot, row in enumerate(results[arm])]
        calls = {arm: updates[arm] + len(results[arm]) for arm in arms}
        for arm in arms:
            cumulative_updates[arm] += updates[arm]
            cumulative_attempts[arm] += attempts[arm]
            cumulative_successes[arm] += successes[arm]
            cumulative_calls[arm] += calls[arm]
        limit = width * options.fresh_per_branch
        record = dict(cycle=wave, updates_by_arm=updates, cumulative_updates_by_arm=dict(cumulative_updates),
            fresh_scoring_policy=scoring_policy, fresh_attempt_allowance_by_arm=dict.fromkeys(arms, limit),
            fresh_attempt_limit=limit, fresh_shortlist_available=available,
            fresh_attempt_shortfall=limit - min(attempts.values()),
            fresh_attempt_shortfall_by_arm={arm: limit - attempts[arm] for arm in arms},
            fresh_attempts=attempts, cumulative_fresh_attempts_by_arm=dict(cumulative_attempts),
            fresh_scored=successes, cumulative_fresh_scored_by_arm=dict(cumulative_successes),
            relaxed_calls_by_arm=calls, cumulative_relaxed_calls_by_arm=dict(cumulative_calls),
            best=best, top_three_mean=top_three, diverse_high_scorers=diverse,
            diverse_high_score_leaders=diverse_leaders,
            scorer_start_seconds=scorer_seconds, optimization_seconds_by_arm=optimization_seconds,
            fresh_seconds_by_arm=fresh_seconds, elapsed_seconds=perf_counter() - wave_started)
        if width_study:
            record["allocations"] = allocation_readout
        else:
            record["adaptive_fallbacks"] = fallback_counts["adaptive"]
        write_json(root / "batch.json", dict(branches=readouts, native_identity=scorer.identity,
            optimization_updates=sum(updates.values()), new_fresh_attempts=sum(attempts.values())))
        write_json(root / "wave-result.json", record)
        cycles.append(record)
        print(json.dumps(record, allow_nan=False), flush=True)
    if _model_identity(decoder) != identity or any(parameter.grad is not None for parameter in decoder.parameters()):
        raise ValueError("Frozen decoder parameters or gradients changed")
    checkpoints = (3, 6) if width_study else (4, 8)
    result = dict(status="time_budget" if stop else "completed", planned=budget, cycles=cycles,
        fresh_scoring_policy=scoring_policy,
        checkpoints={f"wave_{wave}": cycles[wave - 1] for wave in checkpoints if len(cycles) >= wave},
        completed_updates_by_arm=cumulative_updates, fresh_attempts_by_arm=cumulative_attempts,
        fresh_scored_by_arm=cumulative_successes, relaxed_calls_by_arm=cumulative_calls,
        elapsed_seconds=perf_counter() - started, frozen_decoder_unchanged=True)
    if stop is not None:
        result["time_budget_stop"] = stop
    write_json(output / "result.json", result)
    return result


def main(argv=None):
    started = perf_counter()
    args = restart_args(__doc__, argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    budget = plan_budget(config, args.arm)
    if file_sha256(args.plan) != args.plan_sha256:
        raise ValueError("Matched search launch plan changed")
    launch = json.loads(args.plan.read_text(encoding="utf-8"))
    if file_sha256(args.config) != launch["config_sha256"]:
        raise ValueError("Matched search config differs from its staged launch plan")
    if (launch["planned_updates"][args.arm] != budget["updates"]
            or launch["planned_fresh_attempts"][args.arm] != budget["fresh_attempts"]):
        raise ValueError("Matched search budget differs from its staged launch plan")
    _, arm_options, width, _ = resolve_config(config, args.arm)
    fresh = json.loads(args.fresh_config.read_text(encoding="utf-8"))
    if width * next(iter(arm_options.values())).fresh_per_branch > fresh["candidate_limit"]:
        raise ValueError("Matched search exceeds the fresh-score call limit")
    decoder, _ = load_model(args.checkpoint, "cpu")
    return run_matched(decoder, native_factory(args), partial(fresh_score_records, args),
                       config, args.arm, args.output, args.checkpoint_sha256, started=started)


if __name__ == "__main__":
    main()
