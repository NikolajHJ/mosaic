"""Allocate real Boltz restart trajectories to fixed or selectively extended children."""

import math
from dataclasses import replace
from pathlib import Path
import random
from time import perf_counter

from latentmol.design.protenix_restarts import write_json
from latentmol.design.protenix_search_branches import Branch, execute_branches
from latentmol.design.protenix_search_frontier import SearchPoint
from latentmol.guidance.boltz_search import hard_graph_record, hard_pose_key


POLICY_LENGTHS = {"fixed75": 75, "fixed150": 150, "fixed400": 400, "promote": 100}


def allocation_budget(policy, count):
    if policy not in POLICY_LENGTHS or type(count) is not int or count < 1:
        raise ValueError(f"Require one of {tuple(POLICY_LENGTHS)} and at least one start")
    if policy == "promote" and count % 6:
        raise ValueError("Promotion uses cohorts of six starts")
    return count * POLICY_LENGTHS[policy] + (count // 6 * 2 * 300 if policy == "promote" else 0)


def _rank_panel(best):
    return sorted(best.values(), key=lambda row: (row["loss"], row["smiles"]))


class ScoreLedger:
    """One pose-aware FP32 cache per arm, optionally retaining restart latents."""

    def __init__(self, fresh_score, excluded, budget, *, keep_parents=False):
        self.fresh_score = fresh_score
        self.excluded = set(excluded)
        self.budget = budget
        self.by_pose = {}
        self.best = {}
        self.parents = {} if keep_parents else None

    def score_segment(self, result, path, *, offset, allowance, prefix=""):
        # Shared executor keeps the lowest relaxed-loss observation of each SMILES
        # in this segment. Match this shortlist for all policies. Across segments,
        # different fixed-pose assignments of that molecule can be scored again.
        observed = {}
        pending = []
        replayable = {}
        for candidate in result.valid:
            if candidate.smiles in self.excluded:
                continue
            graph = hard_graph_record(candidate.hard)
            key = hard_pose_key(graph)
            observed[key] = candidate.smiles
            if self.parents is not None and candidate.step not in result.nonreplayable_steps:
                replayable[key] = candidate
            if key in self.by_pose or len(pending) >= allowance:
                continue
            pending.append(dict(smiles=candidate.smiles, hard_graph=graph, scoring_key=key,
                                arm=prefix + candidate.branch, step=offset + candidate.step,
                                role="child_trajectory", start=result.branch.name))
        if len(self.by_pose) + len(pending) > self.budget:
            raise RuntimeError("Hard-scoring allowance exceeded")
        if pending:
            scored = self.fresh_score(path, pending)
            expected = {row["scoring_key"]: row["smiles"] for row in pending}
            if (len(scored) != len(pending)
                    or {row["scoring_key"]: row["smiles"] for row in scored} != expected):
                raise ValueError("Hard scorer returned different pose keys or molecules")
            for row in scored:
                loss = row["loss"]
                if loss is not None and not math.isfinite(loss):
                    raise ValueError("Nonfinite fresh-score result")
                ranked = {k: row[k] for k in ("scoring_key", "smiles", "loss", "arm", "step")}
                self.by_pose[row["scoring_key"]] = ranked
                if loss is None:
                    continue
                best = self.best.get(row["smiles"])
                if best is None or loss < best["loss"]:
                    self.best[row["smiles"]] = ranked
        for key, candidate in replayable.items():
            loss = self.by_pose.get(key, {}).get("loss")
            if loss is None:
                continue
            best_parent = self.parents.get(candidate.smiles)
            if best_parent is None or loss < best_parent.fresh_loss:
                saved = replace(candidate, branch=prefix + candidate.branch, step=offset + candidate.step)
                source = result.branch.parent
                self.parents[candidate.smiles] = SearchPoint(saved, source.active, loss,
                                                            source.ancestry + (saved.branch,))
        return observed, len(pending)

    def branch_loss(self, observed, previous):
        eligible = [self.by_pose[key]["loss"] for key, smiles in observed.items()
                    if smiles not in previous and key in self.by_pose and self.by_pose[key]["loss"] is not None]
        return min(eligible, default=None)


def _promotions(branch_losses, rng):
    """Break fresh-score ties randomly; retain one independent wildcard."""
    scored = [index for index, loss in enumerate(branch_losses) if loss is not None]
    if scored:
        best_loss = min(branch_losses[index] for index in scored)
        pool = [index for index in scored if branch_losses[index] == best_loss]
    else:
        pool = list(range(len(branch_losses)))
    winner = rng.choice(pool)
    wildcard = rng.choice([index for index in range(len(branch_losses)) if index != winner])
    return [winner, wildcard]


def run_allocation(decoder, scorer, design, starts, output, *, policy, fresh_budget, seed,
                   ready=None, excluded_smiles=(), fresh_score=None, ledger=None, prefix="",
                   optimization_space="vae_latent"):
    """Run a matched allocation arm and return causal score milestones.

    Input points already represent the requested (possibly perturbed) starts.
    The 300 additional updates of a promoted branch resume the exact step-100
    terminal latent and occupancy, even when the terminal molecule is invalid.
    The shared executor evaluates that boundary twice; count both observations.
    """
    starts = list(starts)
    if optimization_space not in ("vae_latent", "relaxed_graph_logits"):
        raise ValueError("Unknown allocation optimization space")
    target_updates = allocation_budget(policy, len(starts))
    if (type(fresh_budget) is not int or fresh_budget < 1 or type(seed) is not int
            or not 0 <= seed < 2**63 - len(starts)):
        raise ValueError("Invalid hard-score allowance or seed")
    if (design.schedule_horizon < (400 if policy == "promote" else POLICY_LENGTHS[policy])
            or design.noise_initial != 0 or design.temperature_initial != design.temperature_final
            or design.chemistry_initial != design.chemistry_final or design.max_step_norm_final is not None):
        raise ValueError("Allocation comparison requires fixed schedules, zero update noise and enough horizon")
    root = Path(output) / "allocation"
    root.mkdir(parents=True, exist_ok=False)
    rng = random.Random(seed)
    if ledger is None:
        ledger = ScoreLedger(fresh_score or scorer.score_valid_records, excluded_smiles, fresh_budget)
    elif fresh_score is not None or excluded_smiles:
        raise ValueError("Scorer and exclusions are already set on the shared ledger")
    initial_attempts = len(ledger.by_pose)
    initial_successes = sum(row["loss"] is not None for row in ledger.by_pose.values())
    if initial_attempts + fresh_budget > ledger.budget:
        raise ValueError("Generation allowance exceeds the arm's remaining hard-score budget")
    started = perf_counter()
    milestones = []
    promotions = []
    allocated_updates = relaxed_observations = 0
    cumulative_allowance = 0
    group_size = 6 if policy == "promote" else len(starts)
    groups = len(starts) // group_size

    def run_segment(point, child, steps, offset, group, allowance):
        nonlocal allocated_updates, relaxed_observations, cumulative_allowance
        phase = "continuation" if offset else "initial"
        stage = root / f"cohort-{group + 1:02d}-{phase}"
        branch = Branch(f"child-{child + 1:02d}", point, seed + child, policy="continuous",
                        updates=steps, noise_seed=seed + child)
        tick = perf_counter()
        coordinates = decoder
        if optimization_space == "relaxed_graph_logits":
            from latentmol.design.boltz_graph_restarts import DirectGraphCoordinates
            spec = scorer.spec if decoder is None else decoder.spec
            coordinates = DirectGraphCoordinates(spec, point.active)
        result, = execute_branches(coordinates, scorer, design, stage, [branch], batch_size=1, cpu_workers=1,
                                   overlap=False, repair_limit=0, invalid_limit=0, profile_steps=True,
                                   ready=ready if not milestones else None)
        optimized = perf_counter()
        if result.terminal is None or result.summary["updates"] != steps:
            raise RuntimeError("Child did not reach the declared segment boundary")
        allocated_updates += steps
        relaxed_observations += steps + 1
        # Carry untouched slots forward for every policy. The cumulative
        # allowance at each segment boundary never exceeds the common cap.
        cumulative_allowance += allowance
        seen, attempts = ledger.score_segment(result, stage / branch.name, offset=offset,
                                               allowance=initial_attempts + cumulative_allowance - len(ledger.by_pose),
                                               prefix=prefix)
        panel = _rank_panel(ledger.best)
        end = perf_counter()
        entry = dict(child=child, phase=phase, start_step=offset, end_step=offset + steps,
                     updates=allocated_updates, relaxed_observations=relaxed_observations,
                     fresh_attempts=len(ledger.by_pose) - initial_attempts, segment_fresh_attempts=attempts,
                     distinct_scored=len(panel), best_loss=panel[0]["loss"] if panel else None,
                     tenth_loss=panel[9]["loss"] if len(panel) >= 10 else None,
                     elapsed_seconds=end - started, optimization_seconds=optimized - tick,
                     fresh_and_summary_seconds=end - optimized,
                     valid_observations=result.summary["valid_states"], distinct_valid=result.summary["distinct_valid"])
        milestones.append(entry)
        write_json(stage / branch.name / "assessment.json", entry)
        return result.terminal, seen

    for group in range(groups):
        cohort_start = group * group_size
        first_block = fresh_budget * group // groups
        group_budget = fresh_budget * (group + 1) // groups - first_block
        prefix_budget = group_budget // 2 if policy == "promote" else group_budget
        previous = set(ledger.best)
        terminals = []
        observed = []
        for i in range(group_size):
            child = cohort_start + i
            allowance = prefix_budget * (i + 1) // group_size - prefix_budget * i // group_size
            terminal, seen = run_segment(starts[child], child, POLICY_LENGTHS[policy], 0, group, allowance)
            terminals.append(terminal)
            observed.append(seen)
        if policy != "promote":
            continue
        losses = [ledger.branch_loss(seen, previous) for seen in observed]
        selected = _promotions(losses, rng)
        decision = dict(cohort=group + 1, prefix_updates=100, novelty_reference="source exclusions and prior cohorts",
                        branch_indices=list(range(cohort_start, cohort_start + group_size)),
                        prefix_best_new_fresh_losses=losses, merit_child=cohort_start + selected[0],
                        wildcard_child=cohort_start + selected[1], merit_fallback_random=all(x is None for x in losses),
                        fresh_calls_at_decision=len(ledger.by_pose), elapsed_at_decision=perf_counter() - started)
        promotions.append(decision)
        write_json(root / f"cohort-{group + 1:02d}-promotion.json", decision)
        tail_budget = group_budget - prefix_budget
        for pos, i in enumerate(selected):
            child = cohort_start + i
            start = starts[child]
            continued = SearchPoint(terminals[i], start.active, None, start.ancestry + (terminals[i].branch,))
            allowance = tail_budget * (pos + 1) // 2 - tail_budget * pos // 2
            run_segment(continued, child, 300, 100, group, allowance)
    if allocated_updates != target_updates or cumulative_allowance != fresh_budget:
        raise RuntimeError("Allocation experiment did not finish its declared budget")
    panel = _rank_panel(ledger.best)
    time_to_ten = next((row["elapsed_seconds"] for row in milestones if row["distinct_scored"] >= 10), None)
    return dict(status="completed", policy=policy, seed=seed, updates=allocated_updates,
                relaxed_observations=relaxed_observations, fresh_budget=fresh_budget,
                fresh_attempts=len(ledger.by_pose) - initial_attempts,
                fresh_successes=sum(row["loss"] is not None for row in ledger.by_pose.values()) - initial_successes,
                distinct_scored=len(panel), top20=panel[:20], best_loss=panel[0]["loss"] if panel else None,
                tenth_loss=panel[9]["loss"] if len(panel) >= 10 else None,
                elapsed_seconds=perf_counter() - started, time_to_ten_seconds=time_to_ten,
                measurement_scope="Elapsed search wall time through completed child-segment scoring; excludes setup",
                milestones=milestones, promotions=promotions)
