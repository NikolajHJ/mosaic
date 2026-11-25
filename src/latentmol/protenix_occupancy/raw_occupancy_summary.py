"""Paired endpoint comparisons for raw-decoder occupancy experiments."""

import math
import random
from statistics import mean, quantiles


ARMS = ("gradient", "reversed_gradient")
TIE_TOLERANCE = 1e-6


def _loss(scores, seed):
    value = scores.get(seed, {}).get("loss")
    if value is not None and not math.isfinite(value):
        raise ValueError("Fresh geometry loss must be finite or explicitly unavailable")
    return value


def _comparison(differences, comparator):
    win_key = "reversed_wins" if comparator == "reversed_gradient" else f"{comparator}_wins"
    return dict(paired_starts=len(differences),
                mean_paired_difference=mean(differences) if differences else None,
                guided_wins=sum(value < -TIE_TOLERANCE for value in differences),
                ties=sum(abs(value) <= TIE_TOLERANCE for value in differences),
                **{win_key: sum(value > TIE_TOLERANCE for value in differences)})


def _trajectory_counts(start_count, arm, *, include_chemistry):
    accepted = [row for row in arm["updates"] if row["status"] == "accepted"]
    trials = [row["trials"][-1] for row in accepted]
    final_count = arm["final"]["atom_count"]
    counts = dict(final_atom_count=final_count, net_atom_delta=final_count - start_count,
                  attempted_updates=len(arm["updates"]), accepted_updates=len(accepted),
                  atom_adding_updates=sum(trial["atom_delta"] > 0 for trial in trials),
                  atom_removing_updates=sum(trial["atom_delta"] < 0 for trial in trials))
    if include_chemistry:
        counts["molecule_changing_updates"] = sum(
            row["baseline"]["smiles"] != trial["smiles"] for row, trial in zip(accepted, trials))
    return counts


def summarize(starts, *, comparator="reversed_gradient", bootstrap_seed=0, draws=2000):
    """Average geometry seeds within each start, then compare matched arms.

    The primary comparison uses complete endpoint pairs across all fresh
    seeds. A missing geometry is retained in coverage, and the bootstrap
    resamples whole starts to keep their geometry replicates together.
    Additional controls also report each arm's fresh loss, change from the
    initial molecule, and count and chemistry edits.
    """
    if type(draws) is not int or draws < 2:
        raise ValueError("A bootstrap requires at least two draws")
    if (not isinstance(comparator, str) or not comparator
            or comparator in {"gradient", "guided", "initial", "difference"}):
        raise ValueError("A comparator needs a distinct arm name")
    arm_names = ("gradient", comparator)
    include_chemistry = arm_names != ARMS
    seeds = sorted({seed for start in starts
                    for scores in (start["initial_fresh_by_seed"],
                                   *(start["arms"][name]["fresh_by_seed"] for name in arm_names))
                    for seed in scores}, key=int)
    per_seed = {seed: [] for seed in seeds}
    differences, by_start = [], []
    for start in starts:
        count = start["initial"]["atom_count"]
        arms = {name: _trajectory_counts(count, start["arms"][name],
                                         include_chemistry=include_chemistry) for name in arm_names}
        seed_rows = {}
        for seed in seeds:
            initial = _loss(start["initial_fresh_by_seed"], seed)
            guided, control = (_loss(start["arms"][name]["fresh_by_seed"], seed) for name in arm_names)
            difference = guided - control if guided is not None and control is not None else None
            seed_rows[seed] = {"initial": initial, "gradient": guided, comparator: control,
                               "difference": difference}
            if difference is not None:
                per_seed[seed].append(difference)
        values = [row["difference"] for row in seed_rows.values() if row["difference"] is not None]
        difference = mean(values) if values and len(values) == len(seeds) else None
        if difference is not None:
            differences.append(difference)
        by_start.append(dict(draw=start["draw"], initial_atom_count=count, arms=arms,
                             paired_fresh_seeds=len(values), mean_paired_difference=difference,
                             fresh_by_seed=seed_rows))

    interval = None
    if len(differences) >= 2:
        rng = random.Random(bootstrap_seed)
        samples = [mean(rng.choices(differences, k=len(differences))) for _ in range(draws)]
        bounds = quantiles(samples, n=40, method="inclusive")
        interval = [bounds[0], bounds[-1]]

    arm_totals = {}
    for name in arm_names:
        counts = [row["arms"][name] for row in by_start]
        arm_totals[name] = dict(
            starts_with_net_growth=sum(row["net_atom_delta"] > 0 for row in counts),
            starts_with_net_shrinkage=sum(row["net_atom_delta"] < 0 for row in counts),
            starts_with_any_count_change=sum(
                row["atom_adding_updates"] + row["atom_removing_updates"] > 0 for row in counts),
            attempted_updates=sum(row["attempted_updates"] for row in counts),
            accepted_updates=sum(row["accepted_updates"] for row in counts))
        if include_chemistry:
            losses, changes = [], []
            for start in by_start:
                seed_rows = start["fresh_by_seed"].values()
                values = [row[name] for row in seed_rows]
                initial = [row["initial"] for row in seed_rows]
                if values and all(value is not None for value in values):
                    losses.append(mean(values))
                    if all(value is not None for value in initial):
                        changes.append(losses[-1] - mean(initial))
            arm_totals[name].update(
                fresh_complete_starts=len(losses),
                mean_fresh_loss=mean(losses) if losses else None,
                fresh_change_complete_starts=len(changes),
                mean_fresh_change_from_initial=mean(changes) if changes else None,
                molecule_changing_updates=sum(row["molecule_changing_updates"] for row in counts),
                atom_adding_updates=sum(row["atom_adding_updates"] for row in counts),
                atom_removing_updates=sum(row["atom_removing_updates"] for row in counts),
                starts_with_addition=sum(row["atom_adding_updates"] > 0 for row in counts),
                starts_with_removal=sum(row["atom_removing_updates"] > 0 for row in counts))
    return dict(starts=len(starts), fresh_seeds=seeds, **_comparison(differences, comparator),
                paired_difference_bootstrap_95=interval, bootstrap_seed=bootstrap_seed,
                bootstrap_draws=draws, tie_tolerance=TIE_TOLERANCE,
                per_seed={seed: _comparison(values, comparator) for seed, values in per_seed.items()},
                arms=arm_totals, by_start=by_start)
