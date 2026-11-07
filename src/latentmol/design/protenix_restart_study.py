"""Matched exploratory comparisons of Protenix restart and allocation policies."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import partial
import json
import math
import os
from pathlib import Path

from latentmol.design.latent_protenix import LatentDesignConfig, _model_identity
from latentmol.design.protenix_restart_parents import RestartParent, load_parent_panel, make_prior_parent
from latentmol.design.protenix_restart_proposals import propose_start
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

    @property
    def name(self):
        return f"parent-{self.parent_index:02d}-{self.arm}-{self.repeat:02d}"


def budgets(comparison):
    if "variants" in comparison:
        runs = [updates for variant in comparison["variants"].values() for updates in variant["updates"]]
        branches = comparison["parents"] * len(runs)
        updates = comparison["parents"] * sum(runs)
    else:
        branches = 2 * comparison["parents"] * comparison["cycles"] * comparison["branches_per_parent"]
        updates = branches * comparison["updates"]
    if branches < 1 or not 1 <= updates <= 10_000:
        raise ValueError(f"Exploratory comparison exceeds its 10,000-update cap: {updates}")
    return dict(branches=branches, optimization_updates=updates, relaxed_calls=updates + branches)


def paired_branches(parents, comparison, design, seed):
    for index, parent in enumerate(parents):
        for arm, variant in comparison["variants"].items():
            for repeat, updates in enumerate(variant["updates"]):
                yield Branch(arm, parent, index, repeat, seed + 100 * index + repeat,
                    updates, variant.get("cap", design.max_step_norm), variant["policy"], variant["shortlist"])


def select_candidates(ranked, limit, compare):
    relaxed = ranked[:limit]
    count = (limit + 1) // 2
    top = ranked[:count]
    remaining = sorted(ranked[count:], key=lambda row: (row.step, row.smiles))
    slots = min(limit - len(top), len(remaining))
    timed = [remaining[(2 * i + 1) * len(remaining) // (2 * slots)] for i in range(slots)]
    selected = {"mixed": top + timed}
    if compare:
        selected["relaxed"] = relaxed
    return selected


def parent_record(parent):
    return dict(parent.candidate.record(parent.fresh_loss),
                active_slots=parent.active.nonzero().flatten().tolist(), provenance=parent.provenance)


def run_batch(decoder, make_scorer, fresh_score, design, root, branches, cache, *,
              compare_shortlists=False, first_batch=True):
    root.mkdir()
    selections = {}
    scorer = make_scorer(decoder.spec, branches[0].parent.active, root)
    try:
        for index, branch in enumerate(branches):
            parent = branch.parent
            proposal = propose_start(decoder, parent.candidate, parent.active,
                policy=branch.policy, seed=branch.seed, temperature=design.temperature_initial)
            config = replace(design, updates=branch.updates, max_step_norm=branch.cap)

            def select(ranked, limit):
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
                ready=os.environ.get("DBMOL_LAUNCH_READY") if first_batch and index == 0 else None)
            print(f"{root.name} {branch.name} updates={branch.updates}", flush=True)
    finally:
        scorer.close()
    planned = {name: {policy: [dict(smiles=r.smiles, step=r.step, relaxed_loss=r.relaxed_loss)
                                for r in rows] for policy, rows in lists.items()}
               for name, lists in selections.items()}
    write_json(root / "shortlists.json", planned)
    pending = {}
    for branch in branches:
        for rows in selections[branch.name].values():
            for row in rows:
                if row.smiles not in cache:
                    pending.setdefault(row.smiles,
                        dict(smiles=row.smiles, start=root.name, arm=branch.name, role="edit", step=row.step))
    if pending:
        fresh = fresh_score(root, list(pending.values()))
        if len(fresh) != len(pending) or {row["smiles"] for row in fresh} != pending.keys():
            raise ValueError("Hard scoring returned a different candidate set")
        for row in fresh:
            loss = row["loss"]
            if loss is not None and not math.isfinite(loss):
                raise ValueError("Nonfinite fresh contact loss")
            cache[row["smiles"]] = dict(loss=loss, status=row["status"])
    readouts = []
    for branch in branches:
        data = json.loads((root / branch.name / "summary.json").read_text())
        readouts.append(dict(name=branch.name, arm=branch.arm, parent_index=branch.parent_index,
            repeat=branch.repeat,
            parent_smiles=branch.parent.candidate.smiles, parent_fresh_loss=branch.parent.fresh_loss,
            parent_provenance=branch.parent.provenance, active_slots=branch.parent.active.nonzero().flatten().tolist(),
            policy=branch.policy, seed=branch.seed, max_step_norm=branch.cap, summary=data,
            candidates={policy: [row | cache[row["smiles"]] for row in rows]
                        for policy, rows in planned[branch.name].items()}))
    write_json(root / "batch.json", dict(new_fresh_attempts=len(pending), native_identity=scorer.identity,
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


def lineage_study(decoder, make_scorer, fresh_score, design, output, parents, comparison, cache, seed, prior_seed):
    archives = {arm: {p.candidate.smiles: p for p in parents} for arm in ("elite", "independent")}
    used_priors = {p.candidate.smiles for p in parents}
    cycles = []
    for cycle in range(comparison["cycles"]):
        branches = []
        ordered = sorted(archives["elite"].values(), key=lambda p: (p.fresh_loss, p.candidate.smiles))
        for index, parent in enumerate(ordered):
            for repeat in range(comparison["branches_per_parent"]):
                for _ in range(256):
                    prior = make_prior_parent(decoder, prior_seed, temperature=design.temperature_initial,
                                              occupied_atoms=design.occupied_atoms)
                    prior_seed += 1
                    if prior is not None and prior.candidate.smiles not in used_priors:
                        used_priors.add(prior.candidate.smiles)
                        break
                else:
                    raise ValueError("Could not find a distinct valid independent prior within 256 draws")
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


def run_study(decoder, make_scorer, fresh_score, config, task_name, output, checkpoint_sha256):
    task = config["tasks"][task_name]
    comparison = config["comparisons"][task["comparison"]]
    budget = budgets(comparison)
    design = LatentDesignConfig(**config["design"])
    if design.learning_rate != 150 or design.chemistry_initial != design.chemistry_final:
        raise ValueError("Require LR 150 and a common constant chemistry weight")
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
    write_json(output / "run.json", dict(task=task_name, budget=budget, settings=asdict(design),
        comparison=comparison, panel_offset=offset, decoder_sha256=identity,
        checkpoint_sha256=checkpoint_sha256, job_id=os.environ.get("LSB_JOBID"),
        graph_spec=decoder.spec.to_dict(), parents=[parent_record(p) for p in parents]))
    cache = {p.candidate.smiles: dict(loss=p.fresh_loss, status="archived_parent") for p in parents}
    seed = config["perturbation_seed"] + 100000 * offset
    if "variants" in comparison:
        branches = list(paired_branches(parents, comparison, design, seed))
        _, readouts = run_batch(decoder, make_scorer, fresh_score, design, output / "paired", branches,
                                cache, compare_shortlists=comparison.get("compare_shortlists", False))
        result = dict(paired=paired_summary(readouts))
        if comparison.get("compare_shortlists", False):
            result["matched_first_branch"] = paired_summary([row for row in readouts if row["repeat"] == 0])
    else:
        cycles = lineage_study(decoder, make_scorer, fresh_score, design, output, parents, comparison,
                               cache, seed, config["prior_seed"] + 100000 * offset)
        result = dict(cycles=cycles)
    if _model_identity(decoder) != identity or any(p.grad is not None for p in decoder.parameters()):
        raise ValueError("Frozen decoder parameters or gradients changed")
    result.update(status="completed", budget=budget, new_fresh_attempts=len(cache) - len(parents))
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
