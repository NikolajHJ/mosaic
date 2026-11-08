"""Run a configurable Protenix restart search or print its resolved plan."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields, replace
from functools import partial
from importlib import import_module
import json
import math
import os
from pathlib import Path
from time import perf_counter

import torch

from latentmol.design.latent_protenix import LatentDesignConfig, _model_identity, choose_active, hard_from_graph
from latentmol.design.protenix_restart_parents import load_parent_panel
from latentmol.design.protenix_restart_proposals import POLICIES
from latentmol.design.protenix_restart_selection import (
    audit_molecule, max_pairwise_similarity, select_candidates, select_diverse_output, select_restart_parents,
)
from latentmol.design.protenix_restart_study import draw_prior
from latentmol.design.protenix_restarts import Candidate, fresh_score_records, native_factory, restart_args, write_json
from latentmol.design.protenix_search_branches import Branch, execute_branches
from latentmol.design.protenix_search_frontier import SearchPoint, select_frontier, select_invalid_frontier
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256
from latentmol.graph_policy import graph_from_decoder


@dataclass(frozen=True)
class SearchOptions:
    initialization: str = "archived"
    parents: int = 3
    children: int = 2
    waves: int = 5
    batch_size: int = 1
    cpu_workers: int = 1
    overlap: bool = False
    parent_policy: str = "frontier"
    perturbation: str = "latent"
    shortlist: str = "mixed"
    fresh_per_branch: int = 6
    invalid_children: int = 0
    exact_children: int = 0
    repair_edit_limit: int = 2
    min_step_gap: int = 100
    max_similarity: float = 0.85
    ancestor_lookback: int = 3
    max_ancestor_fraction: float | None = None
    fade_updates: int = 25
    diversity_limit: int = 0
    diversity_max_similarity: float = 0.60
    high_score_ceiling: float | None = None
    audit_chemistry: bool = False
    invalid_pool_per_branch: int = 8

    def __post_init__(self):
        for name in ("parents", "children", "waves", "batch_size", "cpu_workers", "fade_updates",
                     "fresh_per_branch", "repair_edit_limit", "ancestor_lookback", "invalid_pool_per_branch"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("invalid_children", "exact_children", "min_step_gap", "diversity_limit"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if (self.initialization not in {"archived", "bootstrap"}
                or self.parent_policy not in {"frontier", "top", "diverse", "fixed", "independent"}
                or self.perturbation not in POLICIES or self.shortlist not in {"mixed", "relaxed"}
                or self.batch_size not in (1, 2, 4, 8)
                or type(self.overlap) is not bool or type(self.audit_chemistry) is not bool):
            raise ValueError("Unknown initialization, parent, perturbation, shortlist, or overlap policy")
        for name in ("max_similarity", "diversity_max_similarity", "max_ancestor_fraction"):
            value = getattr(self, name)
            if name == "max_ancestor_fraction" and value is None:
                continue
            if type(value) not in (float, int) or not 0 <= value <= 1 or (name == "max_ancestor_fraction" and not value):
                raise ValueError(f"Invalid fraction: {name}")
        if (self.invalid_children > 1 or self.invalid_children > self.exact_children
                or self.exact_children > self.parents * self.children
                or (self.invalid_children and self.parent_policy != "frontier")
                or (self.parent_policy in {"fixed", "independent"} and self.initialization != "archived")):
            raise ValueError("Invalid exact/near-valid budget or initialization for the parent policy")
        if (self.high_score_ceiling is not None and (type(self.high_score_ceiling) not in (int, float)
                                                    or not math.isfinite(self.high_score_ceiling))):
            raise ValueError("High-score ceiling must be finite")
        if (self.parent_policy == "frontier" and self.max_ancestor_fraction is not None
                and math.floor(self.parents * self.children * self.max_ancestor_fraction) < self.children):
            raise ValueError("Ancestry cap leaves no room for one parent; adjust max_ancestor_fraction")

    def budget(self, updates):
        branches = self.waves * self.parents * self.children
        if self.initialization == "bootstrap":
            branches = 1 + (self.waves - 1) * self.parents * self.children
        return dict(branches=branches, updates=branches * updates,
                    fresh_attempts=branches * self.fresh_per_branch)


def legacy_options(config, arm):
    campaign = config["campaign"] | config.get("campaign_arms", {}).get(arm, {})
    return dict(initialization="bootstrap", waves=campaign["cycles"], parents=campaign["elite_count"],
        children=campaign["branches_per_elite"], fade_updates=campaign.get("fade_updates", 25),
        fresh_per_branch=campaign.get("shortlist_limit", 16), perturbation=campaign.get("restart_policy", "all-row"),
        shortlist=campaign.get("shortlist_policy", "relaxed"), parent_policy=campaign.get("parent_policy", "top"),
        diversity_limit=campaign.get("final_diversity_limit", 0),
        diversity_max_similarity=campaign.get("final_diversity_max_similarity", 0.60),
        audit_chemistry=campaign.get("audit_chemistry", False))


def resolve_config(config, arm, overrides=()):
    if config.get("study") == "restart_search":
        variant = config["arms"][arm]
        design = config["design"] | variant.get("design", {})
        search = config.get("search", {}) | variant.get("search", {})
        seed = config.get("perturbation_seed", 951700)
        hooks = config.get("hooks", {}) | variant.get("hooks", {})
    elif "campaign" in config and "arms" in config:
        design = config["design"] | config["arms"][arm]
        search = legacy_options(config, arm)
        seed = config["campaign"].get("perturbation_seed", 951700)
        hooks = {}
    else:
        raise ValueError("Expected a restart search or historical cyclic campaign config")
    sections = {"design": (design, LatentDesignConfig), "search": (search, SearchOptions)}
    for item in overrides:
        key, separator, value = item.partition("=")
        section, dot, field = key.partition(".")
        if (not separator or not dot or section not in sections
                or field not in {entry.name for entry in fields(sections[section][1])}):
            raise ValueError(f"Unknown override {item!r}. Use design.FIELD=VALUE or search.FIELD=VALUE")
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        sections[section][0][field] = value
    options = SearchOptions(**search)
    design = LatentDesignConfig(**design)
    cohorts = config.get("parent_cohorts", [])
    if options.initialization == "archived" and not cohorts:
        raise ValueError("Archived initialization requires parent_cohorts")
    if set(hooks) - {"parent_selector", "shortlist_selector"} or any(
            not isinstance(value, str) or value.count(":") != 1 or not all(value.split(":")) for value in hooks.values()):
        raise ValueError("Hooks use module:callable for parent_selector or shortlist_selector")
    settings = dict(arm=arm, design=asdict(design), search=asdict(options), budget=options.budget(design.updates),
        parent_cohorts=cohorts, panel_seed=config.get("panel_seed", 982701), perturbation_seed=seed,
        prior_seed=config.get("prior_seed", 985000), hooks=hooks)
    for key in ("panel_seed", "perturbation_seed", "prior_seed"):
        if type(settings[key]) is not int or not 0 <= settings[key] < 2**63:
            raise ValueError(f"Invalid seed: {key}")
    return design, options, settings


def load_hook(reference):
    if reference is None:
        return None
    module, name = reference.split(":")
    hook = getattr(import_module(module), name)
    if not callable(hook):
        raise ValueError(f"Expected a callable search hook: {reference}")
    return hook


def point_record(point):
    return point.candidate.record(point.fresh_loss) | dict(
        active_slots=point.active.nonzero().flatten().tolist(), ancestry=list(point.ancestry),
        repair_edits=point.repair_edits, repair_smiles=point.repair_smiles,
        repair_target_sha256=point.repair_target_sha256)


@torch.no_grad()
def bootstrap(decoder, design):
    point = torch.randn((1, decoder.config.latent), generator=torch.Generator().manual_seed(design.seed))
    decoded = decoder.decode(point)
    active = choose_active(decoded, decoder.spec, design.occupied_atoms)
    hard = hard_from_graph(graph_from_decoder(decoded, decoder.spec, fixed_active=active,
                                               temperature=design.temperature_initial))
    return SearchPoint(Candidate(None, point, hard, 0., "bootstrap", 0), active, None, ("bootstrap",))


def frontier_args(options, wave, history):
    return dict(wave=wave, slots=options.parents, children_per_parent=options.children,
        history=history, min_step_gap=options.min_step_gap, ancestor_lookback=options.ancestor_lookback - 1,
        max_ancestor_fraction=options.max_ancestor_fraction)


def choose_parents(archive, options, wave, history, selector):
    eligible = list(archive.values())
    if selector:
        selected = list(selector(tuple(eligible), wave=wave, options=options, history=history))
        if (len(selected) > options.parents or len({id(point) for point in selected}) != len(selected)
                or {id(point) for point in selected} - {id(point) for point in eligible}):
            raise ValueError("Parent hook must return distinct eligible points within the parent budget")
        return selected
    if options.parent_policy == "frontier":
        return select_frontier(eligible, max_similarity=options.max_similarity,
                               **frontier_args(options, wave, history))
    ranked = sorted(archive, key=lambda smiles: (archive[smiles].fresh_loss, smiles))
    if options.parent_policy == "diverse":
        ranked = select_restart_parents({smiles: (point.candidate, point.fresh_loss)
                                         for smiles, point in archive.items()}, options.parents)
    return [archive[smiles] for smiles in list(ranked)[:options.parents]]


def score_wave(root, results, fresh_score, archive, attempted, options, selector):
    pending = {}
    for result in results:
        ranked = [row for row in result.valid if row.smiles not in attempted and row.smiles not in pending]
        if selector:
            selected = list(selector(tuple(ranked), options.fresh_per_branch))
            if (len(selected) > options.fresh_per_branch or len({id(row) for row in selected}) != len(selected)
                    or {id(row) for row in selected} - {id(row) for row in ranked}):
                raise ValueError("Shortlist hook must return distinct eligible candidates within the scoring budget")
        elif options.shortlist == "mixed":
            selected = select_candidates(ranked, options.fresh_per_branch, False)["mixed"]
        else:
            selected = ranked[:options.fresh_per_branch]
        for row in selected:
            pending[row.smiles] = (row, result)
    if not pending:
        return [], set()
    records = [dict(smiles=smiles, start=root.name, arm=row.branch, role="edit", step=row.step)
               for smiles, (row, _) in pending.items()]
    if options.audit_chemistry:
        for row in records:
            row["chemistry_audit"] = audit_molecule(row["smiles"])
    scores = fresh_score(root, records)
    if len(scores) != len(pending) or {row["smiles"] for row in scores} != pending.keys():
        raise ValueError("Fresh scorer returned a different candidate set")
    fresh_losses, successes = [], set()
    for scored in scores:
        smiles, loss = scored["smiles"], scored["loss"]
        if loss is None:
            continue
        if not math.isfinite(loss):
            raise ValueError("Nonfinite fresh contact loss")
        row, result = pending[smiles]
        fresh_losses.append(loss)
        successes.add(smiles)
        parent = result.branch.parent
        archive[smiles] = SearchPoint(row, parent.active, loss, parent.ancestry + (row.branch,))
    attempted.update(pending)
    return fresh_losses, successes


def final_diversity(output, archive, options):
    if not options.diversity_limit:
        return None
    scores = {smiles: (point.candidate, point.fresh_loss) for smiles, point in archive.items()}
    ranked = sorted(scores, key=lambda smiles: (scores[smiles][1], smiles))[:options.diversity_limit]
    diverse = select_diverse_output(scores, options.diversity_limit, max_similarity=options.diversity_max_similarity)
    panels = {}
    for name, selected in (("score_only", ranked), ("diverse", diverse)):
        entries = [point_record(archive[smiles]) for smiles in selected]
        panels[name] = dict(molecules=entries, count=len(entries), max_pairwise_similarity=max_pairwise_similarity(selected),
            mean_fresh_loss=sum(row["fresh_loss"] for row in entries) / len(entries) if entries else None)
    write_json(output / "diversity.json", panels)
    return {name: {key: value for key, value in panel.items() if key != "molecules"} for name, panel in panels.items()}


def run_search(decoder, scorer_factory, fresh_score, design, options, settings, output, checkpoint_sha256):
    decoder.eval().requires_grad_(False)
    if any(parameter.grad is not None for parameter in decoder.parameters()):
        raise ValueError("Frozen decoder carries preexisting gradients")
    identity = _model_identity(decoder)
    selector = load_hook(settings["hooks"].get("parent_selector"))
    shortlist = load_hook(settings["hooks"].get("shortlist_selector"))
    founders = []
    founder_records = []
    if options.initialization == "archived":
        parents = load_parent_panel(decoder, [Path(path) for path in settings["parent_cohorts"]],
            count=options.parents, seed=settings["panel_seed"], temperature=design.temperature_initial,
            occupied_atoms=design.occupied_atoms, checkpoint_sha256=checkpoint_sha256)
        for index, parent in enumerate(parents, 1):
            founder = f"founder-{index:02d}"
            point = SearchPoint(replace(parent.candidate, branch=founder), parent.active, parent.fresh_loss, (founder,))
            founders.append(point)
            founder_records.append(point_record(point) | dict(provenance=parent.provenance))
    output.mkdir(parents=True)
    write_json(output / "run.json", settings | dict(runner="restart_search", checkpoint_sha256=checkpoint_sha256,
        decoder_sha256=identity, graph_spec=decoder.spec.to_dict(), job_id=os.environ.get("LSB_JOBID"),
        parents=founder_records, cycles=options.waves))
    archive = {point.candidate.smiles: point for point in founders}
    founder_smiles = set(archive)
    attempted, used_priors, ineligible = set(founder_smiles), set(founder_smiles), set()
    history, invalid_pool, summaries = {}, [], []
    completed_updates = fresh_attempts = 0
    prior_seed = settings["prior_seed"]
    status = "completed"
    for wave in range(1, options.waves + 1):
        started = perf_counter()
        phase = f"cycle-{wave:02d}"
        if wave == 1 and options.initialization == "bootstrap":
            selected = [bootstrap(decoder, design)]
            children = [(selected[0], "bootstrap")]
        elif options.parent_policy == "independent":
            selected, children = [], []
            for _ in range(options.parents * options.children):
                prior, prior_seed = draw_prior(decoder, design, used_priors, prior_seed)
                point = SearchPoint(prior.candidate, prior.active, None, (prior.candidate.branch,))
                children.append((point, "none"))
        else:
            eligible = {smiles: point for smiles, point in archive.items() if smiles not in ineligible}
            selected = (founders if (wave == 1 and founders) or options.parent_policy == "fixed"
                        else choose_parents(eligible, options, wave, history, selector))
            children = [(point, options.perturbation) for point in selected for _ in range(options.children)]
        if not children:
            status = "frontier_exhausted"
            break
        exact = []
        if children[0][1] != "bootstrap" and options.parent_policy != "independent":
            positions = sorted(range(len(children)), key=lambda i: (i % options.children, i))
            exact = positions[:options.exact_children]
            for index in exact:
                children[index] = (children[index][0], "none")
        invalid = None
        if options.invalid_children and exact:
            invalid = select_invalid_frontier(invalid_pool, selected=selected,
                max_repair_edits=options.repair_edit_limit, replacing_parent=children[exact[0]][0],
                **frontier_args(options, wave, history))
            if invalid is not None:
                children[exact[0]] = (invalid, "raw-invalid")
        root = output / phase
        root.mkdir()
        branches = []
        for index, (point, policy) in enumerate(children, 1):
            seed = settings["perturbation_seed"] + wave * 10_000 + index - 1
            branches.append(Branch(f"branch-{index:02d}", point, seed, policy=policy,
                noise_seed=design.noise_seed + wave * 10_000 + index - 1, fade_updates=options.fade_updates))
            if policy != "bootstrap":
                history.setdefault(point.candidate.branch, set()).add(point.candidate.step)
        requested = 1 if children[0][1] == "bootstrap" else options.parents * options.children
        write_json(root / "wave-plan.json", dict(wave=wave, requested_branches=requested,
            invalid_requested=bool(options.invalid_children and exact), invalid_selected=invalid is not None,
            branches=[dict(name=branch.name, policy=branch.policy, seed=branch.seed,
                           noise_seed=branch.noise_seed, source=point_record(branch.parent)) for branch in branches]))
        scorer = scorer_factory(decoder.spec, branches[0].parent.active, root)
        try:
            results = execute_branches(decoder, scorer, design, root, branches, batch_size=options.batch_size,
                cpu_workers=options.cpu_workers, overlap=options.overlap,
                repair_limit=options.repair_edit_limit if options.invalid_children else 0,
                invalid_limit=options.invalid_pool_per_branch,
                ready=os.environ.get("DBMOL_LAUNCH_READY") if wave == 1 else None)
        finally:
            scorer.close()
        before = len(attempted)
        fresh_losses, successes = score_wave(root, results, fresh_score, archive, attempted, options, shortlist)
        for result in results:
            invalid_pool.extend(result.invalid)
            for row in result.valid:
                if row.step in result.nonreplayable_steps and options.perturbation != "all-row":
                    if row.smiles in successes and archive[row.smiles].candidate is row:
                        ineligible.add(row.smiles)
                elif row.smiles in ineligible:
                    parent = result.branch.parent
                    archive[row.smiles] = SearchPoint(row, parent.active, archive[row.smiles].fresh_loss,
                                                      parent.ancestry + (row.branch,))
                    ineligible.remove(row.smiles)
        wave_updates = sum(result.summary["updates"] for result in results)
        if wave_updates != len(branches) * design.updates:
            raise ValueError("Wave completed an unexpected number of optimizer updates")
        completed_updates += wave_updates
        attempts = len(attempted) - before
        fresh_attempts += attempts
        leaders = sorted(archive.values(), key=lambda point: (point.fresh_loss, point.candidate.smiles))
        top_three = leaders[:3]
        invalid_path = f"{phase}/{branches[exact[0]].name}" if invalid is not None else None
        invalid_losses = [archive[smiles].fresh_loss for smiles in successes
                          if invalid_path and archive[smiles].candidate.branch == invalid_path]
        high = {point.candidate.smiles: (point.candidate, point.fresh_loss) for point in leaders
                if options.high_score_ceiling is not None and point.fresh_loss <= options.high_score_ceiling
                and point.candidate.smiles not in founder_smiles}
        high_count = len(select_diverse_output(high, len(high), max_similarity=options.diversity_max_similarity)) if high else 0
        entry = dict(cycle=wave, branches=len(branches), requested_branches=requested, updates=wave_updates,
            cumulative_updates=completed_updates, fresh_attempts=attempts, fresh_scored=len(fresh_losses),
            cumulative_fresh_attempts=fresh_attempts, best_new_fresh_loss=min(fresh_losses, default=None),
            best_fresh_loss=leaders[0].fresh_loss if leaders else None,
            top_three_mean_fresh_loss=sum(point.fresh_loss for point in top_three) / len(top_three) if top_three else None,
            distinct_high_scorers=high_count if options.high_score_ceiling is not None else None,
            invalid_selected=invalid is not None, invalid_fallback=bool(options.invalid_children and exact and invalid is None),
            invalid_first_valid_step=results[exact[0]].summary.get("first_valid_step") if invalid is not None else None,
            invalid_fresh_scored=len(invalid_losses), invalid_best_new_fresh_loss=min(invalid_losses, default=None),
            near_valid_pool_size=len(invalid_pool),
            max_started_depth=max(len(point.ancestry) - 1 for point, _ in children),
            elapsed_seconds=perf_counter() - started)
        write_json(root / "archive.json", entry | dict(native_identity=getattr(scorer, "identity", None),
            elites=[point_record(point) for point in leaders[:options.parents]],
            restart_parents=[point_record(point) for point in selected]))
        summaries.append(entry)
        print(json.dumps(entry, allow_nan=False), flush=True)
    if _model_identity(decoder) != identity or any(parameter.grad is not None for parameter in decoder.parameters()):
        raise ValueError("Frozen decoder parameters or gradients changed")
    result = dict(status=status, planned=settings["budget"], cycles=summaries,
        completed_updates=completed_updates, unique_fresh_attempts=fresh_attempts,
        final_diversity=final_diversity(output, archive, options), frozen_decoder_unchanged=True)
    write_json(output / "result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
        epilog="Native runs also take the standard allocated Protenix arguments such as --output and --checkpoint.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", required=True, help="Preset name from the config")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.FIELD=VALUE")
    parser.add_argument("--plan-only", action="store_true", help="Validate inputs and print settings without a GPU")
    args, native = parser.parse_known_args(argv)
    design, options, settings = resolve_config(json.loads(args.config.read_text(encoding="utf-8")),
                                                args.arm, args.overrides)
    settings["config_source"] = dict(path=str(args.config), sha256=file_sha256(args.config), overrides=args.overrides)
    if args.plan_only:
        if native:
            parser.error(f"Unexpected plan arguments: {' '.join(native)}")
        print(json.dumps(settings, indent=2, allow_nan=False))
        return
    runtime = restart_args(__doc__, ["--config", str(args.config), "--arm", args.arm, *native])
    if file_sha256(runtime.plan) != runtime.plan_sha256:
        raise ValueError("GPU launch plan changed")
    launch_plan = json.loads(runtime.plan.read_text(encoding="utf-8"))
    if settings["budget"]["updates"] > launch_plan["planned_updates"][args.arm]:
        raise ValueError("Resolved updates exceed the staged launch budget")
    fresh_config = json.loads(runtime.fresh_config.read_text(encoding="utf-8"))
    if options.parents * options.children * options.fresh_per_branch > fresh_config["candidate_limit"]:
        raise ValueError("Wave fresh-score budget exceeds the scoring config limit")
    if (cpu_slots := os.environ.get("LSB_DJOB_NUMPROC", "")).isdigit() and options.cpu_workers > int(cpu_slots):
        raise ValueError("CPU workers exceed the allocated CPU slots")
    if options.batch_size > 1 or options.cpu_workers > 1 or options.overlap:
        torch.set_num_threads(1)
    decoder, _ = load_model(runtime.checkpoint, "cpu")
    run_search(decoder, native_factory(runtime, timeout=2400 if options.batch_size > 1 else 900),
               partial(fresh_score_records, runtime),
               design, options, settings, runtime.output, runtime.checkpoint_sha256)


if __name__ == "__main__":
    main()
