"""Run a configurable Protenix restart search or print its resolved plan."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict, dataclass, fields, replace
from functools import partial
from importlib import import_module
import json
import math
import os
from pathlib import Path
from time import perf_counter

import torch

from latentmol.design.bootstrap_screen import bootstrap, screened_bootstrap
from latentmol.design.latent_protenix import LatentDesignConfig, _model_identity
from latentmol.design.protenix_restart_parents import load_parent_panel
from latentmol.design.protenix_restart_proposals import POLICIES
from latentmol.design.protenix_restart_selection import (
    audit_molecule, max_pairwise_similarity, select_candidates, select_diverse_output, select_restart_parents,
)
from latentmol.design.protenix_restart_study import draw_prior
from latentmol.design.protenix_restarts import fresh_score_records, native_factory, restart_args, write_json
from latentmol.design.protenix_search_branches import Branch, execute_branches, refill_exact_starts
from latentmol.design.protenix_search_frontier import (
    SearchPoint, best_frontier_parent, select_frontier, select_invalid_frontier,
)
from latentmol.design.protenix_search_state import load_search_state, save_search_state, search_identity
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256


@dataclass(frozen=True)
class SearchOptions:
    initialization: str = "archived"
    bootstrap_branches: int = 1
    parents: int = 3
    children: int = 2
    waves: int = 5
    batch_size: int = 1
    cpu_workers: int = 1
    overlap: bool = False
    profile_steps: bool = False
    parent_policy: str = "frontier"
    perturbation: str = "latent"
    shortlist: str = "mixed"
    fresh_per_branch: int = 6
    hard_scoring: str = "native_reimport"
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
    time_budget_seconds: int | None = None
    frontier_keep_best: bool = False
    duplicate_refill_attempts: int = 0
    bootstrap_screen_pool: int = 0
    fresh_per_wave: tuple[int, ...] | None = None

    def __post_init__(self):
        for name in ("parents", "children", "waves", "batch_size", "cpu_workers", "fade_updates", "bootstrap_branches",
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
                or self.batch_size not in (1, 2, 3, 4, 8)
                or type(self.overlap) is not bool or type(self.audit_chemistry) is not bool
                or type(self.profile_steps) is not bool):
            raise ValueError("Unknown initialization, parent, perturbation, shortlist, or overlap policy")
        if self.hard_scoring not in {"native_reimport", "fixed_pose_dbmol"}:
            raise ValueError("Unknown hard scoring/cache policy")
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
        if self.time_budget_seconds is not None and (type(self.time_budget_seconds) is not int or self.time_budget_seconds < 1):
            raise ValueError("Time budget must be a positive integer in seconds")
        if type(self.frontier_keep_best) is not bool or (self.frontier_keep_best and self.parent_policy != "frontier"):
            raise ValueError("Best-parent reservation is a boolean frontier option")
        if type(self.duplicate_refill_attempts) is not int or not 0 <= self.duplicate_refill_attempts <= 32:
            raise ValueError("Exact start refill accepts 0 to 32 retries per branch slot")
        if (type(self.bootstrap_screen_pool) is not int or self.bootstrap_screen_pool < 0
                or (self.bootstrap_screen_pool and (self.initialization != "bootstrap"
                    or not 2 <= self.bootstrap_branches <= self.bootstrap_screen_pool))):
            raise ValueError("Bootstrap screening requires a bootstrap pool covering at least two branch slots")
        if (self.parent_policy == "frontier" and self.max_ancestor_fraction is not None
                and math.floor(self.parents * self.children * self.max_ancestor_fraction) < self.children):
            raise ValueError("Ancestry cap leaves no room for one parent; adjust max_ancestor_fraction")
        if self.fresh_per_wave is not None:
            if (not isinstance(self.fresh_per_wave, (list, tuple)) or len(self.fresh_per_wave) != self.waves
                    or any(type(limit) is not int or not 1 <= limit <= slots * self.fresh_per_branch
                           for limit, slots in zip(self.fresh_per_wave, self.wave_widths()))):
                raise ValueError("Specify one feasible positive fresh-score cap per wave")
            object.__setattr__(self, "fresh_per_wave", tuple(self.fresh_per_wave))

    def wave_widths(self):
        width = self.parents * self.children
        if self.initialization == "bootstrap":
            return (self.bootstrap_branches,) + (width,) * (self.waves - 1)
        return (width,) * self.waves

    def budget(self, updates):
        branches = sum(self.wave_widths())
        fresh_attempts = (sum(self.fresh_per_wave) if self.fresh_per_wave is not None
                          else branches * self.fresh_per_branch)
        return dict(branches=branches, updates=branches * updates,
                    fresh_attempts=fresh_attempts)


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
    design = LatentDesignConfig.for_restart(design)
    if options.duplicate_refill_attempts and design.noise_initial != 0:
        raise ValueError("Exact start refill requires zero update noise")
    if options.frontier_keep_best and hooks.get("parent_selector") is not None:
        raise ValueError("Choose either the built-in best-parent reservation or a custom selector")
    cohorts = config.get("parent_cohorts", [])
    encoded = config.get("encoded_founder")
    if encoded and (cohorts or options.initialization != "archived" or options.parents != 1
                    or config.get("optimization_space") != "vae_latent" or not options.diversity_limit
                    or options.hard_scoring != "native_reimport"
                    or config.get("scorer", {}).get("kind") == "boltz2_dbmol"):
        raise ValueError("An encoded founder requires a single-parent native latent search and output panel")
    if options.initialization == "archived" and not (cohorts or encoded):
        raise ValueError("Archived initialization requires parent_cohorts")
    if set(hooks) - {"parent_selector", "shortlist_selector"} or any(
            not isinstance(value, str) or value.count(":") != 1 or not all(value.split(":")) for value in hooks.values()):
        raise ValueError("Hooks use module:callable for parent_selector or shortlist_selector")
    settings = dict(arm=arm, design=asdict(design), search=asdict(options), budget=options.budget(design.updates),
        parent_cohorts=cohorts, panel_seed=config.get("panel_seed", 982701), perturbation_seed=seed,
        prior_seed=config.get("prior_seed", 985000), hooks=hooks)
    if encoded:
        seeds = config.get("confirmation_geometry_seeds", [])
        if (len(set(seeds)) != len(seeds) or any(type(s) is not int or not 0 < s < 2**31 or s == 1201 for s in seeds)):
            raise ValueError("Held-out geometry seeds must be distinct and different from the discovery seed")
        settings.update(encoded_founder=encoded, confirmation_geometry_seeds=seeds)
        settings["budget"].update(founder_fresh_attempts=1, confirmation_fresh_attempts=4 * len(seeds))
    if "nesso_blend" in config:
        blend = config["nesso_blend"]
        if (not encoded or options.batch_size != 1 or design.occupied_atoms != 25
                or not isinstance(blend, dict) or set(blend) != {"affinity_share", "seed", "assets_dir"}
                or not isinstance(blend["assets_dir"], str) or not blend["assets_dir"]
                or type(blend["seed"]) is not int or not 0 <= blend["seed"] < 2**32
                or type(blend["affinity_share"]) not in (int, float) or not 0 < blend["affinity_share"] < 1):
            raise ValueError("Nesso blend needs a 25-atom encoded founder, serial children and valid settings")
        settings["nesso_blend"] = blend
    if not options.bootstrap_screen_pool:
        # Keep resume identities for pre-screening configs unchanged.
        settings["search"].pop("bootstrap_screen_pool")
    if options.fresh_per_wave is None:
        settings["search"].pop("fresh_per_wave")
    if config.get("study") == "restart_search" and "target" in config["arms"][arm]:
        from latentmol.guidance.protenix_pocket import validate_pocket
        target = config["arms"][arm]["target"]
        validate_pocket(target["pocket"])
        if target["initialized_atoms"] != design.occupied_atoms:
            raise ValueError("Target atom count and design occupancy differ")
        settings["target"] = target
    if "scorer" in config:
        settings["scorer"] = config["scorer"]
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
        return selected, None
    if options.parent_policy == "frontier":
        args = frontier_args(options, wave, history) | dict(max_similarity=options.max_similarity)
        selected = select_frontier(eligible, **args)
        if options.frontier_keep_best:
            best = best_frontier_parent(eligible, wave=wave, history=history, min_step_gap=options.min_step_gap)
            if best is not None and all(point is not best for point in selected):
                return select_frontier(eligible, reserved=best, **args), best
        return selected, None
    ranked = sorted(archive, key=lambda smiles: (archive[smiles].fresh_loss, smiles))
    if options.parent_policy == "diverse":
        ranked = select_restart_parents({smiles: (point.candidate, point.fresh_loss)
                                         for smiles, point in archive.items()}, options.parents)
    return [archive[smiles] for smiles in list(ranked)[:options.parents]], None


def score_wave(root, results, fresh_score, archive, attempted, options, selector, *, limit=None):
    posed = options.hard_scoring == "fixed_pose_dbmol"
    hard_records = {}

    def scoring_key(row):
        if not posed:
            return row.smiles
        from latentmol.guidance.boltz_search import hard_pose_key
        graph = {key: getattr(row.hard, key)[0].detach().cpu().tolist() for key in ("atoms", "charges", "bonds")}
        key = hard_pose_key(graph)
        hard_records[key] = graph
        return key

    pending = {}
    shortlist_positions = {}
    for result in results:
        ranked = [row for row in result.valid if (key := scoring_key(row)) not in attempted and key not in pending]
        if selector:
            selected = list(selector(tuple(ranked), options.fresh_per_branch))
            if (len(selected) > options.fresh_per_branch or len({id(row) for row in selected}) != len(selected)
                    or {id(row) for row in selected} - {id(row) for row in ranked}):
                raise ValueError("Shortlist hook must return distinct eligible candidates within the scoring budget")
        elif options.shortlist == "mixed":
            selected = select_candidates(ranked, options.fresh_per_branch, False)["mixed"]
        else:
            selected = ranked[:options.fresh_per_branch]
        for position, row in enumerate(selected):
            key = scoring_key(row)
            pending[key] = (row, result)
            if limit is not None:
                shortlist_positions[key] = position
    if limit is not None and len(pending) > limit:
        # Give each branch a first score before spending slots on secondary picks.
        keys = sorted(pending, key=shortlist_positions.__getitem__)[:limit]
        pending = {key: pending[key] for key in keys}
    if not pending:
        return [], set()
    records = [dict(smiles=row.smiles, start=root.name, arm=row.branch, role="edit", step=row.step,
                   **(dict(scoring_key=key, hard_graph=hard_records[key]) if posed else {}))
               for key, (row, _) in pending.items()]
    if options.audit_chemistry:
        for row in records:
            row["chemistry_audit"] = audit_molecule(row["smiles"])
    scores = fresh_score(root, records)
    key_field = "scoring_key" if posed else "smiles"
    if (len(scores) != len(pending) or {row[key_field] for row in scores} != pending.keys()
            or any(row["smiles"] != pending[row[key_field]][0].smiles for row in scores)):
        raise ValueError("Fresh scorer returned a different candidate set")
    fresh_losses, successes = [], set()
    for scored in scores:
        smiles, loss = scored["smiles"], scored["loss"]
        if loss is None:
            continue
        if not math.isfinite(loss):
            raise ValueError("Nonfinite hard-scored loss")
        row, result = pending[scored[key_field]]
        fresh_losses.append(loss)
        successes.add(smiles)
        parent = result.branch.parent
        if not posed or smiles not in archive or loss < archive[smiles].fresh_loss:
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


def run_search(decoder, scorer_factory, fresh_score, design, options, settings, output, checkpoint_sha256, *,
               run_started=None, resume_state=None):
    if run_started is None:
        run_started = perf_counter()
    decoder.eval().requires_grad_(False)
    if any(parameter.grad is not None for parameter in decoder.parameters()):
        raise ValueError("Frozen decoder carries preexisting gradients")
    identity = _model_identity(decoder)
    state_identity = search_identity(settings, checkpoint_sha256, identity, decoder.spec)
    restored = load_search_state(resume_state, state_identity, decoder.spec) if resume_state else None
    selector = load_hook(settings["hooks"].get("parent_selector"))
    shortlist = load_hook(settings["hooks"].get("shortlist_selector"))
    founders = []
    founder_records = []
    if restored:
        founders, founder_records = restored["founders"], restored["founder_records"]
    elif options.initialization == "archived":
        if encoded := settings.get("encoded_founder"):
            from latentmol.design.protenix_encoded_founder import score_encoded_founder
            parents = [score_encoded_founder(decoder, encoded, design, checkpoint_sha256, output, fresh_score)]
        else:
            parents = load_parent_panel(decoder, [Path(path) for path in settings["parent_cohorts"]],
                count=options.parents, seed=settings["panel_seed"], temperature=design.temperature_initial,
                occupied_atoms=design.occupied_atoms, checkpoint_sha256=checkpoint_sha256)
        for index, parent in enumerate(parents, 1):
            founder = f"founder-{index:02d}"
            point = SearchPoint(replace(parent.candidate, branch=founder), parent.active, parent.fresh_loss, (founder,))
            founders.append(point)
            founder_records.append(point_record(point) | dict(provenance=parent.provenance))
    archive = {point.candidate.smiles: point for point in founders}
    founder_smiles = set(archive)
    attempted, used_priors, ineligible = set(founder_smiles), set(founder_smiles), set()
    seen_starts = set()
    history, invalid_pool, summaries = {}, [], []
    completed_updates = fresh_attempts = 0
    prior_seed = settings["prior_seed"]
    previous_segments = []
    continuation = {}
    if restored:
        archive, invalid_pool, history = (restored[key] for key in ("archive", "invalid_pool", "history"))
        attempted, used_priors, ineligible, seen_starts = (
            restored[key] for key in ("attempted", "used_priors", "ineligible", "seen_starts"))
        prior_seed, summaries, previous_segments = (
            restored[key] for key in ("prior_seed", "summaries", "segments"))
        completed_updates = summaries[-1]["cumulative_updates"]
        fresh_attempts = summaries[-1]["cumulative_fresh_attempts"]
        continuation = dict(resumed_from=str(Path(resume_state).resolve()), previous_waves=summaries)
    first_wave = len(summaries) + 1
    if first_wave > options.waves:
        raise ValueError("Search checkpoint has already reached its configured wave limit")
    segments = previous_segments + [str(output.resolve())]
    if restored or "encoded_founder" not in settings:
        output.mkdir(parents=True)
    write_json(output / "run.json", settings | dict(runner="restart_search", checkpoint_sha256=checkpoint_sha256,
        decoder_sha256=identity, graph_spec=decoder.spec.to_dict(), job_id=os.environ.get("LSB_JOBID"),
        parents=founder_records, cycles=options.waves, segments=segments, **continuation))
    status = "completed"
    time_stop = None
    for wave in range(first_wave, options.waves + 1):
        started = perf_counter()
        if options.time_budget_seconds is not None and summaries:
            recent_wave_seconds = max(row["elapsed_seconds"] for row in summaries[-3:])
            remaining = options.time_budget_seconds - (started - run_started)
            reserve = 1.20 * recent_wave_seconds
            if reserve >= remaining:
                status = "time_budget"
                time_stop = dict(next_wave=wave, remaining_seconds=remaining, estimated_wave_seconds=reserve)
                break
        phase = f"cycle-{wave:02d}"
        reserved = None
        screening = None
        if wave == 1 and options.initialization == "bootstrap":
            if options.bootstrap_screen_pool:
                selected, screening = screened_bootstrap(decoder, design, seed=design.seed,
                    count=options.bootstrap_branches, pool_size=options.bootstrap_screen_pool,
                    near_edit_limit=options.repair_edit_limit)
            else:
                selected = ([bootstrap(decoder, design)] if options.bootstrap_branches == 1 else
                            [bootstrap(decoder, design, seed=design.seed + i, name=f"bootstrap-{i + 1:02d}")
                             for i in range(options.bootstrap_branches)])
            children = [(point, "bootstrap") for point in selected]
        elif options.parent_policy == "independent":
            selected, children = [], []
            for _ in range(options.parents * options.children):
                prior, prior_seed = draw_prior(decoder, design, used_priors, prior_seed)
                point = SearchPoint(prior.candidate, prior.active, None, (prior.candidate.branch,))
                children.append((point, "none"))
        else:
            eligible = {smiles: point for smiles, point in archive.items() if smiles not in ineligible}
            selected, reserved = ((founders, None) if (wave == 1 and founders) or options.parent_policy == "fixed"
                                  else choose_parents(eligible, options, wave, history, selector))
            planned_children = [(point, repeat) for point in selected
                                for repeat in range(1 if point is reserved else options.children)]
            children = [(point, options.perturbation) for point, _ in planned_children]
        if not children:
            status = "frontier_exhausted"
            break
        exact = []
        if children[0][1] != "bootstrap" and options.parent_policy != "independent":
            positions = sorted(range(len(children)), key=lambda i: (planned_children[i][1], i))
            exact = positions[:options.exact_children]
            for index in exact:
                children[index] = (children[index][0], "none")
        invalid = None
        invalid_index = next((i for i in exact if children[i][0] is not reserved), None)
        if options.invalid_children and invalid_index is not None:
            invalid = select_invalid_frontier(invalid_pool, selected=selected,
                max_repair_edits=options.repair_edit_limit, replacing_parent=children[invalid_index][0], reserved=reserved,
                **frontier_args(options, wave, history))
            if invalid is not None:
                children[invalid_index] = (invalid, "raw-invalid")
        root = output / phase
        root.mkdir()
        if screening is not None:
            write_json(root / "bootstrap-screen.json", screening)
        branches = []
        for index, (point, policy) in enumerate(children, 1):
            seed = settings["perturbation_seed"] + wave * 10_000 + index - 1
            branches.append(Branch(f"branch-{index:02d}", point, seed, policy=policy,
                noise_seed=design.noise_seed + wave * 10_000 + index - 1, fade_updates=options.fade_updates))
        collisions = []
        if options.duplicate_refill_attempts:
            branches, collisions = refill_exact_starts(decoder, design, branches, seen_starts,
                attempts=options.duplicate_refill_attempts, fallback_policy=options.perturbation,
                repair_limit=options.repair_edit_limit)
        for branch in branches:
            if branch.policy != "bootstrap":
                point = branch.parent
                history.setdefault(point.candidate.branch, set()).add(point.candidate.step)
        requested = options.bootstrap_branches if children[0][1] == "bootstrap" else options.parents * options.children
        invalid_result_index = next((i for i, branch in enumerate(branches) if branch.policy == "raw-invalid"), None)
        reservation, dedupe = {}, {}
        if options.frontier_keep_best:
            reservation["reserved_best"] = (dict(branch=reserved.candidate.branch, step=reserved.candidate.step,
                                                 smiles=reserved.candidate.smiles, fresh_loss=reserved.fresh_loss)
                                            if reserved is not None else None)
        if options.duplicate_refill_attempts:
            dedupe.update(duplicate_starts=collisions, duplicate_shortfall=sum(not row["refilled"] for row in collisions))
        wave_cap = options.fresh_per_wave[wave - 1] if options.fresh_per_wave is not None else None
        cap_receipt = {"fresh_attempt_cap": wave_cap} if wave_cap is not None else {}
        write_json(root / "wave-plan.json", dict(wave=wave, requested_branches=requested,
            invalid_requested=bool(options.invalid_children and exact), invalid_selected=invalid_result_index is not None,
            branches=[dict(name=branch.name, policy=branch.policy, seed=branch.seed,
                           noise_seed=branch.noise_seed, source=point_record(branch.parent)) for branch in branches])
            | reservation | dedupe | cap_receipt)
        if not branches:
            status = "duplicate_frontier_exhausted"
            break
        scorer = scorer_factory(decoder.spec, branches[0].parent.active, root)
        try:
            results = execute_branches(decoder, scorer, design, root, branches, batch_size=options.batch_size,
                cpu_workers=options.cpu_workers, overlap=options.overlap,
                profile_steps=options.profile_steps,
                repair_limit=options.repair_edit_limit if options.invalid_children else 0,
                invalid_limit=options.invalid_pool_per_branch,
                ready=os.environ.get("DBMOL_LAUNCH_READY") if wave == first_wave else None)
        finally:
            scorer.close()
        before = len(attempted)
        fresh_losses, successes = score_wave(root, results, fresh_score, archive, attempted,
                                             options, shortlist, limit=wave_cap)
        for result in results:
            invalid_pool.extend(result.invalid)
            for row in result.valid:
                if row.step in result.nonreplayable_steps and (options.perturbation != "all-row" or options.frontier_keep_best):
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
        invalid_path = (f"{phase}/{branches[invalid_result_index].name}" if invalid_result_index is not None else None)
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
            invalid_selected=invalid_path is not None, invalid_fallback=bool(options.invalid_children and exact and not invalid_path),
            invalid_first_valid_step=(results[invalid_result_index].summary.get("first_valid_step")
                                      if invalid_result_index is not None else None),
            invalid_fresh_scored=len(invalid_losses), invalid_best_new_fresh_loss=min(invalid_losses, default=None),
            near_valid_pool_size=len(invalid_pool),
            max_started_depth=max(len(branch.parent.ancestry) - 1 for branch in branches),
            elapsed_seconds=perf_counter() - started) | reservation | dedupe
        write_json(root / "archive.json", entry | dict(native_identity=getattr(scorer, "identity", None),
            elites=[point_record(point) for point in leaders[:options.parents]],
            restart_parents=[point_record(point) for point in selected]))
        summaries.append(entry)
        save_search_state(output / "search-state.pt", state_identity, founders=founders, archive=archive,
            invalid_pool=invalid_pool, attempted=attempted, used_priors=used_priors, ineligible=ineligible,
            history=history, seen_starts=seen_starts, prior_seed=prior_seed, summaries=summaries,
            founder_records=founder_records, segments=segments)
        print(json.dumps(entry, allow_nan=False), flush=True)
    if _model_identity(decoder) != identity or any(parameter.grad is not None for parameter in decoder.parameters()):
        raise ValueError("Frozen decoder parameters or gradients changed")
    panel_archive = {k:v for k,v in archive.items() if k not in founder_smiles} if "encoded_founder" in settings else archive
    result = dict(status=status, planned=settings["budget"], cycles=summaries,
        completed_updates=completed_updates, unique_fresh_attempts=fresh_attempts,
        elapsed_seconds=perf_counter() - run_started,
        final_diversity=final_diversity(output, panel_archive, options), frozen_decoder_unchanged=True)
    if "vae_checkpoint" in settings:
        result.update(optimization_space=settings["optimization_space"], vae_checkpoint=settings["vae_checkpoint"])
    if time_stop is not None:
        result["time_budget_stop"] = time_stop
    write_json(output / "result.json", result)
    return result


def main(argv=None):
    started = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
        epilog="Native runs also take the standard allocated Protenix arguments such as --output and --checkpoint.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", required=True, help="Preset name from the config")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.FIELD=VALUE")
    parser.add_argument("--plan-only", action="store_true", help="Validate inputs and print settings without a GPU")
    parser.add_argument("--resume-state", type=Path, help="Read a completed-wave checkpoint into a new output root")
    parser.add_argument("--resume-sha256", help="Expected SHA-256 of the completed-wave checkpoint")
    args, native = parser.parse_known_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("study") == "synthesis_pilot":
        from latentmol.design.protenix_synthesis_study import resolve_pilot, run_pilot, pilot_variants
        if args.overrides or args.resume_state or args.resume_sha256:
            parser.error("Synthesis panels use explicit config variants and do not resume a search wave")
        design, budget = resolve_pilot(config, args.arm)
        if args.plan_only:
            if native:
                parser.error("Unexpected native arguments in plan-only mode")
            preview = dict(arm=args.arm, budget=budget, design=asdict(design),
                           variants=pilot_variants(config, args.arm), task=config["tasks"][args.arm])
            if config.get("founder_policy"):
                preview["protocol"] = {key: config[key] for key in (
                    "founder_policy", "parent_panel", "shortlist_policy", "sa_mode", "parent_count",
                    "waves", "recovery_wave", "fresh_per_parent", "confirm_top_hits", "vina_parent_panel")
                    if key in config}
            print(json.dumps(preview, indent=2))
            return
        runtime = restart_args(__doc__, ["--config", str(args.config), "--arm", args.arm, *native])
        if file_sha256(runtime.plan) != runtime.plan_sha256:
            raise ValueError("GPU launch plan changed")
        plan = json.loads(runtime.plan.read_text(encoding="utf-8"))
        if (file_sha256(args.config) != plan["config_sha256"]
                or budget["updates"] != plan["planned_updates"][args.arm]
                or budget["fresh_attempts"] != plan["planned_fresh_attempts"][args.arm]):
            raise ValueError("Synthesis panel differs from its staged budget or config")
        if config.get("confirm_top_hits") and (budget["confirmation_fresh_attempts"]
                != plan.get("planned_confirmation_fresh_attempts", {}).get(args.arm)):
            raise ValueError("Synthesis confirmation allowance differs from the launch plan")
        if "native_vina_attempts" in budget and (budget["native_vina_attempts"]
                != plan.get("planned_native_vina_attempts", {}).get(args.arm)):
            raise ValueError("Native Vina redocking allowance differs from the launch plan")
        decoder, _ = load_model(runtime.checkpoint, "cpu")
        result = run_pilot(decoder, native_factory(runtime), partial(fresh_score_records, runtime),
                         config, args.arm, runtime.output, runtime.checkpoint_sha256)
        if config.get("confirm_top_hits"):
            from latentmol.design.protenix_hit_confirmation import confirm_search_hits
            confirm_search_hits(runtime, config, result)
        return result
    design, options, settings = resolve_config(config,
                                                args.arm, args.overrides)
    settings["config_source"] = dict(path=str(args.config), sha256=file_sha256(args.config), overrides=args.overrides)
    if args.plan_only:
        if native or args.resume_state or args.resume_sha256:
            parser.error(f"Unexpected plan arguments: {' '.join(native)}")
        print(json.dumps(settings, indent=2, allow_nan=False))
        return
    if config.get("scorer", {}).get("kind") == "boltz2_dbmol":
        from latentmol.design.boltz_restart import run_config
        return run_config(args, native, config, design, options, settings, started)
    runtime = restart_args(__doc__, ["--config", str(args.config), "--arm", args.arm, *native])
    if bool(args.resume_state) != bool(args.resume_sha256):
        parser.error("--resume-state and --resume-sha256 must be supplied together")
    if args.resume_state and file_sha256(args.resume_state) != args.resume_sha256:
        raise ValueError("Search checkpoint checksum differs from the declared value")
    if file_sha256(runtime.plan) != runtime.plan_sha256:
        raise ValueError("GPU launch plan changed")
    launch_plan = json.loads(runtime.plan.read_text(encoding="utf-8"))
    requested_resume = [str(args.resume_state), args.resume_sha256] if args.resume_state else None
    if launch_plan.get("continuations", {}).get(args.arm) != requested_resume:
        raise ValueError("Search checkpoint differs from the staged launch plan")
    if settings["budget"]["updates"] > launch_plan["planned_updates"][args.arm]:
        raise ValueError("Resolved updates exceed the staged launch budget")
    if target := settings.get("target"):
        from latentmol.guidance.protenix_target_context import prepare_runtime
        prepare_runtime(runtime, target, design.occupied_atoms)
    fresh_config = json.loads(runtime.fresh_config.read_text(encoding="utf-8"))
    settings["scoring_configs"] = dict(native=file_sha256(runtime.native_config),
                                       fresh=file_sha256(runtime.fresh_config))
    wave_caps = options.fresh_per_wave or tuple(width * options.fresh_per_branch for width in options.wave_widths())
    if max(wave_caps) > fresh_config["candidate_limit"]:
        raise ValueError("Wave fresh-score budget exceeds the scoring config limit")
    if (cpu_slots := os.environ.get("LSB_DJOB_NUMPROC", "")).isdigit() and options.cpu_workers > int(cpu_slots):
        raise ValueError("CPU workers exceed the allocated CPU slots")
    if options.batch_size > 1 or options.cpu_workers > 1 or options.overlap:
        torch.set_num_threads(1)
    decoder, checkpoint = load_model(runtime.checkpoint, "cpu")
    if requested := config.get("vae_checkpoint"):
        if (requested != launch_plan.get("vae_checkpoint")
                or requested["path"] != str(runtime.checkpoint)
                or requested["sha256"] != runtime.checkpoint_sha256
                or requested["updates"] != checkpoint.get("continuation", {}).get("updates")):
            raise ValueError("Loaded VAE does not match the requested checkpoint and update")
        if config["optimization_space"] != "vae_latent":
            raise ValueError("The native VAE restart search optimizes the latent vector")
        settings.update(optimization_space=config["optimization_space"], vae_checkpoint=requested)
    del checkpoint
    native_config = json.loads(runtime.native_config.read_text(encoding="utf-8"))
    factory = native_factory(runtime, timeout=native_config.get(
        "startup_timeout_seconds", 2400 if options.batch_size > 1 else 900))
    with ExitStack() as stack:
        session = None
        if blend := settings.get("nesso_blend"):
            from latentmol.design.protenix_nesso_blend import NessoBlendSession
            session = NessoBlendSession(blend, runtime, decoder, design, settings["encoded_founder"], factory, stack)
            factory = session.scorer
        result = run_search(decoder, factory, partial(fresh_score_records, runtime),
                   design, options, settings, runtime.output, runtime.checkpoint_sha256,
                   run_started=started, resume_state=args.resume_state)
        if encoded := settings.get("encoded_founder"):
            from latentmol.design.protenix_encoded_founder import summarize_encoded_search
            summarize_encoded_search(runtime, encoded, settings["confirmation_geometry_seeds"])
        if session is not None:
            from latentmol.design.protenix_nesso_readout import write_nesso_readout
            write_nesso_readout(runtime.output, session.fresh, seed=blend["seed"],
                                model_sha256=session.worker.identity["model_sha256"])
    if target:
        panel = json.loads((runtime.output / "diversity.json").read_text())["score_only"]["molecules"]
        write_json(runtime.output / "selected-molecules.json", dict(target=target,
            requested_primary=10, requested_distinct=20, available=len(panel),
            primary=panel[:10], additional=panel[10:20],
            search_status=result["status"], ranking="Ascending fresh native pocket-contact loss"))


if __name__ == "__main__":
    main()
