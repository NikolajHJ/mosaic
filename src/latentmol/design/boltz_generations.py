"""Compare fixed and selective child allocation with an evolving Boltz archive."""
from pathlib import Path
from functools import partial
from time import perf_counter

import torch

from latentmol.design.boltz_allocation_pilot import ScoreLedger, allocation_budget, run_allocation
from latentmol.design.boltz_speed import _point_digest
from latentmol.design.protenix_restarts import write_json
from latentmol.design.protenix_restart_selection import max_pairwise_similarity
from latentmol.design.protenix_search_state import _pack


def generation_budget(policy, updates_per_generation=3600, generations=3, fresh_per_generation=48,
                      parents=3, children_per_parent=None, last_parent_similarity=None):
    if any(type(v) is not int or v < 1 for v in (updates_per_generation, generations, fresh_per_generation, parents)):
        raise ValueError("Generation counts and budgets must be positive integers")
    if last_parent_similarity is not None and (parents < 2 or not 0 <= last_parent_similarity <= 1):
        raise ValueError("The diversity slot requires at least two parents and a similarity between zero and one")
    group_size = 6 if policy == "promote" else 1
    unit = allocation_budget(policy, group_size)
    if updates_per_generation % unit:
        raise ValueError(f"Generation budget must fit complete {unit}-update groups for {policy}")
    starts = (updates_per_generation // unit) * group_size
    if children_per_parent is not None and (type(children_per_parent) is not int or children_per_parent < 1
                                            or parents * children_per_parent != starts):
        raise ValueError("Parent and child counts must match the complete generation budget")
    return dict(starts_per_generation=starts, updates=updates_per_generation * generations,
                fresh_budget=fresh_per_generation * generations)


def ranked_elites(archive, parents=3, last_parent_similarity=None):
    ranked = sorted(archive.values(), key=lambda p: (p.fresh_loss, p.candidate.smiles))
    panel = ranked[:parents]
    if last_parent_similarity is not None and len(ranked) > parents:
        for candidate in ranked[parents - 1:]:
            if all(max_pairwise_similarity([candidate.candidate.smiles, p.candidate.smiles])
                   <= last_parent_similarity for p in panel[:-1]):
                panel[-1] = candidate
                break
    return panel


def parent_record(point, optimization_space="vae_latent"):
    key = "graph_logits_mask_sha256" if optimization_space == "relaxed_graph_logits" else "latent_mask_sha256"
    return dict(smiles=point.candidate.smiles, fresh_loss=point.fresh_loss, source=point.candidate.branch,
                step=point.candidate.step, ancestry=list(point.ancestry),
                **{key: _point_digest(point.candidate.latent, point.active)})


def run_generations(decoder, scorer, design, state, output, *, policy, seed, generations=3,
                    updates_per_generation=3600, fresh_per_generation=48, parents=3, children_per_parent=None,
                    last_parent_similarity=None, ready=None, optimization_space="vae_latent",
                    reject_unstable_identity=False):
    """Search one independent arm from frozen founders and publish generation checkpoints.

    Eligible parents come from the union of original and
    newly scored, replayable molecules. A separate output panel excludes every
    molecule present in the frozen input. Unused scoring slots expire per wave.
    """
    from latentmol.design.boltz_search_methods import archive_starts

    if optimization_space not in ("relaxed_graph_logits", "vae_latent"):
        raise ValueError("Unknown generation search space")
    direct = optimization_space == "relaxed_graph_logits"
    native_founders = state.get("coordinate_space") == "relaxed_graph_logits"
    if native_founders and (not direct or decoder is not None):
        raise ValueError("Native graph founders require a decoder-free graph-space search")
    founder_latents = state["archive"]
    pack_elite = _pack
    if direct:
        from latentmol.design.boltz_graph_restarts import (
            convert_point, direct_archive_starts, graph_founder_state, pack_graph_point,
        )
        spec = scorer.spec if native_founders else decoder.spec
        if not native_founders:
            state = graph_founder_state(decoder, design, state)
        pack_elite = pack_graph_point
    budget = generation_budget(policy, updates_per_generation, generations, fresh_per_generation,
                               parents, children_per_parent, last_parent_similarity)
    count = budget["starts_per_generation"]
    if type(seed) is not int or not 0 <= seed < 2**63 - generations * 100_000 - count - 10_000:
        raise ValueError("Generation seed does not fit the per-generation proposal streams")
    base = {smi: p for smi, p in state["archive"].items() if smi not in state["ineligible"]}
    if len(base) < parents:
        raise ValueError("The founder archive has fewer eligible molecules than requested parents")
    founder_attempts = state.get("founder_fresh_attempts", 0)
    if (native_founders or founder_attempts) and (
            founder_attempts != len(base) or not 0 < founder_attempts < fresh_per_generation):
        raise ValueError("Freshly scored founders must fit within the first generation's scoring allowance")
    fresh_score = scorer.score_valid_records
    if reject_unstable_identity:
        fresh_score = partial(fresh_score, reject_unstable_identity=True)
    ledger = ScoreLedger(fresh_score, state.get("excluded_smiles", state["archive"].keys()),
                         budget["fresh_budget"] - founder_attempts, keep_parents=True)
    root = Path(output) / "generations"
    root.mkdir(parents=True, exist_ok=False)
    start_time = perf_counter()
    first_hashes, milestones, waves = [], [], []
    seen_starts = set()
    completed_updates = total_observations = 0
    for index in range(generations):
        name = f"generation-{index + 1:02d}"
        folder = root / name
        folder.mkdir()
        wave_seed = seed + index * 100_000
        archive = base | ledger.parents
        panel = ranked_elites(archive, parents, last_parent_similarity)
        before = [parent_record(p, optimization_space) for p in panel]
        selected = {p.candidate.smiles: p for p in panel}
        selection = dict(archive=selected, ineligible=set())
        matched = {}
        if direct and index == 0 and not native_founders:
            # Generate the exact same children as the control, then release the VAE from the search path.
            selection["archive"] = {s: founder_latents[s] for s in selected}
            originals, proposals = archive_starts(decoder, design, selection, count, wave_seed, seen=set(),
                                                 fallback_prior=True, parent_count=parents)
            matched["paired_vae_start_hashes"] = [_point_digest(p.candidate.latent, p.active) for p in originals]
            starts = [convert_point(decoder, design, p) for p in originals]
        elif direct:
            starts, proposals = direct_archive_starts(spec, design, selection, count, wave_seed,
                                                     seen=seen_starts, parent_count=parents)
        else:
            starts, proposals = archive_starts(decoder, design, selection, count, wave_seed, seen=seen_starts,
                                              fallback_prior=True, parent_count=parents)
        hashes = [_point_digest(p.candidate.latent, p.active) for p in starts]
        if not index:
            first_hashes = hashes
            if direct:
                if len(set(hashes)) != count:
                    raise ValueError("Paired first-wave starts collapsed to duplicate graph coordinates")
                seen_starts.update(hashes)
        write_json(folder / "starts.json", dict(generation=index + 1, seed=wave_seed, elites=before,
                                                optimization_space=optimization_space,
                                                proposals=proposals, start_hashes=hashes, **matched))
        prior_fresh = founder_attempts + len(ledger.by_pose)
        wave_budget = fresh_per_generation - (founder_attempts if index == 0 else 0)
        elapsed = perf_counter() - start_time
        wave = run_allocation(decoder, scorer, design, starts, folder, policy=policy,
                              fresh_budget=wave_budget, seed=wave_seed, ready=ready if not index else None,
                              ledger=ledger, prefix=f"generations/{name}/allocation/",
                              optimization_space=optimization_space)
        if wave["updates"] != updates_per_generation or not 0 <= wave["fresh_attempts"] <= wave_budget:
            raise RuntimeError("Generation did not match declared update and scoring budgets")
        for row in wave["milestones"]:
            milestones.append(row | dict(generation=index + 1, updates=completed_updates + row["updates"],
                                         relaxed_observations=total_observations + row["relaxed_observations"],
                                         fresh_attempts=prior_fresh + row["fresh_attempts"],
                                         elapsed_seconds=elapsed + row["elapsed_seconds"]))
        completed_updates += wave["updates"]
        total_observations += wave["relaxed_observations"]
        after = ranked_elites(base | ledger.parents, parents, last_parent_similarity)
        # Save replayable coordinates and pose masks alongside their score and provenance.
        with (folder / "elites.pt").open("xb") as snapshot:
            torch.save([pack_elite(p) for p in after], snapshot)
        entry = dict(generation=index + 1, cumulative_updates=completed_updates,
                     wave_fresh_attempts=wave["fresh_attempts"],
                     wave_founder_fresh_attempts=founder_attempts if index == 0 else 0,
                     cumulative_fresh_attempts=founder_attempts + len(ledger.by_pose),
                     cumulative_distinct=wave["distinct_scored"], best_novel=wave["best_loss"],
                     tenth_novel=wave["tenth_loss"], selected_parents=before,
                     next_parents=[parent_record(p, optimization_space) for p in after],
                     elapsed_seconds=perf_counter() - start_time)
        write_json(folder / "result.json", dict(summary=entry, allocation=wave))
        waves.append(entry)
    return dict(status="completed", optimization_space=optimization_space, policy=policy, seed=seed,
                waves=waves, updates=completed_updates,
                fresh_budget=budget["fresh_budget"], founder_fresh_attempts=founder_attempts,
                child_fresh_attempts=len(ledger.by_pose), fresh_attempts=founder_attempts + len(ledger.by_pose),
                fresh_successes=founder_attempts + sum(r["loss"] is not None for r in ledger.by_pose.values()),
                relaxed_observations=total_observations, distinct_scored=wave["distinct_scored"],
                best_loss=wave["best_loss"], tenth_loss=wave["tenth_loss"], top20=wave["top20"],
                time_to_ten_seconds=next((r["elapsed_seconds"] for r in milestones if r["distinct_scored"] >= 10), None),
                elapsed_seconds=perf_counter() - start_time, start_hashes=first_hashes, milestones=milestones)
