"""Execute independent restart branches in waves through one native scorer."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from functools import partial
import json
import os
from pathlib import Path

import torch

from latentmol.chemistry import inspect_graph
from latentmol.design.latent_protenix import (
    LatentDesignConfig, RawBlocks, _json, finish_point, hard_from_graph, latent_step,
    prepare_point, schedules, structural_target,
)
from latentmol.design.protenix_restart_proposals import propose_start
from latentmol.design.protenix_restarts import Candidate, RestartOffset, write_json
from latentmol.design.protenix_search_frontier import SearchPoint
from latentmol.graph_policy import graph_from_decoder
from latentmol.multimodel.study import hard_digest


@dataclass(frozen=True)
class Branch:
    name: str
    parent: SearchPoint
    seed: int
    policy: str = "latent"
    updates: int | None = None
    noise_seed: int | None = None
    fade_updates: int = 25


@dataclass(frozen=True)
class BranchResult:
    branch: Branch
    valid: list[Candidate]
    invalid: list[SearchPoint]
    summary: dict
    nonreplayable_steps: frozenset[int]


@dataclass
class _LiveBranch:
    branch: Branch
    path: Path
    config: LatentDesignConfig
    point: torch.Tensor
    offset: RestartOffset | None
    noise: torch.Generator
    raw: RawBlocks
    metrics: object
    raw_parent_identity: str | None
    valid: dict = field(default_factory=dict)
    invalid: dict = field(default_factory=dict)
    nonreplayable_steps: set = field(default_factory=set)
    valid_states: int = 0
    bounded_steps: int = 0
    first_valid_step: int | None = None


def _start_branch(decoder, root, branch, config, repair_limit, files):
    source = branch.parent
    candidate = source.candidate
    active = source.active
    temperature = schedules(config, 0)["temperature"]
    latent, offset = candidate.latent.detach().clone(), None
    diagnostics = dict(policy=branch.policy, seed=branch.seed)
    bootstrap = branch.policy == "bootstrap"
    raw_parent_identity = None
    if branch.policy == "raw-invalid":
        if (candidate.smiles is not None or source.fresh_loss is not None or not source.repair_smiles
                or type(source.repair_edits) is not int or not 1 <= source.repair_edits <= repair_limit
                or not source.repair_target_sha256):
            raise ValueError("Raw-invalid branch requires an eligible recorded repair witness")
        diagnostics["repair_witness"] = dict(smiles=source.repair_smiles, edits=source.repair_edits,
                                             sha256=source.repair_target_sha256)
        raw_parent_identity = hard_digest(candidate.hard)
    elif branch.policy == "all-row":
        if candidate.smiles is None:
            raise ValueError("All-row perturbation requires a saved valid molecule")
        offset = RestartOffset.from_parent(decoder, latent, candidate.hard, active, seed=branch.seed,
                                           temperature=temperature, fade_updates=branch.fade_updates)
    elif not bootstrap:
        latent, offset, diagnostics = propose_start(
            decoder, candidate, active, policy=branch.policy, seed=branch.seed,
            temperature=temperature, fade_updates=branch.fade_updates,
        )
    path = root / branch.name
    path.mkdir()
    write_json(path / "start.json", dict(
        parent=None if bootstrap else candidate.smiles,
        parent_source=None if bootstrap else dict(branch=candidate.branch, step=candidate.step),
        ancestry=list(source.ancestry), active_slots=active.nonzero().flatten().tolist(),
        perturbation_seed=None if bootstrap else branch.seed, noise_seed=config.noise_seed,
        fade_updates=branch.fade_updates if offset is not None else 0, proposal=diagnostics,
    ))
    raw = RawBlocks(path / "raw-blocks", decoder.spec.max_nodes, 100)
    files.callback(raw.flush)
    metrics = files.enter_context((path / "metrics.jsonl").open("x", encoding="utf-8", buffering=1))
    return _LiveBranch(branch, path, config, latent, offset,
                       torch.Generator().manual_seed(config.noise_seed), raw, metrics, raw_parent_identity)


@dataclass(frozen=True)
class _Observation:
    candidate: Candidate
    record: dict
    next_point: torch.Tensor
    repair: SearchPoint | None
    replayable: bool


def _finish_step(decoder, row, *, repair_limit):
    live, prepared, score, target_bundle = row
    evaluation = finish_point(decoder, prepared, score, target_bundle=target_bundle)
    graph, config, step = prepared.graph, prepared.config, prepared.step
    hard = hard_from_graph(graph)
    checked = inspect_graph(hard.atoms[0], hard.charges[0], hard.bonds[0], decoder.spec)
    smiles = checked.smiles
    binding = float(evaluation["binding"])
    branch_path = f"{live.path.parent.name}/{live.path.name}"
    candidate = Candidate(smiles, prepared.point.detach(), hard, binding, branch_path, step)
    replayable = True
    if live.offset is not None and step < live.offset.fade_updates:
        with torch.no_grad():
            bare = hard_from_graph(graph_from_decoder(
                decoder.decode(candidate.latent), decoder.spec, temperature=prepared.schedule["temperature"],
                fixed_active=prepared.active,
            ))
        replayable = all(torch.equal(getattr(bare, field), getattr(hard, field))
                         for field in ("atoms", "charges", "bonds"))

    repair = evaluation["repair"]
    edit_count, target = repair["edit_count"], evaluation["target_bundle"][0]
    expansion = None
    if (smiles is None and replayable and target is not None and type(edit_count) is int
            and 1 <= edit_count <= repair_limit):
        witness = target.graph
        repaired = inspect_graph(witness.atoms[0], witness.charges[0], witness.bonds[0], decoder.spec)
        if repaired.smiles:
            expansion = SearchPoint(candidate, prepared.active, None,
                                    live.branch.parent.ancestry + (branch_path,),
                                    edit_count, repaired.smiles, target.identity)
    if step == 0 and live.branch.policy == "raw-invalid":
        parent = live.branch.parent
        if (expansion is None or hard_digest(hard) != live.raw_parent_identity
                or expansion.repair_edits != parent.repair_edits or expansion.repair_smiles != parent.repair_smiles
                or expansion.repair_target_sha256 != parent.repair_target_sha256):
            raise ValueError("Raw-invalid start did not replay its graph and repair witness")

    next_point, update = candidate.latent, None
    if step < config.updates:
        next_point, update = latent_step(candidate.latent, evaluation["binding_gradient"],
                                         evaluation["gradient"], config, step, live.noise)
    record = dict(step=step, smiles=smiles, raw_valid=smiles is not None, failure=checked.failure_reason,
                  relaxed_loss=binding, chemistry_loss=float(evaluation["chemistry"]),
                  repair_status=repair["status"], repair_edit_count=edit_count,
                  repair_target_sha256=repair["target_sha256"], chemistry_weight=prepared.schedule["chemistry_weight"],
                  gradient_norms=evaluation["gradient_norms"], update=update,
                  hard_bond_count=int(hard.bonds[0].triu(diagonal=1).count_nonzero()),
                  soft_bond_mass=float((1 - graph.bonds[0, :, :, 0]).triu(diagonal=1).sum().detach()))
    if "valence_excess" in evaluation:
        record.update(valence_excess=float(evaluation["valence_excess"]),
                      valence_excess_weight=config.valence_excess_weight)
    if expansion is not None:
        record["repair_witness_smiles"] = expansion.repair_smiles
    return _Observation(candidate, record, next_point, expansion, replayable)


def _record_observation(live, observation, *, invalid_limit):
    candidate = observation.candidate
    live.point = observation.next_point
    live.raw.add(candidate.step, candidate.latent, candidate.hard)
    update = observation.record["update"]
    if update is not None:
        live.bounded_steps += update["bound_factor"] < 1
    if not observation.replayable:
        live.nonreplayable_steps.add(candidate.step)
    if candidate.smiles is not None:
        if live.first_valid_step is None:
            live.first_valid_step = candidate.step
        live.valid_states += 1
        previous = live.valid.get(candidate.smiles)
        if previous is None or candidate.relaxed_loss < previous.relaxed_loss:
            live.valid[candidate.smiles] = candidate
    if (source := observation.repair) is not None:
        identity = hard_digest(candidate.hard)
        rank = (source.repair_edits, candidate.relaxed_loss, identity)
        existing = live.invalid.get(identity)
        if identity != live.raw_parent_identity and (existing is None or rank < existing[0]):
            live.invalid[identity] = rank, source
            if len(live.invalid) > invalid_limit:
                worst = max(live.invalid, key=lambda key: live.invalid[key][0])
                del live.invalid[worst]
    live.metrics.write(json.dumps(observation.record, allow_nan=False) + "\n")


def execute_branches(decoder, scorer, design, root, branches, *, batch_size, cpu_workers, overlap,
                     repair_limit, ready=None, invalid_limit=16):
    """Run each wave to completion with a distinct RNG and artifact directory per branch."""
    branches = list(branches)
    names = [branch.name for branch in branches]
    if (type(batch_size) is not int or batch_size not in (1, 2, 4, 8)
            or type(cpu_workers) is not int or cpu_workers < 1 or type(overlap) is not bool
            or any(type(limit) is not int or limit < 0 for limit in (repair_limit, invalid_limit))):
        raise ValueError("Invalid branch execution limits")
    if len(set(names)) != len(names) or any(not name or name in (".", "..") or "/" in name or "\\" in name
                                           or Path(name).name != name for name in names):
        raise ValueError("Require distinct, local branch output names")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if any((root / name).exists() for name in names):
        raise FileExistsError("A branch output already exists")
    configs = [replace(design, updates=design.updates if branch.updates is None else branch.updates,
                       noise_seed=design.noise_seed + branch.seed if branch.noise_seed is None else branch.noise_seed)
               for branch in branches]
    if len({config.noise_seed for config in configs}) != len(configs):
        raise ValueError("Require an independent noise seed per branch")
    results = []
    with ExitStack() as resources:
        pool = (resources.enter_context(ThreadPoolExecutor(max_workers=cpu_workers))
                if overlap or cpu_workers > 1 else None)
        apply = map if pool is None else pool.map
        for first in range(0, len(branches), batch_size):
            wave = slice(first, first + batch_size)
            with ExitStack() as files:
                live = [_start_branch(decoder, root, branch, config, repair_limit, files)
                        for branch, config in zip(branches[wave], configs[wave], strict=True)]
                for step in range(max(branch.config.updates for branch in live) + 1):
                    running = [branch for branch in live if step <= branch.config.updates]
                    prepared = list(apply(lambda branch: prepare_point(
                        decoder, branch.point, branch.branch.parent.active, branch.config, step,
                        logit_offset=branch.offset), running))
                    targets = ([pool.submit(structural_target, point.graph, decoder.spec, point.active,
                                            point.config, step) for point in prepared] if overlap else [])
                    if len(prepared) == 1:
                        point = prepared[0]
                        scores = [scorer.score_at(point.graph, point.active)]
                    else:
                        scores = scorer.score_batch([point.graph for point in prepared],
                                                    [point.active for point in prepared])
                    bundles = [future.result() for future in targets] if overlap else [None] * len(running)
                    rows = zip(running, prepared, scores, bundles, strict=True)
                    observations = list(apply(partial(_finish_step, decoder, repair_limit=repair_limit), rows))
                    if first == 0 and step == 0 and ready is not None:
                        _json(ready, dict(status="models_and_inputs_ready",
                            nonce=os.environ.get("DBMOL_LAUNCH_NONCE"), job_id=os.environ.get("LSB_JOBID")))
                    for branch, observation in zip(running, observations, strict=True):
                        _record_observation(branch, observation, invalid_limit=invalid_limit)
            for branch in live:
                valid = sorted(branch.valid.values(), key=lambda row: (row.relaxed_loss, row.smiles))
                invalid = [source for _, source in sorted(branch.invalid.values(), key=lambda row: row[0])]
                summary = dict(updates=branch.config.updates, valid_states=branch.valid_states,
                    first_valid_step=branch.first_valid_step,
                    distinct_valid=len(valid), unscored_distinct_valid=len(valid), fresh_shortlist=[],
                    bounded_steps=branch.bounded_steps, eligible_invalid=len(invalid),
                    nonreplayable_steps=sorted(branch.nonreplayable_steps))
                write_json(branch.path / "summary.json", summary)
                results.append(BranchResult(branch.branch, valid, invalid, summary,
                                            frozenset(branch.nonreplayable_steps)))
    return results
