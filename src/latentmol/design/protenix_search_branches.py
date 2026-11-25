"""Execute independent restart branches in waves through one native scorer."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from functools import partial
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

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
class _BranchStart:
    latent: torch.Tensor
    offset: RestartOffset | None
    diagnostics: dict
    raw_parent_identity: str | None


@dataclass(frozen=True)
class Branch:
    name: str
    parent: SearchPoint
    seed: int
    policy: str = "latent"
    updates: int | None = None
    noise_seed: int | None = None
    fade_updates: int = 25
    prepared: _BranchStart | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class BranchResult:
    branch: Branch
    valid: list[Candidate]
    invalid: list[SearchPoint]
    summary: dict
    nonreplayable_steps: frozenset[int]
    terminal: Candidate | None = None


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
    terminal: Candidate | None = None
    guidance: object | None = None


def _prepare_branch_start(decoder, branch, config, repair_limit):
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
    elif branch.policy == "continuous":
        diagnostics["exact_boundary"] = True
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
    return _BranchStart(latent, offset, diagnostics, raw_parent_identity)


def _exact_start_key(start, active):
    identity = hashlib.sha256()
    tensors = [start.latent, active]
    if start.offset is not None:
        delta = start.offset.delta
        offsets = (delta.atoms, delta.charges, delta.bonds)
        if any(torch.count_nonzero(tensor) for tensor in offsets):
            identity.update(str(start.offset.fade_updates).encode() + b"|")
            tensors.extend(offsets)
    for tensor in tensors:
        tensor = tensor.detach().cpu().contiguous()
        identity.update(str((tensor.dtype, tuple(tensor.shape))).encode() + b"|")
        identity.update(tensor.view(torch.uint8).numpy().tobytes())
    return identity.hexdigest()


def refill_exact_starts(decoder, design, branches, seen, *, attempts, fallback_policy, repair_limit):
    """Keep unique deterministic starts and retry with matched source slots."""
    if design.noise_initial != 0:
        raise ValueError("Exact start refill requires zero update noise")
    accepted, collisions = [], []
    for branch in branches:
        current = branch
        duplicate_attempts = 0
        for attempt in range(attempts + 1):
            start = _prepare_branch_start(decoder, current, design, repair_limit)
            key = _exact_start_key(start, current.parent.active)
            if key not in seen:
                seen.add(key)
                accepted.append(replace(current, prepared=start))
                break
            duplicate_attempts += 1
            policy = fallback_policy if current.policy == "none" else current.policy
            if policy in {"none", "bootstrap", "raw-invalid", "continuous"} or attempt == attempts:
                break
            salt = f"{branch.seed}:{branch.name}:{attempt + 1}".encode()
            seed = int.from_bytes(hashlib.sha256(salt).digest()[:8], "big") % (2**63)
            current = replace(current, seed=seed, policy=policy)
        if duplicate_attempts:
            collisions.append(dict(branch=branch.name, rejected_starts=duplicate_attempts,
                refilled=len(accepted) > 0 and accepted[-1].name == branch.name))
    return accepted, collisions


def _start_branch(decoder, root, branch, config, repair_limit, files):
    source = branch.parent
    candidate = source.candidate
    active = source.active
    bootstrap = branch.policy == "bootstrap"
    start = branch.prepared or _prepare_branch_start(decoder, branch, config, repair_limit)
    latent, offset = start.latent, start.offset
    path = root / branch.name
    path.mkdir()
    write_json(path / "start.json", dict(
        parent=None if bootstrap else candidate.smiles,
        parent_source=None if bootstrap else dict(branch=candidate.branch, step=candidate.step),
        ancestry=list(source.ancestry), active_slots=active.nonzero().flatten().tolist(),
        perturbation_seed=None if bootstrap or branch.policy == "continuous" else branch.seed,
        noise_seed=config.noise_seed,
        fade_updates=branch.fade_updates if offset is not None else 0, proposal=start.diagnostics,
    ))
    raw = RawBlocks(path / "raw-blocks", decoder.spec.max_nodes, 100,
                    coordinate_key=getattr(decoder, "raw_coordinate_key", "latent"))
    files.callback(raw.flush)
    metrics = files.enter_context((path / "metrics.jsonl").open("x", encoding="utf-8", buffering=1))
    return _LiveBranch(branch, path, config, latent, offset,
                       torch.Generator().manual_seed(config.noise_seed), raw, metrics, start.raw_parent_identity)


@dataclass(frozen=True)
class _Observation:
    candidate: Candidate
    record: dict
    next_point: torch.Tensor
    repair: SearchPoint | None
    replayable: bool


def _finish_step(decoder, row, *, repair_limit):
    live, prepared, score, target_bundle = row
    evaluation = finish_point(decoder, prepared, score, target_bundle=target_bundle, guidance=live.guidance)
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
    if step == 0 and live.branch.policy == "continuous":
        parent_hard = live.branch.parent.candidate.hard
        if any(not torch.equal(getattr(hard, field), getattr(parent_hard, field))
               for field in ("atoms", "charges", "bonds")):
            raise ValueError("Continuous start did not replay the previous terminal hard labels")

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
    if "synthesis" in evaluation:
        record["synthesis"] = evaluation["synthesis"]
    if isinstance(evaluation["native_aux"], dict) and "gradient_reuse" in evaluation["native_aux"]:
        record["gradient_reuse"] = evaluation["native_aux"]["gradient_reuse"]
    if isinstance(evaluation["native_aux"], dict) and "model_blend" in evaluation["native_aux"]:
        record["model_blend"] = evaluation["native_aux"]["model_blend"]
    if expansion is not None:
        record["repair_witness_smiles"] = expansion.repair_smiles
    return _Observation(candidate, record, next_point, expansion, replayable)


def _record_observation(live, observation, *, invalid_limit):
    candidate = observation.candidate
    live.point = observation.next_point
    if candidate.step == live.config.updates:
        live.terminal = candidate
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
                     repair_limit, ready=None, invalid_limit=16, guidance_factory=None, profile_steps=False):
    """Run each wave to completion with a distinct RNG and artifact directory per branch."""
    branches = list(branches)
    names = [branch.name for branch in branches]
    if (type(batch_size) is not int or batch_size not in (1, 2, 3, 4, 8)
            or type(cpu_workers) is not int or cpu_workers < 1 or type(overlap) is not bool
            or any(type(limit) is not int or limit < 0 for limit in (repair_limit, invalid_limit))):
        raise ValueError("Invalid branch execution limits")
    if len(set(names)) != len(names) or any(not name or name in (".", "..") or "/" in name or "\\" in name
                                           or Path(name).name != name for name in names):
        raise ValueError("Require distinct, local branch output names")
    if any(branch.policy == "continuous" for branch in branches) and (
            design.noise_initial != 0 or design.temperature_initial != design.temperature_final
            or design.chemistry_initial != design.chemistry_final or design.max_step_norm_final is not None):
        raise ValueError("Continuous replay requires zero update noise and fixed temperature, chemistry and step cap")
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
                if guidance_factory is not None:
                    for branch in live:
                        branch.guidance = guidance_factory(branch.branch)
                for step in range(max(branch.config.updates for branch in live) + 1):
                    iteration_started = perf_counter()
                    cuda_profile = profile_steps and torch.cuda.is_initialized()
                    if cuda_profile and first == 0 and step == 0:
                        torch.cuda.reset_peak_memory_stats()
                    running = [branch for branch in live if step <= branch.config.updates]
                    prepared = list(apply(lambda branch: prepare_point(
                        decoder, branch.point, branch.branch.parent.active, branch.config, step,
                        logit_offset=branch.offset), running))
                    targets = ([pool.submit(structural_target, point.graph, decoder.spec, point.active,
                                            point.config, step) for point in prepared] if overlap else [])
                    if len(prepared) == 1:
                        point = prepared[0]
                        if hasattr(scorer, "score_branch_step"):
                            scores = [scorer.score_branch_step(point.graph, point.active, branch=str(running[0].path),
                                step=step, terminal=step == point.config.updates)]
                        else:
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
                    native_metrics = getattr(scorer, "last_worker_metrics", None)
                    for index, (branch, observation) in enumerate(zip(running, observations, strict=True)):
                        if profile_steps:
                            observation.record["timing"] = dict(iteration_seconds=perf_counter()-iteration_started,
                                batch_size=len(running),
                                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if cuda_profile else None,
                                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved() if cuda_profile else None)
                            if native_metrics is not None and len(native_metrics) == len(running):
                                observation.record["timing"]["native_worker"] = dict(native_metrics[index])
                        _record_observation(branch, observation, invalid_limit=invalid_limit)
                    # A resident Torch scorer can retain both large model
                    # trunks through its scalar autograd output. Drop those
                    # references before constructing the next forward graph.
                    del scores, rows
            for branch in live:
                if branch.terminal is None:
                    raise ValueError("Branch did not record a terminal state")
                valid = sorted(branch.valid.values(), key=lambda row: (row.relaxed_loss, row.smiles))
                invalid = [source for _, source in sorted(branch.invalid.values(), key=lambda row: row[0])]
                summary = dict(updates=branch.config.updates, valid_states=branch.valid_states,
                    first_valid_step=branch.first_valid_step,
                    distinct_valid=len(valid), unscored_distinct_valid=len(valid), fresh_shortlist=[],
                    bounded_steps=branch.bounded_steps, eligible_invalid=len(invalid),
                    nonreplayable_steps=sorted(branch.nonreplayable_steps))
                write_json(branch.path / "summary.json", summary)
                results.append(BranchResult(branch.branch, valid, invalid, summary,
                                            frozenset(branch.nonreplayable_steps), branch.terminal))
    return results
