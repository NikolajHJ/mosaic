"""Compare one-hop hard graph edits and latent restarts at equal fresh scoring cost."""

from collections import Counter
from dataclasses import asdict
from functools import partial
from itertools import combinations, zip_longest
import json
import math
import os
from pathlib import Path
import random
from time import monotonic

from latentmol.chemistry import inspect_graph
from latentmol.design.latent_protenix import LatentDesignConfig, _model_identity
from latentmol.design.protenix_restart_parents import load_parent_panel
from latentmol.design.protenix_restart_proposals import propose_start
from latentmol.design.protenix_restart_selection import audit_molecule, select_candidates
from latentmol.design.protenix_restart_study import parent_record
from latentmol.design.protenix_restarts import (
    fresh_score_records, native_factory, restart_args, run_branch, write_json,
)
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256


def graph_edit_pool(parent, spec, *, seed, budget, excluded=()):
    """Propose edits to a fixed hard graph and keep connected, valid unique molecules."""
    atoms, charges, bonds = (getattr(parent.hard, key)[0].detach().cpu().numpy()
                             for key in ("atoms", "charges", "bonds"))
    if inspect_graph(atoms, charges, bonds, spec).smiles != parent.smiles or not parent.smiles:
        raise ValueError("Graph edit parent must match its archived connected molecule")
    if type(budget) is not int or not 1 <= budget <= 4096:
        raise ValueError("Graph edit budget must be an integer in [1, 4096]")
    active = [i for i, atom in enumerate(atoms) if atom != 0]
    single = spec.bond_names.index("SINGLE")
    atom_edits = [
        dict(kind="atom_substitution", slot=slot, atom_label=label)
        for slot in active for label in range(1, len(spec.atomic_numbers)) if label != atoms[slot]
    ]
    bond_edits = [
        dict(kind="bond_reattachment", removed=[anchor, moved], added=[moved, target])
        for left, right in combinations(active, 2) if bonds[left, right] == single
        for anchor, moved in ((left, right), (right, left))
        for target in active if target not in (anchor, moved) and bonds[moved, target] == 0
    ]
    rng = random.Random(seed)
    rng.shuffle(atom_edits)
    rng.shuffle(bond_edits)
    seen = set(excluded) | {parent.smiles}
    checked = 0
    rejected = Counter()
    accepted = {"atom_substitution": [], "bond_reattachment": []}
    for pair in zip_longest(atom_edits, bond_edits):
        for edit in pair:
            if edit is None or checked == budget:
                continue
            checked += 1
            edited_atoms, edited_bonds = atoms.copy(), bonds.copy()
            if edit["kind"] == "atom_substitution":
                edited_atoms[edit["slot"]] = edit["atom_label"]
            else:
                left, right = edit["removed"]
                edited_bonds[left, right] = edited_bonds[right, left] = 0
                left, right = edit["added"]
                edited_bonds[left, right] = edited_bonds[right, left] = single
            result = inspect_graph(edited_atoms, charges, edited_bonds, spec)
            if not result.smiles:
                rejected[result.failure_reason] += 1
            elif result.smiles in seen:
                rejected["duplicate"] += 1
            else:
                seen.add(result.smiles)
                accepted[edit["kind"]].append(dict(smiles=result.smiles, edit=edit, step=checked))
        if checked == budget:
            break
    # Give both edit families a chance to enter the scored pool if their valid yields differ.
    candidates = [row for pair in zip_longest(*accepted.values()) for row in pair if row is not None]
    return candidates, dict(checked=checked, rejected=dict(rejected),
                            accepted={kind: len(rows) for kind, rows in accepted.items()})


def _score_arm(fresh_score, root, records):
    started = monotonic()
    root.mkdir()
    rows = fresh_score(root, records) if records else []
    def keys(items):
        return {(row["start"], row["arm"], row["smiles"]) for row in items}

    if len(rows) != len(records) or keys(rows) != keys(records):
        raise ValueError("Fresh scoring returned a different candidate set")
    for row in rows:
        loss = row["loss"]
        if ((row["status"] == "scored") != (loss is not None)
                or (loss is not None and not math.isfinite(loss))):
            raise ValueError("Fresh scoring returned inconsistent or nonfinite losses")
    return rows, monotonic() - started


def run_study(decoder, make_scorer, fresh_score, config, task_name, output, checkpoint_sha256):
    design = LatentDesignConfig.for_restart(config["design"])
    task = config["tasks"][task_name]
    count, offset = task["parents"], task["panel_offset"]
    limit, graph_budget = config["fresh_attempts"], config["graph_attempts"]
    if (config["study"] != "graph_edit" or design.noise_initial != 0
            or any(type(v) is not int or v < 1 for v in (count, limit, graph_budget))
            or type(offset) is not int or not 0 <= offset <= config["panel_size"] - count
            or not limit <= graph_budget <= 4096 or count * design.updates > 10_000
            or count * limit > 850):
        raise ValueError("Invalid graph edit comparison or budget")
    started = monotonic()
    decoder.eval().requires_grad_(False)
    if any(parameter.grad is not None for parameter in decoder.parameters()):
        raise ValueError("Frozen decoder carries preexisting gradients")
    identity = _model_identity(decoder)
    panel = load_parent_panel(decoder, [Path(path) for path in config["parent_cohorts"]],
        count=config["panel_size"], seed=config["panel_seed"], temperature=design.temperature_initial,
        occupied_atoms=design.occupied_atoms, checkpoint_sha256=checkpoint_sha256)
    parents = panel[offset:offset + count]
    if (len(parents) != count or len({parent.candidate.smiles for parent in parents}) != count
            or any(parent.fresh_loss is None or not math.isfinite(parent.fresh_loss) for parent in parents)):
        raise ValueError("Missing distinct scored parents for declared panel slice")
    output.mkdir(parents=True)
    paired = output / "paired"
    paired.mkdir()
    write_json(output / "run.json", dict(task=task_name, settings=asdict(design),
        panel_offset=offset, fresh_attempt_limit_per_parent=limit, graph_attempt_limit_per_parent=graph_budget,
        perturbation_seed=config["perturbation_seed"], checkpoint_sha256=checkpoint_sha256,
        decoder_sha256=identity, job_id=os.environ.get("LSB_JOBID"),
        graph_spec=decoder.spec.to_dict(), parents=[parent_record(parent) for parent in parents]))
    records = {"latent": [], "graph": []}
    excluded = {arm: {parent.candidate.smiles for parent in parents} for arm in records}
    groups = {}
    scorer = make_scorer(decoder.spec, parents[0].active, paired)
    try:
        for index, parent in enumerate(parents):
            name = f"parent-{index:02d}"
            seed = config["perturbation_seed"] + 100 * (offset + index)
            before = monotonic()
            graph, graph_counts = graph_edit_pool(parent.candidate, decoder.spec, seed=seed,
                budget=graph_budget, excluded=excluded["graph"])
            graph_seconds = monotonic() - before
            before = monotonic()
            proposal = propose_start(decoder, parent.candidate, parent.active, policy="latent",
                                     seed=seed, temperature=design.temperature_initial)
            branch = paired / f"{name}-latent-00"
            matched_limit = min(limit, len(graph))
            latent = run_branch(decoder, partial(scorer.score_at, active=parent.active), design,
                branch, parent.candidate.latent, parent.active, parent=parent.candidate,
                proposal=proposal, perturbation_seed=seed,
                shortlist_limit=matched_limit, scored=excluded["latent"],
                select_shortlist=lambda rows, size: select_candidates(rows, size, False)["mixed"],
                ready=os.environ.get("DBMOL_LAUNCH_READY") if index == 0 else None)
            updates = json.loads((branch / "summary.json").read_text(encoding="utf-8"))["updates"]
            if updates != design.updates or len(latent) > matched_limit:
                raise ValueError("Latent branch exceeded the paired proposal or update budget")
            latent_seconds = monotonic() - before
            groups[name] = dict(parent_smiles=parent.candidate.smiles, baseline=parent.fresh_loss,
                fresh_attempt_shortfall=limit - len(latent), arms={
                    "latent": dict(optimization_updates=updates, proposal_seconds=latent_seconds),
                    "graph": dict(optimization_updates=0, proposal_seconds=graph_seconds, **graph_counts),
                })
            latent_records = [dict(smiles=row.smiles, step=row.step, relaxed_loss=row.relaxed_loss)
                              for row in latent]
            for arm, chosen in (("latent", latent_records), ("graph", graph[:len(latent)])):
                for row in chosen:
                    smiles = row["smiles"]
                    if smiles in excluded[arm]:
                        raise ValueError(f"Duplicate fresh candidate in {arm}: {smiles}")
                    excluded[arm].add(smiles)
                    records[arm].append(dict(row, start=name, arm=arm, role="edit",
                                             chemistry_audit=audit_molecule(smiles)))
    finally:
        scorer.close()

    scored = {}
    for arm, selected in records.items():
        rows, elapsed = _score_arm(fresh_score, output / f"fresh-{arm}", selected)
        for name, group in groups.items():
            candidates = [row for row in rows if row["start"] == name]
            successful = [row for row in candidates if row["loss"] is not None]
            best = min((row["loss"] for row in successful), default=None)
            group["arms"][arm].update(fresh_attempts=len(candidates), fresh_scored=len(successful),
                best_candidate_loss=best, improvement=None if best is None else group["baseline"] - best,
                candidates=candidates)
        scored[arm] = dict(fresh_attempts=len(rows), fresh_scored=sum(row["loss"] is not None for row in rows),
            fresh_seconds=elapsed, proposal_seconds=sum(group["arms"][arm]["proposal_seconds"]
                                                       for group in groups.values()),
            optimization_updates=sum(group["arms"][arm]["optimization_updates"] for group in groups.values()))
    if any(len({arm["fresh_attempts"] for arm in group["arms"].values()}) != 1 for group in groups.values()):
        raise ValueError("Graph and latent arms used different fresh attempts for a parent")
    if _model_identity(decoder) != identity or any(parameter.grad is not None for parameter in decoder.parameters()):
        raise ValueError("Frozen decoder parameters or gradients changed")
    result = dict(status="completed", task=task_name, fresh_attempt_limit_per_parent=limit,
                  wall_seconds=monotonic() - started, by_parent=groups, by_arm=scored)
    write_json(output / "result.json", result)
    return result


def main(argv=None):
    args = restart_args(__doc__, argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    parents = config["tasks"][args.arm]["parents"]
    if (file_sha256(args.plan) != args.plan_sha256
            or parents * config["design"]["updates"] >
            json.loads(args.plan.read_text(encoding="utf-8"))["planned_updates"][args.arm]):
        raise ValueError("Graph edit study exceeds or differs from its staged launch plan")
    fresh = json.loads(args.fresh_config.read_text(encoding="utf-8"))
    if parents * config["fresh_attempts"] > fresh["candidate_limit"]:
        raise ValueError("Graph edit study exceeds the fresh-score call limit")
    decoder, _ = load_model(args.checkpoint, "cpu")
    return run_study(decoder, native_factory(args), partial(fresh_score_records, args), config,
                     args.arm, args.output, args.checkpoint_sha256)


if __name__ == "__main__":
    main()
