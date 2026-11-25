"""Matched BRD4 restart search directly on neutral atom and unordered bond probabilities."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

from repair_frontier import ParentFrontier, hard_graph_key, near_valid_key
from study import hard_smiles, write


def project(rows, floor):
    """Euclidean row projection onto {p >= floor, sum(p) = 1}."""
    rows = np.asarray(rows, dtype=np.float64)
    width = rows.shape[1]
    if not 0 < floor < 1 / width or not np.isfinite(rows).all():
        raise ValueError("Invalid simplex floor or nonfinite proposal")
    values = rows - rows.max(axis=1, keepdims=True)
    mass = 1 - width * floor
    ordered = np.sort(values, axis=1)[:, ::-1]
    offsets = (np.cumsum(ordered, axis=1) - mass) / np.arange(1, width + 1)
    last = np.sum(ordered > offsets, axis=1) - 1
    threshold = offsets[np.arange(len(rows)), last, None]
    return (floor + np.maximum(0., values - threshold)).astype(np.float32)


def update(graph, gradient, cfg):
    displacements = [-cfg["learning_rate"] * d.astype(np.float64) for d in gradient]
    norm = math.sqrt(sum(float(np.vdot(d, d)) for d in displacements))
    if not math.isfinite(norm):
        raise FloatingPointError("Nonfinite combined molecular probability gradient")
    scale = min(1., cfg["max_step_norm"] / norm) if norm else 1.
    proposed = tuple(project(x + scale * d, cfg["probability_floor"]) for x, d in zip(graph, displacements))
    distance = math.sqrt(sum(float(np.sum((y-x)**2)) for x, y in zip(graph, proposed)))
    return proposed, dict(unclipped_step_norm=norm, clip_factor=scale, probability_step_norm=distance)


def projected_repair_comparison(graph, contact, repair, cfg):
    """Compare hypothetical updates on one witnessed state using the actual cap and projection."""
    def norm(rows):
        return math.sqrt(sum(float(np.sum(np.square(row, dtype=np.float64))) for row in rows))

    proposals, step_norms = {}, {}
    for weight in (1, 5, 10):
        combined = tuple(d + weight * r for d, r in zip(contact, repair))
        proposals[weight], stats = update(graph, combined, cfg)
        step_norms[weight] = stats["probability_step_norm"]

    comparisons = []
    for low, high in ((1, 5), (5, 10)):
        pairs = list(zip(proposals[low], proposals[high]))
        distance = norm([a - b for a, b in pairs])
        scale = max(step_norms[low], step_norms[high])
        comparisons.append(dict(weights=[low, high], projected_l2_difference=distance,
                                relative_to_larger_step=distance / scale if scale else 0.,
                                changed_argmax_rows=sum(int(np.count_nonzero(a.argmax(1) != b.argmax(1)))
                                                        for a, b in pairs)))
    return dict(contact_gradient_norm=norm(contact), repair_gradient_norm=norm(repair),
                projected_step_norms=step_norms, comparisons=comparisons)


def perturb(graph, mass, generator, floor):
    """Perturb one occupied atom and soften one existing bond; preserve the parent's hard readout."""
    new = tuple(x.copy() for x in graph)
    atom, bond = new
    atom_index = int(generator.integers(len(atom)))
    occupied = np.flatnonzero(bond.argmax(1) != 0)
    decisions = [(atom, atom_index, int(generator.integers(1, atom.shape[1])))]
    if len(occupied):
        decisions.append((bond, int(generator.choice(occupied)), None))
    for rows, index, offset in decisions:
        row = rows[index]
        winner = int(row.argmax())
        target = 0 if offset is None else (winner + offset) % len(row)
        goal = np.eye(len(row), dtype=np.float32)[target]
        candidate = project(((1 - mass)*row + mass*goal)[None], floor)[0]
        if candidate.argmax() != winner:
            goal = np.eye(len(row), dtype=np.float32)[winner]
            candidate = project(((1 - mass)*row + mass*goal)[None], floor)[0]
        rows[index] = candidate
    if hard_graph_key(new) != hard_graph_key(graph):
        raise ValueError("Restart changed the parent's hard molecule")
    return new


def digest_graph(graph):
    h = hashlib.sha256()
    for array in graph:
        h.update(array.tobytes())
    return h.hexdigest()


def finite_contact(backend, graph):
    loss, ga, gb = backend.gradient(*graph)
    if (not math.isfinite(loss) or any(d.shape != x.shape or not np.isfinite(d).all()
                                      for d, x in zip((ga, gb), graph))):
        raise FloatingPointError("Invalid native Protenix contact or derivatives")
    return float(loss), (ga, gb)


def evaluate_segment(backend, oracle, frontier, graph, cfg, weight, steps, branch, folder,
                     founder_sulfur, ready=None):
    folder.mkdir()
    snapshots, valid, nearby, roles = [], {}, {}, {}
    statuses = Counter(dict.fromkeys(("already_valid", "verified_witness", "no_witness"), 0))
    diagnostic_samples = 0
    sulfur_sum = valid_sources = 0
    sequence = []
    with (folder / "trajectory.jsonl").open("x", buffering=1) as stream:
        for step in range(steps + 1):
            loss, gradient = finite_contact(backend, graph)
            repair, da, db, info = oracle.evaluate(*graph)
            if (not math.isfinite(repair) or any(d.shape != x.shape or not np.isfinite(d).all()
                                                 for d, x in zip((da, db), graph))):
                raise FloatingPointError("Invalid repair loss or derivative")
            point = dict(info, graph=graph, branch=branch, step=step, relaxed_loss=loss)
            smi = info["smiles"]
            key = hard_graph_key(graph)
            sequence.append(key)
            sulfur = info["hard_sulfur"] - founder_sulfur
            if step == 0:
                roles["initial"] = point
            if smi:
                if "best_relaxed_valid" not in roles or loss < roles["best_relaxed_valid"]["relaxed_loss"]:
                    roles["best_relaxed_valid"] = point
                if key not in valid or loss < valid[key]["relaxed_loss"] or valid[key]["step"] == 0:
                    valid[key] = point
            elif step:
                if key not in nearby or near_valid_key(point) < near_valid_key(nearby[key]):
                    nearby[key] = point
                if len(nearby) > 20:
                    del nearby[max(nearby, key=lambda k: (near_valid_key(nearby[k]), k))]
            snapshots.append(tuple(x.argmax(axis=1).astype(np.uint8) for x in graph))
            row = dict(info, step=step, graph_key=key, relaxed_contact_loss=loss, repair_loss=repair,
                       total_loss=loss + weight * repair, hard_sulfur_drift=sulfur)
            if hasattr(backend, "binding_metrics"):
                row["binding"] = backend.binding_metrics
                weighted_loss = backend.binding_metrics["weighted_model_value"] + weight * repair
                if backend.binding_metrics["policy"] == "adaptive":
                    del row["total_loss"]
                    row["stepwise_weighted_loss"] = weighted_loss
                else:
                    row["total_loss"] = weighted_loss
            if step < steps:
                valid_sources += int(bool(smi))
                sulfur_sum += sulfur
                statuses[info["status"]] += 1
                if (cfg.get("compare_repair_weights", True) and info["status"] == "verified_witness"
                        and (statuses["verified_witness"] - 1) % 25 == 0):
                    row["repair_weight_diagnostic"] = projected_repair_comparison(graph, gradient, (da, db), cfg)
                    diagnostic_samples += 1
                combined = tuple(d + weight*r for d, r in zip(gradient, (da, db)))
                graph, row["update"] = update(graph, combined, cfg)
            else:
                roles["endpoint"] = point
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            if ready and step == 0:
                ready()
    with (folder / "argmax-states.npz").open("xb") as stream:
        np.savez_compressed(stream, atoms=np.stack([a for a, _ in snapshots]), bonds=np.stack([b for _, b in snapshots]))
    hard_readouts = []
    for role, p in roles.items():
        hard_readouts.append(dict(role=role, branch=branch, step=p["step"], smiles=p["smiles"],
                                  relaxed_contact_loss=p["relaxed_loss"],
                                  hard_contact_loss=frontier.hard_contact(p["graph"]) if p["smiles"] else None))
    write(folder / "hard-readouts.json", hard_readouts)
    summary = dict(branch=branch, updates=steps, valid_update_states=valid_sources,
                   fraction_valid_steps=valid_sources/steps, distinct_valid=len({p["smiles"] for p in valid.values()}),
                   mean_sulfur_drift=sulfur_sum/steps, sulfur_drift_sum=sulfur_sum, endpoint_sulfur_drift=sulfur,
                   repair_status_counts=dict(statuses), repair_weight_diagnostic_samples=diagnostic_samples)
    write(folder / "summary.json", summary)
    return list(valid.values()), list(nearby.values()), summary, hard_readouts, sequence


def run_arm(backend, oracle, frontier, initial, cfg, weight, seed, policy, folder, ready=None,
            score_archive=None):
    folder.mkdir()
    original = hard_smiles(initial)
    if not original:
        raise ValueError("The sparse founder must be valid and connected")
    start_loss = frontier.hard_contact(initial)
    initial_s = int(np.count_nonzero(initial[0].argmax(1) == 3))
    starts = [(initial, None, 0.)]
    summaries, readouts, candidates, prefixes = [], [], {}, []
    expanded = set()
    parent_limit = cfg["parents"]
    width = parent_limit * cfg["children_per_parent"]
    for wave in range(cfg["child_generations"] + 1):
        group = folder / f"generation-{wave:02d}"
        group.mkdir()
        write(group / "starts.json", [dict(slot=i, source=None if p is None else
              dict(branch=p["branch"], step=p["step"], graph_key=hard_graph_key(p["graph"])),
              restart_mass=m, start_sha256=digest_graph(g)) for i, (g, p, m) in enumerate(starts)])
        possible, near = [], []
        siblings = defaultdict(list)
        for i, (g, parent, _) in enumerate(starts):
            name = f"generation-{wave:02d}/child-{i:02d}"
            valid, invalid, summary, roles, path = evaluate_segment(
                backend, oracle, frontier, g, cfg, weight, cfg["child_updates"] if wave else cfg["founder_updates"],
                name, group / f"child-{i:02d}", initial_s, ready=ready if wave == i == 0 else None)
            summaries.append(summary)
            readouts.extend(roles)
            for p in valid:
                loss = frontier.hard_contact(p["graph"])
                smi = p["smiles"]
                if smi not in candidates or loss < candidates[smi]["selected_hard_contact_loss"]:
                    candidates[smi] = dict(smiles=smi, branch=name, step=p["step"],
                                           selected_hard_contact_loss=loss, score_change=loss-start_loss)
            possible.extend(valid)
            near.extend(invalid)
            if parent is not None:
                siblings[(parent["branch"], parent["step"])].append((name, path))
        merged = set()
        divergence = []
        for key, paths in siblings.items():
            first = next((j for j, labels in enumerate(zip(*(s for _, s in paths))) if len(set(labels)) > 1), None)
            if first is None and len(paths) > 1:
                merged.update(n for n, _ in paths)
            divergence.append(dict(parent=list(key), children=[n for n, _ in paths], first_different_hard_step=first))
        write(group / "sibling-paths.json", divergence)
        hits = sorted((r for r in candidates.values() if r["smiles"] != original and r["score_change"] < 0),
                      key=lambda r: (r["selected_hard_contact_loss"], r["smiles"]))
        prefixes.append(dict(generation=wave, updates=sum(s["updates"] for s in summaries),
                             distinct_valid=len(candidates), improved=len(hits), best=hits[0] if hits else None))
        write(group / "progress.json", prefixes[-1])
        if score_archive is not None:
            score_archive(group, candidates)
        if wave == cfg["child_generations"]:
            break
        parents, ranking = frontier.select(possible, near, count=parent_limit,
                          vina_weight=cfg["search_versions"][policy]["vina_weight"], expanded_keys=expanded)
        write(group / "parent-scores.json", ranking)
        if not parents:
            raise ValueError("No post-update graph available even in the near-valid fallback")
        expanded.update(r["graph_key"] for r in ranking["selected"])
        starts, used = [], set()
        for slot in range(width):
            p = parents[slot % len(parents)]
            graph = p["graph"]
            mass = 0.
            if slot >= len(parents):
                mass = cfg["restart_mass_after_merge"] if p["branch"] in merged else cfg["restart_mass"]
                rng = np.random.default_rng([seed, wave, slot])
                for _ in range(32):
                    graph = perturb(p["graph"], mass, rng, cfg["probability_floor"])
                    if digest_graph(graph) not in used:
                        break
            identifier = digest_graph(graph)
            if identifier in used:
                raise ValueError("Could not produce distinct relaxed sibling starts")
            used.add(identifier)
            starts.append((graph, p, mass))
    steps = sum(s["updates"] for s in summaries)
    valid_source_count = sum(s["valid_update_states"] for s in summaries)
    final = [s for s in summaries if s["branch"].startswith(f"generation-{wave:02d}/")]
    result = dict(seed=seed, repair_weight=weight, starting_smiles=original, starting_hard_contact_loss=start_loss,
                  selected=hits[0] if hits else None, improved_distinct=len(hits), distinct_valid=len(candidates),
                  distinct_new_valid=len(candidates)-1, updates=steps, fraction_valid_steps=valid_source_count/steps,
                  mean_sulfur_drift=sum(s["sulfur_drift_sum"] for s in summaries)/steps,
                  mean_final_leaf_endpoint_sulfur_drift=float(np.mean([s["endpoint_sulfur_drift"] for s in final])),
                  endpoint_sulfur_drifts={s["branch"]:s["endpoint_sulfur_drift"] for s in final},
                  founder_only=prefixes[0], generations=prefixes)
    write(folder / "scored-molecules.json", sorted(candidates.values(),key=lambda r:r["selected_hard_contact_loss"]))
    write(folder / "hard-readouts.json", readouts)
    write(folder / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "plan", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--task", help="Task identifier from the staged GPU plan")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_bytes())
    cfg = json.loads(args.config.read_bytes())
    task = plan["tasks"][args.task or f"{args.policy}-{args.seed}"]
    weights = task.get("repair_weights", cfg["repair_weights"])
    updates = cfg["founder_updates"] + cfg["child_generations"]*cfg["parents"]*cfg["children_per_parent"]*cfg["child_updates"]
    if (hashlib.sha256(args.plan.read_bytes()).hexdigest() != args.plan_sha256 or
            hashlib.sha256(args.config.read_bytes()).hexdigest() != plan["config_sha256"] or
            str(args.output) != task["output"] or task["updates"] != updates * len(weights) or
            not weights or len(weights) != len(set(weights)) or not set(weights).issubset(cfg["repair_weights"]) or
            args.seed != task["seed"] or args.policy != task["policy"] or
            args.seed not in cfg["seeds"] or args.policy not in cfg["search_versions"] or
            cfg["contact_weight"] != 1 or cfg["update_noise"] or cfg["atom_preservation"] or
            cfg["substructure_preservation"] or cfg["connectivity_operator"] != "none" or
            not os.getenv("LSB_JOBID","").isdigit()):
        raise ValueError("Incompatible native job, configuration, or search budget")
    from stage3_math import make_initial_graph
    from protenix_native import make_backend
    from repair_signal import RepairOracle
    from latentmol.design.protenix_vina_parents import VinaParentPanel

    args.output.mkdir(parents=True, exist_ok=False)
    founder = make_initial_graph(args.seed, "sparse", n_atoms=cfg["occupied_atoms"])
    graph = tuple(project(np.asarray(x),cfg["probability_floor"]) for x in (founder.X, founder.E_upper))
    smi = hard_smiles(graph)
    if not smi:
        raise ValueError("Sparse initialization failed chemistry validation")
    write(args.output/"founder.json", dict(seed=args.seed, smiles=smi, start_sha256=digest_graph(graph),
                                          atoms=graph[0].argmax(1).tolist(), bonds=graph[1].argmax(1).tolist()))
    backend = make_backend(args.seed,args.output)
    def ready():
        if path := os.getenv("DBMOL_LAUNCH_READY"):
            write(Path(path), dict(status="models_and_inputs_ready", job_id=os.environ["LSB_JOBID"],
                                   nonce=os.environ["DBMOL_LAUNCH_NONCE"]))
    try:
        frontier = ParentFrontier(backend, VinaParentPanel(cfg["vina_parent_panel"]))
        with RepairOracle(python=os.environ["DIRECT_REPAIR_PYTHON"]) as oracle:
            arms = {}
            for index, weight in enumerate(weights):
                arms[str(weight)] = run_arm(backend, oracle, frontier, graph, cfg, weight, args.seed, args.policy,
                                            args.output / f"repair-{weight:g}", ready if index == 0 else None)
    finally:
        audit = backend.close()
    write(args.output/"result.json", dict(status="completed", score="predicted pocket contact (lower is better)",
          seed=args.seed, policy=args.policy, optimization_space=cfg["optimization_space"], settings=cfg,
          optimized_repair_weights=weights, founder_sha256=digest_graph(graph),
          native_identity=backend.identity, native_audit=audit, arms=arms, docking=frontier.docking.accounting()))


if __name__ == "__main__":
    main()
