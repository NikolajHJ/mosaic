"""Paired direct-logit experiments on a prepared Boltz2 pocket."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
from time import perf_counter

import torch

from latentmol.chemistry import inspect_graph
from latentmol.experiments import file_sha256
from latentmol.guidance.boltz_search import BoltzSearchScorer, hard_graph_record, hard_pose_key
from latentmol.representation import GraphSpec, HardGraphBatch


def task_budget(cfg, task):
    if cfg.get("study") != "boltz_direct_graph" or cfg.get("optimization_space") != "relaxed_graph_logits":
        raise ValueError("Select the direct graph-logit optimizer")
    if type(cfg["tasks"][task]["seed"]) is not int:
        raise ValueError("A direct-graph task needs an integer seed")
    n = len(cfg["variants"])
    if n < 1 or any(type(cfg[k]) is not int or cfg[k] < 1 for k in ("updates", "fresh_budget")):
        raise ValueError("Each variant needs positive update and fresh-score budgets")
    if not (0 < cfg["lr"] < math.inf and 0 < cfg["step_cap"] < math.inf):
        raise ValueError("Direct-logit step size and cap must be positive and finite")
    weight_keys = {"motif_weight", "triangle_weight"}
    if "pharmaconet_map" in cfg:
        asset = cfg["pharmaconet_map"]
        if (set(asset) != {"path", "sha256"} or not asset["path"] or len(asset["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in asset["sha256"])):
            raise ValueError("Declare the PharmacoNet map path and SHA256")
        weight_keys.add("pharmaconet_weight")
    for name, weights in cfg["variants"].items():
        if (not name.isidentifier() or set(weights) != weight_keys
                or not all(0 <= w < math.inf for w in weights.values())):
            raise ValueError(f"Invalid direct-graph variant: {name}")
    return n * cfg["updates"], n * cfg["fresh_budget"]


def make_founder(spec, active, coordinates, seed):
    """Construct a valid sparse graph with only pose coordinates and a seed.

    Randomized Kruskal selects a spanning tree on spatial neighbors within
    1.95 Å. Heteroatoms respect node degree and occupy nonadjacent vertices.
    Nothing here reads the reference ligand's bonds, elements or charges.
    """
    active = active.detach().cpu()
    slots = active.nonzero().flatten().tolist()
    count = len(slots)
    xyz = coordinates.detach().cpu().double()
    if (active.shape != (spec.max_nodes,) or active.dtype != torch.bool
            or xyz.shape != (count, 3) or not xyz.isfinite().all() or count < 7):
        raise ValueError("Founder requires finite coordinates for the occupied slots")
    rng = random.Random(seed)
    distance = torch.cdist(xyz, xyz)
    edges = sorted((float(distance[i, j]) * rng.uniform(0.8, 1.2), i, j)
                   for i in range(count) for j in range(i) if 0 < distance[i, j] <= 1.95)
    groups = list(range(count))
    degrees = [0] * count
    tree = []
    for _, i, j in edges:
        if groups[i] == groups[j] or max(degrees[i], degrees[j]) >= 4:
            continue
        old, new = groups[i], groups[j]
        groups = [new if g == old else g for g in groups]
        degrees[i] += 1
        degrees[j] += 1
        tree.append((i, j))
        if len(tree) == count - 1:
            break
    if len(tree) != count - 1:
        raise ValueError("Pose slots cannot form a connected degree-limited tree at the declared cutoff")
    neighbors = [set() for _ in slots]
    for i, j in tree:
        neighbors[i].add(j)
        neighbors[j].add(i)
    for _ in range(64):
        blocked = set()
        chosen = []
        for max_degree in (2, 2, 2, 3, 3, 3):
            eligible = [i for i in range(count) if i not in blocked and degrees[i] <= max_degree]
            if not eligible:
                break
            i = rng.choice(eligible)
            chosen.append(i)
            blocked.update(neighbors[i] | {i})
        if len(chosen) == 6:
            break
    else:
        raise ValueError("Could not place three nonadjacent oxygens and three nonadjacent nitrogens on the tree")
    oxygen, nitrogen = set(chosen[:3]), set(chosen[3:])
    atoms = torch.zeros(1, spec.max_nodes, dtype=torch.long)
    charges = torch.full_like(atoms, spec.formal_charges.index(0))
    bonds = torch.zeros(1, spec.max_nodes, spec.max_nodes, dtype=torch.long)
    for i, slot in enumerate(slots):
        element = 8 if i in oxygen else 7 if i in nitrogen else 6
        atoms[0, slot] = spec.atomic_numbers.index(element)
    for i, j in tree:
        bonds[0, slots[i], slots[j]] = bonds[0, slots[j], slots[i]] = spec.bond_names.index("SINGLE")
    return HardGraphBatch(atoms, charges, bonds)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run(args, cfg):
    from latentmol.design.boltz_direct_graph import run_child

    started = perf_counter()
    job = os.environ.get("LSB_JOBID", "")
    if not job.isdigit():
        raise RuntimeError("The Boltz direct-logit pilot requires an allocated GPU")
    plan = json.loads(args.plan.read_text())
    budget = task_budget(cfg, args.arm)
    if (file_sha256(args.plan) != args.plan_sha256 or file_sha256(args.config) != plan["config_sha256"]
            or str(args.output) != plan["arms"][args.arm]
            or budget != (plan["planned_updates"][args.arm], plan["planned_fresh_attempts"][args.arm])
            or plan["optimization_space"] != cfg["optimization_space"]):
        raise ValueError("Direct-logit inputs or budgets differ from the staged plan")
    prepared = cfg["prepared_config"]
    if file_sha256(Path(prepared["path"])) != prepared["sha256"]:
        raise ValueError("Prepared Boltz pocket configuration changed")
    native = json.loads(Path(prepared["path"]).read_text())
    assets = json.loads(args.boltz_assets.read_text())
    if (native["checkpoint"] != assets["checkpoint"] or native["checkpoint_sha256"] != assets["checkpoint_sha256"]
            or native["dbmol"]["confidence_checkpoint"] != assets["confidence_checkpoint"]):
        raise ValueError("Boltz weights differ from the staged affinity and confidence checkpoints")
    native["execution"] = cfg["scorer"]["execution"]
    if native["execution"]["tf32"] or native["execution"]["autocast"] != "none":
        raise ValueError("Use FP32 for the paired optimization and scoring")
    args.output.mkdir(exist_ok=False)
    seed = cfg["tasks"][args.arm]["seed"]
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    spec = GraphSpec.from_dict(cfg["graph_spec"])
    scorer = BoltzSearchScorer.load(native, spec, device="cuda", source_sha256=os.environ["LATENTMOL_SOURCE_SHA256"])
    try:
        context = scorer.prepared
        pocket = context.metadata["target"]["pocket"]
        if (context.metadata["target_id"] != cfg["target_id"] or pocket["pdb_id"] != cfg["scorer"]["pocket_pdb_id"]
                or pocket["protein_chain"] != cfg["chain"] or int(context.active.sum()) != cfg["occupied_atoms"]):
            raise ValueError("Prepared pocket and requested direct-graph target differ")
        coords = context.coordinates[0, 0, context.slot_to_atom[context.active]]
        pharmaconet = None
        map_metadata, map_result = {}, {}
        if "pharmaconet_map" in cfg:
            from latentmol.guidance.pharmaconet import PocketFeatureLoss, registered_reference
            asset = cfg["pharmaconet_map"]
            raw = Path(asset["path"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != asset["sha256"]:
                raise ValueError("PharmacoNet map differs from the paired preset")
            map_data = json.loads(raw)
            names, crystal_xyz, error = registered_reference(map_data, context.metadata,
                context.active.nonzero().flatten().tolist(), coords)
            pharmaconet = PocketFeatureLoss(map_data["hotspots"], crystal_xyz)
            with (args.output / "pharmaconet-map.json").open("xb") as stream:
                stream.write(raw)
            map_metadata = dict(reference_atom_names=names, pharmaconet_reference_coordinates=crystal_xyz.tolist(),
                                pharmaconet_registration_max_error_angstrom=error)
            map_result = dict(pharmaconet_map=asset, pharmaconet_map_sha256=asset["sha256"],
                              pharmaconet_registration_max_error_angstrom=error)
        founder = make_founder(spec, context.active, coords, seed)
        inspected = inspect_graph(founder.atoms[0], founder.charges[0], founder.bonds[0], spec)
        if not inspected.smiles:
            raise ValueError(f"Invalid direct-graph founder: {inspected.failure_reason}")
        graph = hard_graph_record(founder)
        write_json(args.output / "founder.json", dict(seed=seed, smiles=inspected.smiles, hard_graph=graph,
                   pose_key=hard_pose_key(graph), active_slots=context.active.nonzero().flatten().tolist(),
                   initialization="Random spatial spanning tree with nonadjacent neutral N/O on a carbon skeleton",
                   coordinate_source="Fixed reference coordinates only", max_initial_bond_distance_angstrom=1.95,
                   **map_metadata))
        results = {}
        for name, weights in cfg["variants"].items():
            torch.manual_seed(seed)
            results[name] = run_child(scorer, spec, founder, args.output / name, seed=seed, updates=cfg["updates"],
                                      lr=cfg["lr"], step_cap=cfg["step_cap"], fresh_budget=cfg["fresh_budget"],
                                      excluded_smiles=[inspected.smiles], pharmaconet=pharmaconet, **weights,
                                      ready=None if results else os.environ.get("DBMOL_LAUNCH_READY"))
            if len({row["start_sha256"] for row in results.values()}) != 1:
                raise ValueError("Paired variants used different initial logits or occupancy")
    finally:
        shutdown = scorer.close()
    result = dict(status="completed", job_id=job, seed=seed, optimization_space=cfg["optimization_space"],
                  founder_smiles=inspected.smiles, variants=results, planned_updates=budget[0],
                  planned_fresh_attempts=budget[1], prepared_config=prepared, scorer=scorer.identity,
                  frozen=shutdown, device=torch.cuda.get_device_name(), elapsed_seconds=perf_counter() - started,
                  **map_result)
    write_json(args.output / "result.json", result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--plan-only", action="store_true")
    for key in ("output", "boltz-assets", "plan"):
        parser.add_argument("--" + key, type=Path)
    parser.add_argument("--plan-sha256")
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text())
    budget = task_budget(cfg, args.arm)
    if args.plan_only:
        print(json.dumps(dict(seed=cfg["tasks"][args.arm]["seed"], variants=cfg["variants"],
                              updates=budget[0], fresh_budget=budget[1], optimization_space=cfg["optimization_space"])))
        return
    if not all((args.output, args.boltz_assets, args.plan, args.plan_sha256)):
        parser.error("Supply the allocated output, Boltz assets and staged plan with checksum")
    run(args, cfg)


if __name__ == "__main__":
    main()
