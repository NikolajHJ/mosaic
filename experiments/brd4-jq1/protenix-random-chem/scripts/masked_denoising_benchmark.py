"""Compare three read-only proposal strategies at common saved Mosaic states."""

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
from statistics import median
import sys
import time

from score_transfer_audit import PROFILE, binding, bound, read, require, write


VERSION = "mosaic-masked-denoising-v1"
METHODS = ("independent", "structural", "masked_defog")
POINTS = ("sparse23-invalid", "sparse41-valid", "shared-repaired")
ATTEMPTS = 64
SCORE_CAP = 16


def check_plan(plan, plan_path):
    require(plan["version"] == VERSION and plan["optimizer_updates"] == 0,
            "Expected read-only benchmark")
    require(tuple(point["id"] for point in plan["points"]) == POINTS,
            "Expected the three predeclared checkpoints")
    require(plan["proposals_per_cell"] == ATTEMPTS and plan["hard_score_cap_per_cell"] == SCORE_CAP,
            "Benchmark budget changed")
    root = plan_path.resolve().parent.parent
    require(Path(plan["output"]) == root / "payload", "Output escapes owned run")
    config = plan["denoiser"]
    require(config["steps"] == 8 and config["start_time"] == 0.55 and
            Path(config["probe"]) == root / "source/defog_probe.py" and
            Path(config["log"]) == root / "payload/defog-worker.log",
            "Unexpected denoiser profile or path")
    require(all(Path(config[name]).exists() for name in ("python", "source", "checkpoint", "probe")) and
            (Path(plan["source_root"]) / "src/mosaic/small_molecule/chemistry/structural.py").is_file(),
            "Frozen worker or original structural builder is missing")
    records = [plan["context_ready"], plan["pocket"], *plan["source_files"]]
    for point in plan["points"]:
        require(type(point["step"]) is int and point["step"] >= 0 and
                type(point["expected_argmax_valid"]) is bool, "Invalid checkpoint metadata")
        records.append(point["snapshot"])
    for record in records:
        bound(record)
    return records


def score_shortlist(rows):
    """First canonical representative in a fixed seeded attempt stream, before scores."""
    unique = {}
    for row in rows:
        smiles = row.get("canonical_smiles")
        if row.get("valid_connected") and smiles:
            unique.setdefault(smiles, row)
    return len(unique), list(unique.values())[:SCORE_CAP]


def patch_masks(candidate, x, e, pairs, spec, diagnostic, components, np):
    """Open at most two atom sites, their existing bonds and nearby alternatives."""
    a, b = candidate
    a = np.asarray(a) + 1
    bonds = np.zeros((25, 25), dtype=np.int64)
    bonds[pairs] = b
    bonds += bonds.T
    defect = diagnostic(a, np.full(25, spec.formal_charges.index(0)), bonds, spec)
    ordered = np.sort(x, axis=1)
    gap = ordered[:, -1] - ordered[:, -2]
    implicated = {i for problem in defect.get("problems", []) for i in problem["slots"]}
    groups = components(a, bonds)
    bridge = None
    if len(groups) > 1:
        largest = set(max(groups, key=len))
        candidates = [k for k, (i, j) in enumerate(zip(*pairs))
                      if (i in largest) != (j in largest)]
        edge = max(candidates, key=lambda k: (1 - e[k, 0], -k))
        bridge = tuple(int(pairs[axis][edge]) for axis in range(2))
    sites = sorted(implicated, key=lambda i: (gap[i], i))[:2]
    if not sites and bridge is None:
        sites = [int(np.argmin(gap))]
    atom_mask = np.zeros(25, dtype=bool)
    atom_mask[sites] = True
    bond_mask = np.zeros_like(bonds, dtype=bool)
    for site in sites:
        bond_mask[site, :] |= bonds[site, :] > 0
        alternatives = [k for k, (i, j) in enumerate(zip(*pairs))
                        if site in (i, j) and not bonds[i, j]]
        for k in sorted(alternatives, key=lambda k: (e[k, 0], k))[:2]:
            bond_mask[pairs[0][k], pairs[1][k]] = True
    if bridge is not None:
        bond_mask[bridge] = True
    bond_mask |= bond_mask.T
    return a, bonds, atom_mask, bond_mask


def run(plan, plan_path, records):
    job = os.environ.get("LSB_JOBID", "")
    require(job.isdigit() and os.environ.get("LSB_QUEUE") == plan["queue"] and
            os.environ.get("XLA_FLAGS") == PROFILE, "Original allocated FP32 scorer profile required")
    routing = Path(plan["routing_source"]) / "scripts/cuda_ownership.py"
    module = importlib.util.spec_from_file_location("_masked_denoising_cuda", routing)
    owner = importlib.util.module_from_spec(module)
    module.loader.exec_module(owner)
    os.environ["CUDA_VISIBLE_DEVICES"] = owner.resolve_visibility(os.environ)["selected_cuda_visible_devices"]
    sys.path.insert(0, str(Path(plan["source_root"]) / "src"))
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    from rdkit import rdBase
    from mosaic.small_molecule.chemistry._reference.quality import diagnose_graph
    from mosaic.small_molecule.chemistry.preparation import components
    from mosaic.small_molecule.chemistry.state import mosaic_spec
    from mosaic.small_molecule.gradient_checks import array_digest
    from mosaic.small_molecule.io import hard_readout, load_context
    from mosaic.small_molecule.losses import NativeContactLoss
    from mosaic.small_molecule.pocket import map_protenix_pocket
    from mosaic.small_molecule.protenix import FrozenProtenixTiny
    from mosaic.small_molecule.state import RelaxedGraph, graph_from_hard, validate_graph
    from masked_denoiser import load_denoiser
    from masked_denoising_controls import independent_proposal, proposal_seed, structural_proposal

    require(not jax.config.x64_enabled and jax.default_backend() == "gpu" and len(jax.devices("gpu")) == 1,
            "Require original FP32 and one allocated GPU")
    jax.config.update("jax_default_matmul_precision", "highest")
    output = Path(plan["output"])
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    result = {"status": "running", "version": VERSION, "job_id": job, "plan": binding(plan_path),
              "optimizer_updates": 0, "proposals_per_cell": ATTEMPTS, "hard_score_cap_per_cell": SCORE_CAP,
              "score_policy": "First distinct valid canonical molecules in the seeded attempt stream; unused slots stay unused",
              "score_scope": "Frozen Protenix Tiny 8A pocket contact on hard graphs", "cells": [], "checkpoints": []}
    worker = model = context = frozen = None

    def persist():
        write(output / "result.json", result)
        require(sum(p.stat().st_size for p in output.iterdir() if p.is_file()) <= plan["output_cap_bytes"],
                "Benchmark output cap exceeded")

    def tick():
        require(time.monotonic() - started < plan["internal_seconds"], "Benchmark time cap exceeded")

    def inspect(candidate):
        graph = graph_from_hard(*candidate)
        with rdBase.BlockLogs():
            info = hard_readout(graph)
        return {"valid_connected": info["sanitized"] and not info["disconnected"],
                "canonical_smiles": info["canonical_smiles"], "fragments": info["fragment_count"],
                "error": info["conversion_error"]}

    try:
        ready = read(plan["context_ready"]["path"])
        context = load_context(Path(plan["context_ready"]["path"]).parent / "context", ready["context_files"])
        model = FrozenProtenixTiny.load(cache_dir=Path(plan["model_cache"]))
        frozen = {"model_digest": array_digest(model), "context_digest": array_digest(context)}
        require(frozen == plan["frozen_identity"], "Saved frozen scorer differs")
        result["frozen_before"] = frozen
        loss = model.build_loss(context=context, loss=NativeContactLoss(
            8., map_protenix_pocket(context, read(plan["pocket"]["path"]))))
        key = jax.random.key(17)
        spec = mosaic_spec(25)
        pairs = np.triu_indices(25, 1)
        worker = load_denoiser(plan["denoiser"])

        @eqx.filter_jit
        def forward(graph):
            value, aux = loss(graph, key=key)
            finite = jnp.isfinite(value)
            for leaf in jax.tree_util.tree_leaves(aux):
                if eqx.is_array(leaf):
                    finite = finite & jnp.all(jnp.isfinite(leaf))
            return value, finite

        # Warm the same hard-scorer shape once outside the matched cell budgets.
        result["baseline_hard_calls"] = 0
        for point_index, point in enumerate(plan["points"]):
            tick()
            with np.load(point["snapshot"]["path"], allow_pickle=False) as saved:
                require(saved["step"].shape == () and int(saved["step"]) == point["step"] and
                        np.array_equal(saved["key_data"], np.asarray(jax.random.key_data(key))),
                        "Saved step or scorer key differs")
                x, e = (saved[name].copy() for name in ("X", "E_upper"))
            require(x.shape == (25, 6) and e.shape == (300, 5) and x.dtype == e.dtype == np.float32,
                    "Saved graph shape or dtype differs")
            validate_graph(RelaxedGraph(jnp.asarray(x), jnp.asarray(e)))
            argmax = (x.argmax(1), e.argmax(1))
            raw = inspect(argmax)
            require(raw["valid_connected"] == point["expected_argmax_valid"], "Saved argmax does not match cohort")
            value, finite = forward(graph_from_hard(*argmax))
            require(bool(finite), "Nonfinite checkpoint hard forward")
            result["baseline_hard_calls"] += 1
            result["checkpoints"].append({**point, **raw, "argmax_contact_score": -float(value)})
            for method in METHODS:
                rows = []
                for attempt in range(ATTEMPTS):
                    tick()
                    seed = proposal_seed(plan["seed"], point_index, attempt)
                    now = time.monotonic()
                    source = independent_proposal(x, e, seed)
                    candidate, status, extra = source, "drawn", {}
                    if method == "structural":
                        candidate, status = structural_proposal(source, x, e)
                    elif method == "masked_defog":
                        a, b, am, bm = patch_masks(source, x, e, pairs, spec, diagnose_graph, components, np)
                        new_a, new_b = worker.propose(a, b, am, bm, seed + 2**32)
                        require(np.array_equal(new_a[~am], a[~am]) and np.array_equal(new_b[~bm], b[~bm]),
                                "Denoiser edited outside its patch")
                        candidate = (new_a - 1, new_b[pairs])
                        status = "denoised"
                        extra = {"editable_atoms": np.flatnonzero(am).tolist(),
                                 "editable_upper_bonds": np.flatnonzero(bm[pairs]).tolist()}
                    row = {"attempt": attempt, "seed": int(seed), "status": status,
                           "proposal_seconds": time.monotonic() - now, **extra}
                    if candidate is None:
                        row["valid_connected"] = False
                    else:
                        a, b = map(np.asarray, candidate)
                        require(a.shape == (25,) and b.shape == (300,) and a.dtype.kind in "iu" and b.dtype.kind in "iu"
                                and ((0 <= a) & (a < 6)).all() and ((0 <= b) & (b < 5)).all(),
                                "Proposal changed the neutral N25 categorical schema")
                        row.update(atoms=a.tolist(), bonds=b.tolist(),
                                   atom_edits=int(np.count_nonzero(a != source[0])),
                                   bond_edits=int(np.count_nonzero(b != source[1])),
                                   sulfur_atoms=int(np.count_nonzero(a == 3)), **inspect(candidate))
                        if method == "structural":
                            require(row["valid_connected"], "Structural witness did not pass hard readout")
                    rows.append(row)
                unique, shortlist = score_shortlist(rows)
                score_started = time.monotonic()
                for row in shortlist:
                    tick()
                    value, finite = forward(graph_from_hard(row["atoms"], row["bonds"]))
                    require(bool(finite) and math.isfinite(float(value)), "Nonfinite proposal hard forward")
                    row["contact_score"] = -float(value)
                scores = [row["contact_score"] for row in shortlist]
                result["cells"].append({"checkpoint": point["id"], "method": method, "attempts": len(rows),
                                        "valid": sum(row["valid_connected"] for row in rows), "unique_valid": unique,
                                        "hard_calls": len(scores), "unused_hard_calls": SCORE_CAP - len(scores),
                                        "median_contact": median(scores) if scores else None,
                                        "best_contact": max(scores, default=None),
                                        "proposal_seconds": sum(row["proposal_seconds"] for row in rows),
                                        "scoring_seconds": time.monotonic() - score_started, "proposals": rows})
                persist()
        result["status"] = "completed"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        try:
            if worker is not None:
                worker.close()
                result["denoiser_audit"] = worker.audit
                if result["status"] == "completed":
                    require(worker.audit["completion"]["proposals"] == len(POINTS) * ATTEMPTS and
                            len(result["cells"]) == len(POINTS) * len(METHODS), "Incomplete comparison")
            if frozen is not None:
                result["frozen_after"] = {"model_digest": array_digest(model), "context_digest": array_digest(context)}
                require(result["frozen_after"] == frozen, "Frozen scorer changed")
            for record in records:
                bound(record)
            result["inputs_unchanged"] = True
        except BaseException as error:
            result.update(status="failed", preservation_error=repr(error))
            raise
        finally:
            result["elapsed_seconds"] = time.monotonic() - started
            persist()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--startup-only", action="store_true")
    args = parser.parse_args()
    plan = read(args.plan)
    records = check_plan(plan, args.plan)
    if args.startup_only:
        print(json.dumps({"status": "metadata_startup_passed_no_models", "cells": len(POINTS) * len(METHODS)}))
    else:
        result = run(plan, args.plan, records)
        print(json.dumps({"status": result["status"], "cells": len(result["cells"]),
                          "optimizer_updates": result["optimizer_updates"]}))


if __name__ == "__main__":
    main()
