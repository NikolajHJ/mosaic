"""Allocated fixed-point repair timing only; never certifies native readiness.

Use the exact same fixed chain/uniform point and internal-fluorine invalid assay
at six and 25 slots. No optimizer, Boltz inference or training executes. Record
warmup separately and retain every measured sample rather than selecting scales.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--science", type=Path, required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Numerical timing requires allocated DTU HPC")
    import subprocess
    import sys
    subprocess.run([sys.executable, str(args.science / "scripts/verify_stage0_release.py"), str(args.science), args.manifest], check=True)
    import equinox as eqx
    import jax
    import numpy as np
    from mosaic.small_molecule.chemistry import GraphLogits, mosaic_spec, FailureGuidanceConfig, prepare_failure_guidance, failure_aware_repair_loss
    from mosaic.small_molecule.chemistry.initialization import initialize_graph
    from mosaic.small_molecule.chemistry._reference.probability_repair import ProbabilityRepairConfig
    from mosaic.small_molecule.chemistry._reference.repair_progress import ProgressSearchConfig
    from mosaic.small_molecule.state import graph_from_hard, RelaxedGraph

    if jax.config.jax_enable_x64 or any(d.platform != "cpu" for d in jax.devices()):
        raise RuntimeError("Require original FP32 chemistry on allocated CPU")
    config_path = args.science / "examples/configs/brd4_25_boltz2_chemistry_10000.json"
    config = json.loads(config_path.read_text())
    args.output.mkdir(exist_ok=False)
    result = {"status": "running", "job_id": os.environ["LSB_JOBID"], "science_manifest_sha256": args.manifest,
              "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(), "samples_per_case": 3,
              "scope": "Fixed-point repair preparation plus one chemistry value/gradient only; no native score, line search or optimization",
              "native_readiness": "not_evaluated", "full_chemistry_run": "held_for_runtime_review", "cases": []}
    search = ProbabilityRepairConfig(**config["search"])
    progress = ProgressSearchConfig(**config["progress_search"])
    settings = FailureGuidanceConfig(**config["loss"])
    for n in (6, 25):
        hard = graph_from_hard([0] * n, [int(j == i + 1) for i in range(n) for j in range(i + 1, n)])
        initial = GraphLogits.from_graph(initialize_graph(hard, config["initialization"]))
        graph = initial.graph()
        x = graph.X.at[1, 0].set(0.5 / 6).at[1, 4].set(0.5 + 0.5 / 6)
        invalid = GraphLogits.from_graph(RelaxedGraph(x, graph.E_upper))
        spec = mosaic_spec(n)
        for label, state in (("initial_chain_uniform", initial), ("internal_fluorine_assay", invalid)):
            case = {"n_atoms": n, "unordered_pairs": n * (n - 1) // 2, "point": label, "timings": []}
            snapshot = args.output / f"{n}-{label}.npz"
            np.savez_compressed(snapshot, X=np.asarray(state.graph().X), E_upper=np.asarray(state.graph().E_upper), atoms=np.asarray(state.atoms), bonds=np.asarray(state.bonds))
            case["snapshot_sha256"] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
            for repeat in range(4):
                started = time.perf_counter()
                prepared = prepare_failure_guidance(state.decoded(), spec, update=0, search=search, progress_search=progress)
                after_prepare = time.perf_counter()
                value, gradient = eqx.filter_value_and_grad(lambda z: failure_aware_repair_loss(z.decoded(), prepared, spec, settings)[0])(state)
                value = float(value)
                for leaf in jax.tree.leaves(gradient):
                    leaf.block_until_ready()
                ended = time.perf_counter()
                sample = {"warmup": repeat == 0, "preparation_seconds": after_prepare - started,
                          "chemistry_value_gradient_seconds": ended - after_prepare, "total_seconds": ended - started,
                          "loss": value, "preparation_identity": prepared.identity,
                          "original_search_seconds": sum(a for a, b in prepared.search_seconds),
                          "progress_search_seconds": sum(b for a, b in prepared.search_seconds)}
                case["timings"].append(sample)
                if repeat == 0:
                    (args.output / f"{n}-{label}-preparation.json").write_text(json.dumps(prepared.record(), indent=2) + "\n")
            measured = case["timings"][1:]
            case["mean_component_seconds"] = sum(s["total_seconds"] for s in measured) / len(measured)
            case["component_10000_calls_seconds"] = 10000 * case["mean_component_seconds"]
            case["limitation"] = "Warmup excluded but recorded. One chemistry gradient per fixed point; full steps may require more evaluations and change search difficulty. Not end-to-end runtime."
            result["cases"].append(case)
            (args.output / "timing.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps({k: case[k] for k in ("n_atoms", "point", "mean_component_seconds", "component_10000_calls_seconds")}), flush=True)
    result["status"] = "recorded"
    (args.output / "timing.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
