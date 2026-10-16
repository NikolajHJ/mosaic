"""Like-for-like frozen hard-graph controls for an authenticated completed study."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--summary-sha256", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Hard-control scoring requires allocated DTU HPC")
    import hashlib

    source = args.study / "summary.json"
    if hashlib.sha256(source.read_bytes()).hexdigest() != args.summary_sha256:
        raise ValueError("Native summary binding differs")
    study = json.loads(source.read_text())
    if (
        study["status"] != "passed"
        or study["config"]["checks"]["steps"] != 1000
        or len(study["cases"]) != 3
    ):
        raise ValueError("Requires the passed three-case 1000-update study")
    import jax

    jax.config.update(
        "jax_default_matmul_precision", study["config"]["matmul_precision"]
    )
    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("Allocated GPU required")
    from mosaic.small_molecule.gradient_checks import array_digest, _value
    from mosaic.small_molecule.io import (
        load_context,
        load_graph,
        save_graph,
        hard_readout,
        write_json,
    )
    from mosaic.small_molecule.losses import DBMolContactLoss
    from mosaic.small_molecule.protenix import FrozenProtenixTiny
    from mosaic.small_molecule.provenance import source_identity, environment_identity
    from mosaic.small_molecule.state import graph_from_hard

    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "job_id": os.environ["LSB_JOBID"],
        "native_summary_sha256": args.summary_sha256,
        "native_config_sha256": study["config_sha256"],
        "fixture_sha256": study["fixture_sha256"],
        "source": source_identity(Path(__file__).resolve().parents[1]),
        "environment": environment_identity(),
        "controls": [],
        "policy": "hard one-hot graphs only; same frozen model/context/key/objective; raw endpoint argmax, no repair; native score is not affinity and can score invalid graphs",
    }
    try:
        model = FrozenProtenixTiny.load(cache_dir=args.cache_dir)
        context = load_context(args.study / "context", study["context_files"])
        if model.identities != study["model"]:
            raise ValueError("Frozen checkpoint identity differs")
        before_model, before_context = array_digest(model), array_digest(context)
        if (
            before_model != study["model_digest_before"]
            or before_context != study["context_digest_before"]
        ):
            raise ValueError("Frozen model/context bytes differ")
        report.update(
            model_digest_before=before_model, context_digest_before=before_context
        )
        objective = DBMolContactLoss(
            pocket_token_indices=tuple(study["pocket_token_indices"]),
            cutoff_angstrom=study["config"]["contact_cutoff_angstrom"],
            log_epsilon=study["config"]["log_epsilon"],
        )
        loss = model.build_loss(context=context, loss=objective)
        graphs = [
            (
                "benzene",
                None,
                load_graph(
                    args.study / "reference_graph.npz",
                    study["reference"]["graph_sha256"],
                ),
            ),
            (
                "initial hard chain",
                None,
                load_graph(args.study / "hard_graph.npz", study["hard_graph_sha256"]),
            ),
        ]
        for index, case in enumerate(study["cases"]):
            if case["status"] != "passed" or len(case["states"]) != 1001:
                raise ValueError("Incomplete native case")
            readout = case["states"][-1]["readout"]
            # These categorical labels are authenticated by the native summary;
            # constructing their one-hot graph is detached evaluation only.
            graph = graph_from_hard(
                readout["element_indices"], readout["bond_upper_indices"]
            )
            graphs.append((f"hard endpoint {index}", case["mixing"], graph))
        scores = {}
        for index, (name, mixing, graph) in enumerate(graphs):
            identity = array_digest(graph)
            reused = scores.get(identity)
            if reused is None:
                value = _value(
                    loss,
                    graph,
                    jax.random.key(study["config"]["seed"]),
                    args.output / f"control-{index}.npz",
                    compact=True,
                )
                scores[identity] = (value, name)
            else:
                value = reused[0]
            report["controls"].append(
                {
                    "name": name,
                    "mixing": mixing,
                    "graph_digest": identity,
                    "graph_sha256": save_graph(
                        args.output / f"graph-{index}.npz", graph
                    ),
                    "primal_loss": value,
                    "readout": hard_readout(graph),
                    "reused_identical_graph_score": reused[1] if reused else None,
                }
            )
            write_json(args.output / "summary.json", report)
        if (
            abs(
                report["controls"][0]["primal_loss"] - study["reference"]["primal_loss"]
            )
            > 2e-5
        ):
            raise AssertionError(
                "Hard reference replay exceeds existing primal parity tolerance"
            )
        report.update(
            model_digest_after=array_digest(model),
            context_digest_after=array_digest(context),
        )
        if (
            report["model_digest_after"] != before_model
            or report["context_digest_after"] != before_context
        ):
            raise AssertionError("Frozen model/context changed")
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(args.output / "summary.json", report)
    print(json.dumps({"status": report["status"], "controls": report["controls"]}))


if __name__ == "__main__":
    main()
