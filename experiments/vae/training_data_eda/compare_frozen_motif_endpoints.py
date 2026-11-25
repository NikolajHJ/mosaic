"""Score two frozen pilot endpoints on new matched Gaussian codes and categorical draws."""

import argparse
from collections import Counter
from contextlib import ExitStack
import gzip
import json
import math
from pathlib import Path
import time

from rdkit import rdBase
import torch

from experiments.vae.training_data_eda.teacher_motif_training_pilot import hard_records
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256
from latentmol.sampling import sample_hard_graph
from latentmol.train import state_digest


LABELS = ("valid", "aromatic", "amide", "ring3", "ring8plus", "ring12plus")
ARMS = ("control", "treatment")
MODES = ("argmax", "categorical")


def endpoint(folder, arm):
    report = json.loads((folder / "summary.json").read_bytes())
    model, payload = load_model(folder / "endpoint.pt", "cuda")
    if (not payload.get("diagnostic_only") or payload.get("pilot_updates") != 128
            or state_digest(model.state_dict()) != report["arms"][arm]["parameter_sha"]):
        raise ValueError(f"Endpoint and training report disagree for {arm}")
    return model.eval().requires_grad_(False), report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("control", "treatment", "output"):
        p.add_argument("--" + k, required=True, type=Path)
    p.add_argument("--latent-seed", required=True, type=int)
    p.add_argument("--categorical-seed", required=True, type=int)
    p.add_argument("--draws", default=8192, type=int)
    p.add_argument("--batch", default=16, type=int)
    p.add_argument("--saved-control", type=Path, help="Reuse this control's previous raw observations on identical seeds")
    args = p.parse_args()
    if min(args.batch, args.draws) < 1 or args.latent_seed == args.categorical_seed:
        p.error("Draws and batch size must be positive and seeds distinct")
    torch.set_num_threads(8)
    args.output.mkdir(parents=True, exist_ok=False)
    models = {}
    reports = {}
    for arm in ARMS:
        models[arm], reports[arm] = endpoint(getattr(args, arm), arm)
        summary = reports[arm]
        used_seeds = {summary["seed"] + i for i in range(5)} | {
            summary["fresh_evaluation_seed"], summary["categorical_seed"]}
        if used_seeds & {args.latent_seed, args.categorical_seed}:
            raise ValueError("New frozen evaluation reuses an earlier seed")
    keys = ("checkpoint", "reference", "train_indices_sha256", "seed", "updates", "device", "rdkit", "torch")
    if (any(reports["control"][k] != reports["treatment"][k] for k in keys)
            or models["control"].spec != models["treatment"].spec):
        raise ValueError("Endpoints must derive from the same matched training pilot")
    if args.saved_control:
        previous = json.loads((args.saved_control / "summary.json").read_bytes())
        expected = {"source": reports["control"]["checkpoint"], "latent_seed": args.latent_seed,
                    "categorical_seed": args.categorical_seed, "batch": args.batch, "device": torch.cuda.get_device_name(),
                    "torch": torch.__version__, "rdkit": rdBase.rdkitVersion, "raw_attempts_per_mode_and_arm": args.draws}
        if (previous["endpoint_files"]["control"] != file_sha256(args.control / "endpoint.pt")
                or not previous["matched_categorical_rng"] or any(previous[k] != v for k, v in expected.items())):
            raise ValueError("Saved control does not match this evaluation configuration")
    codes = torch.randn(args.draws, models["control"].config.latent,
                        generator=torch.Generator().manual_seed(args.latent_seed))
    generators = {arm: torch.Generator(device="cuda").manual_seed(args.categorical_seed) for arm in ARMS}
    paired = {mode: {key: Counter() for key in LABELS} for mode in MODES}
    unique = {(mode, arm): {"scaffolds": set(), "smiles": set()} for mode in MODES for arm in ARMS}
    raw = {(mode, arm): Counter() for mode in MODES for arm in ARMS}
    start = time.monotonic()
    with ExitStack() as stack, torch.inference_mode(), rdBase.BlockLogs():
        streams = {(mode, arm): stack.enter_context(gzip.open(
            args.output / f"{arm}-{mode}.jsonl.gz", "xt", compresslevel=3, encoding="utf-8"))
            for mode in MODES for arm in ARMS if not args.saved_control or arm == "treatment"}
        inputs = ({mode: stack.enter_context(gzip.open(args.saved_control / f"control-{mode}.jsonl.gz", "rt"))
                   for mode in MODES} if args.saved_control else {})
        for first in range(0, args.draws, args.batch):
            block = codes[first:first + args.batch].to("cuda")
            current = {}
            for arm in ARMS:
                model = models[arm]
                saved = arm == "control" and inputs
                if not saved:
                    decoded = model.decode(block)
                    graphs = {"argmax": decoded.harden(model.spec),
                              "categorical": sample_hard_graph(decoded, model.spec, generator=generators[arm])}
                for mode in MODES:
                    if saved:
                        rows = [json.loads(next(inputs[mode])) for _ in range(len(block))]
                        if any(row["draw"] != first+i for i, row in enumerate(rows)):
                            raise ValueError("Saved control draw IDs differ")
                    else:
                        rows = hard_records(graphs[mode].to("cpu"), model.spec)
                    current[mode, arm] = rows
                    outcomes = unique[mode, arm]
                    for i, row in enumerate(rows):
                        if not saved:
                            streams[mode, arm].write(json.dumps({"draw": first + i, **row}) + "\n")
                        if row["valid"]:
                            outcomes["smiles"].add(row["smiles"])
                            if row["scaffold"]:
                                outcomes["scaffolds"].add(row["scaffold"])
                        explicit = any(edge[2] == model.spec.bond_names.index("AROMATIC") for edge in row["edges"])
                        raw[mode, arm]["any_explicit_aromatic"] += explicit
                        raw[mode, arm]["invalid_with_explicit_aromatic"] += explicit and not row["valid"]
            for mode in MODES:
                for left, right in zip(current[mode, "control"], current[mode, "treatment"], strict=True):
                    for label in LABELS:
                        stats = paired[mode][label]
                        stats["control"] += left[label]
                        stats["treatment"] += right[label]
                        stats["gained"] += right[label] and not left[label]
                        stats["lost"] += left[label] and not right[label]
            n = first + len(block)
            if n % 512 == 0 or n == args.draws:
                print(n, "/", args.draws, round(time.monotonic() - start, 1), "seconds", flush=True)
        if any(next(stream, None) is not None for stream in inputs.values()):
            raise ValueError("Saved control contains unexpected extra observations")
    if not args.saved_control and not torch.equal(generators["control"].get_state(), generators["treatment"].get_state()):
        raise RuntimeError("Endpoint decoders consumed different categorical random streams")
    for labels in paired.values():
        for stats in labels.values():
            net = stats["gained"] - stats["lost"]
            delta = net / args.draws
            stderr = math.sqrt(max(0., (stats["gained"] + stats["lost"]) / args.draws - delta**2)
                               / max(1, args.draws - 1))
            stats["net"] = net
            stats["difference_pp"] = 100 * delta
            stats["paired_95pct_normal_interval_pp"] = [100 * (delta - 1.96 * stderr), 100 * (delta + 1.96 * stderr)]
    report = {
        "source": reports["control"]["checkpoint"], "raw_attempts_per_mode_and_arm": args.draws,
        "latent_seed": args.latent_seed, "categorical_seed": args.categorical_seed, "batch": args.batch,
        "device": torch.cuda.get_device_name(), "torch": torch.__version__, "rdkit": rdBase.rdkitVersion,
        "matched_categorical_rng": True if not args.saved_control else None,
        "saved_control": str(args.saved_control) if args.saved_control else None,
        "seconds": time.monotonic() - start,
        "endpoint_files": {a: file_sha256(getattr(args, a) / "endpoint.pt") for a in ARMS},
        "paired": paired,
        "diversity": {m: {a: {k: len(v) for k, v in unique[m, a].items()} | dict(raw[m, a])
                          for a in ARMS} for m in MODES}}
    with (args.output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report["paired"]["categorical"]), flush=True)


if __name__ == "__main__":
    main()
