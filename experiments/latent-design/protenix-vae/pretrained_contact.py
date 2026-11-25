"""Fixed-size Graph LDA ZINC VAE latent search against the frozen BRD4 contact objective."""

import argparse
import json
import math
import os
from pathlib import Path

import torch

from latentmol.chemistry import encode_smiles, inspect_graph
from latentmol.design.latent_protenix import _model_identity, hard_from_graph
from latentmol.design.temperature_candidates import select_valid_candidates
from latentmol.guidance.mosaic_protenix import MosaicProtenixScorer
from latentmol.multimodel.external import file_sha256
from latentmol.reporting.capture import capture_shared_graph
from latentmol.reporting.optimization import render_report


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False, indent=2)


def record(stream, value):
    stream.write(json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n")
    stream.flush()


def trajectory(model, scorer, z0, state, spec, cfg, folder, *, random_lengths=None, seed=0):
    """Random control follows isotropic directions with the realized SGD lengths, one for each update."""
    folder.mkdir(parents=True, exist_ok=False)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    point = z0.detach().clone()
    count = spec.max_nodes
    slots = list(range(count))
    lengths, rows, losses = [], [], []
    updates, rate, temperature = cfg["updates"], cfg["learning_rate"], cfg["temperature"]
    if random_lengths is not None and len(random_lengths) != updates:
        raise ValueError("Expected one matched random displacement per update")
    with (folder / "metrics.jsonl").open("x") as metrics, (folder / "candidates.jsonl").open("x") as candidates:
        for step in range(updates + 1):
            point = point.detach().requires_grad_(True)
            graph = model.decode(point, state, temperature=temperature)
            value, aux = scorer(graph)
            if value.numel() != 1 or not bool(value.isfinite()) or not value.requires_grad:
                raise ValueError("Expected one finite attached native contact objective")
            gradient, = torch.autograd.grad(value, point)
            if not bool(gradient.isfinite().all()):
                raise FloatingPointError("Nonfinite derivative of contact with respect to latent")
            hard = hard_from_graph(graph)
            labels = {field: getattr(hard, field)[0].detach().cpu() for field in ("atoms", "charges", "bonds")}
            result = inspect_graph(labels["atoms"], labels["charges"], labels["bonds"], spec)
            observation = dict(step=step, smiles=result.smiles, valid=result.smiles is not None)
            rows.append(observation)
            record(candidates, dict(**observation, kind="argmax", draw=0, failure=result.failure_reason,
                graph_spec=spec.to_dict(), labels={field: tensor.tolist() for field, tensor in labels.items()}))
            losses.append(float(value.detach()))
            metric = dict(step=step, score=dict(loss=losses[-1], binding_loss=losses[-1],
                gradient_norm=float(gradient.norm())), selection=dict(active_slots=slots, count=count),
                latent=point.detach()[0].cpu().tolist(), latent_displacement=float((point.detach()-z0).norm()),
                native_aux=aux, temperature=temperature)
            if step % cfg.get("snapshot_interval", 8) == 0 or step == updates:
                capture_shared_graph(folder, step, graph, spec, active_slots=slots, temperature=temperature)
            if step < updates:
                if random_lengths is None:
                    displacement = -rate * gradient
                else:
                    direction = torch.randn(point.shape, generator=generator, dtype=point.dtype).to(point.device)
                    displacement = direction * (random_lengths[step] / direction.norm().clamp_min(1.e-12))
                next_point = point.detach() + displacement
                if not bool(next_point.isfinite().all()):
                    raise FloatingPointError("Nonfinite candidate latent")
                lengths.append(float((next_point - point.detach()).norm()))
                metric["next_step_length"] = lengths[-1]
                point = next_point
            record(metrics, metric)
    selection = select_valid_candidates(rows, edit_limit=cfg["edit_limit"])
    stats = dict(initial_decoded_smiles=rows[0]["smiles"], final_decoded_smiles=rows[-1]["smiles"],
                 relaxed_initial_loss=losses[0], relaxed_final_loss=losses[-1], relaxed_best_loss=min(losses),
                 relaxed_contact_gain=losses[0]-losses[-1], native_evaluations=len(losses),
                 valid_states=selection["valid_observations"], distinct_valid_edits=selection["distinct_valid_edits"])
    return stats, selection["candidates"], lengths


def run(args):
    from latentmol.models.pretrained_graph_vae import GraphLDAZincVAE

    if not os.environ.get("LSB_JOBID", "").isdigit():
        raise RuntimeError("Native Protenix pilot requires an allocated GPU job")
    cfg = read_json(args.config)
    if (type(cfg["updates"]) is not int or cfg["updates"] < 1 or cfg["occupied_atoms"] != 25 or
            any(not math.isfinite(cfg[key]) or cfg[key] <= 0 for key in ("learning_rate", "temperature")) or
            not 1 <= cfg["edit_limit"] <= 3 or cfg.get("snapshot_interval", 8) < 1):
        raise ValueError("Invalid pilot budget or unsupported fixed BRD4 context")
    starts = cfg["starts"]
    names = [s["name"] for s in starts]
    if not names or len(set(names)) != len(names) or any(not name or not name.isascii() or not all(
            c.isalnum() or c in "-_" for c in name) for name in names):
        raise ValueError("Start names must be unique single path components")
    args.output.mkdir(parents=True, exist_ok=False)
    model = GraphLDAZincVAE.load(args.source, args.checkpoint, device="cpu").eval().requires_grad_(False)
    initial_identity = _model_identity(model)
    starts_encoded = []
    for start in starts:
        with torch.no_grad():
            z, state = model.encode_start(start["smiles"])
        spec = model.spec_for(state)
        if spec.max_nodes != cfg["occupied_atoms"] or z.shape != (1, 90) or not bool(z.isfinite().all()):
            raise ValueError(f"Incompatible encoded start {start['name']}")
        canonical = encode_smiles(start["smiles"], spec)["smiles"]
        starts_encoded.append((start["name"], canonical, z.detach(), state, spec))
    spec = starts_encoded[0][-1]
    if (any(entry[-1] != spec for entry in starts_encoded) or
            len({entry[1] for entry in starts_encoded}) != len(starts_encoded)):
        raise ValueError("Starts must use the same graph schema and distinct molecular identities")
    command = [args.mosaic_python, "-m", "latentmol.guidance.mosaic_protenix_worker"]
    native = read_json(args.native_config)
    checkpoint_sha256 = file_sha256(args.checkpoint)
    panel, fresh, metadata = [], [], []
    with MosaicProtenixScorer(spec, torch.ones(spec.max_nodes, dtype=torch.bool), command=command, config=native,
            stderr_path=args.output / "native.stderr.log", timeout=600, startup_timeout=600) as scorer:
        for index, (name, source_smiles, z, state, spec) in enumerate(starts_encoded):
            case = args.output / name
            exact = None
            lengths = None
            methods = {}
            fresh.append(dict(start=name, arm="encoded_input", role="initial", step=0, smiles=source_smiles))
            for arm in ("gradient", "random"):
                stats, chosen, actual = trajectory(model, scorer, z, state, spec, cfg, case / "arms" / arm,
                    random_lengths=lengths, seed=cfg["seed"] + index)
                matches = stats["initial_decoded_smiles"] == source_smiles
                if arm == "gradient":
                    lengths, exact = actual, matches
                elif exact != matches or stats["initial_decoded_smiles"] != methods["gradient"]["initial_decoded_smiles"]:
                    raise RuntimeError("Paired trajectories did not reproduce the same starting graph")
                methods[arm] = stats
                fresh.extend(dict(row, start=name, arm=arm) for row in chosen)
            panel.append(dict(name=name, source_smiles=source_smiles, exact_reconstruction=exact, arms=methods))
            save(case / "summary.json", dict(arms=methods))
            metadata.append((case, dict(run_id=name, status="completed", config=dict(start=dict(
                kind="pretrained_posterior_mean", name=name, smiles=source_smiles), optimizer=dict(
                steps=cfg["updates"], learning_rate=cfg["learning_rate"])), gradient_scope="Frozen VAE and BRD4 contact",
                checkpoint_sha256=checkpoint_sha256, native_identity=scorer.identity)))
            print(json.dumps(panel[-1]), flush=True)
    if _model_identity(model) != initial_identity or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("Frozen pretrained weights or gradients changed")
    for case, item in metadata:
        save(case / "run.json", item)
    save(args.output / "relaxed-summary.json", dict(config=cfg, starts=panel, native_identity=scorer.identity,
         checkpoint_sha256=checkpoint_sha256, decoder_unchanged=True, native_unchanged=True,
         objective="negative mean per-ligand-atom maximum pocket contact probability below 8 angstroms",
         job_id=os.environ["LSB_JOBID"]))
    save(args.output / "fresh-candidates.json", dict(candidates=fresh))


def combined_summary(panel, fresh):
    """Separate contact on the relaxed graph, the hard decoded start, and the input molecule."""
    cases = []
    rows = fresh["candidates"]
    for case in panel["starts"]:
        name = case["name"]
        group = [r for r in rows if r["start"] == name]
        original, = (r for r in group if r["arm"] == "encoded_input" and r["role"] == "initial")
        methods = {}
        for arm, relaxed in case["arms"].items():
            members = [r for r in group if r["arm"] == arm]
            edits = [r for r in members if r["role"] != "initial"]
            novel = [r for r in edits if r["loss"] is not None and r["smiles"] != original["smiles"]]
            deltas = [r["loss"] - original["loss"] for r in novel] if original["loss"] is not None else []
            decoded_deltas = [r["loss_delta_from_valid_start"] for r in edits if r["loss_delta_from_valid_start"] is not None]
            methods[arm] = dict(**relaxed, attempted_fresh_edits=len(edits),
                scored_fresh_edits=sum(r["loss"] is not None for r in edits),
                best_fresh_loss_delta_vs_decoded_start=min(decoded_deltas, default=None),
                best_fresh_loss_delta_vs_encoded_input=min(deltas, default=None),
                distinct_fresh_improvements_vs_encoded_input=sum(d < 0 for d in deltas))
        cases.append(dict(name=name, exact_reconstruction=case["exact_reconstruction"],
            original_status=original["status"], original_loss=original["loss"], arms=methods))
    return dict(starts=cases, exact_reconstructions=sum(c["exact_reconstruction"] for c in cases),
                fresh_unique_candidates=fresh["unique_candidates"],
                interpretation="Relaxed and fresh objectives have different aggregation and geometry. Negative fresh deltas are improvements within the fresh metric. Missing deltas mean no comparable scores.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarize", type=Path)
    parser.add_argument("--plan")
    parser.add_argument("--plan-sha256")
    for name in ("config", "source", "checkpoint", "native-config", "output"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--mosaic-python")
    args = parser.parse_args(argv)
    if args.summarize:
        root = args.summarize
        panel = read_json(root / "relaxed-summary.json")
        fresh = read_json(root / "fresh" / "summary.json")
        if fresh.get("status") != "completed":
            raise ValueError("Fresh native scoring did not finish")
        combined = combined_summary(panel, fresh)
        save(root / "comparison.json", combined)
        render_report([root / case["name"] for case in combined["starts"]], root / "interactive-report",
                      title="Pretrained graph VAE trajectories on BRD4")
    else:
        if not all((args.config, args.source, args.checkpoint, args.native_config, args.output, args.mosaic_python)):
            parser.error("Optimization requires config, source, checkpoint, native config, output and Mosaic interpreter")
        run(args)


if __name__ == "__main__":
    main()
