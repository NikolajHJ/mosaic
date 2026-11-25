"""Sweep fixed-context Protenix descent and matched random latent directions."""

import argparse
import json
import math
import os
from pathlib import Path
import time

import torch

from latentmol.datasets.combined import CombinedCorpus
from latentmol.datasets.data import collate_graphs
from latentmol.design.embedding_diagnostics import softmax_sensitivity
from latentmol.design.latent_landscape import categorical_rows, changed, graph_observation, unit
from latentmol.design.latent_protenix import _json, _model_identity, hard_from_graph
from latentmol.design.protenix_embedding_probe import load_points
from latentmol.evaluation.evaluate import load_model
from latentmol.graph_policy import graph_from_decoder
from latentmol.guidance.mosaic_protenix import MosaicProtenixScorer
from latentmol.multimodel.external import file_sha256


def centers(model, corpus, config, atom_count):
    """Select a seeded validation panel with the native atom count."""
    for source_index, source in enumerate(("moses", "chembl")):
        data = corpus.split("val", source=source)
        rng = torch.Generator().manual_seed(config["seed"] + source_index)
        selected = 0
        for row in torch.randperm(len(data), generator=rng).tolist():
            hard = collate_graphs([data[row]])
            if int(hard.atoms.count_nonzero()) != atom_count:
                continue
            with torch.no_grad():
                point, _ = model.encode(hard.to_relaxed(model.spec))
            yield f"{source}-{row}", source, point, str(data.smiles[row])
            selected += 1
            if selected == config["heldout_per_source"]:
                break
        if selected != config["heldout_per_source"]:
            raise ValueError(f"Insufficient {source} validation centers with {atom_count} atoms")
    for name, _, point in load_points(config["native_centers"], model):
        family = "saved" if name.startswith("mapping-") else "prior"
        yield name, family, point, None


def row_movement(reference, trial, temperature):
    top_two = trial.topk(2, dim=-1).values
    shift = trial - reference
    shift = shift - shift.mean(-1, keepdim=True)
    probability_shift = (trial / temperature).softmax(-1) - (reference / temperature).softmax(-1)
    return dict(
        logit_rms=float(shift.square().mean().sqrt()),
        minimum_top_two_margin=float((top_two[..., 0] - top_two[..., 1]).min()),
        maximum_probability_change=float(probability_shift.abs().max()),
        changed_labels=int((trial.argmax(-1) != reference.argmax(-1)).sum()),
    )


def first_edits(baseline, samples):
    """Return first changes on the sampled grid, with separate validity."""
    edit, valid_edit = None, None
    for sample in samples:
        graph = sample["graph"]
        if changed(baseline, graph):
            if edit is None:
                edit = sample["radius"]
            if graph["smiles"] is not None:
                valid_edit = sample["radius"]
                break
    return dict(first_edit=edit, first_valid_edit=valid_edit)


def slot_edits(base, trial):
    """Identify atom, charge and undirected bond label transitions by slot."""
    edits = {}
    for name in ("atoms", "charges", "bonds"):
        before, after = getattr(base, name)[0], getattr(trial, name)[0]
        mask = before != after
        if name == "bonds":
            mask = torch.triu(mask, diagonal=1)
        slots = mask.nonzero(as_tuple=False).tolist()
        edits[name] = [dict(slots=slot, previous=int(before[tuple(slot)]),
                            current=int(after[tuple(slot)])) for slot in slots]
    return edits


def sweep(model, center, scorer, active, temperature, radii, random_direction):
    if center.shape != (1, model.config.latent) or not center.isfinite().all():
        raise ValueError("Incompatible or nonfinite latent center")

    def evaluate(point):
        decoded = model.decode(point)
        if not all(value.isfinite().all() for value in (decoded.atoms, decoded.charges, decoded.bonds)):
            raise FloatingPointError("Decoder produced nonfinite logits")
        graph = graph_from_decoder(decoded, model.spec, temperature=temperature, fixed_active=active)
        loss, _ = scorer(graph)
        if loss.numel() != 1 or not loss.isfinite():
            raise FloatingPointError("Native contact objective must be finite and scalar")
        return decoded, graph, loss

    def labels(decoded, graph):
        return dict(raw=decoded.harden(model.spec), fixed=hard_from_graph(graph))

    def observations(graphs):
        return {name: graph_observation(hard, model.spec) for name, hard in graphs.items()}

    point = center.detach().requires_grad_(True)
    decoded, graph, loss = evaluate(point)
    if not loss.requires_grad:
        raise RuntimeError("Required native contact gradient path detached")
    gradient, = torch.autograd.grad(loss, point)
    if not gradient.isfinite().all():
        raise FloatingPointError("Native contact latent gradient is nonfinite")
    directions = {"descent": unit(-gradient), "random_plus": random_direction,
                  "random_minus": -random_direction}
    base_labels = labels(decoded, graph)
    base = observations(base_labels)
    rows = categorical_rows(decoded, active)
    reference = {name: value.detach() for name, value in rows(decoded).items()}
    sensitivity = {name: softmax_sensitivity(value, temperature) for name, value in reference.items()}
    trials = {}
    with torch.no_grad():
        for direction, vector in directions.items():
            if vector is None:
                trials[direction] = dict(samples=[], first=None)
                continue
            samples = []
            for radius in radii:
                moved = center + radius * vector
                trial_decoded, trial_graph, trial_loss = evaluate(moved)
                trial_labels = labels(trial_decoded, trial_graph)
                measures = {name: row_movement(reference[name], value, temperature)
                            for name, value in rows(trial_decoded).items()}
                samples.append(dict(radius=radius, represented_l2=float((moved - center).norm()),
                                    loss=float(trial_loss), loss_delta=float(trial_loss - loss.detach()),
                                    graphs=observations(trial_labels), row_movement=measures,
                                    slot_edits={kind: slot_edits(base_labels[kind], labels)
                                                for kind, labels in trial_labels.items()}))
            trials[direction] = dict(samples=samples, first={
                kind: first_edits(base[kind], [
                    dict(radius=sample["radius"], graph=sample["graphs"][kind]) for sample in samples
                ]) for kind in base})
    return dict(temperature=temperature, baseline=base, loss=float(loss.detach()),
                gradient_norm=float(gradient.double().norm()), gradient=gradient.flatten().tolist(),
                sensitivity=sensitivity, trials=trials)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mosaic-config", type=Path, required=True)
    parser.add_argument("--mosaic-python", required=True)
    parser.add_argument("--plan")
    parser.add_argument("--plan-sha256")
    args = parser.parse_args(argv)
    if not os.environ.get("LSB_JOBID", "").isdigit():
        raise ValueError("Native scoring requires an allocated GPU job")
    torch.set_num_threads(4)
    config = json.loads(args.config.read_text())
    radii, temperatures = config["radii"], config["temperatures"]
    if (not radii or radii != sorted(set(radii)) or not temperatures
            or any(not math.isfinite(v) or v <= 0 for v in [*radii, *temperatures])
            or config["heldout_per_source"] < 1):
        raise ValueError("Require positive ordered radii, temperatures and held-out count")
    if file_sha256(args.checkpoint) != config["checkpoint_sha256"]:
        raise ValueError("VAE checkpoint hash differs")
    corpus = CombinedCorpus(config["data"], expected_sha256=config["dataset_sha256"])
    model, _ = load_model(args.checkpoint, "cpu", expected_spec=corpus.spec)
    model.requires_grad_(False)
    before = _model_identity(model)
    slots = config["active_slots"]
    if slots != sorted(set(slots)) or not slots or slots[0] < 0 or slots[-1] >= model.spec.max_nodes:
        raise ValueError("Require distinct ordered active slots within the graph schema")
    active = torch.zeros(model.spec.max_nodes, dtype=torch.bool)
    active[slots] = True
    panel = list(centers(model, corpus, config, len(slots)))
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    records = []
    with MosaicProtenixScorer(model.spec, active,
            command=[args.mosaic_python, "-m", "latentmol.guidance.mosaic_protenix_worker"],
            config=json.loads(args.mosaic_config.read_text()), stderr_path=args.output / "native.stderr.log",
            timeout=360, startup_timeout=900) as scorer:
        if proof := os.environ.get("DBMOL_LAUNCH_READY"):
            _json(Path(proof), dict(status="models_and_inputs_ready", job_id=os.environ["LSB_JOBID"],
                                   nonce=os.environ["DBMOL_LAUNCH_NONCE"]))
        for index, (name, family, point, source_smiles) in enumerate(panel):
            rng = torch.Generator().manual_seed(config["seed"] + 10_000 + index)
            direction = unit(torch.randn(point.shape, generator=rng))
            record = dict(name=name, family=family, source_smiles=source_smiles,
                          center=point.flatten().tolist(), random_direction=direction.flatten().tolist(),
                          temperatures=[])
            for temperature in temperatures:
                record["temperatures"].append(sweep(model, point, scorer, active, temperature, radii, direction))
                _json(args.output / f"{name}.json", record)
            records.append(record)
    if _model_identity(model) != before or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise RuntimeError("Frozen VAE changed")
    summary = [
        dict(center=record["name"], family=record["family"], temperature=arm["temperature"],
             baseline=arm["baseline"], loss=arm["loss"], gradient_norm=arm["gradient_norm"],
             first_edits={name: direction["first"] for name, direction in arm["trials"].items()})
        for record in records for arm in record["temperatures"]
    ]
    _json(args.output / "summary.json", dict(
        status="completed", job_id=os.environ["LSB_JOBID"], seconds=time.monotonic() - started,
        checkpoint_sha256=config["checkpoint_sha256"], dataset_sha256=corpus.dataset_sha256,
        seed=config["seed"], active_slots=slots, radii=radii, native_identity=scorer.identity,
        boundary_scope="first observed edit on the sampled positive-radius grid", results=summary,
    ))


if __name__ == "__main__":
    main()
