"""Probe decoder saturation, hard-graph plateaus and local score gradients."""

import argparse
from contextlib import nullcontext
from itertools import combinations
import json
import math
import os
from pathlib import Path
from statistics import median
import time

import torch

from latentmol.chemistry import encode_smiles, inspect_graph
from latentmol.design.embedding_diagnostics import softmax_sensitivity
from latentmol.design.latent_protenix import _json, _model_identity, hard_from_graph
from latentmol.design.protenix_embedding_probe import cosine, load_points
from latentmol.evaluation.evaluate import load_model
from latentmol.graph_policy import graph_from_decoder
from latentmol.guidance.mosaic_protenix import MosaicProtenixScorer
from latentmol.multimodel.external import file_sha256, tensor_tree_sha256
from latentmol.representation import HardGraphBatch


def unit(vector):
    norm = vector.norm()
    return vector / norm if norm else None


def graph_observation(hard, spec):
    labels = tuple(value[0] for value in (hard.atoms, hard.charges, hard.bonds))
    inspected = inspect_graph(*labels, spec)
    return dict(signature=tensor_tree_sha256(list(labels)), smiles=inspected.smiles,
                failure=inspected.failure_reason, atoms=int(labels[0].count_nonzero()))


def changed(base, trial):
    if base["smiles"] is not None and trial["smiles"] is not None:
        return base["smiles"] != trial["smiles"]
    return base["signature"] != trial["signature"]


def categorical_rows(center, active):
    """Fix row selections at the center, including occupied and absent bonds."""
    occupied = center.atoms[0].argmax(-1) != 0 if active is None else active
    pairs = torch.triu(occupied[:, None] & occupied[None, :], diagonal=1)
    present = center.symmetric_bond_logits[0].argmax(-1) != 0

    def rows(decoded):
        atom = decoded.atoms[0]
        groups = dict(
            atom=atom[occupied] if active is None else atom[occupied, 1:],
            charge=decoded.charges[0, occupied],
            present_bond=decoded.symmetric_bond_logits[0, pairs & present],
            no_bond=decoded.symmetric_bond_logits[0, pairs & ~present],
        )
        if active is None:
            groups["padding_atom"] = atom[~occupied]
        return {name: value for name, value in groups.items() if value.numel()}

    return rows


def contrasts(rows, seed):
    rng = torch.Generator().manual_seed(seed)
    weights = {}
    for name, values in rows.items():
        weight = torch.randn(values.shape, generator=rng, dtype=values.dtype)
        weight -= weight.mean(-1, keepdim=True)
        weights[name] = weight / math.sqrt(weight.shape[0])
    return weights


def probe(model, center, radii, direction, temperature, kind, rows, weights,
          scorer=None, active=None):
    def observe(point, gradient=False):
        point = point.detach().requires_grad_(gradient)
        with torch.set_grad_enabled(gradient):
            decoded = model.decode(point)
            if any(not value.isfinite().all() for value in (decoded.atoms, decoded.charges, decoded.bonds)):
                raise ValueError("Nonfinite decoder logits")
            fixed_hard = None
            if kind == "native_contact":
                graph = graph_from_decoder(decoded, model.spec, temperature=temperature, fixed_active=active)
                loss, _ = scorer(graph)
                fixed_hard = hard_from_graph(graph)
            else:
                values = rows(decoded)
                loss = sum((weights[key] * (value if kind == "logits" else
                           (value / temperature).softmax(-1))).sum() for key, value in values.items())
            if loss.numel() != 1 or not loss.isfinite():
                raise ValueError("Expected a finite scalar objective")
            g = None
            if gradient:
                if not loss.requires_grad:
                    raise ValueError("Landscape objective detached")
                g, = torch.autograd.grad(loss, point)
                if not g.isfinite().all():
                    raise ValueError("Nonfinite latent gradient")
        raw = graph_observation(decoded.harden(model.spec), model.spec)
        fixed = graph_observation(fixed_hard, model.spec) if fixed_hard is not None else None
        return dict(loss=float(loss.detach()), raw=raw, fixed=fixed,
                    gradient_norm=float(g.norm()) if g is not None else None), g

    base, g = observe(center, gradient=True)
    repeat, repeat_g = observe(center, gradient=True)
    epsilon = torch.finfo(center.dtype).eps
    resolution = max(4 * abs(repeat["loss"] - base["loss"]),
                     4 * epsilon * max(abs(base["loss"]), abs(repeat["loss"]), epsilon))
    descent = unit(-g)
    trials = []
    for radius in radii:
        plus_point, minus_point = center + radius * direction, center - radius * direction
        plus, plus_g = observe(plus_point, gradient=True)
        minus, _ = observe(minus_point)
        gap = plus["loss"] - minus["loss"]
        # Compare the central secant with the AD prediction on the represented displacement.
        linear_gap = float((g * (plus_point - minus_point)).sum())
        native_step = observe(center + radius * descent)[0] if descent is not None else base
        descent_change = native_step["loss"] - base["loss"]
        best_random_change = min(plus["loss"], minus["loss"]) - base["loss"]
        trials.append(dict(
            radius=radius, neighbor_cosine=cosine(g, plus_g), random_plus=plus, random_minus=minus,
            descent=native_step, descent_loss_change=descent_change, best_random_loss_change=best_random_change,
            descent_improves=descent_change < -resolution,
            descent_beats_both_random=descent_change < best_random_change - resolution,
            secant_gap=gap, predicted_gap=linear_gap,
            secant_sign_agrees=(gap * linear_gap > 0 if min(abs(gap), abs(linear_gap)) > resolution else None),
            random_hard_changed=changed(base["raw"], plus["raw"]),
            descent_hard_changed=changed(base["raw"], native_step["raw"]),
            fixed_descent_hard_changed=(changed(base["fixed"], native_step["fixed"]) if base["fixed"] else None),
        ))
    return dict(center=base, repeat_loss_difference=repeat["loss"] - base["loss"],
                repeat_gradient_cosine=cosine(g, repeat_g), noise_floor=resolution, trials=trials), g


def summarize(records):
    def med(values):
        values = [v for v in values if v is not None]
        return median(values) if values else None

    groups = {}
    for key, temperature, family in sorted({(r["objective"], r["temperature"], r["family"]) for r in records}):
        group = [r for r in records if (r["objective"], r["temperature"], r["family"]) == (key, temperature, family)]
        results = []
        for radius in [t["radius"] for t in group[0]["trials"]]:
            trials = [t for r in group for t in r["trials"] if t["radius"] == radius]
            results.append(dict(
                radius=radius, trials=len(trials),
                cosine=med(t["neighbor_cosine"] for t in trials),
                hard_changed=sum(t["random_hard_changed"] for t in trials),
                descent_hard_changed=sum(t["descent_hard_changed"] for t in trials),
                descent_improved=sum(t["descent_improves"] for t in trials),
                descent_beat_random=sum(t["descent_beats_both_random"] for t in trials),
                descent_loss_change=med(t["descent_loss_change"] for t in trials),
                resolved_secants=sum(t["secant_sign_agrees"] is not None for t in trials),
                agreeing_secants=sum(t["secant_sign_agrees"] is True for t in trials),
            ))
        gradients = [torch.tensor(r["center_gradient"]) for r in group]
        groups[f"{key}@{temperature:g}/{family}"] = dict(
            centers=len(group), center_gradient_norm=med(r["center"]["gradient_norm"] for r in group),
            repeat_gradient_cosine=med(r["repeat_gradient_cosine"] for r in group),
            between_center_gradient_cosine=(med(cosine(a, b) for a, b in combinations(gradients, 2))
                                            if key == "native_contact" else None),
            saturated_fraction_099={k: med(r["sensitivity"].get(k, {}).get("saturated_fraction_099") for r in group)
                for k in sorted({k for r in group for k in r["sensitivity"]})},
            softmax_jacobian={k: med(r["sensitivity"].get(k, {}).get("jacobian_frobenius_mean") for r in group)
                for k in sorted({k for r in group for k in r["sensitivity"]})},
            radii=results)
    return groups


def centers(model, config, native):
    if native:
        for name, step, point in load_points(config["native_centers"], model):
            yield name, "saved" if name.startswith("mapping-") else "prior", point
        return
    for seed in config["prior_seeds"]:
        yield f"prior-{seed}", "prior", torch.randn((1, model.config.latent),
                                                   generator=torch.Generator().manual_seed(seed))
    for i, smiles in enumerate(config["encoded_smiles"]):
        data = encode_smiles(smiles, model.spec)
        labels = HardGraphBatch(*(torch.as_tensor(data[k]).long()[None] for k in
                                  ("atom_types", "charges", "bond_types")))
        with torch.no_grad():
            point, _ = model.encode(labels.to_relaxed(model.spec))
        yield f"encoded-{i}", "encoded", point


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mosaic-config", type=Path)
    parser.add_argument("--mosaic-python")
    parser.add_argument("--plan")
    parser.add_argument("--plan-sha256")
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    config = json.loads(args.config.read_text())
    started = time.monotonic()
    native = args.mosaic_config is not None
    if native and (not args.mosaic_python or not os.environ.get("LSB_JOBID", "").isdigit()):
        raise ValueError("Native scoring requires Mosaic Python and an allocated GPU job")
    if file_sha256(args.checkpoint) != config["checkpoint_sha256"]:
        raise ValueError("VAE checkpoint hash differs")
    model, _ = load_model(args.checkpoint, "cpu")
    model.requires_grad_(False)
    before = _model_identity(model)
    args.output.mkdir(parents=True, exist_ok=False)
    active = None
    context = nullcontext(None)
    if native:
        active = torch.zeros(model.spec.max_nodes, dtype=torch.bool)
        active[config["active_slots"]] = True
        context = MosaicProtenixScorer(model.spec, active,
            command=[args.mosaic_python, "-m", "latentmol.guidance.mosaic_protenix_worker"],
            config=json.loads(args.mosaic_config.read_text()), stderr_path=args.output / "native.stderr.log",
            timeout=360, startup_timeout=900)
    records = []
    with context as scorer:
        if proof := os.environ.get("DBMOL_LAUNCH_READY"):
            _json(Path(proof), dict(status="models_and_inputs_ready", job_id=os.environ["LSB_JOBID"],
                                   nonce=os.environ["DBMOL_LAUNCH_NONCE"]))
        for i, (name, family, point) in enumerate(centers(model, config, native)):
            if point.shape != (1, model.config.latent) or not point.isfinite().all():
                raise ValueError("Incompatible/nonfinite latent center")
            with torch.no_grad():
                decoded = model.decode(point)
                rows = categorical_rows(decoded, active)
                weights = contrasts(rows(decoded), config["seed"])
            rng = torch.Generator().manual_seed(config["seed"] + i)
            direction = unit(torch.randn(point.shape, generator=rng))
            objectives = [("native_contact" if native else "probabilities", t) for t in config["temperatures"]]
            if not native:
                objectives.append(("logits", 1.0))
            for kind, temperature in objectives:
                record, gradient = probe(model, point, config["radii"], direction, temperature,
                                         kind, rows, weights, scorer, active)
                record.update(name=name, family=family, objective=kind, temperature=temperature,
                              center_gradient=gradient.flatten().tolist(),
                              sensitivity={k: softmax_sensitivity(v, temperature)
                                           for k, v in rows(decoded).items()})
                records.append(record)
                _json(args.output / f"{name}-{kind}-{temperature:g}.json", record)
    if _model_identity(model) != before or any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise RuntimeError("Frozen VAE changed")
    _json(args.output / "summary.json", dict(status="completed", native=native, seed=config["seed"],
          seconds=time.monotonic() - started, checkpoint_sha256=config["checkpoint_sha256"],
          native_identity=scorer.identity if scorer is not None else None, groups=summarize(records)))


if __name__ == "__main__":
    main()
