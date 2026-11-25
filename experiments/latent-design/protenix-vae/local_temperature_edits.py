"""Measure whether a latent descent step can add one N/O at fixed heavy-atom count.

The differentiable size and N/O-count surrogate and the RDKit readout are
independent of slot ordering. Score every forward trial at T=1 for comparison,
regardless of the temperature used for its gradient. Keep every raw trial.
"""

import argparse
import json
import random
from pathlib import Path
from time import monotonic

from rdkit import Chem, rdBase
import torch

from experiments.vae.training_data_eda.teacher_motif_training_pilot import encode
from latentmol.chemistry import inspect_graph
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256
from latentmol.train import state_digest


SEED = 2026112289
CHECKPOINT = Path("outputs/runs/e48-s02-early-stop-600116/checkpoint-000600116.pt")
PANEL = Path("outputs/runs/vae-review-20261119-47682960/pubchem-val-panel.json")
LADDERS = {"short": (0.025, 0.05, 0.1), "wide": (0.1, 1.0, 2.0)}
RADII = sorted(set().union(*LADDERS.values()))


def raw_records(decoded, spec, polar):
    batch = decoded.harden(spec).to("cpu")
    records = []
    for atom, charge, bond in zip(batch.atoms, batch.charges, batch.bonds, strict=True):
        graph = inspect_graph(atom, charge, bond, spec)
        size = int(atom.ne(0).sum())
        count = int(((atom == polar[0]) | (atom == polar[1])).sum())
        smiles = graph.smiles
        failure = graph.failure_reason
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                failure, smiles = "smiles_roundtrip", None
            else:
                rd_size = mol.GetNumHeavyAtoms()
                rd_count = sum(a.GetAtomicNum() in (7, 8) for a in mol.GetAtoms())
                if (size, count) != (rd_size, rd_count):
                    raise ValueError("Canonical molecule does not preserve graph counts")
        records.append(dict(smiles=smiles, failure=failure, size=size, count=count))
    return records


def expected_counts(decoded, temperature, polar):
    # Expected N/O count sums the joint per-slot element marginal directly.
    p = (decoded.atoms / temperature).softmax(dim=-1)
    count = p[..., polar[0]].sum(-1) + p[..., polar[1]].sum(-1)
    size = (1 - p[..., 0]).sum(-1)
    return count, size


def objective(decoded, t, goal, polar, *, reference_counts=None):
    count, size = expected_counts(decoded, t, polar)
    if reference_counts is not None:
        # Matched forward readout with a deliberately warmer backward surrogate.
        # Trial selection and hard counts still use the ordinary forward definitions.
        count = reference_counts[0] + (count - count.detach())
        size = reference_counts[1] + (size - size.detach())
    return (size - goal["size"]).square() + (count - goal["count"]).square()


def probe(model, center, polar, random_direction, warm_backward):
    point = center[None].detach().requires_grad_(True)
    decoded = model.decode(point)
    parent, = raw_records(decoded, model.spec, polar)
    goal = dict(size=parent["size"], count=parent["count"] + 1)
    if goal["count"] > goal["size"]:
        raise ValueError("Changing one element to N/O is impossible at this parent")
    directions = {} if warm_backward else {"random": random_direction}
    norms = {}
    temperatures = (1.5, 2.0) if warm_backward else (1.0, 1.5, 2.0)
    ref = None
    if warm_backward:
        ref = tuple(v.detach() for v in expected_counts(decoded, 1.0, polar))
    for index, temperature in enumerate(temperatures):
        loss = objective(decoded, temperature, goal, polar, reference_counts=ref)
        grad, = torch.autograd.grad(loss.sum(), point, retain_graph=index < len(temperatures)-1)
        norm = grad.norm().item()
        if not grad.isfinite().all():
            raise ValueError("Nonfinite decoder gradient")
        key = f"T{temperature:g}" + ("_backward_only" if warm_backward else "")
        norms[key] = norm
        directions[key] = -grad[0] / norm if norm else torch.zeros_like(center)
    base = objective(decoded, 1.0, goal, polar).detach().item()
    tol = 8 * torch.finfo(point.dtype).eps * max(1., abs(base))
    arms = {}
    with torch.no_grad():
        for name, direction in directions.items():
            samples = torch.stack([center + r * direction for r in RADII])
            # Splitting keeps local VRAM bounded even when another probe shares the GPU.
            trials = []
            for radii, points in zip((RADII[:3], RADII[3:]), samples.split(3), strict=True):
                result = model.decode(points)
                scores = objective(result, 1.0, goal, polar).cpu().tolist()
                for radius, score, trial in zip(radii, scores, raw_records(result, model.spec, polar), strict=True):
                    trial.update(radius=radius, score=score, relaxed_improved=score < base - tol,
                                 hard_score=(trial["size"]-goal["size"])**2 + (trial["count"]-goal["count"])**2)
                    trials.append(trial)
            if not all(torch.isfinite(torch.tensor(x["score"])) for x in trials):
                raise ValueError("Nonfinite trial score")
            arms[name] = trials
    return dict(parent=parent, goal=goal, base=base, resolution=tol, norms=norms, arms=arms)


def summarize(rows):
    """Primary endpoint needs a valid parent, a distinct valid edit, and hard score 1→0."""
    valid = [r for r in rows if r["parent"]["smiles"]]
    arms = {}
    for name in rows[0]["arms"]:
        for ladder, radii in LADDERS.items():
            counters = dict(valid_parents=len(valid), any_valid_edit=0, any_hit=0,
                            any_hit_and_relaxed_improved=0, total_distinct_valid_edits=0,
                            selected_valid_edit=0, selected_hit=0, relaxed_descent=0)
            for r in valid:
                trials = [x for x in r["arms"][name] if x["radius"] in radii]
                edits = [x for x in trials if x["smiles"] and x["smiles"] != r["parent"]["smiles"]]
                hits = [x for x in edits if x["hard_score"] == 0]
                counters["any_valid_edit"] += bool(edits)
                counters["any_hit"] += bool(hits)
                counters["any_hit_and_relaxed_improved"] += any(x["relaxed_improved"] for x in hits)
                counters["total_distinct_valid_edits"] += len({x["smiles"] for x in edits})
                # Select using the T=1 continuous surrogate only, without peeking at chemistry.
                best = min(trials, key=lambda x: x["score"])
                if best["relaxed_improved"]:
                    counters["relaxed_descent"] += 1
                    counters["selected_valid_edit"] += best in edits
                    counters["selected_hit"] += best in hits
            arms[f"{name}/{ladder}"] = counters
    return dict(starts=len(rows), valid_parents=len(valid), arms=arms)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--starts", type=int, default=24)
    parser.add_argument("--warm-backward-only", action="store_true",
                        help="Test warmer backward derivatives with T=1 forward counts")
    args = parser.parse_args()
    if not 1 <= args.starts <= 512:
        parser.error("--starts must be in [1, 512]")
    args.output.mkdir(parents=True, exist_ok=False)
    start = monotonic()
    sha = file_sha256(CHECKPOINT)
    if sha != "be683894d49976ee7570b25bda9fefac6b8c53b9d9b55e767ea570b7d147690a":
        raise ValueError("Unexpected checkpoint")
    torch.set_num_threads(2)
    model, state = load_model(CHECKPOINT, args.device)
    model.eval().requires_grad_(False)
    before = state_digest(model.state_dict())
    if state["continuation"]["updates"] != 600116:
        raise ValueError("Unexpected update")
    rows = random.Random(SEED).sample(json.loads(PANEL.read_bytes())["rows"], args.starts)
    smiles = [r[2] for r in rows]
    labels = encode(smiles, model.spec).to(args.device)
    with torch.no_grad():
        means, _ = model.encoder(labels, from_canonical_cache=True)
    priors = torch.randn(means.shape, generator=torch.Generator().manual_seed(SEED)).to(args.device)
    polar = tuple(model.spec.atomic_numbers.index(n) for n in (7, 8))
    generator = torch.Generator().manual_seed(SEED + 1)
    observations = []
    for family, centers in (("encoded", means), ("prior", priors)):
        for i, center in enumerate(centers):
            control = torch.randn(center.shape, generator=generator).to(args.device)
            with rdBase.BlockLogs():
                rec = probe(model, center, polar, control / control.norm(), args.warm_backward_only)
            rec.update(family=family, index=i, encoded_source=rows[i] if family == "encoded" else None)
            observations.append(rec)
        print(f"Finished {family} with {len(centers)} centers", flush=True)
    if state_digest(model.state_dict()) != before or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("Frozen model changed")
    summary = {fam: summarize([r for r in observations if r["family"] == fam])
               for fam in ("encoded", "prior")}
    report = dict(checkpoint_sha256=sha, panel_sha256=file_sha256(PANEL), seed=SEED, updates=600116,
                  variant="warm_backward_only" if args.warm_backward_only else "true_temperature",
                  ladders=LADDERS, records=observations, summary=summary,
                  seconds=monotonic()-start, device=args.device, torch_version=str(torch.__version__))
    (args.output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
