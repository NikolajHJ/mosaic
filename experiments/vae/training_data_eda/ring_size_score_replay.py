"""Measure the additional ring score on paired Gaussian draws from a frozen checkpoint."""

import argparse
import json
from pathlib import Path
from statistics import median
import time

import numpy as np
from rdkit import Chem, rdBase
import torch

from latentmol.chemistry import inspect_graph
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256, write_json
from latentmol.guidance.functional_groups import prepare_pooled_motif_score
from latentmol.guidance.ring_size_reference import COUNTS_BY_HAC, REFERENCE_ID
from latentmol.train import state_digest


def check_draws(base, rings, spec):
    """Independently recount each molecule and its size-matched reference."""
    counts = dict(valid=0, supported=0, ring=0, triangle=0, large=0, invalid=0)
    expected = np.zeros(2)
    for start, hard in base.samples.items():
        second = rings.samples[start]
        for key in ("atoms", "charges", "bonds"):
            if not torch.equal(getattr(hard, key), getattr(second, key)):
                raise ValueError("Enabling ring guidance changed the sampled raw graphs")
        for atoms, charges, bonds in zip(hard.atoms.cpu(), hard.charges.cpu(), hard.bonds.cpu(), strict=True):
            inspected = inspect_graph(atoms, charges, bonds, spec)
            mol = Chem.MolFromSmiles(inspected.smiles) if inspected.smiles else None
            if mol is None:
                counts["invalid"] += 1
                continue
            counts["valid"] += 1
            n, triangles, large = COUNTS_BY_HAC[mol.GetNumHeavyAtoms()]
            if n == 0:
                continue
            sizes = [len(ring) for ring in mol.GetRingInfo().BondRings()]
            counts["supported"] += 1
            counts["ring"] += bool(sizes)
            counts["triangle"] += 3 in sizes
            counts["large"] += any(s >= 8 for s in sizes)
            expected += np.array((triangles, large)) / n
    return dict(counts, triangle_reference=expected[0]/counts["supported"],
                large_reference=expected[1]/counts["supported"])


def compare_gradients(model, prior, base, rings, microbatch, weight):
    """Differentiate both pooled frequency surrogates at identical labels and dropout masks."""
    params = list(model.decoder.parameters())
    sums = [torch.zeros(sum(p.numel() for p in params), device=prior.device) for _ in range(2)]
    for start in range(0, len(prior), microbatch):
        live = model.decode(prior[start:start+microbatch])
        for arm, pooled in enumerate((base, rings)):
            scalar = weight * pooled.surrogate(start, live, model.spec)
            grads = torch.autograd.grad(scalar, params, retain_graph=(arm == 0), allow_unused=True)
            pos = 0
            for param, grad in zip(params, grads, strict=True):
                if grad is not None:
                    sums[arm][pos:pos+param.numel()].add_(grad.detach().reshape(-1))
                pos += param.numel()
    a,b = sums
    delta = b-a
    if not all(x.isfinite().all() and x.norm()>0 for x in (a,b,delta)):
        raise ValueError("The combined score or incremental ring component had zero or nonfinite gradient")
    return dict(control_norm=float(a.norm()), combined_norm=float(b.norm()), ring_norm=float(delta.norm()),
                ring_to_control=float(delta.norm()/a.norm()),
                ring_control_cosine=float((delta @ a)/(delta.norm()*a.norm())))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "receipt", "base-config", "output"):
        p.add_argument("--"+name, required=True, type=Path)
    p.add_argument("--panels", type=int, default=5, help="First pair is used for warmup; remaining pairs alternate order")
    p.add_argument("--microbatch", type=int, default=16)
    p.add_argument("--ring-weight", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=2026112261)
    args = p.parse_args()
    if args.panels < 3 or args.microbatch < 1 or args.output.exists():
        p.error("Require >=3 panel pairs, a positive microbatch size and a new output file")
    policy = json.loads(args.base_config.read_bytes())["functional_groups"]
    receipt = json.loads(args.receipt.read_bytes())
    if (file_sha256(args.checkpoint) != receipt["sha256"] or "ring_size" in policy or
            policy["score_draws"] != 240):
        raise ValueError("Source checkpoint or pooled control policy differs from expected")
    alternative = dict(policy, ring_size=dict(reference=REFERENCE_ID, weight=args.ring_weight))
    torch.set_num_threads(2)
    model, state = load_model(args.checkpoint, "cuda")
    model.train()
    digest = state_digest(model.state_dict())
    if state["continuation"]["updates"] != receipt["updates"]:
        raise ValueError("Unexpected checkpoint cursor")
    results = []
    with rdBase.BlockLogs():
        for i in range(args.panels):
            torch.manual_seed(args.seed + i)
            z = torch.randn(240,model.config.latent,device="cuda")
            scores, elapsed = {}, {}
            for arm, cfg in ((("rings", alternative),("base", policy)) if i%2 else
                             (("base", policy),("rings", alternative))):
                generator = torch.Generator(device="cuda").manual_seed(args.seed + 100 + i)
                torch.cuda.synchronize()
                then = time.perf_counter()
                scores[arm] = prepare_pooled_motif_score(model, z, cfg, args.microbatch, generator)
                torch.cuda.synchronize()
                elapsed[arm] = time.perf_counter() - then
            base, rings = scores["base"], scores["rings"]
            counted = check_draws(base, rings, model.spec)
            if not np.array_equal(base.labels, rings.labels):
                raise ValueError("Motif hit labels differ for paired raw draws")
            for observed, expected in (("ring3_hits", "triangle"), ("ring8plus_hits","large"),
                                       ("ring_reference_supported_valid","supported"), ("ring_any_hits","ring"),
                                       ("ring3_reference_rate","triangle_reference"),
                                       ("ring8plus_reference_rate","large_reference")):
                if not np.isclose(rings.stats[observed], counted[expected], rtol=0, atol=1e-12):
                    raise ValueError(f"Independent {observed} disagrees with pooled trainer")
            result = dict(index=i, timings=elapsed, counts=counted, stats=rings.stats)
            if i == 0:
                result["gradient"] = compare_gradients(model,z,base,rings,args.microbatch,policy["weight"])
            results.append(result)
            print(i, elapsed, result.get("gradient",""), flush=True)
    if state_digest(model.state_dict()) != digest:
        raise RuntimeError("The replay changed checkpoint model weights")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    deltas = [r["timings"]["rings"]-r["timings"]["base"] for r in results[1:]]
    write_json(args.output,dict(checkpoint=receipt,reference=REFERENCE_ID,device=torch.cuda.get_device_name(),
               torch=torch.__version__,rdkit=rdBase.rdkitVersion,seed=args.seed,paired_panels=args.panels,
               microbatch=args.microbatch,draws=240,ring_weight=args.ring_weight,outer_weight=policy["weight"],
               median_extra_pooled_seconds=median(deltas),mean_extra_pooled_seconds=sum(deltas)/len(deltas),
               mean_extra_seconds_per_update=(sum(deltas)/len(deltas))/policy["every_updates"],
               results=results))


if __name__ == "__main__":
    main()
