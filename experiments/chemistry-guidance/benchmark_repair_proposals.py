"""Frozen-checkpoint target discovery comparison; no training or HPC submission.

Run --help. The same saved model probabilities and raw graphs feed all arms.
Synthetic dense/disconnected stress cases are reported separately from prior
samples. Candidate-check caps are explicit; elapsed runtime includes proposal
generation, raw checks and final witness creation. It is not a GPU epoch ETA.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from latentmol.atom_preserving_fast import search_atom_preserving_repair_fast
from latentmol.atom_preserving_chemistry import AtomPreservingRepairTeacher, repair_loss
from latentmol.evaluation.evaluate import load_model
from latentmol.guidance.repair_proposals import ProposalRepairTeacher, RepairProposalConfig, search_with_proposals
from latentmol.guidance.repair_astar import AStarRepairTeacher, AStarRepairConfig, search_astar
from latentmol.probability_repair import ProbabilityRepairConfig
from latentmol.representation import GraphSpec
from latentmol.train import state_digest


def arm_policy(value):
    if value is None:
        return None
    settings = dict(value)
    backend = settings.pop("backend", "proposals")
    if backend == "astar":
        return AStarRepairConfig(**settings)
    if backend == "proposals":
        return RepairProposalConfig(**settings)
    raise ValueError("Unknown repair benchmark backend: " + str(backend))


def run_search(raw, probabilities, spec, search, policy):
    if policy is None:
        return search_atom_preserving_repair_fast(raw, probabilities, spec, search)
    if isinstance(policy, AStarRepairConfig):
        return search_astar(raw, probabilities, spec, search, policy)
    return search_with_proposals(raw, probabilities, spec, search, policy)


def make_teacher(spec, search, policy):
    if policy is None:
        return AtomPreservingRepairTeacher(spec, search, backend="optimized")
    if isinstance(policy, AStarRepairConfig):
        return AStarRepairTeacher(spec, search, astar=policy)
    return ProposalRepairTeacher(spec, search, proposals=policy)


def stress_cases(spec):
    for name, count, dense in (("fragments8", 8, False), ("dense8", 8, True), ("dense16", 16, True)):
        if spec.max_nodes < count:
            continue
        atoms = np.zeros(spec.max_nodes, dtype=np.int64)
        atoms[:count] = spec.atomic_numbers.index(6)
        charges = np.full(spec.max_nodes, spec.formal_charges.index(0), dtype=np.int64)
        bonds = np.zeros((spec.max_nodes, spec.max_nodes), dtype=np.int64)
        for i in range(count):
            for j in range(i+1, count):
                if dense or i//2 == j//2:
                    bonds[i, j] = bonds[j, i] = spec.bond_names.index("SINGLE")
        raw = (atoms, charges, bonds)
        laws = []
        for labels, classes in zip(raw, (len(spec.atomic_numbers), len(spec.formal_charges), len(spec.bond_names))):
            p = np.full((*labels.shape, classes), .001, dtype=np.float64)
            np.put_along_axis(p, labels[..., None], 1. - .001*(classes-1), axis=-1)
            laws.append(p)
        yield name, raw, tuple(laws)


@torch.no_grad()
def cases(args, model):
    if model is not None:
        codes = torch.randn(args.count, model.config.latent,
                            generator=torch.Generator().manual_seed(args.seed))
        for start in range(0, args.count, args.batch_size):
            decoded = model.decode(codes[start:start+args.batch_size].to(args.device))
            hard = decoded.harden(model.spec)
            probabilities = tuple(v.cpu().double().numpy() for v in (decoded.atoms.softmax(-1),
                decoded.charges.softmax(-1), decoded.symmetric_bond_logits.softmax(-1)))
            for i in range(len(hard.atoms)):
                raw = tuple(v[i].cpu().numpy().copy() for v in (hard.atoms, hard.charges, hard.bonds))
                yield f"prior-{start+i}", raw, tuple(v[i].copy() for v in probabilities)
    yield from stress_cases(model.spec if model is not None else GraphSpec())


def main(args):
    if min(args.count, args.batch_size, args.threads, args.seconds) < 1:
        raise ValueError("Require positive counts and budget")
    if args.gradient_smoke and args.checkpoint is None:
        raise ValueError("Decoder gradient smoke requires a checkpoint")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    config = json.loads(args.config.read_text())
    search = ProbabilityRepairConfig(**config["search"])
    policies = {name: arm_policy(value) for name, value in config["arms"].items()}
    if not policies:
        raise ValueError("Benchmark needs at least one configured arm")
    model, payload, before = None, {}, None
    if args.checkpoint:
        model, payload = load_model(args.checkpoint, args.device)
        model.requires_grad_(False).eval()
        before = state_digest(model.state_dict())
    spec = model.spec if model is not None else GraphSpec()
    started = time.monotonic()
    records = []
    with (args.output / "attempts.jsonl").open("x", encoding="utf-8") as stream:
        for index, (name, raw, p) in enumerate(cases(args, model)):
            # Rotate order to reduce systematic first-arm warm-cache advantage.
            names = list(policies); names = names[index % len(names):] + names[:index % len(names)]
            for arm in names:
                if time.monotonic() - started > args.seconds:
                    raise TimeoutError("Explicit frozen comparison budget exhausted")
                tick = time.perf_counter()
                result = run_search(raw, p, spec, search, policies[arm])
                elapsed = time.perf_counter()-tick
                witness = result.get("weighted_witness")
                if witness is not None:
                    if witness["labels"]["atoms"] != raw[0].tolist() or witness["labels"]["charges"] != raw[1].tolist():
                        raise RuntimeError("Proposal changed a fixed identity")
                row = dict(case=name, arm=arm, seconds=elapsed, status=result["status"],
                    checks=result["checks"], raw_valid=result["raw_valid"],
                    edits=None if witness is None else witness["edit_count"],
                    weighted_cost=None if witness is None else witness["weighted_cost"],
                    source=result.get("accepted_source"),
                    raw_graph={k:v.tolist() for k,v in zip(("atoms", "charges", "bonds"),raw)},
                    result=result)
                stream.write(json.dumps(row, allow_nan=False)+"\n");stream.flush()
                records.append(row)
            if index % 32 == 0:
                print(json.dumps(dict(completed_cases=index+1, seconds=time.monotonic()-started)), flush=True)
    groups = {}
    for arm in policies:
        for group in ("prior", "stress"):
            rows = [r for r in records if r["arm"]==arm and r["case"].startswith("prior-") == (group=="prior")]
            if not rows:
                continue
            failed = [r for r in rows if not r["raw_valid"]]
            repaired = [r for r in failed if r["status"]=="witness_found"]
            seconds = np.array([r["seconds"] for r in rows])
            groups[f"{arm}/{group}"] = dict(attempts=len(rows), originally_invalid=len(failed),
                repaired=len(repaired), coverage=None if not failed else len(repaired)/len(failed),
                elapsed_seconds=float(seconds.sum()), median_seconds=float(np.median(seconds)),
                p95_seconds=float(np.quantile(seconds,.95)), candidate_checks=sum(r["checks"] for r in rows),
                sources=dict(Counter(r["source"] for r in repaired)),
                mean_witness_edits=None if not repaired else float(np.mean([r["edits"] for r in repaired])),
                mean_weighted_cost=None if not repaired else float(np.mean([r["weighted_cost"] for r in repaired])),
                terminations=dict(Counter(r["result"].get("termination", r["status"]) for r in rows)),
                generated_states=sum(r["result"].get("generated_states", 0) for r in rows))
    gradients = {}
    invalid = next((r for r in records if r["case"].startswith("prior-") and not r["raw_valid"]), None)
    if args.gradient_smoke and invalid is not None:
        index = int(invalid["case"].split("-")[1])
        code = torch.randn(args.count, model.config.latent,
                           generator=torch.Generator().manual_seed(args.seed))[index:index+1].to(args.device)
        for arm, policy in policies.items():
            z = code.detach().clone().requires_grad_()
            output = model.decode(z)
            teacher = make_teacher(spec, search, policy)
            with teacher:
                targets = teacher.build_targets(output, candidate_ids=[index], update=0)
                loss = repair_loss(output, targets, spec)
                grad, = torch.autograd.grad(loss, z)
            if not torch.isfinite(loss) or not torch.isfinite(grad).all():
                raise RuntimeError("Nonfinite real-decoder repair gradient")
            gradients[arm] = dict(case=invalid["case"], loss=float(loss.detach()),
                latent_gradient_norm=float(grad.norm()), flat=bool((grad==0).all()),
                accepted=targets.receipt["status_counts"].get("accepted", 0))
        if any(p.grad is not None for p in model.parameters()):
            raise RuntimeError("Frozen model accumulated an unintended parameter gradient")
    if model is not None and state_digest(model.state_dict()) != before:
        raise RuntimeError("Frozen checkpoint model changed")
    result = dict(config=config, effective_proposals={k:None if p is None else asdict(p) for k,p in policies.items()},
        seed=args.seed, prior_count=args.count if model is not None else 0,
        epoch=payload.get("epoch"), device=args.device, threads=args.threads,
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() if args.checkpoint else None,
        model_unchanged=True, seconds=time.monotonic()-started, groups=groups,
        decoder_gradient_smoke=gradients,
        limitations=["Frozen target-discovery benchmark, not a training or validity improvement",
            "Prior and artificial stress graphs are separate populations",
            "No closest-repair or fixed-identity chemical feasibility guarantee",
            "Native candidate budgets exclude raw checks and independent teacher target checks",
            "Large coordinated targets can need a stricter edit cap in future training comparisons"])
    (args.output/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path(__file__).with_name("repair_structural.json"))
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--count",type=int,default=256)
    parser.add_argument("--seed",type=int,default=904017)
    parser.add_argument("--batch-size",type=int,default=16)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    parser.add_argument("--seconds",type=int,default=600)
    parser.add_argument("--gradient-smoke", action="store_true")
    main(parser.parse_args())
