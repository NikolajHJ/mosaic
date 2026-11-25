"""Local matched training test of retaining only edge existence in the frozen teacher's bond loss."""

import argparse
import csv
import gc
import gzip
import hashlib
import json
import math
from contextlib import nullcontext
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
from rdkit import Chem, rdBase
import torch
from torch.nn import functional as F

from experiments.vae.motif_pilot import chemistry, describe
from latentmol.chemistry import encode_canonical_smiles, inspect_graph
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256
from latentmol.representation import HardGraphBatch
from latentmol.sampling import sample_hard_graph
from latentmol.train import learning_rate_at_epoch, state_digest
from latentmol.training.combined_continuation import _fresh
from latentmol.training import projected_continuation


def existence_only_parent_loss(decoded, hard, valid):
    """Retain production atom/charge CE and binary marginal edge CE with identical normalization."""
    active = valid[:, None] & (hard.atoms != 0)
    if not active.any():
        return decoded.bonds.reshape(-1)[0] * 0
    atom = F.cross_entropy(decoded.atoms[..., 1:][active], hard.atoms[active] - 1)
    charge = F.cross_entropy(decoded.charges[active], hard.charges[active])
    pairs = (active[:, :, None] & active[:, None, :]).triu(1)
    edge = decoded.bonds.reshape(-1)[0] * 0
    if pairs.any():
        logits = decoded.symmetric_bond_logits[pairs]
        present = hard.bonds[pairs].ne(0)
        weights = torch.where(present, torch.ones_like(logits[:, 0]), .1)
        binary = F.binary_cross_entropy_with_logits(
            torch.logsumexp(logits[:, 1:], -1) - logits[:, 0], present.to(logits.dtype), reduction="none")
        edge = (weights * binary).sum() / weights.sum()
    return (atom + charge + 2 * edge) * valid.to(atom.dtype).mean()


def encode(smiles, spec):
    data = [encode_canonical_smiles(sm, spec) for sm in smiles]
    return HardGraphBatch(*(torch.from_numpy(np.stack([row[key] for row in data])).long()
                            for key in ("atom_types", "charges", "bond_types")))


def hard_records(hard, spec):
    result = []
    for a, c, b in zip(hard.atoms.cpu().numpy(), hard.charges.cpu().numpy(), hard.bonds.cpu().numpy(), strict=True):
        inspected = inspect_graph(a, c, b, spec)
        active = np.flatnonzero(a)
        i, j = np.nonzero(np.triu(b, 1))
        smiles, failure = inspected.smiles, inspected.failure_reason
        # A sanitized graph can exceptionally emit a SMILES that fails a strict round trip.
        parsed = Chem.MolFromSmiles(smiles) if smiles else None
        valid_smiles = smiles if parsed is not None else None
        if smiles and parsed is None:
            failure = "smiles_roundtrip"
        rings = parsed.GetRingInfo().AtomRings() if parsed is not None else []
        result.append(dict(smiles=smiles, failure=failure, **chemistry(valid_smiles),
                           ring3=any(len(r) == 3 for r in rings), ring8plus=any(len(r) >= 8 for r in rings),
                           ring12plus=any(len(r) >= 12 for r in rings),
                           nodes=np.column_stack((active, a[active], c[active])).tolist(),
                           edges=np.column_stack((i, j, b[i, j])).tolist()))
    return result


@torch.no_grad()
def evaluate(model, z, validation, out, *, categorical_seed=None):
    model.eval()
    records = []
    generator = (None if categorical_seed is None else
                 torch.Generator(device="cuda").manual_seed(categorical_seed))
    for block in z.split(8):
        decoded = model.decode(block.to("cuda"))
        hard = (decoded.harden(model.spec) if generator is None else
                sample_hard_graph(decoded, model.spec, generator=generator))
        records.extend(hard_records(hard, model.spec))
    recs = []
    for graphs, sm in validation:
        mu, _ = model.encoder(graphs.to("cuda"), from_canonical_cache=True)
        items = hard_records(model.decode(mu).harden(model.spec), model.spec)
        for row, target in zip(items, sm, strict=True):
            row.update(target=target, exact=row["smiles"]==target)
        recs.extend(items)
    with gzip.open(out, "xt", encoding="utf-8") as f:
        for name, rows in (("prior", records), ("reconstruction", recs)):
            for idx, row in enumerate(rows):
                f.write(json.dumps(dict(panel=name, index=idx, **row))+"\n")
    model.train()
    return (records, recs), dict(prior=describe(records), reconstruction=describe(recs),
                                 exact=sum(r["exact"] for r in recs))


def digest_rng(generators):
    return {name: hashlib.sha256(data.cpu().numpy().tobytes()).hexdigest() for name, data in
            (("cpu", torch.get_rng_state()), ("cuda", torch.cuda.get_rng_state()),
             ("prior", generators["fresh_prior"].get_state()))}


def paired_graph_changes(control, treatment):
    pairs = list(zip(control, treatment, strict=True))
    return dict(draws=len(pairs), changed_slot_or_bond_labels=sum(
        a["nodes"] != b["nodes"] or a["edges"] != b["edges"] for a,b in pairs),
        both_valid=sum(a["valid"] and b["valid"] for a,b in pairs),
        both_valid_different_smiles=sum(a["valid"] and b["valid"] and a["smiles"] != b["smiles"]
                                       for a,b in pairs),
        both_valid_changed_scaffold=sum(a["valid"] and b["valid"] and a["scaffold"] != b["scaffold"]
                                         for a,b in pairs))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["checkpoint", "reference", "config", "train-panel", "heldout-panel", "output"]:
        p.add_argument("--"+key, type=Path, required=True)
    p.add_argument("--updates", type=int, default=128)
    p.add_argument("--seed", type=int, default=2026112168)
    p.add_argument("--draws", type=int, default=1024)
    p.add_argument("--completion-arm", choices=("control", "treatment"),
        help="Run one arm of the motif-completion comparison and score a second independent panel.")
    p.add_argument("--control-summary", type=Path,
        help="Require an exact match to the historical control's inputs, training rows and sampling streams.")
    p.add_argument("--new-platform-control", action="store_true",
        help="Regenerate a matched control on another device/software version. Pair treatment to its new summary.")
    p.add_argument("--ring-encoding", choices=("kekule",), help="Override the treatment's aromatic bond labels.")
    p.add_argument("--completion-pattern", action="store_true",
        help="Test sparse aromatic pattern completion and protection, with relative amide weight 0.10.")
    p.add_argument("--neighbor-weight", type=float,
        help="Use only natural motif-neighborhood guidance alongside the original sampled score.")
    args = p.parse_args()
    if bool(args.completion_arm) != bool(args.control_summary):
        p.error("The completion pilot requires --completion-arm and --control-summary together")
    if args.new_platform_control and args.completion_arm != "control":
        p.error("--new-platform-control applies only to a fresh control")
    if args.ring_encoding and args.completion_arm != "treatment":
        p.error("--ring-encoding applies only to the treatment")
    if args.completion_pattern and (args.completion_arm != "treatment" or args.ring_encoding):
        p.error("Pattern completion requires the treatment and the explicit aromatic encoding")
    if args.neighbor_weight is not None and (
            args.completion_arm != "treatment" or args.ring_encoding or args.completion_pattern
            or not math.isfinite(args.neighbor_weight) or args.neighbor_weight <= 0):
        p.error("Specify a positive neighbor weight on a treatment without motif completion")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(3)
    no = 64
    microbatch = 16
    config = json.loads(args.config.read_bytes())
    config["functional_groups"] = dict(config["functional_groups"], draws=microbatch)
    parent, payload = load_model(args.checkpoint, "cpu")
    start = payload["continuation"]["updates"]
    if (start != 580498 or parent.config.extra_bond_rounds != 3 or not torch.cuda.is_available()
            or file_sha256(args.reference) != config["prior_only"]["guidance"]["reference"]["sha256"]):
        raise ValueError("Pilot requires the seven-round 5,000-update checkpoint and the configured frozen reference")
    reference, _ = load_model(args.reference, "cuda", parent.spec)
    reference.eval().requires_grad_(False)
    parent_digest = state_digest(parent.state_dict())
    ref_digest = state_digest(reference.state_dict())
    with gzip.open(args.train_panel, "rt") as stream:
        train = [row["smiles"] for row in csv.DictReader(stream, delimiter="\t")]
    panel = json.loads(args.heldout_panel.read_bytes())
    heldout = [row[2] for row in panel["rows"]]
    if set(heldout).intersection(train):
        raise ValueError("Training panel overlaps held-out identities")
    selected = np.random.default_rng(args.seed).choice(len(train), args.updates * no, replace=False)
    held = np.random.default_rng(args.seed+1).choice(len(heldout), 128, replace=False)
    val = [([heldout[i] for i in ids]) for ids in np.array_split(held, 16)]
    val = [(encode(sm, parent.spec), sm) for sm in val]
    batches = []
    for ids in selected.reshape(-1, no):
        batches.append(encode([train[i] for i in ids], parent.spec))
    print("Prepared", len(selected), "unique rows,", len(held), "held-out rows", flush=True)
    z = torch.randn(args.draws, parent.config.latent, generator=torch.Generator().manual_seed(args.seed+2))
    summary = dict(checkpoint=file_sha256(args.checkpoint), reference=file_sha256(args.reference), config=config,
        train_panel=file_sha256(args.train_panel), heldout_panel=file_sha256(args.heldout_panel),
        train_indices_sha256=hashlib.sha256(selected.tobytes()).hexdigest(), seed=args.seed,
        updates=args.updates, real_per_update=no, priors_per_update=no, microbatch=microbatch,
        gaussian_evaluation=args.draws, heldout_evaluation=len(held), device=torch.cuda.get_device_name(),
        torch=torch.__version__, rdkit=rdBase.rdkitVersion, arms={})
    records = {}
    if args.completion_arm:
        old = json.loads(args.control_summary.read_bytes())
        for key in ("checkpoint", "reference", "config", "train_panel", "heldout_panel", "train_indices_sha256",
                    "seed", "updates", "real_per_update", "priors_per_update", "microbatch",
                    "gaussian_evaluation", "heldout_evaluation", "device", "torch", "rdkit"):
            if args.new_platform_control and key in ("device", "torch", "rdkit"):
                continue
            if summary[key] != old[key]:
                raise ValueError(f"Cannot match the historical control for {key}")
        if args.completion_arm == "treatment":
            if args.neighbor_weight is None:
                config["functional_groups"]["completion_weight"] = 1.0
            else:
                config["functional_groups"]["neighbor_weight"] = args.neighbor_weight
            if args.ring_encoding:
                config["functional_groups"]["ring_encoding"] = args.ring_encoding
            if args.completion_pattern:
                config["functional_groups"].update(completion_mode="pattern", amide_relative_weight=.10)
        if args.new_platform_control:
            summary["new_platform_control"] = True
        summary["source_control"] = file_sha256(args.control_summary)
        summary["fresh_evaluation_seed"] = 2026112235
        summary["categorical_seed"] = 2026112236
        fresh_z = torch.randn(args.draws, parent.config.latent, generator=torch.Generator().manual_seed(2026112235))
    for arm in ((args.completion_arm,) if args.completion_arm else ("control", "existence_only")):
        model, optimizer, training, generators = _fresh(parent,payload,None,no,"cuda",None,
            atom_refinement_dropout=config["atom_refinement_dropout"])
        if training.loss.atom_order != config["atom_order"] or state_digest(model.state_dict()) != parent_digest:
            raise ValueError("Working model differs from saved weights or reconstruction policy")
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_at_epoch(training, 48)
        torch.manual_seed(args.seed+3)
        generators["fresh_prior"].manual_seed(args.seed+4)
        if arm == "control" and not args.completion_arm:
            with rdBase.BlockLogs():
                records["start"], summary["start"] = evaluate(model, z, val, args.output/"start.jsonl.gz")
            print("Start panel:",summary["start"]["prior"]["aromatic"],"exact",summary["start"]["exact"],flush=True)
        before = digest_rng(generators)
        changed = patch.object(projected_continuation, "conditional_parent_loss", existence_only_parent_loss)
        t = time.monotonic()
        with rdBase.BlockLogs(), (changed if arm == "existence_only" else nullcontext()), (args.output/(arm+"-training.jsonl")).open("x") as log:
            for step, hard in enumerate(batches):
                metrics = projected_continuation.projected_update(config, model, reference, optimizer, hard, training,
                    None,None,generators,start+step,48,microbatch)
                log.write(json.dumps(dict(step=step+1, **metrics))+"\n")
                if step==0 or (step+1)%16==0:
                    log.flush()
                    print(arm,step+1,"/",args.updates,round(time.monotonic()-t,2),"s",flush=True)
        after = digest_rng(generators)
        seconds = time.monotonic()-t
        with rdBase.BlockLogs():
            records[arm], result = evaluate(model, z, val, args.output/(arm+".jsonl.gz"))
        summary["arms"][arm] = dict(result, train_seconds=seconds,rng_start=before,rng_end=after,
             parameter_sha=state_digest(model.state_dict()),optimizer_states=len(optimizer.state),
             learning_rate=optimizer.param_groups[0]["lr"])
        if args.completion_arm:
            # Save only evaluation weights, which lets later panels reuse the exact endpoint.
            torch.save({key: payload[key] for key in ("format_version", "model_config", "graph_spec")} |
                       {"model_state": model.state_dict(), "diagnostic_only": True,
                        "source_update": start, "pilot_updates": args.updates}, args.output / "endpoint.pt")
            with rdBase.BlockLogs():
                (fresh, _), fresh_result = evaluate(model, fresh_z, [], args.output / (arm+"-fresh.jsonl.gz"))
                (sampled, _), sample_result = evaluate(model, fresh_z, [], args.output / (arm+"-sampled.jsonl.gz"),
                                                       categorical_seed=summary["categorical_seed"])
            summary["arms"][arm]["fresh_prior"] = fresh_result["prior"]
            summary["arms"][arm]["fresh_categorical"] = sample_result["prior"]
            summary["arms"][arm]["ring_counts"] = {
                name: {key: sum(row[key] for row in rows) for key in ("valid", "ring3", "ring8plus", "ring12plus")}
                for name, rows in (("primary", records[arm][0]), ("fresh", fresh), ("categorical", sampled))}
            same_rng = all(summary["arms"][arm][key] == old["arms"]["control"][key]
                           for key in ("rng_start", "rng_end"))
            summary["matched_control_rng"] = same_rng
            if not same_rng and not args.new_platform_control:
                raise RuntimeError("Random streams differ from the historical control")
            if arm == "control":
                summary["reproduced_control_weights"] = (summary["arms"][arm]["parameter_sha"]
                                                           == old["arms"]["control"]["parameter_sha"])
        if state_digest(reference.state_dict()) != ref_digest or state_digest(parent.state_dict())!=parent_digest:
            raise RuntimeError("Frozen teacher or source checkpoint weights changed")
        print(arm, "complete", result["prior"]["aromatic"], result["prior"]["amide"], result["exact"], flush=True)
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    if args.completion_arm:
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
        return
    for moment in ("rng_start", "rng_end"):
        if summary["arms"]["control"][moment]!=summary["arms"]["existence_only"][moment]:
            raise RuntimeError("Arms consumed different training randomness")
    summary["paired"] = {}
    summary["paired_prior_graphs"] = paired_graph_changes(records["control"][0], records["existence_only"][0])
    for key, idx in (("valid",0),("aromatic",0),("amide",0),("exact",1)):
        pairs = list(zip(records["control"][idx], records["existence_only"][idx], strict=True))
        summary["paired"][key] = dict(gained=sum(b[key] and not a[key] for a,b in pairs),
                                      lost=sum(a[key] and not b[key] for a,b in pairs))
    with (args.output/"summary.json").open("x") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
        f.write("\n")
    print("Paired",summary["paired"],flush=True)


if __name__=="__main__":
    main()
