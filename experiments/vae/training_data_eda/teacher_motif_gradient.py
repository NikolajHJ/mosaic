"""Does the frozen teacher penalize increasing the student's aromatic bond logits?

Recreate its graph labels on a shared fixed prior panel and perturb one scalar
on frozen student logits. Decompose the actual bond cross-entropy into edge
existence and bond type conditional on presence. No parameters change.
"""

import argparse
import gzip
import json
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from rdkit import Chem, rdBase
from torch.nn import functional as F

from latentmol.chemistry_classification import classify_repair_graph
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256
from latentmol.guidance.valence_projection import projection_targets
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch
from latentmol.training.prior_repair import conditional_parent_loss


def teacher_panel(path, count):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["panel"] == "prior_argmax":
                rows.append(row)
            if len(rows) == count:
                break
    if [r["index"] for r in rows] != list(range(count)):
        raise ValueError("Teacher archive must supply the ordered prior_argmax panel")
    return rows


def teacher_targets(rows, spec):
    n = len(rows)
    atoms = torch.zeros((n, spec.max_nodes), dtype=torch.long)
    charges = torch.full_like(atoms, spec.formal_charges.index(0))
    bonds = torch.zeros((n, spec.max_nodes, spec.max_nodes), dtype=torch.long)
    eligible = []
    for k, row in enumerate(rows):
        for slot, element, charge in row["nodes"]:
            atoms[k, slot], charges[k, slot] = element, charge
        for i, j, label in row["edges"]:
            bonds[k, i, j] = bonds[k, j, i] = label
        result = classify_repair_graph(atoms[k], charges[k], bonds[k], spec)
        if result["smiles"] != row["smiles"]:
            raise ValueError("Current chemistry inspection disagrees with archived graph")
        eligible.append(result["strict_teacher_valid"])
    return HardGraphBatch(atoms, charges, bonds), torch.tensor(eligible)


def raw_cycle_mask(hard):
    """Use topology alone, including cyclic edges on unsanitizable graphs."""
    mask = torch.zeros_like(hard.bonds, dtype=torch.bool)
    for k, adjacency in enumerate(hard.bonds):
        mol = Chem.RWMol()
        for _ in range(adjacency.shape[0]):
            mol.AddAtom(Chem.Atom(6))
        for i, j in zip(*torch.where(adjacency.triu(1) != 0), strict=True):
            mol.AddBond(int(i), int(j), Chem.BondType.SINGLE)
        for cycle in Chem.GetSymmSSSR(mol.GetMol()):
            ring = list(cycle)
            for i, j in zip(ring, ring[1:] + ring[:1], strict=True):
                mask[k, i, j] = mask[k, j, i] = True
    return mask


def projection_derivative(decoded, spec, aromatic):
    """Hold the existing projection's selected pair labels fixed as during training."""
    slots = decoded.atoms.shape[1]
    i, j = torch.triu_indices(slots, slots, 1)
    logits = decoded.symmetric_bond_logits[:, i, j]
    original, new, _ = projection_targets(
        decoded.atoms.argmax(-1).numpy(), decoded.charges.argmax(-1).numpy(), logits.numpy(), spec, 8)
    rows, edges = np.where(original != new)
    if not len(rows):
        return 0., 0
    counts = torch.from_numpy(np.bincount(rows, minlength=len(logits))[rows])
    labels = torch.from_numpy(new[rows, edges])
    derivative = logits[rows, edges].softmax(-1)[:, aromatic] - (labels == aromatic).double()
    return float((derivative / counts).sum() / len(logits)), len(rows)


def measure_group(decoded, hard, valid, spec):
    aromatic = spec.bond_names.index("AROMATIC")
    active = (hard.atoms != 0) & valid[:, None]
    pairs = (active[:, :, None] & active[:, None, :]).triu(1)
    if not pairs.any() or not valid.any():
        raise ValueError("No eligible occupied teacher pairs in requested group")
    cycles = raw_cycle_mask(decoded.harden(spec))[pairs]
    fraction = valid.double().mean()
    targets = hard.bonds[pairs]
    present = targets != 0
    weights = torch.ones_like(targets, dtype=torch.double)
    weights[~present] = 0.1
    normalization = 2 * fraction / weights.sum()
    labels = torch.zeros(len(spec.bond_names), dtype=torch.double)
    labels[aromatic] = 1
    logits = decoded.symmetric_bond_logits[pairs]

    atom_ce = F.cross_entropy(decoded.atoms[..., 1:][active], hard.atoms[active] - 1)
    charge_ce = F.cross_entropy(decoded.charges[active], hard.charges[active])
    node_loss = fraction * (atom_ce + charge_ce)

    def objectives(delta):
        shifted = logits + delta * labels
        log_odds_exists = shifted[:, 1:].logsumexp(-1) - shifted[:, 0]
        existence = F.binary_cross_entropy_with_logits(log_odds_exists, present.double(), reduction="none")
        type_ce = F.cross_entropy(shifted[present, 1:], targets[present] - 1, reduction="none")
        existence = normalization * (weights * existence).sum()
        bond_type = normalization * (weights[present] * type_ce).sum()
        native = conditional_parent_loss(
            DecoderOutput(decoded.atoms, decoded.charges, decoded.bonds + delta * labels), hard, valid)
        return {"native": native, "existence_only": node_loss + existence, "conditional_type": bond_type}

    delta = torch.tensor(0., dtype=torch.double, requires_grad=True)
    losses = objectives(delta)
    gradient = {name: float(torch.autograd.grad(value, delta, retain_graph=True)[0]) for name, value in losses.items()}
    difference = float((losses["native"] - losses["existence_only"] - losses["conditional_type"]).detach())
    h = 1e-4
    plus, minus = objectives(h), objectives(-h)
    fd = {name: float((plus[name] - minus[name]) / (2 * h)) for name in losses}
    if (abs(difference) > 5e-9 or abs(gradient["native"] - gradient["existence_only"] - gradient["conditional_type"]) > 1e-9
            or any(abs(gradient[name] - fd[name]) > 1e-7 for name in fd)):
        raise ArithmeticError("Split, native differentiation or centered difference disagrees")

    # Binary-edge logit sensitivity comes through logsumexp of all positive
    # bond classes. A right-kind/wrong-kind edge can obtain a topology signal
    # even though this term is indifferent to its bond type at fixed existence.
    probs = logits.softmax(-1)
    conditional_prob = logits[:, 1:].softmax(-1)[:, aromatic - 1]
    full = normalization * weights * (probs[:, aromatic] - (targets == aromatic).double())
    type_gradient = normalization * weights * present * (conditional_prob - (targets == aromatic).double())
    existence_gradient = full - type_gradient
    by_pairs = {}
    for name, mask in [("all", torch.ones_like(cycles)), ("cycle", cycles), ("off_cycle", ~cycles)]:
        by_pairs[name] = dict(pairs=int(mask.sum()), explicit_aromatic_labels=int((targets[mask] == aromatic).sum()),
                              native=float(full[mask].sum()), conditional_type=float(type_gradient[mask].sum()),
                              existence_only=float(existence_gradient[mask].sum()))
    if any(abs(gradient[k] - by_pairs["all"][k]) > 1e-9 for k in gradient):
        raise ArithmeticError("Analytic per-pair decomposition differs from autograd")
    projection, changed = projection_derivative(decoded, spec, aromatic)
    return dict(teacher_eligible=int(valid.sum()), loss={k:float(v.detach()) for k,v in losses.items()},
                node_loss=float(node_loss), split_loss_residual=difference, gradient=gradient, centered_fd=fd,
                by_pairs=by_pairs, projection_weighted_gradient=.375 * projection, projection_changed_edges=changed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-graphs", type=Path, required=True)
    parser.add_argument("--teacher-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--group", type=int, default=16)
    parser.add_argument("--decode-batch", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.count < 1 or args.group < 1 or args.count % args.group or min(args.decode_batch,args.threads) < 1:
        parser.error("Use positive budgets and complete teacher groups")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    metadata = json.loads(args.teacher_summary.read_bytes())
    spec = GraphSpec.from_dict(metadata["spec"])
    model, checkpoint = load_model(args.checkpoint, "cpu", spec)
    model.requires_grad_(False)
    if metadata["rdkit"] != rdBase.rdkitVersion or metadata["torch"] != torch.__version__:
        raise ValueError("Replay requires the archived panel's CPU Torch and RDKit versions")
    rows = teacher_panel(args.teacher_graphs, args.count)
    ck_id = args.teacher_graphs.name.removesuffix(".jsonl.gz")
    ref = metadata["checkpoints"][ck_id]
    n = ref["populations"]["prior_argmax"]["draws"]
    z = torch.randn(n, model.config.latent, generator=torch.Generator().manual_seed(metadata["seed"]))[:args.count]
    groups = []
    with rdBase.BlockLogs():
        for start in range(0, args.count, args.group):
            hard, valid = teacher_targets(rows[start:start+args.group], spec)
            with torch.no_grad():
                preds = [model.decode(block) for block in z[start:start+args.group].split(args.decode_batch)]
                decoded = DecoderOutput(*(torch.cat([getattr(p, field) for p in preds]).double()
                                          for field in ("atoms", "charges", "bonds")))
            groups.append(measure_group(decoded, hard, valid, spec))
    summary = {name: dict(mean=mean(g["gradient"][name] for g in groups),
                          positive=sum(g["gradient"][name] > 0 for g in groups),
                          negative=sum(g["gradient"][name] < 0 for g in groups),
                          cycle_contribution=mean(g["by_pairs"]["cycle"][name] for g in groups))
               for name in groups[0]["gradient"]}
    result = dict(checkpoint_sha256=file_sha256(args.checkpoint), student_update=checkpoint["continuation"]["updates"],
                  teacher_checkpoint_sha256=ref["sha256"], teacher_graphs_sha256=file_sha256(args.teacher_graphs),
                  prior_seed=metadata["seed"], draws=args.count, group_size=args.group,
                  torch=torch.__version__, rdkit=rdBase.rdkitVersion, groups=groups, summary=summary,
                  mean_weighted_projection_gradient=mean(g["projection_weighted_gradient"] for g in groups),
                  interpretation="Positive derivative means decreasing the common aromatic-class bias is a local "
                                 "descent direction for this term. Existence-only retains the native atom, charge, "
                                 "normalization, NONE weight and eligibility scaling. Other objectives, optimizer "
                                 "preconditioning, training dropout and shared-parameter updates are not simulated.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2); out.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
