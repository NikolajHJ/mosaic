"""Experimental ablation of conditional SINGLE supervision on verified student six-membered C/N rings."""

import argparse
import json
from pathlib import Path
import statistics
import time

from rdkit import rdBase
import torch
from torch.nn import functional as F

from experiments.vae.training_data_eda.teacher_motif_gradient import teacher_panel, teacher_targets
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256
from latentmol.guidance.ring_teacher import ring_single_mask, ring_teacher_loss
from latentmol.representation import DecoderOutput, GraphSpec
from latentmol.training.prior_repair import conditional_parent_loss


def independent_split(decoded, hard, valid, masked):
    """Rebuild the same objective from node terms, all-pair existence BCE and retained type CE."""
    active = valid[:,None] & hard.atoms.ne(0)
    pairs = (active[:,:,None] & active[:,None,:]).triu(1)
    present = (hard.bonds != 0) & pairs
    keep = present.clone()
    if masked:
        r,i,j = torch.tensor(masked).T
        assert ((hard.bonds[r,i,j] == 1) & pairs[r,i,j]).all()
        keep[r,i,j] = False
    l = decoded.symmetric_bond_logits
    nodes = F.cross_entropy(decoded.atoms[...,1:][active],hard.atoms[active]-1)
    nodes += F.cross_entropy(decoded.charges[active],hard.charges[active])
    w = torch.where(present[pairs], l.new_ones(()), l.new_tensor(.1))
    odds = l[pairs][:,1:].logsumexp(-1) - l[pairs][:,0]
    existence = F.binary_cross_entropy_with_logits(odds, present[pairs].to(odds.dtype), weight=w, reduction="sum")
    types = F.cross_entropy(l[keep][:,1:], hard.bonds[keep]-1,reduction="sum") if keep.any() else 0
    return valid.double().mean() * (nodes + 2*(existence + types)/w.sum())


def replay_group(decoded, hard, valid, spec):
    start = time.perf_counter()
    masked, counts = ring_single_mask(decoded,hard,valid,spec)
    selection = time.perf_counter()-start
    live = tuple(x.detach().requires_grad_() for x in (decoded.atoms,decoded.charges,decoded.bonds))
    output = DecoderOutput(*live)
    native = conditional_parent_loss(output,hard,valid)
    treatment = ring_teacher_loss(output,hard,valid,masked)
    direct = independent_split(output,hard,valid,masked)
    residual_value = float((direct-treatment).detach().abs())
    gn, gt, gd = [torch.autograd.grad(l,live,retain_graph=True) for l in (native,treatment,direct)]
    residual_grad = max(float((a-b).abs().max()) for a,b in zip(gt,gd))
    mask = torch.zeros_like(hard.bonds,dtype=torch.bool)
    if masked:
        r,i,j = torch.tensor(masked).T
        mask[r,i,j] = mask[r,j,i] = True
    if (not torch.equal(gn[0],gt[0]) or not torch.equal(gn[1],gt[1]) or
            not torch.equal(gn[2][~mask],gt[2][~mask]) or not torch.equal(gn[2][...,0],gt[2][...,0])
            or max(residual_value,residual_grad) > 1e-11):
        raise AssertionError("Node, offsite bond, existence-logit or exact loss normalization changed")
    aro = spec.bond_names.index("AROMATIC")
    return dict(teacher_valid=int(valid.sum()), **counts, masked_pairs=len(masked),
                selected_seconds=selection, native=float(native.detach()), treatment=float(treatment.detach()),
                native_aro_direction=float(gn[2][...,aro].sum()), treatment_aro_direction=float(gt[2][...,aro].sum()),
                value_residual=residual_value,max_gradient_residual=residual_grad)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("checkpoint", "teacher-graphs", "teacher-summary", "output"):
        p.add_argument("--"+field,type=Path,required=True)
    p.add_argument("--count",type=int,default=128)
    args = p.parse_args()
    if args.count <= 0 or args.count % 16:
        p.error("Use positive complete 16-prior teacher groups")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    metadata = json.loads(args.teacher_summary.read_bytes())
    spec = GraphSpec.from_dict(metadata["spec"])
    model, state = load_model(args.checkpoint,"cpu",spec)
    model.eval().requires_grad_(False)
    rows = teacher_panel(args.teacher_graphs,args.count)
    ref = metadata["checkpoints"][args.teacher_graphs.name.removesuffix(".jsonl.gz")]
    n = ref["populations"]["prior_argmax"]["draws"]
    z = torch.randn(n,model.config.latent,generator=torch.Generator().manual_seed(metadata["seed"]))
    groups = []
    with rdBase.BlockLogs():
        for k in range(0,args.count,16):
            hard, valid = teacher_targets(rows[k:k+16],spec)
            with torch.no_grad():
                preds = [model.decode(block) for block in z[k:k+16].split(4)]
                decoded = DecoderOutput(*(torch.cat([getattr(p,field) for p in preds]).double()
                                          for field in ("atoms","charges","bonds")))
            groups.append(replay_group(decoded,hard,valid,spec))
    report = dict(checkpoint=file_sha256(args.checkpoint),update=state["continuation"]["updates"],
                  teacher_graphs=file_sha256(args.teacher_graphs),teacher_seed=metadata["seed"],
                  draws=args.count,batch=16,torch=torch.__version__,rdkit=rdBase.rdkitVersion,groups=groups)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(report,stream,indent=2)
    print(json.dumps(dict(masked=sum(r["masked_pairs"] for r in groups),
        native=statistics.mean(r["native_aro_direction"] for r in groups),
        treatment=statistics.mean(r["treatment_aro_direction"] for r in groups),
        median_seconds=statistics.median(r["selected_seconds"] for r in groups))))


if __name__ == "__main__":
    main()
