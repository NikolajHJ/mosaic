"""How much of the prior deficit can a uniform inference-only shift in the aromatic label recover?"""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch
from rdkit import rdBase

from generation_motifs import inspect_batch, motifs
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256
from latentmol.representation import DecoderOutput


@torch.inference_mode()
def probe(path, n, seed):
    model, _ = load_model(path, "cpu")
    model.eval()
    z = torch.randn(n, model.config.latent, generator=torch.Generator().manual_seed(seed))
    aromatic = model.spec.bond_names.index("AROMATIC")
    counts = {shift: Counter() for shift in (0., 1., 2., 4.)}
    for piece in z.split(12):
        output = model.decode(piece)
        for shift, counter in counts.items():
            shifted = output.bonds.clone()
            shifted[..., aromatic] += shift
            hard = DecoderOutput(output.atoms, output.charges, shifted).harden(model.spec)
            rows = list(inspect_batch(hard, None, model.spec))
            valid = np.array([bool(r["smiles"]) for r in rows])
            aromatic_valid = np.array([bool(r["smiles"] and motifs(r["smiles"])["aromatic_ring"]) for r in rows])
            if shift == 0:
                baseline_valid, baseline_aromatic = valid, aromatic_valid
            counter.update(dict(draws=len(rows), valid=int(valid.sum()), valid_aromatic=int(aromatic_valid.sum()),
                newfound_aromatic=int((~baseline_aromatic & aromatic_valid).sum()),
                originally_valid_now_invalid=int((baseline_valid & ~valid).sum()),
                raw_any_aromatic_edge=sum(r["aromatic_edges"] > 0 for r in rows),
                raw_any_aromatic_cycle=sum(r["aromatic_cycle"] for r in rows)))
    return dict(sha256=file_sha256(path), shifts=counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026112047)
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.threads)
    with rdBase.BlockLogs():
        result = {p.stem: probe(p, args.count, args.seed) for p in args.checkpoint}
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(dict(seed=args.seed, count=args.count, checkpoints=result), out, indent=2)
    print(json.dumps(result), flush=True)
