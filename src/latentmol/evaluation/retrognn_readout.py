"""Evaluate the released Molecule.one RetroGNN classifier on a blind SMILES panel."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
from statistics import median

import torch


WEIGHTS_SHA256 = "32d33e8b569b7a5ce26aeaa5aec2c9c5f67a1ce2e5bb798a427e16266b85884a"
REFERENCE_SHA256 = "9dfb591fe12e8011c6777cdc288c0f618789c80f822f05e93ae1ab6191d19467"
CHEMPROP_REVISION = "7d173c134b286acd4c039bf5cdb80a642fb66c47"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_panel(path):
    """Sample released examples to check score direction independently of the blind panel."""
    if sha256(path) != REFERENCE_SHA256:
        raise ValueError("Unexpected RetroGNN reference dataset")
    groups = {False: [], True: []}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            score = float(row["Score"])
            if score == 0 or score >= 8:
                groups[score > 0].append((row["Smiles"], score))
    rng = random.Random(171117)
    return [dict(id=f"ref-{int(solved)}-{i:02d}", smiles=smiles, published_score=score)
            for solved, group in groups.items()
            for i, (smiles, score) in enumerate(rng.sample(group, 12))]


def run(panel_path, checkpoint, output, reference=None):
    from chemprop.models import MoleculeModel
    from rdkit import Chem

    if output.exists():
        raise FileExistsError(output)
    if sha256(checkpoint) != WEIGHTS_SHA256:
        raise ValueError("Unexpected M1 classification checkpoint")
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    controls = reference_panel(reference) if reference else []
    rows = panel + controls
    if not panel or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("A nonempty panel with unique IDs is required")
    for row in rows:
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None or not mol.GetNumAtoms():
            raise ValueError(f"Invalid panel molecule {row['id']}")
    with torch.serialization.safe_globals([argparse.Namespace]):
        state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    args = state["args"]
    args.device, args.cuda = torch.device("cpu"), False
    model = MoleculeModel(args)
    model.load_state_dict(state["state_dict"], strict=True)
    model.eval().requires_grad_(False)

    # Preserve pre-sigmoid ranking even where float32 probabilities saturate.
    logits = []
    handle = model.ffn.register_forward_hook(lambda module, inputs, result: logits.append(result))
    predictions = []
    try:
        with torch.inference_mode():
            for start in range(0, len(rows), 16):
                batch = rows[start:start + 16]
                prob = model([row["smiles"] for row in batch])
                logit = logits.pop()
                if (prob.shape != (len(batch), 1) or not logit.isfinite().all()
                        or not torch.equal(prob, logit.sigmoid())):
                    raise ValueError("Inconsistent RetroGNN classifier output")
                predictions.extend(dict(row, class1_logit=l.item(), class1_score=p.item())
                                   for row, l, p in zip(batch, logit, prob))
    finally:
        handle.remove()
    check_rows = predictions[len(panel):]
    check = {}
    if controls:
        zero = [r["class1_logit"] for r in check_rows if r["published_score"] == 0]
        positive = [r["class1_logit"] for r in check_rows if r["published_score"] > 0]
        check = dict(zero_score_median_logit=median(zero), high_score_median_logit=median(positive),
                     fraction_high_ranked_above_zero=sum((x > y) + 0.5 * (x == y)
                         for x in positive for y in zero) / (len(positive) * len(zero)))
    report = dict(model="RetroGNN m1_classification", checkpoint_sha256=WEIGHTS_SHA256,
                  chemprop_revision=CHEMPROP_REVISION, panel_sha256=sha256(panel_path),
                  predictions=predictions[:len(panel)], reference=check_rows, direction_check=check)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    opts = parser.parse_args()
    torch.set_num_threads(2)
    run(opts.panel, opts.checkpoint, opts.output, opts.reference_data)
