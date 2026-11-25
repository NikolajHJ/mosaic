"""Replay posterior means and isolate hard graph errors in the canonical training slots."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np
from rdkit import rdBase
import torch

from latentmol.chemistry import inspect_graph
from latentmol.design.same_molecule_starts import CHECKPOINT_597604
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256
from latentmol.guidance.valence_projection import projection_targets
from experiments.vae.training_data_eda.teacher_motif_training_pilot import encode


def evaluate_labels(labels, spec, target):
    checked = inspect_graph(*labels, spec)
    return dict(valid=checked.smiles is not None, exact=checked.smiles == target,
                reason=checked.failure_reason, detail=checked.detail)


def diagnose(pred, ref, spec, smiles):
    atoms, charges, bonds = pred
    target_atoms, target_charges, target_bonds = ref
    extra = (bonds != 0) & (target_bonds == 0)
    missing = (bonds == 0) & (target_bonds != 0)
    order = (bonds != 0) & (target_bonds != 0) & (bonds != target_bonds)
    # These substitutions use the known parent to localize errors. They are
    # unavailable during generation and use decoder slots, not graph edit distance.
    variants = {
        "raw": pred,
        "given_atom_types_and_charges": (target_atoms, target_charges, bonds),
        "given_bonds": (atoms, charges, target_bonds),
        "remove_extra_bonds": (atoms, charges, np.where(extra, 0, bonds)),
        "restore_missing_bonds": (atoms, charges, np.where(missing, target_bonds, bonds)),
        "restore_bond_orders": (atoms, charges, np.where(order, target_bonds, bonds)),
    }
    return dict(variants={key: evaluate_labels(labels, spec, smiles) for key, labels in variants.items()},
                atoms=int(np.count_nonzero(target_atoms)),
                wrong_atom_slots=int(np.count_nonzero(atoms != target_atoms)),
                wrong_charge_slots=int(np.count_nonzero((charges != target_charges) & (target_atoms != 0))),
                extra_bonds=int(np.triu(extra, 1).sum()), missing_bonds=int(np.triu(missing, 1).sum()),
                wrong_bond_orders=int(np.triu(order, 1).sum()))


def predicted_bond_variants(decoded, hard, spec):
    """Three readouts from the predictions alone, including the teacher used in training."""
    i, j = np.triu_indices(spec.max_nodes, 1)
    logits = decoded.symmetric_bond_logits[:, i, j].float().cpu().numpy()
    atoms, charges = hard.atoms.numpy(), hard.charges.numpy()
    _, projected, _ = projection_targets(atoms, charges, logits, spec, max_edits=8)
    result = {"valence_projection_8": projected}
    for factor in (2, 4):
        shifted = logits.copy()
        shifted[..., 0] += math.log(factor)
        result[f"no_bond_odds_x{factor}"] = shifted.argmax(-1)
    for name, upper in result.items():
        active = (atoms[:, i] != 0) & (atoms[:, j] != 0)
        matrix = np.zeros_like(hard.bonds.numpy())
        matrix[:, i, j] = matrix[:, j, i] = np.where(active, upper, 0)
        yield name, matrix


def summarize(rows):
    variants = {}
    for key in rows[0]["variants"]:
        results = [r["variants"][key] for r in rows]
        variants[key] = dict(valid=sum(x["valid"] for x in results), exact=sum(x["exact"] for x in results),
                             failures=Counter(x["detail"] or x["reason"] for x in results if not x["valid"]))
    errs = {key: dict(rows=sum(r[key] > 0 for r in rows), total=sum(r[key] for r in rows))
            for key in ("wrong_atom_slots", "wrong_charge_slots", "extra_bonds", "missing_bonds", "wrong_bond_orders")}
    return dict(n=len(rows), variants=variants, errors=errs)


@torch.inference_mode()
def replay(checkpoint, panel_path, heldout_path, device):
    panel = json.loads(panel_path.read_bytes())
    if file_sha256(checkpoint) != CHECKPOINT_597604 or panel["checkpoint_sha256"] != CHECKPOINT_597604:
        raise ValueError("Requires checkpoint 597604 and its prior screening panel")
    model, state = load_model(checkpoint, device)
    if state["continuation"]["updates"] != 597604:
        raise ValueError("Checkpoint update mismatch")
    model.eval().requires_grad_(False)
    priors = [dict(cohort="generated", index=a["index"], smiles=a["smiles"], prior_seed=a["prior_seed"],
                   cpu_mean_exact=a["stage"] == "posterior_sample") for a in panel["attempts"]
              if a["stage"] in ("reencoded_mean", "posterior_sample")]
    heldout = [dict(cohort="heldout", index=i, smiles=s) for i, _, s in json.loads(heldout_path.read_bytes())["rows"]]
    rows = []
    original_priors = []
    for cohort in (priors, heldout):
        for start in range(0, len(cohort), 8):
            batch = cohort[start:start+8]
            truth = encode([r["smiles"] for r in batch], model.spec)
            mu, _ = model.encoder(truth.to(device), from_canonical_cache=True)
            decoded = model.decode(mu)
            if not all(x.isfinite().all() for x in (mu, decoded.atoms, decoded.charges, decoded.bonds)):
                raise FloatingPointError("Nonfinite encoding or decoding")
            hard = decoded.harden(model.spec).to("cpu")
            alternatives = dict(predicted_bond_variants(decoded, hard, model.spec))
            for i, record in enumerate(batch):
                fields = ("atoms", "charges", "bonds")
                expected = [getattr(truth, k)[i].numpy() for k in fields]
                predicted = [getattr(hard, k)[i].numpy() for k in fields]
                diag = diagnose(predicted, expected, model.spec, record["smiles"])
                for name, bonds in alternatives.items():
                    diag["variants"][name] = evaluate_labels((*predicted[:2], bonds[i]), model.spec, record["smiles"])
                rows.append(dict(**record, **diag))
            if cohort is priors:
                # A bond-score correction that helps the re-encoding may destroy a valid prior decode.
                z = torch.stack([torch.randn(model.config.latent, generator=torch.Generator().manual_seed(r["prior_seed"]))
                                 for r in batch]).to(device)
                direct = model.decode(z)
                if not all(x.isfinite().all() for x in (direct.atoms, direct.charges, direct.bonds)):
                    raise FloatingPointError("Nonfinite prior decoding")
                direct_hard = direct.harden(model.spec).to("cpu")
                alt = dict(raw=direct_hard.bonds.numpy(), **dict(predicted_bond_variants(direct, direct_hard, model.spec)))
                for i, record in enumerate(batch):
                    results = {name: evaluate_labels((direct_hard.atoms[i], direct_hard.charges[i], bonds[i]),
                                                    model.spec, record["smiles"]) for name, bonds in alt.items()}
                    original_priors.append(dict(index=record["index"], variants=results))
    groups = {name: [r for r in rows if r["cohort"] == name] for name in ("generated", "heldout")}
    groups["heldout_25_atoms"] = [r for r in groups["heldout"] if r["atoms"] == 25]
    groups["generated_failed_sanitization"] = [r for r in groups["generated"]
                                                if r["variants"]["raw"]["reason"] == "sanitization"]
    return dict(update=597604, device=device, checkpoint_sha256=CHECKPOINT_597604,
                heldout_sha256=file_sha256(heldout_path), prior_panel_sha256=file_sha256(panel_path),
                original_priors=original_priors,
                original_prior_summary={k: {m: sum(r["variants"][k][m] for r in original_priors)
                                            for m in ("valid", "exact")} for k in original_priors[0]["variants"]},
                cpu_exact_disagreements=sum(r["cpu_mean_exact"] != r["variants"]["raw"]["exact"]
                                            for r in groups["generated"]), rows=rows,
                summary={name: summarize(items) for name, items in groups.items() if items})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for item in ("checkpoint", "panel", "heldout", "out"):
        ap.add_argument("--"+item, type=Path, required=True)
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cpu")
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(2)
    with rdBase.BlockLogs():
        out = replay(args.checkpoint, args.panel, args.heldout, args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(out, stream, allow_nan=False)
    print(json.dumps(out["summary"], indent=2))
