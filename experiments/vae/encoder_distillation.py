"""Checkpoint 623116 pilot: amortize successful latent refinements with a frozen decoder."""

import argparse
import csv
from copy import deepcopy
import gzip
import json
from pathlib import Path
import random

from rdkit import rdBase
import torch
from torch.nn import functional as F

from experiments.vae.refined_latent_bank import build_teacher_bank
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256
from latentmol.representation import HardGraphBatch
from latentmol.train import state_digest
from latentmol.training.losses import LossConfig, reconstruction_losses


RECOVERY = Path("outputs/runs/e48-inverse-recovery-623116-20261124a")
CHECKPOINT = RECOVERY / "checkpoint-000623116.pt"
SHA256 = "504fccee8954f684bace24c84e3b04af239260927531248b3ce5f974725d5df8"
VAL = Path("outputs/runs/vae-review-20261119-47682960/pubchem-val-panel.json")
REAL_TRAIN = Path("outputs/runs/vae-training-data-eda-20261120a/inputs/panel.tsv.gz")
TEACHER_SEED, EVAL_SEED, TRAIN_SEED = 2026112470, 2026112471, 2026112472
STEPS, BATCH, STUDENT_LR = 128, 16, 0.00002


def write_json(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, allow_nan=False)


def prepare(model, config, output):
    output.mkdir(parents=True, exist_ok=False)
    prior_study_names = {r["target"] for r in json.loads((RECOVERY / "recovery.json").read_bytes())["parents"]}
    heldout = {r[2] for r in json.loads(VAL.read_bytes())["rows"]}
    bank = build_teacher_bank(model, config, seed=TEACHER_SEED, draws=384, batch_size=12,
                              exclude_smiles=prior_study_names | heldout)
    bank["source_sha256"] = SHA256
    with (output / "bank.pt").open("xb") as f:
        torch.save(bank, f)
    write_json(output / "bank-summary.json", bank["summary"])


def select(graph, rows):
    return HardGraphBatch(graph.atoms[rows], graph.charges[rows], graph.bonds[rows])


def bank_graph(bank):
    return HardGraphBatch(bank["atoms"], bank["charges"], bank["bonds"])


@torch.no_grad()
def real_anchors(model, exclude, device):
    from experiments.vae.training_data_eda.teacher_motif_training_pilot import encode

    with gzip.open(REAL_TRAIN, "rt") as f:
        names = list(dict.fromkeys(row["smiles"] for row in csv.DictReader(f, delimiter="\t")
                                   if row["smiles"] not in exclude))
    names = random.Random(TRAIN_SEED).sample(names, STEPS * BATCH)
    graph = encode(names, model.spec)
    means = []
    for start in range(0, len(names), BATCH):
        batch = select(graph, slice(start, start + BATCH)).to(device)
        means.append(model.encoder(batch, from_canonical_cache=True)[0].cpu())
    return graph, torch.cat(means), names


def paired(before, after):
    if [(r["source_row"], r["target"]) for r in before["rows"]] != [
        (r["source_row"], r["target"]) for r in after["rows"]
    ]:
        raise RuntimeError("Evaluation cohort changed")
    pairs = list(zip(before["rows"], after["rows"], strict=True))
    return dict(n=len(pairs), exact=after["exact"], valid=after["valid"], extra=after["extra"],
                missing=after["missing"], gained=sum(b["exact"] and not a["exact"] for a, b in pairs),
                lost=sum(a["exact"] and not b["exact"] for a, b in pairs))


@torch.no_grad()
def teacher_fit(student, bank, rows, device):
    """Measure learning on the actual teacher examples separately from fresh-panel performance."""
    from latentmol.training.occupancy_monitor import _inspect

    generated = bank_graph(bank)
    exact = {"anchor": 0, "rescued": 0}
    starts, errors, cosines = [], [], []
    for block in rows.split(BATCH):
        pred, _ = student.encoder(select(generated, block).to(device), from_canonical_cache=True)
        observed = _inspect(student.decode(pred).harden(student.spec), student.spec)
        for i, item in zip(block.tolist(), observed, strict=True):
            exact[bank["kind"][i]] += int(item.smiles == bank["smiles"][i])
        moved = bank["hit_steps"][block] > 0
        if moved.any():
            origin = bank["mu0"][block][moved]
            target = bank["zteacher"][block][moved]
            prediction = pred.cpu()[moved]
            starts.extend((origin - target).norm(dim=-1).tolist())
            errors.extend((prediction - target).norm(dim=-1).tolist())
            cosines.extend(F.cosine_similarity(prediction - origin, target - origin).tolist())
    return dict(exact=exact, rescued=len(starts), anchors=len(rows)-len(starts),
                rescued_mean_distance_before=sum(starts)/len(starts) if starts else None,
                rescued_mean_distance_after=sum(errors)/len(errors) if errors else None,
                rescued_mean_direction_cosine=sum(cosines)/len(cosines) if cosines else None)


def fit_arm(base, bank, indices, replay, replay_mu, config, panels, baseline, arm, dest, device):
    from experiments.vae.encoder_distill_panel import evaluate_panels

    student = deepcopy(base)
    student.encoder.requires_grad_(True)
    student.encoder.logvar.requires_grad_(False)
    params = [p for p in student.encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=STUDENT_LR)
    generated = bank_graph(bank)
    tracking, snapshots = [], {}
    for i, rows in enumerate(indices, start=1):
        targets = select(generated, rows).to(device)
        real_rows = slice((i - 1) * BATCH, i * BATCH)
        pred, _ = student.encoder(targets, from_canonical_cache=True)
        real_pred, _ = student.encoder(select(replay, real_rows).to(device), from_canonical_cache=True)
        teacher = bank["zteacher"][rows].to(device)
        graph_loss = reconstruction_losses(student.decode(pred), targets, config)["reconstruction"]
        latent_loss = F.mse_loss(pred, teacher)
        anchor_loss = F.mse_loss(real_pred, replay_mu[real_rows].to(device))
        loss = graph_loss + 2 * anchor_loss
        if arm == "distilled":
            loss = loss + 2 * latent_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite encoder loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if student.encoder.mu.weight.grad is None:
            raise RuntimeError("Encoder mean detached")
        grad = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        optimizer.step()
        tracking.append(dict(update=i, graph=float(graph_loss.detach()), teacher_mse=float(latent_loss.detach()),
                             real_mse=float(anchor_loss.detach()), grad_norm=float(grad)))
        if i in (32, STEPS):
            scored = evaluate_panels(student, panels, batch=BATCH)
            snapshots[i] = {k: paired(baseline[k], v) for k, v in scored.items()}
            write_json(dest / f"{arm}-{i}-rows.json", scored)
            print(arm, i, snapshots[i], flush=True)
    if (any(p.grad is not None for p in student.decoder.parameters())
            or state_digest(student.decoder.state_dict()) != state_digest(base.decoder.state_dict())
            or state_digest(student.encoder.logvar.state_dict()) != state_digest(base.encoder.logvar.state_dict())):
        raise RuntimeError("Frozen decoder or posterior variance head changed")
    write_json(dest / f"{arm}-teacher-fit.json", teacher_fit(student, bank, indices.unique(), device))
    with (dest / f"{arm}-encoder.pt").open("xb") as f:
        torch.save(dict(source_sha256=SHA256, encoder=student.encoder.cpu().state_dict()), f)
    write_json(dest / f"{arm}-train.json", tracking)
    return snapshots


def fit(model, config, output, device):
    from experiments.vae.encoder_distill_panel import build_panels, evaluate_panels

    dest = output / "comparison"
    dest.mkdir(exist_ok=False)
    bank = torch.load(output / "bank.pt", map_location="cpu", weights_only=True)
    if bank["source_sha256"] != SHA256 or bank["summary"]["seed"] != TEACHER_SEED:
        raise ValueError("Unexpected teacher bank")
    panels, excluded, provenance = build_panels(model, VAL, seed=EVAL_SEED, draws=512, heldout_count=512, batch=BATCH)
    available = [i for i, name in enumerate(bank["smiles"]) if name not in excluded]
    if len(available) < BATCH or len(set(bank["smiles"])) != len(bank["smiles"]):
        raise ValueError("Insufficient independent teacher targets")
    # Both arms see the same shuffled passes over teacher-qualified molecules and identical real replay.
    rng, order = random.Random(TRAIN_SEED), []
    while len(order) < STEPS * BATCH:
        order.extend(rng.sample(available, len(available)))
    indices = torch.tensor(order[:STEPS * BATCH]).reshape(STEPS, BATCH)
    seen = {r["smiles"] for r in bank["screening"]}
    replay, replay_mu, replay_names = real_anchors(model, seen | excluded, device)
    write_json(dest / "sampling.json", dict(real_replay=replay_names, bank_indices=indices.tolist(),
                                            panels=provenance))
    baseline = evaluate_panels(model, panels, batch=BATCH)
    write_json(dest / "baseline-rows.json", baseline)
    results = {arm: fit_arm(model, bank, indices, replay, replay_mu, config, panels, baseline, arm, dest, device)
               for arm in ("distilled", "graph-only")}
    endpoint = {arm: json.loads((dest / f"{arm}-{STEPS}-rows.json").read_bytes()) for arm in results}
    comparison = {k: paired(endpoint["graph-only"][k], v) for k, v in endpoint["distilled"].items()}
    result = dict(checkpoint=623116, source_sha256=SHA256, teacher=bank["summary"], teachers_used=len(available),
                  teacher_eval_overlap=len(bank["smiles"])-len(available), seeds=[TEACHER_SEED,EVAL_SEED,TRAIN_SEED],
                  heldout_panel_sha256=file_sha256(VAL), real_train_source_sha256=file_sha256(REAL_TRAIN),
                  batch=BATCH, steps=STEPS, lr=STUDENT_LR, real_mean_weight=2., refined_mean_weight=2.,
                  device=device, gpu=torch.cuda.get_device_name() if device=="cuda" else None,
                  frozen_decoder_verified=True, frozen_logvar_head_verified=True,
                  baseline={k: paired(v,v) for k,v in baseline.items()}, results=results,
                  distilled_vs_graph_only=comparison)
    write_json(dest / "summary.json", result)
    print("Completed", comparison, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "fit"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if file_sha256(CHECKPOINT) != SHA256:
        raise ValueError("Incorrect checkpoint digest")
    torch.set_num_threads(2)
    model, state = load_model(CHECKPOINT, "cpu")
    if state["continuation"]["updates"] != 623116:
        raise ValueError("Unexpected source update")
    model.eval().requires_grad_(False).to(args.device)
    loss_config = LossConfig(**state["loss_config"])
    with rdBase.BlockLogs():
        if args.stage == "prepare":
            prepare(model, loss_config, args.output)
        else:
            fit(model, loss_config, args.output, args.device)
