"""Compare corpus-frequency or ring-size guidance on short, matched checkpoint forks."""

import argparse
import csv
import gc
import gzip
import json
from pathlib import Path
import time

import numpy as np
from rdkit import rdBase
import torch

from experiments.vae.training_data_eda.teacher_motif_training_pilot import digest_rng, encode, evaluate
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256, write_json
from latentmol.train import learning_rate_at_epoch, state_digest
from latentmol.training.combined_continuation import _fresh
from latentmol.training.projected_continuation import projected_update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint", "receipt", "teacher", "config", "control-config", "real-panel", "heldout", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--updates", type=int, default=2)
    parser.add_argument("--saved-corpus-control", type=Path,
                        help="For a ring-only comparison reuse the identical archived corpus-frequency arm.")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_bytes())
    ctl = json.loads(args.control_config.read_bytes())
    old = ctl.pop("functional_groups")
    if ctl != {k: v for k, v in cfg.items() if k != "functional_groups"}:
        raise ValueError("Both arms must use the same base training recipe")
    old.pop("activation_gate", None)
    new = dict(cfg["functional_groups"])
    gate = new.pop("activation_gate", None)
    treatment = "ring" if "ring_size" in new else "corpus"
    changed = {"ring_size"} if treatment == "ring" else {"aromatic_floor", "amide_floor", "score_draws"}
    if ({k: v for k, v in new.items() if k not in changed} != {k: v for k, v in old.items() if k not in changed}
            or (gate and gate["control"] != old) or (args.saved_corpus_control and treatment != "ring")):
        raise ValueError("Mismatched policies for the selected comparison")
    receipt = json.loads(args.receipt.read_bytes())
    if (file_sha256(args.checkpoint) != receipt["sha256"] or args.updates < 1 or
            file_sha256(args.teacher) != cfg["prior_only"]["guidance"]["reference"]["sha256"]):
        raise ValueError("Unexpected checkpoint or training budget")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(3)
    parent, state = load_model(args.checkpoint, "cpu")
    reference, _ = load_model(args.teacher, "cuda", parent.spec)
    reference.eval().requires_grad_(False)
    source_digest, teacher_digest = (state_digest(m.state_dict()) for m in (parent, reference))
    start = state["continuation"]["updates"]
    batch, micro = cfg["batch_size"], cfg["microbatch"]
    if start != receipt["updates"] or parent.config.extra_bond_rounds != 3:
        raise ValueError("Checkpoint does not match the seven-round starting receipt")
    seed = 2026112253
    with gzip.open(args.real_panel, "rt") as f:
        train = [r["smiles"] for r in csv.DictReader(f, delimiter="\t")]
    holdout = [r[2] for r in json.loads(args.heldout.read_bytes())["rows"]]
    if set(holdout).intersection(train):
        raise ValueError("Training and evaluation SMILES overlap")
    ids = np.random.default_rng(seed).choice(len(train), batch * args.updates, replace=False)
    real = [encode([train[i] for i in group], parent.spec) for group in ids.reshape(-1, batch)]
    vrows = np.random.default_rng(seed + 1).choice(holdout, 128, replace=False).tolist()
    val = [(encode(vrows[i:i+8], parent.spec), vrows[i:i+8]) for i in range(0, len(vrows), 8)]
    latents = torch.randn(1024, parent.config.latent, generator=torch.Generator().manual_seed(seed+2))
    report = dict(start=start, checkpoint_sha256=receipt["sha256"], source_panel=file_sha256(args.real_panel),
                  seed=seed, updates=args.updates, batch=batch, microbatch=micro, result={},
                  device=torch.cuda.get_device_name(), torch=torch.__version__, rdkit=rdBase.rdkitVersion)
    arms = [("control", old), (treatment, new)]
    archived_logs = None
    if args.saved_corpus_control:
        saved = json.loads(args.saved_corpus_control.read_bytes())
        archived = saved["result"]["corpus"]
        if any(saved[k] != v for k,v in report.items() if k != "result") or archived["policy"] != old:
            raise ValueError("Archived corpus arm has different inputs, device, budget or policy")
        archived_logs = [json.loads(line) for line in
                         (args.saved_corpus_control.parent / "corpus-updates.jsonl").read_text().splitlines()]
        if len(archived_logs) != args.updates:
            raise ValueError("Incomplete archived training log")
        report["saved_control"] = dict(path=str(args.saved_corpus_control), sha256=file_sha256(args.saved_corpus_control))
        report["result"]["control"] = archived
        arms = [(treatment, new)]
    for arm, policy in arms:
        settings = dict(cfg, functional_groups=policy)
        model, opt, training, gens = _fresh(parent, state, None, batch, "cuda", None)
        if state_digest(model.state_dict()) != source_digest or training.loss.atom_order != cfg["atom_order"]:
            raise ValueError("Wrong starting weights or reconstruction objective")
        for group in opt.param_groups:
            group["lr"] = learning_rate_at_epoch(training, 48)
        torch.cuda.synchronize()
        events = []
        with rdBase.BlockLogs(), (args.output / (arm + "-updates.jsonl")).open("x") as f:
            for i, hard in enumerate(real):
                begin = time.monotonic()
                m = projected_update(settings, model, reference, opt, hard, training, None, None,
                                     gens, start + i, 48, micro)
                torch.cuda.synchronize()
                m.update(update=start+i+1, seconds=time.monotonic()-begin)
                if archived_logs is not None:
                    ref = archived_logs[i]
                    if "functional_groups" not in ref:
                        if {k:v for k,v in ref.items() if k != "seconds"} != {k:v for k,v in m.items() if k != "seconds"}:
                            raise ValueError("Unscheduled training update differs from the matched fork")
                    else:
                        obs = m["functional_groups"]
                        if (obs["ring_size_weight"] != policy["ring_size"]["weight"] or
                                not np.isclose(obs["penalty"], obs["motif_shortfall_penalty"] +
                                               obs["ring_size_weight"] * obs["ring_size_penalty"])):
                            raise ValueError("Ring-size penalty was not included at its specified weight")
                f.write(json.dumps(m, allow_nan=False) + "\n")
                f.flush()
                events.append(m)
                print(arm, m["update"], round(m["seconds"], 2), flush=True)
            rng = digest_rng(gens)
            _, score = evaluate(model, latents, val, args.output / (arm+"-panel.jsonl.gz"), categorical_seed=seed+3)
        aux = [m["functional_groups"] for m in events if "functional_groups" in m]
        if not aux or any(m["draws"] != policy.get("score_draws", policy["draws"]) for m in aux):
            raise RuntimeError("Expected a scheduled training signal at the configured sample count")
        report["result"][arm] = dict(policy=policy, panel=score, rng=rng, scheduled=len(aux))
        if state_digest(reference.state_dict()) != teacher_digest or state_digest(parent.state_dict()) != source_digest:
            raise RuntimeError("Frozen source or reference changed")
        del model, opt
        gc.collect()
        torch.cuda.empty_cache()
    if report["result"]["control"]["rng"] != report["result"][treatment]["rng"]:
        raise RuntimeError("The training arms consumed different base random streams")
    write_json(args.output / "summary.json", report)


if __name__ == "__main__":
    main()
