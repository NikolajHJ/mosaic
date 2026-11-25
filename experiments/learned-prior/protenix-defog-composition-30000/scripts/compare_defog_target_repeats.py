"""Compare target repeatability on the five archived one-query control graphs."""

import argparse
from itertools import combinations
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from rdkit import Chem, rdBase


def labels(graph):
    x, e = graph
    i, j = torch.triu_indices(x.shape[1], x.shape[1], 1)
    return torch.cat([x.argmax(-1), e[:, i, j].argmax(-1)], dim=1)


def pairwise_edits(rows):
    return [int((left != right).sum()) for left, right in combinations(rows, 2)]


def run(args):
    sys.path.insert(0, str(args.probe_release))
    import probe

    # The archived diagnostic imports its numerical dependencies inside main().
    probe.torch, probe.F, probe.Chem, probe.OmegaConf = torch, F, Chem, OmegaConf
    config = json.loads((args.probe_release / "config.json").read_text())
    model, _ = probe.load_model(args.assets / "DeFoG", args.assets / "guacamol.ckpt", config)
    frozen = probe.state_digest(model)
    if model.conditional or model.cfg.sample.time_distortion != "polydec":
        raise ValueError("Expected the historical unconditional GuacaMol sampling schedule")
    steps = 1000
    model.sample_T = model.cfg.sample.sample_steps = steps
    for name in ("eta", "omega", "rdb", "rdb_crit"):
        setattr(model.rate_matrix_designer, name, getattr(model.cfg.sample, name))
    grid = model.time_distorter.sample_ft(torch.arange(steps + 1)[:, None].float() / steps, "polydec")
    first = int((grid[:, 0] >= 0.8).nonzero()[0].item())
    if first != 553 or float(grid[-1]) != 1 or not (grid[1:] > grid[:-1]).all():
        raise ValueError("Historical grid mismatch")
    args.output.mkdir(exist_ok=False)
    teacher = probe.Probe(model, config, args.output)
    count = 8
    summary = {"start_time": float(grid[first]), "suffix_steps": steps - first,
               "batch_draws": count, "checkpoint_sha256": config["checkpoint_sha256"],
               "source_commit": config["source_commit"], "controls": []}
    with rdBase.BlockLogs(), torch.inference_mode():
        for index, control in enumerate(config["controls"]):
            start = probe.make_control(control)
            repeat_outputs = []
            for k in range(count):
                torch.manual_seed(23000 + k)
                target, receipt = teacher.query("hard_aux_soft_core", start, float(grid[first]))
                if target is None:
                    raise ValueError(receipt["error"])
                repeat_outputs.append(target)
            soft_edits = pairwise_edits(torch.cat([labels(pair) for pair in repeat_outputs]))
            max_repeat_tv = max(probe.measure(a, b)["max_cell_tv"] for a, b in combinations(repeat_outputs, 2))
            # Same initial graph/time in all eight lanes. No corruption, selection or retries.
            x, e = [v.repeat((count,) + (1,) * (v.ndim - 1)) for v in start]
            mask = torch.ones(x.shape[:2], dtype=torch.bool)
            y = x.new_zeros((count, 0))
            torch.manual_seed(72100 + index)
            before = teacher.forwards
            for k in range(first, steps):
                t, s = [grid[u].expand(count, 1).contiguous() for u in (k, k + 1)]
                out, _ = model.sample_p_zs_given_zt(t, s, x, e, y, mask)
                x, e, y = out.X, out.E, out.y
                if not torch.isfinite(x).all() or not torch.isfinite(e).all():
                    raise ValueError("Nonfinite native sampling output")
            if teacher.forwards - before != steps - first:
                raise ValueError("Unexpected number of native denoiser calls")
            terminal = labels((x, e))
            hard_edits = pairwise_edits(terminal)
            i, j = torch.triu_indices(x.shape[1], x.shape[1], 1)
            tx, te = repeat_outputs[0]
            row = dict(control=control["id"], sample_seed=72100 + index, n_atoms=x.shape[1],
                       suffix_labels=terminal.tolist(), input_labels=labels(start)[0].tolist(),
                       soft_atom_probabilities=tx[0].tolist(), soft_bond_probabilities=te[0,i,j].tolist(),
                       suffix_readouts=[probe.raw_readout(x[k:k+1],e[k:k+1]) for k in range(count)],
                       soft_readout=probe.raw_readout(tx,te), suffix_pair_edits=hard_edits,
                       soft_argmax_pair_edits=soft_edits, soft_max_repeat_tv=max_repeat_tv)
            summary["controls"].append(row)
            print(control["id"], "differing endpoint pairs", sum(n > 0 for n in hard_edits),
                  "/", len(hard_edits), "max soft TV", max_repeat_tv, flush=True)
    if probe.state_digest(model) != frozen:
        raise RuntimeError("Frozen checkpoint changed")
    summary["frozen_state_sha256"] = frozen
    probe.write_json(args.output / "results.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("probe-release", "assets", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    run(parser.parse_args())
