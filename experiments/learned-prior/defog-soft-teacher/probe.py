"""Allocated-HPC-only diagnostic of the original GuacaMol DeFoG teacher.

Each successful query makes exactly one frozen denoiser forward. No sampler,
flow trajectory or autograd through the teacher is used. Alternative feature
policies are separately labeled and never represented as native soft support.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback

ATOMS = (6, 7, 8, 9, 5, 35, 17, 53, 15, 16, 34, 14)
MODES = ("native_soft", "hard_view", "hard_aux_soft_core")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def validate_config(c):
    if c["version"] != "defog-one-query-locality-v1" or c["modes"] != list(MODES):
        raise ValueError("Unknown diagnostic contract")
    if not (0 < c["trajectory_step_size"] < 1 and 0 < c["tie_delta"] < 0.1):
        raise ValueError("Invalid local update/tie sizes")
    if not all(0 < v < 1 for v in c["times"] + c["epsilons"]):
        raise ValueError("Times and perturbations must lie inside (0,1)")
    if not all(0 <= v < 1 for v in c["softness"]):
        raise ValueError("Invalid softness")
    if not (1 <= c["trajectory_steps"] <= 20 and 1 <= c["max_queries"] <= 2500
            and 1 <= c["max_seconds"] <= 2400):
        raise ValueError("Probe budget exceeds bounded scope")
    if not c["controls"] or len(c["controls"]) > 8:
        raise ValueError("Invalid cohort")


def validate_graph(x, e):
    if x.ndim != 3 or x.shape[0] != 1 or x.shape[-1] != 12:
        raise ValueError("Expected one native 12-element graph")
    n = x.shape[1]
    if e.shape != (1, n, n, 5) or n < 2 or n > 32:
        raise ValueError("Unsupported count/shape")
    off = ~torch.eye(n, dtype=torch.bool)
    diag = torch.arange(n)
    edges = e[:, off, :]
    if not all(torch.isfinite(v).all() for v in (x, e)) or min(float(x.min()), float(e.min())) < -1e-7:
        raise ValueError("Nonfinite or negative probabilities")
    if not torch.allclose(x.sum(-1), torch.ones_like(x[..., 0]), atol=1e-6, rtol=0):
        raise ValueError("Off-simplex atoms")
    if not torch.allclose(edges.sum(-1), torch.ones_like(edges[..., 0]), atol=1e-6, rtol=0):
        raise ValueError("Off-simplex bonds")
    if bool(e[:, diag, diag].any()) or not torch.equal(e, e.transpose(1, 2)):
        raise ValueError("Asymmetric bonds or nonzero diagonal")


def hard_view(x, e):
    hx = F.one_hot(x.argmax(-1), 12).float()
    he = F.one_hot(e.argmax(-1), 5).float()
    d = torch.arange(x.shape[1])
    he[:, d, d] = 0
    return hx, he


def measure(a, b):
    ax, ae = a
    bx, be = b
    upper = torch.triu_indices(ax.shape[1], ax.shape[1], 1)
    atv = (ax - bx).abs().sum(-1).flatten() / 2
    btv = (ae[:, upper[0], upper[1]] - be[:, upper[0], upper[1]]).abs().sum(-1).flatten() / 2
    return {"atom_mean_tv": float(atv.mean()), "bond_mean_tv": float(btv.mean()),
            "decision_mean_tv": float(torch.cat((atv, btv)).mean()),
            "max_cell_tv": float(torch.cat((atv, btv)).max()),
            "atom_argmax_edits": int((ax.argmax(-1) != bx.argmax(-1)).sum()),
            "bond_argmax_edits": int((ae[:, upper[0], upper[1]].argmax(-1)
                                      != be[:, upper[0], upper[1]].argmax(-1)).sum())}


def make_control(row):
    mol = Chem.MolFromSmiles(row["smiles"])
    if mol is None or any(a.GetFormalCharge() for a in mol.GetAtoms()):
        raise ValueError("Controls must be parseable and neutral")
    n = mol.GetNumAtoms()
    x = F.one_hot(torch.tensor([[ATOMS.index(a.GetAtomicNum()) for a in mol.GetAtoms()]]), 12).float()
    labels = torch.zeros((1, n, n), dtype=torch.long)
    bond_types = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2,
                  Chem.BondType.TRIPLE: 3, Chem.BondType.AROMATIC: 4}
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        labels[0, i, j] = labels[0, j, i] = bond_types[bond.GetBondType()]
    if row.get("invalid_star"):
        # Six carbon atoms, center degree five: representable, chemically invalid.
        labels.zero_()
        labels[0, 0, 1:] = labels[0, 1:, 0] = 1
    e = F.one_hot(labels, 5).float()
    d = torch.arange(n)
    e[:, d, d] = 0
    validate_graph(x, e)
    return x, e


def raw_readout(x, e):
    """Neutral interpretation only; the checkpoint has no formal-charge head."""
    try:
        mol = Chem.RWMol()
        for label in x.argmax(-1)[0].tolist():
            mol.AddAtom(Chem.Atom(ATOMS[label]))
        types = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
                 3: Chem.BondType.TRIPLE, 4: Chem.BondType.AROMATIC}
        labels = e.argmax(-1)[0]
        for i in range(x.shape[1]):
            for j in range(i + 1, x.shape[1]):
                label = int(labels[i, j])
                if label:
                    mol.AddBond(i, j, types[label])
                    if label == 4:
                        mol.GetAtomWithIdx(i).SetIsAromatic(True)
                        mol.GetAtomWithIdx(j).SetIsAromatic(True)
        mol = mol.GetMol()
        Chem.SanitizeMol(mol)
        return {"neutral_sanitized": True, "connected": len(Chem.GetMolFrags(mol)) == 1,
                "smiles": Chem.MolToSmiles(mol)}
    except Exception as exc:
        return {"neutral_sanitized": False, "connected": None, "smiles": None,
                "error": str(exc)[:250]}


def noisy(x, e, t):
    return {"X_t": x, "E_t": e, "y_t": x.new_zeros((1, 0)),
            "node_mask": torch.ones(x.shape[:2], dtype=torch.bool),
            "t": x.new_full((1, 1), t)}


def perturb(x, e, kind, epsilon):
    px, pe = x.clone(), e.clone()
    if kind == "atom":
        target = (int(x[0, 0].argmax()) + 1) % 12
        px[0, 0] *= 1 - epsilon
        px[0, 0, target] += epsilon
    elif kind == "bond":
        # The same atom pair is used at every scale; symmetry is preserved.
        target = (int(e[0, 0, 1].argmax()) + 1) % 5
        pe[0, 0, 1] *= 1 - epsilon
        pe[0, 0, 1, target] += epsilon
        pe[0, 1, 0] = pe[0, 0, 1]
    else:
        raise ValueError(kind)
    validate_graph(px, pe)
    return px, pe


def tie_pair(x, e, kind, delta):
    pair = []
    for sign in (-1, 1):
        px, pe = x.clone(), e.clone()
        if kind == "atom":
            first = int(x[0, 0].argmax())
            second = (first + 1) % 12
            px[0, 0].zero_()
            px[0, 0, first], px[0, 0, second] = 0.5 + sign * delta, 0.5 - sign * delta
        else:
            first = int(e[0, 0, 1].argmax())
            second = (first + 1) % 5
            pe[0, 0, 1].zero_()
            pe[0, 0, 1, first], pe[0, 0, 1, second] = 0.5 + sign * delta, 0.5 - sign * delta
            pe[0, 1, 0] = pe[0, 0, 1]
        validate_graph(px, pe)
        pair.append((px, pe))
    return pair


def state_digest(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str((value.shape, value.dtype)).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def load_model(source, checkpoint, c):
    if digest(checkpoint) != c["checkpoint_sha256"]:
        raise ValueError("Checkpoint hash mismatch")
    head = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip()
    if head != c["source_commit"] or dirty:
        raise ValueError("Upstream source identity differs")
    sys.path[:0] = [str(source), str(source / "src")]
    from graph_discrete_flow_model import GraphDiscreteFlowModel
    model = GraphDiscreteFlowModel.load_from_checkpoint(str(checkpoint), map_location="cpu", strict=False,
        train_metrics=None, sampling_metrics=None, visualization_tools=None)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    loaded = model.state_dict()
    omitted = {"sampling_metrics.n_target_dist", "sampling_metrics.node_target_dist",
               "sampling_metrics.edge_target_dist", "sampling_metrics.valency_target_dist"}
    if set(saved) - set(loaded) != omitted or set(loaded) - set(saved):
        raise ValueError("Unexpected checkpoint omissions")
    if any(v.dtype != saved[k].dtype or v.shape != saved[k].shape or not torch.equal(v, saved[k])
           for k, v in loaded.items()):
        raise ValueError("Restored tensor mismatch")
    if model.cfg.dataset.name != "guacamol" or model.cfg.model.transition != "marginal":
        raise ValueError("Unexpected checkpoint family")
    if model.output_dims != {"X": 12, "E": 5, "y": 0}:
        raise ValueError("Unexpected output vocabulary")
    model.eval().requires_grad_(False)
    return model, {"state_tensors": len(loaded), "omitted_evaluation_buffers": sorted(omitted),
                   "input_dims": dict(model.input_dims), "output_dims": dict(model.output_dims),
                   "cfg": OmegaConf.to_container(model.cfg, resolve=True)}


class Probe:
    def __init__(self, model, c, output):
        self.model, self.c, self.output = model, c, output
        self.rows, self.tensors = [], {}
        self.started = time.monotonic()
        self.forwards = 0
        self.hook = model.model.register_forward_pre_hook(self.count)

    def count(self, *_):
        self.forwards += 1

    def query(self, mode, graph, t, **metadata):
        if len(self.rows) >= self.c["max_queries"] or time.monotonic() - self.started >= self.c["max_seconds"]:
            raise TimeoutError("Declared probe budget exhausted")
        x, e = graph
        validate_graph(x, e)
        row = {"id": len(self.rows), "mode": mode, "time": t, **metadata}
        before = self.forwards
        started = time.monotonic()
        try:
            with torch.inference_mode():
                data = noisy(x, e, t)
                if mode in ("hard_view", "hard_aux_soft_core"):
                    hx, he = hard_view(x, e)
                    feature_data = noisy(hx, he, t)
                    if mode == "hard_view":
                        data = feature_data
                else:
                    feature_data = data
                extra = self.model.compute_extra_data(feature_data)
                pred = self.model.forward(data, extra, data["node_mask"])
                tx, te = pred.X.softmax(-1).detach().clone(), pred.E.softmax(-1).detach().clone()
                d = torch.arange(x.shape[1])
                te[:, d, d] = 0
                validate_graph(tx, te)
                if self.forwards - before != 1 or tx.requires_grad or te.requires_grad:
                    raise RuntimeError("Expected one detached denoiser forward")
                row.update(status="completed", locality=measure(graph, (tx, te)),
                           readout=raw_readout(tx, te))
                result = tx, te
                self.tensors[str(row["id"])] = {"input_X": x.clone(), "input_E": e.clone(),
                                               "teacher_X": tx, "teacher_E": te}
        except (AssertionError, RuntimeError, ValueError) as exc:
            row.update(status="unsupported_or_failed", error_type=type(exc).__name__,
                       error=str(exc)[:800], traceback=traceback.format_exc(limit=9))
            result = None
        row.update(forward_calls=self.forwards - before, wall_seconds=time.monotonic() - started)
        self.rows.append(row)
        with (self.output / "queries.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        return result, row


def comparisons(p, c, model):
    pairs, trajectories, parity = [], [], []
    for control in c["controls"]:
        hard = make_control(control)
        for t in c["times"]:
            hard_outputs = {}
            for softness in c["softness"]:
                x = (1 - softness) * hard[0] + softness * model.limit_dist.X.reshape(1, 1, -1)
                e = (1 - softness) * hard[1] + softness * model.limit_dist.E.reshape(1, 1, 1, -1)
                d = torch.arange(x.shape[1])
                e[:, d, d] = 0
                graph = x, e
                for mode in MODES:
                    meta = dict(control=control["id"], softness=softness)
                    baseline, br = p.query(mode, graph, t, kind="baseline", **meta)
                    torch.manual_seed(c["seed"] + 999)
                    repeat, rr = p.query(mode, graph, t, kind="repeat_different_rng_seed", **meta)
                    if baseline is not None and repeat is not None:
                        pairs.append({"kind": "repeat", "base": br["id"], "other": rr["id"],
                                      "output": measure(baseline, repeat)})
                    if softness == 0 and baseline is not None:
                        hard_outputs[mode] = (baseline, br["id"])
                    for direction in ("atom", "bond"):
                        for epsilon in c["epsilons"]:
                            changed = perturb(x, e, direction, epsilon)
                            target, row = p.query(mode, changed, t, kind="perturbation", direction=direction,
                                                  epsilon=epsilon, **meta)
                            if baseline is not None and target is not None:
                                inp, out = measure(graph, changed), measure(baseline, target)
                                pairs.append({"kind": "perturbation", "base": br["id"], "other": row["id"],
                                              "input": inp, "output": out,
                                              "max_cell_tv_gain": out["max_cell_tv"] / inp["max_cell_tv"]})
            if set(hard_outputs) != set(MODES):
                raise RuntimeError("Native hard control or boundary parity unavailable")
            for mode in MODES[1:]:
                diff = measure(hard_outputs["native_soft"][0], hard_outputs[mode][0])
                parity.append({"control": control["id"], "time": t, "mode": mode, "difference": diff})
                if diff["max_cell_tv"] != 0:
                    raise RuntimeError("Alternative mode changed the exact hard boundary")
            for direction in ("atom", "bond"):
                graphs = tie_pair(*hard, direction, c["tie_delta"])
                for mode in MODES:
                    a, ar = p.query(mode, graphs[0], t, control=control["id"], kind="tie_minus", direction=direction)
                    b, br = p.query(mode, graphs[1], t, control=control["id"], kind="tie_plus", direction=direction)
                    if a is not None and b is not None:
                        inp, out = measure(*graphs), measure(a, b)
                        pairs.append({"kind": "argmax_boundary", "base": ar["id"], "other": br["id"],
                                      "input": inp, "output": out,
                                      "max_cell_tv_gain": out["max_cell_tv"] / inp["max_cell_tv"]})
            # This one-edit probe measures response across a discrete graph change.
            changed = perturb(*hard, "bond", 1.0)
            for mode in MODES:
                b, br = p.query(mode, changed, t, control=control["id"], kind="one_bond_edit")
                if b is not None:
                    a, aid = hard_outputs[mode]
                    pairs.append({"kind": "one_bond_edit", "base": aid, "other": br["id"],
                                  "input": measure(hard, changed), "output": measure(a, b)})
            # Realizable training-noise input; this categorical draw is input preparation,
            # not sampling the teacher output. Its full tensors are saved with the query.
            torch.manual_seed(c["seed"])
            native = model.apply_noise(*hard, hard[0].new_zeros((1, 0)),
                                       torch.ones(hard[0].shape[:2], dtype=torch.bool),
                                       t=hard[0].new_full((1, 1), t))
            native["E_t"][:, d, d] = 0  # Same zero-diagonal feature convention as native training.
            for mode in MODES:
                p.query(mode, (native["X_t"].float(), native["E_t"].float()), t,
                        control=control["id"], kind="native_categorical_corruption")
            # Small teacher-only updates, independent of AF3/Boltz optimization.
            for mode in MODES:
                graph = tuple(0.98 * v + 0.02 * prior for v, prior in (
                    (hard[0], model.limit_dist.X.reshape(1, 1, -1)),
                    (hard[1], model.limit_dist.E.reshape(1, 1, 1, -1))))
                d = torch.arange(graph[0].shape[1])
                graph[1][:, d, d] = 0
                previous = None
                for step in range(c["trajectory_steps"]):
                    target, row = p.query(mode, graph, t, control=control["id"], kind="teacher_only_trajectory", step=step)
                    if target is None:
                        trajectories.append({"query": row["id"], "status": "stopped_unsupported"})
                        break
                    updated = tuple((1 - c["trajectory_step_size"]) * v + c["trajectory_step_size"] * w
                                    for v, w in zip(graph, target))
                    record = {"query": row["id"], "status": "completed", "update": measure(graph, updated),
                              "distance_from_hard_start": measure(hard, updated), "readout": raw_readout(*updated)}
                    if previous is not None:
                        record["teacher_drift"] = measure(previous, target)
                    trajectories.append(record)
                    graph, previous = updated, target
    return {"pairs": pairs, "trajectories": trajectories, "hard_boundary_parity": parity}


def summarize(rows, comparisons_):
    summaries = {}
    lookup = {row["id"]: row for row in rows}
    for mode in MODES:
        selected = [r for r in rows if r["mode"] == mode]
        baseline = [r for r in selected if r["kind"] == "baseline" and r["status"] == "completed"]
        ps = [p for p in comparisons_["pairs"] if lookup[p["base"]]["mode"] == mode]
        summaries[mode] = {"queries": len(selected), "status_counts": dict(Counter(r["status"] for r in selected)),
            "successful_fractional_baselines": sum(r["softness"] > 0 for r in baseline),
            "baseline_max_change_tv": max((r["locality"]["max_cell_tv"] for r in baseline), default=None),
            "baseline_mean_decision_tv": sum(r["locality"]["decision_mean_tv"] for r in baseline) / len(baseline) if baseline else None,
            "repeat_max_tv": max((p["output"]["max_cell_tv"] for p in ps if p["kind"] == "repeat"), default=None),
            "perturbation_max_tv_gain": max((p["max_cell_tv_gain"] for p in ps if p["kind"] == "perturbation"), default=None),
            "argmax_boundary_max_output_tv": max((p["output"]["max_cell_tv"] for p in ps if p["kind"] == "argmax_boundary"), default=None),
            "argmax_boundary_max_tv_gain": max((p["max_cell_tv_gain"] for p in ps if p["kind"] == "argmax_boundary"), default=None),
            "baseline_neutral_sanitized": sum(r["readout"]["neutral_sanitized"] for r in baseline),
            "baseline_count": len(baseline),
            "error_examples": [dict(id=r["id"], control=r["control"], error_type=r["error_type"], error=r["error"], traceback=r["traceback"])
                               for r in selected if r["status"] != "completed"][:2]}
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("Numerical execution requires an allocated DTU job")
    if sys.flags.optimize:
        raise RuntimeError("Native assertions must remain enabled")
    args.output.mkdir(parents=True, exist_ok=False)
    global torch, F, Chem, OmegaConf
    import torch
    from torch.nn import functional as F
    from rdkit import Chem, RDLogger
    from omegaconf import OmegaConf
    RDLogger.DisableLog("rdApp.error")
    c = json.loads(args.config.read_text())
    validate_config(c)
    torch.set_num_threads(min(int(os.environ.get("LSB_DJOB_NUMPROC", "4")), 4))
    torch.set_num_interop_threads(1)
    torch.manual_seed(c["seed"])
    random.seed(c["seed"])
    model, audit = load_model(args.source, args.checkpoint, c)
    before = state_digest(model)
    write_json(args.output / "load-audit.json", audit)
    runtime = {"job": os.environ["LSB_JOBID"], "host": platform.node(), "python": sys.version,
               "torch": torch.__version__, "threads": torch.get_num_threads(), "device": "cpu",
               "config_sha256": digest(args.config), "driver_sha256": digest(__file__),
               "checkpoint_sha256": c["checkpoint_sha256"], "upstream_commit": c["source_commit"]}
    write_json(args.output / "runtime.json", runtime)
    probe = Probe(model, c, args.output)
    try:
        comparison = comparisons(probe, c, model)
        after = state_digest(model)
        checks = {"frozen_state_unchanged": before == after,
                  "all_successes_one_forward": all(r["forward_calls"] == 1 for r in probe.rows if r["status"] == "completed"),
                  "exact_hard_boundary_parity": all(r["difference"]["max_cell_tv"] == 0 for r in comparison["hard_boundary_parity"]),
                  "repeatability": all(p["output"]["max_cell_tv"] <= 1e-6 for p in comparison["pairs"] if p["kind"] == "repeat")}
        write_json(args.output / "comparisons.json", comparison)
        torch.save(probe.tensors, args.output / "probabilities.pt")
        result = {"status": "completed" if all(checks.values()) else "engineering_failed", "checks": checks,
                  "mode_summaries": summarize(probe.rows, comparison), "queries": len(probe.rows),
                  "forward_calls": probe.forwards, "seconds": time.monotonic() - probe.started,
                  "frozen_state_before": before, "frozen_state_after": after,
                  "scientific_acceptance": "diagnostic_only_no_locality_or_chemistry_approval",
                  "limits": ["CPU precision only", "five synthetic fixed-occupancy controls; not a representative molecular cohort",
                             "soft inputs differ from the categorical training distribution", "no formal-charge output",
                             "hard_aux_soft_core is an altered feature policy", "teacher-only steps are not native-score optimization"]}
        write_json(args.output / "summary.json", result)
        print(json.dumps(result, allow_nan=False), flush=True)
        if not all(checks.values()):
            raise RuntimeError("Engineering checks failed")
    except Exception as exc:
        if not (args.output / "summary.json").exists():
            write_json(args.output / "failure.json", {"status": "failed", "error_type": type(exc).__name__,
                "error": str(exc), "traceback": traceback.format_exc(), "queries": len(probe.rows)})
        raise
    finally:
        probe.hook.remove()


if __name__ == "__main__":
    main()
