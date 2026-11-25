"""Bounded BRD4 replay and paired latent flow warm-start experiment."""

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch import nn

from flow_lr_selection import config_digest, load_config, rate_arms
from latentmol.chemistry import inspect_graph
from latentmol.defog.candidate_repair import graph_from_record, graph_record, request_sha256
from latentmol.defog.codec import to_guacamol_hard
from latentmol.design.latent_protenix import (
    LatentDesignConfig, _json, _model_identity, hard_from_graph, run_design, structural_target,
)
from latentmol.design.temperature_candidates import select_valid_candidates
from latentmol.design.temperature_fresh import group_candidates
from latentmol.design.temperature_study import hard_at, start_state
from latentmol.evaluation.evaluate import load_model
from latentmol.graph_policy import graph_from_decoder
from latentmol.guidance.mosaic_protenix import MosaicProtenixScorer
from latentmol.multimodel.external import file_sha256


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def replay(rows, budgets):
    panel = []
    for (start, arm), group in group_candidates(rows).items():
        improvements = [row for row in group if row["role"] != "initial"
                        and row["loss_delta_from_valid_start"] is not None
                        and row["loss_delta_from_valid_start"] < 0]
        panel.append(dict(start=start, arm=arm,
            valid_scored_start=any(row["role"] == "initial" and row["loss"] is not None
                                   for row in group),
            first_improvement_step=min((row["step"] for row in improvements), default=None),
            budgets=[dict(updates=budget,
                unique_improvements=len({row["smiles"] for row in improvements if row["step"] <= budget}),
                best_delta=min((row["loss_delta_from_valid_start"] for row in improvements
                                if row["step"] <= budget), default=None)) for budget in budgets]))
    return panel


def velocity_model(start, end, *, flow, iterations, seed):
    if start.ndim != 2 or start.shape != end.shape or not start.shape[0]:
        raise ValueError("Expected a nonempty aligned matrix of start and endpoint latents")
    delta = end - start
    scale = float(delta.square().mean().sqrt())
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Training pairs require finite, nontrivial displacements")
    width = start.shape[-1]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(width + 1, 64), nn.SiLU(), nn.Linear(64, width))
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.003)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(iterations):
        t = torch.rand((len(start), 1), generator=generator) if flow else start.new_zeros((len(start), 1))
        prediction = net(torch.cat((start + t * delta, t), dim=-1))
        loss = (prediction - delta / scale).square().mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite warm-start training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is None or not bool(p.grad.isfinite().all()) for p in net.parameters()):
            raise ValueError("Missing or nonfinite warm-start model gradient")
        optimizer.step()
    return net.eval(), scale, float(loss.detach())


@torch.no_grad()
def transport(net, start, scale, *, steps, radius):
    point = start.clone()
    for step in range(steps):
        t = point.new_full((len(point), 1), step / steps)
        point += (scale / steps) * net(torch.cat((point, t), dim=-1))
    delta = point - start
    if not bool(delta.isfinite().all()):
        raise ValueError("Nonfinite latent transport")
    factor = (radius / delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1)
    return start + factor * delta


def connected(hard, spec):
    result = inspect_graph(hard.atoms[0], hard.charges[0], hard.bonds[0], spec)
    return result.smiles if result.smiles and "." not in result.smiles else None


@torch.no_grad()
def decode_graph(model, point, active):
    decoded = model.decode(point)
    if any(not bool(getattr(decoded, name).isfinite().all()) for name in ("atoms", "charges", "bonds")):
        raise ValueError("Nonfinite frozen decoder output")
    return graph_from_decoder(decoded, model.spec, temperature=1., fixed_active=active)


def screened_starts(model, config, count):
    starts = {}
    seen = set()
    wanted = config["train_starts"] + config["test_starts"]
    for seed in range(config["seed"], config["seed"] + config["screen_limit"]):
        source = dict(kind="prior", seed=seed)
        state = start_state(model, source, count)
        hard = hard_from_graph(decode_graph(model, state["latent"], state["active"]))
        smiles = connected(hard, model.spec)
        if smiles is None or smiles in seen:
            continue
        starts[str(seed)] = (state, smiles)
        seen.add(smiles)
        if len(starts) == wanted:
            return starts
    raise ValueError(f"Only {len(starts)} distinct connected starts in the screening budget")


def history(output, start, arm):
    return output / start / arm / "arms" / "latent"


def records(folder):
    with (folder / "candidates.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    for row in rows:
        row["valid"] = bool(row["valid"] and row["smiles"] and "." not in row["smiles"])
    return rows


def candidates(folder, start, arm, edit_limit):
    selection = select_valid_candidates(records(folder), edit_limit=edit_limit)
    return [dict(row, start=start, arm=arm) for row in selection["candidates"]]


def short_candidates(rows, start, arm, initial_smiles):
    """Score every new connected identity from the matched short call budget."""
    panel = [dict(start=start, arm=arm, role="initial", step=-1, smiles=initial_smiles)]
    seen = {initial_smiles}
    for row in rows:
        smiles = row["smiles"]
        if row["valid"] and smiles not in seen:
            panel.append(dict(start=start, arm=arm, role="proposal" if row["step"] == 0 else "edit",
                              step=row["step"], smiles=smiles))
            seen.add(smiles)
    return panel


def latent_at(folder, step):
    for path in sorted((folder / "raw-blocks").glob("states-*.npz")):
        with np.load(path, allow_pickle=False) as block:
            indices = np.flatnonzero(block["steps"] == step)
            if indices.size:
                latent = torch.from_numpy(block["latent"][indices.item()])[None]
                if not bool(latent.isfinite().all()):
                    raise ValueError("Saved latent is nonfinite")
                return latent
    raise ValueError(f"Missing saved step {step} in {folder}")


def run_arm(model, design, state, args, native, identity, start, arm):
    folder = args.output / start / arm

    def factory(spec, active):
        return MosaicProtenixScorer(spec, active, command=[args.mosaic_python, "-m",
            "latentmol.guidance.mosaic_protenix_worker"], config=native,
            stderr_path=folder / "native.stderr.log", timeout=180, startup_timeout=240)

    result = run_design(model, factory, design, folder,
                        checkpoint_sha256=identity, start_state=state)
    if result["updates_this_segment"] != design.updates:
        raise RuntimeError("Trajectory did not reach its budget")
    return result


def tune(model, starts, design, args, config, native, identity):
    output = args.output
    arms = rate_arms(config)
    output.mkdir(exist_ok=False)
    trajectories = {}
    rows = []
    for start, (state, smiles) in starts.items():
        trajectories[start] = {}
        for arm, rate in arms.items():
            short = replace(design, updates=config["refine_updates"], learning_rate=rate,
                            noise_seed=config["teacher_noise_offset"] + int(start))
            result = run_arm(model, short, state, args, native, identity, start, arm)
            trajectories[start][arm] = dict(scorer_calls=result["updates_this_segment"] + 1,
                bounded_steps=result["trajectory_statistics"]["bounded_steps"],
                elapsed_seconds=result["elapsed_seconds"])
            rows.extend(short_candidates(records(history(output, start, arm)), start, arm, smiles))
    save(output / "tuning.json", dict(seed=config["seed"], config_digest=args.study_digest,
         checkpoint_sha256=identity, job_id=os.environ["LSB_JOBID"],
         starts={key: smiles for key, (_, smiles) in starts.items()}, trajectories=trajectories))
    save(output / "fresh-candidates.json", dict(candidates=rows))


def teacher(model, starts, design, args, config, native, identity):
    output = args.output
    output.mkdir(exist_ok=False)
    save(output / "panel.json", dict(starts=list(starts), train=list(starts)[:config["train_starts"]],
                                    start_smiles={key: smiles for key, (_, smiles) in starts.items()},
                                    learning_rate=design.learning_rate, lr_selection_sha256=args.selection_sha256,
                                    config=config, checkpoint_sha256=identity))
    rows = []
    for start, (state, _) in starts.items():
        trajectory = (replace(design, noise_seed=config["teacher_noise_offset"] + int(start))
                      if config.get("portfolio_test") else design)
        run_arm(model, trajectory, state, args, native, identity, start, "teacher")
        folder = history(output, start, "teacher")
        observations = records(folder)
        short = observations[:config["refine_updates"] + 1]
        if config.get("portfolio_test"):
            early = short_candidates(short, start, "sgd_short", starts[start][1])
            rows.extend(early)
            early_identities = {row["smiles"] for row in early}
            late = [observations[0], *(row for row in observations[len(short):]
                                      if row["smiles"] not in early_identities)]
            rows.extend(dict(row, start=start, arm="teacher") for row in
                        select_valid_candidates(late, edit_limit=config["edit_limit"])["candidates"])
        else:
            rows.extend(dict(row, start=start, arm="sgd_short") for row in
                        select_valid_candidates(short, edit_limit=config["edit_limit"])["candidates"])
            rows.extend(candidates(folder, start, "teacher", config["edit_limit"]))
    save(output / "fresh-candidates.json", dict(candidates=rows))


def students(model, starts, design, args, config, native, identity):
    output = args.output / "students"
    output.mkdir(exist_ok=False)
    groups = group_candidates(read(args.output / "fresh" / "summary.json")["candidates"])
    training = list(starts)[:config["train_starts"]]
    pairs = []
    for start in training:
        panel = groups[(start, "teacher")] + groups[(start, "sgd_short")]
        improved = [r for r in panel if r["loss_delta_from_valid_start"] is not None
                    and r["loss_delta_from_valid_start"] < -config["min_score_gain"]]
        if improved:
            best = min(improved, key=lambda r: (r["loss_delta_from_valid_start"], r["step"]))
            pairs.append(best)
    rows = []
    models = {}
    followup = config.get("portfolio_test", False)
    if followup and read(args.output / "panel.json")["start_smiles"] != {
        key: smiles for key, (_, smiles) in starts.items()
    }:
        raise ValueError("Held-out start replay differs from the teacher panel")
    if args.selection_sha256 is not None:
        panel = read(args.output / "panel.json")
        if panel["learning_rate"] != design.learning_rate or panel["lr_selection_sha256"] != args.selection_sha256:
            raise ValueError("Student refinement must use the same selected rate as the teacher")
    residual_points = {}
    if len(pairs) >= config["min_pairs"]:
        x = torch.cat([starts[row["start"]][0]["latent"] for row in pairs])
        y = torch.cat([latent_at(history(args.output, row["start"], "teacher"), row["step"])
                       for row in pairs])
        radius = float((y - x).norm(dim=-1).max())
        short = replace(design, updates=config["refine_updates"])
        steps_by_arm = {"residual": 1, "flow": config["flow_steps"]}
        for arm in config.get("student_arms", steps_by_arm):
            steps = steps_by_arm[arm]
            tick = time.monotonic()
            net, scale, loss = velocity_model(x, y, flow=arm == "flow",
                                             iterations=config["fit_iterations"], seed=config["seed"])
            models[arm] = dict(loss=loss, radius=radius, steps=steps, fit_seconds=time.monotonic()-tick)
            for start in list(starts)[config["train_starts"]:]:
                state, smiles = starts[start]
                tick = time.monotonic()
                point = transport(net, state["latent"], scale, steps=steps, radius=radius)
                models[arm].setdefault("proposal_seconds", {})[start] = time.monotonic()-tick
                if followup:
                    paired_design = replace(short, noise_seed=config["restart_noise_offset"] + int(start))
                    run_arm(model, paired_design, dict(state, latent=point), args, native, identity, start, arm)
                    rows.extend(short_candidates(records(history(args.output, start, arm)), start, arm, smiles))
                    if arm == "residual":
                        residual_points[start] = point
                    continue
                run_arm(model, short, dict(state, latent=point), args, native, identity, start, arm)
                panel = candidates(history(args.output, start, arm), start, arm, config["edit_limit"])
                rows.append(dict(start=start, arm=arm, role="initial", step=-1, smiles=smiles))
                # Use the same edit cap as SGD, counting the warm proposal if valid and novel.
                rows.extend(dict(row, role="proposal" if row["step"] == 0 else "edit")
                            for row in [r for r in panel if r["smiles"] != smiles][:config["edit_limit"]])
    if followup:
        for start in list(starts)[config["train_starts"]:]:
            state, smiles = starts[start]
            original = state["latent"]
            paired = replace(design, updates=config["refine_updates"],
                             noise_seed=config["restart_noise_offset"] + int(start))
            controls = {"sgd_restart": original}
            if start in residual_points:
                generator = torch.Generator().manual_seed(config["random_offset"] + int(start))
                direction = torch.randn(original.shape, generator=generator, dtype=original.dtype)
                controls["random_jump"] = original + direction * (
                    (residual_points[start] - original).norm() / direction.norm())
            for arm, point in controls.items():
                run_arm(model, paired, dict(state, latent=point), args, native, identity, start, arm)
                rows.extend(short_candidates(records(history(args.output, start, arm)), start, arm, smiles))
    save(output / "fit.json", dict(status="trained" if models else "insufficient_improved_pairs",
                                  pairs=pairs, models=models))
    save(output / "fresh-candidates.json", dict(candidates=rows))


def defog_case(args, output, name, request, model):
    request["request_sha256"] = request_sha256(request)
    request_path, receipt_path = output / f"{name}-request.json", output / f"{name}-defog.json"
    save(request_path, request)
    result = subprocess.run([args.defog_python, "-m", "latentmol.defog.candidate_repair",
                            "--request", str(request_path), "--output", str(receipt_path)], timeout=240)
    receipt = read(receipt_path)
    if receipt["request_sha256"] != request["request_sha256"] or result.returncode:
        raise RuntimeError(f"DeFoG execution/contract failure: {receipt_path}")
    if ready := os.environ.get("DBMOL_LAUNCH_READY"):
        if receipt["cost"]["native_forward_calls"] and not Path(ready).exists():
            _json(Path(ready), dict(status="models_and_inputs_ready", scope="defog_forward_completed",
                                   job_id=os.environ["LSB_JOBID"], nonce=os.environ["DBMOL_LAUNCH_NONCE"]))
    target = graph_from_record(receipt["target"], model.spec) if receipt["target"] else None
    return target, {key: receipt.get(key) for key in ("status", "reason", "edits", "cost")}


def repairs(model, starts, design, args, config):
    output = args.output / "repair"
    output.mkdir(exist_ok=False)
    rows, reports = [], []
    count = {True: 0, False: 0}
    seen = set()
    saved = config["repair_source"]
    independent = config.get("one_repair_case_per_start", False)
    for start, (state, source) in starts.items():
        parent = Path(source) if independent else Path(saved["path"])
        folder = history(parent, start, saved["arm"])
        observations = records(folder)
        if independent:
            # Alternate the requested class as cohorts contribute cases. Try the
            # other class only when the preferred class has no eligible state.
            want_valid = count[True] * config["repair_invalid"] < count[False] * config["repair_valid"]
            observations.sort(key=lambda row: row["valid"] != want_valid)
        for row in observations:
            valid, step = row["valid"], row["step"]
            limit = config["repair_valid" if valid else "repair_invalid"] if independent else config["repair_per_class"]
            if count[valid] >= limit:
                continue
            hard = hard_at(folder / "raw-blocks", step)
            key = json.dumps(graph_record(hard), separators=(",", ":"))
            if key in seen:
                continue
            seen.add(key)
            report = dict(start=start, step=step, valid_input=valid, input_smiles=row["smiles"])
            try:
                mapped = to_guacamol_hard(hard, model.spec)
            except ValueError as exc:
                reports.append(dict(report, eligibility="unsupported", reason=str(exc)))
                continue
            count[valid] += 1
            name = f"{start}-{step}"
            point = latent_at(folder, step)
            graph = decode_graph(model, point, state["active"])
            if graph_record(hard_from_graph(graph)) != graph_record(hard):
                raise ValueError("Replayed probabilities changed the saved candidate labels")
            tick = time.monotonic()
            target, structural = structural_target(graph, model.spec, state["active"], design, step)
            structural["wall_seconds"] = time.monotonic()-tick
            request = dict(config["defog"], candidate=graph_record(hard), spec=model.spec.to_dict(),
                           occupied_slots=list(mapped.occupied_slots), seed=config["seed"]+sum(count.values()))
            repaired_graphs = {"structural": target.graph if target else hard if valid else None}
            report.update(eligibility="eligible", structural=structural)
            schedules = config.get("repair_schedules", {"defog": request["t0"]})
            for arm, t0 in schedules.items():
                label = f"{name}-{arm}" if independent else name
                repaired_graphs[arm], report[arm] = defog_case(args, output, label, dict(request, t0=t0), model)
            for arm, repaired in repaired_graphs.items():
                if valid:
                    rows.append(dict(start=name, arm=arm, role="initial", step=0, smiles=row["smiles"]))
                smiles = connected(repaired, model.spec) if repaired is not None else None
                report[arm]["connected_smiles"] = smiles
                if smiles and smiles != row["smiles"]:
                    rows.append(dict(start=name, arm=arm, role="repair", step=1, smiles=smiles))
            reports.append(report)
            if independent:
                break
    save(output / "repair.json", dict(cases=reports, eligible_valid=count[True], eligible_invalid=count[False]))
    save(output / "fresh-candidates.json", dict(candidates=rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("replay", "screen", "tune", "teacher", "students", "repairs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--fresh", type=Path)
    parser.add_argument("--mosaic-python")
    parser.add_argument("--mosaic-config", type=Path)
    parser.add_argument("--defog-python")
    parser.add_argument("--plan")
    parser.add_argument("--plan-sha256")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--repair-parents", type=Path, nargs="+")
    parser.add_argument("--lr-selection", type=Path)
    args = parser.parse_args()
    if args.stage == "replay":
        save(args.output, dict(source=str(args.fresh), results=replay(read(args.fresh)["candidates"], [7, 16, 32])))
        return
    if args.stage != "screen" and not os.environ.get("LSB_JOBID", "").isdigit():
        raise RuntimeError("Native experiments require an allocated GPU job")
    torch.set_num_threads(4)
    config = load_config(args.config)
    args.study_digest = config_digest(config)
    args.selection_sha256 = None
    if args.seed is not None:
        config["seed"] = args.seed
    identity = file_sha256(args.checkpoint)
    if identity != config["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint changed")
    model, _ = load_model(args.checkpoint, "cpu")
    model.eval().requires_grad_(False)
    before = _model_identity(model)
    design = replace(LatentDesignConfig.from_mapping(read(args.config.parent / "pilot.json")),
                     temperature_initial=1., temperature_final=1., updates=config["teacher_updates"])
    selection = None
    if args.stage == "tune" or (args.stage == "screen" and config["seed"] in config.get("tuning_seeds", [])):
        if config["seed"] not in config["tuning_seeds"]:
            raise ValueError("Tuning seed is outside the declared panel")
        config = dict(config, train_starts=0, test_starts=config["tuning_starts_per_cohort"])
    if args.stage in ("teacher", "students") and "tuning_seeds" in config:
        if args.lr_selection is None or config["seed"] not in config["cohort_seeds"]:
            raise ValueError("Tuned comparison requires the selected rate and a declared validation cohort")
        selection = read(args.lr_selection)
        if (selection["status"] != "selected" or selection["config_digest"] != args.study_digest
                or selection["tuning_seeds"] != sorted(config["tuning_seeds"])
                or selection["learning_rate"] not in config["learning_rates"]):
            raise ValueError("Learning-rate selection belongs to a different study")
        args.selection_sha256 = file_sha256(args.lr_selection)
        design = replace(design, learning_rate=selection["learning_rate"])
    if args.stage == "repairs":
        if args.repair_parents:
            if not config.get("one_repair_case_per_start"):
                raise ValueError("Independent repair inputs require the follow-up configuration")
            panels = [read(path / "panel.json") for path in args.repair_parents]
            if any(panel["checkpoint_sha256"] != identity or not panel["config"]["portfolio_test"]
                   for panel in panels):
                raise ValueError("Repair input checkpoint or protocol differs")
            heldout = [[start for start in panel["starts"] if start not in panel["train"]] for panel in panels]
            starts = {}
            for group in zip(*heldout, strict=True):
                for start, path in zip(group, args.repair_parents, strict=True):
                    if start in starts:
                        raise ValueError("Repair panel repeats an independent start")
                    starts[start] = (start_state(model, dict(kind="prior", seed=int(start)), design.occupied_atoms), path)
        else:
            starts = {f"prior-{seed}": (start_state(model, dict(kind="prior", seed=seed),
                                                design.occupied_atoms), None)
                  for seed in config["repair_source"]["seeds"]}
        repairs(model, starts, design, args, config)
        if before != _model_identity(model):
            raise ValueError("Frozen decoder state changed")
        return
    starts = screened_starts(model, config, design.occupied_atoms)
    if selection is not None:
        tune_starts = selection["tuning_starts"]
        if (len(tune_starts) != len(config["tuning_seeds"]) * config["tuning_starts_per_cohort"]
                or set(starts) & tune_starts.keys()
                or {smiles for _, smiles in starts.values()} & set(tune_starts.values())):
            raise ValueError("Tuning and comparison must have complete, disjoint starting panels")
    if args.stage == "screen":
        save(args.output, dict(starts={key: smiles for key, (_, smiles) in starts.items()}))
    else:
        stage = {"teacher": teacher, "students": students, "tune": tune}[args.stage]
        stage(model, starts, design, args, config, read(args.mosaic_config), identity)
    if before != _model_identity(model) or any(p.grad is not None for p in model.parameters()):
        raise ValueError("Frozen decoder state or gradients changed")


if __name__ == "__main__":
    main()
