"""Segmented, restarted search with raw-decoder occupancy gradients.

Only an accepted latent move consumes an update. A restart begins a new
trajectory from the prior and never consumes an update. Each trajectory
retains the pilot's exact-chemistry native scoring and acceptance rules.
"""

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
import time

import torch

from latentmol.evaluation.evaluate import load_model
from latentmol.multimodel.external import file_sha256, tensor_tree_sha256
from latentmol.protenix_occupancy.native import load_frozen_tiny
from latentmol.protenix_occupancy.prepare import _bind_native
from latentmol.protenix_occupancy.raw_occupancy import (
    ProtenixOracle, decode, observe, occupancy_step, probe_rows,
)
from latentmol.protenix_occupancy.study import write_json
from latentmol.scoring import require_frozen_eval


_COUNTERS = ("completed_updates", "molecule_changing_updates", "atom_adding_updates",
             "atom_removing_updates", "native_attempts", "trajectories_started", "next_draw",
             "failed_draws", "unproductive_trajectories", "trajectory_updates",
             "trajectory_chemistry_updates", "same_chemistry_updates")


def validate_config(config):
    fields = ("updates", "probe_limit", "max_stage_native_calls", "max_total_native_calls",
              "max_trajectory_updates", "max_same_chemistry_updates", "max_start_draws",
              "max_draws", "max_unproductive_trajectories")
    if any(type(config[key]) is not int or config[key] < 1 for key in fields):
        raise ValueError("Long-run budgets must be positive integers")
    if type(config["seed"]) is not int or not 0 <= config["seed"] <= 2**63 - config["max_draws"]:
        raise ValueError("Prior seeds fall outside the deterministic generator range")
    if not math.isfinite(config["latent_radius"]) or config["latent_radius"] <= 0:
        raise ValueError("Invalid latent radius")
    steps = config["trial_steps"]
    if (not steps or any(not math.isfinite(step) or step <= 0 for step in steps)
            or any(left <= right for left, right in zip(steps, steps[1:]))):
        raise ValueError("Trial steps must be finite, positive and strictly decreasing")
    fresh_seeds = config["fresh_geometry_seeds"]
    seeds = [config["geometry_seed"], *fresh_seeds]
    if (not fresh_seeds or any(type(seed) is not int or not 0 <= seed < 2**31 - 1 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Require distinct supported optimization and fresh geometry seeds")
    # Reserve an uncached baseline, both probe lists, every trial and the
    # fresh initial, final, best overall and best guided molecules.
    reserve = 1 + 2 * config["probe_limit"] + len(steps) + 4 * len(fresh_seeds)
    if min(config["max_stage_native_calls"], config["max_total_native_calls"]) < reserve:
        raise ValueError("Native score budget cannot fit a single bounded operation")
    return reserve


def _load_models(config_path, config):
    assets_path = config_path.parent / config["assets"]
    assets = json.loads(assets_path.read_text(encoding="utf-8"))
    checkpoint_path = config_path.parent / assets["assets_config"]
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    pocket_path = Path(assets["pocket"]["path"])
    if (file_sha256(pocket_path) != assets["pocket"]["sha256"]
            or file_sha256(checkpoint["checkpoint"]) != checkpoint["checkpoint_sha256"]):
        raise ValueError("Pocket or VAE checkpoint identity differs")
    torch.set_num_threads(4)
    model, _ = load_model(Path(checkpoint["checkpoint"]), "cuda")
    model.requires_grad_(False)
    require_frozen_eval(model)
    _bind_native(Path(checkpoint["native"]["source_root"]), checkpoint["native"]["source_sha256"],
                 Path(assets["ccd_root"]))
    native = load_frozen_tiny(**checkpoint["native"])
    identity = dict(config=file_sha256(config_path), assets=file_sha256(assets_path),
                    checkpoint_config=file_sha256(checkpoint_path), vae=checkpoint["checkpoint_sha256"],
                    pocket=assets["pocket"]["sha256"], native=checkpoint["native"]["weights_sha256"])
    return model, native, json.loads(pocket_path.read_text(encoding="utf-8")), identity


def _restore(resume, identity, model, config):
    if resume is None:
        state = dict.fromkeys(_COUNTERS, 0)
        state.update(version=1, identity=identity, active_draw=None, point=None, anchor=None,
                     initial=None, final=None, final_point=None, best=None, best_guided=None,
                     initial_fresh=None)
        return state, None, None
    state = json.loads(resume.read_text(encoding="utf-8"))
    if (state.get("version") != 1 or state.get("identity") != identity
            or state.get("stage_status") != "segment_completed"):
        raise ValueError("Resume requires an incomplete checkpoint with identical inputs")
    if (any(type(state.get(key)) is not int or state[key] < 0 for key in _COUNTERS)
            or not state["molecule_changing_updates"] <= state["completed_updates"] < config["updates"]
            or state["native_attempts"] > config["max_total_native_calls"]
            or state["next_draw"] > config["max_draws"]):
        raise ValueError("Invalid resume counters")
    if state["active_draw"] is None:
        if state["point"] is not None or state["anchor"] is not None:
            raise ValueError("Inactive checkpoint carries a live trajectory")
        return state, None, None
    if (type(state["active_draw"]) is not int or not 0 <= state["active_draw"] < state["next_draw"]
            or state["point"] is None or state["anchor"] is None or state["final"] is None):
        raise ValueError("Incomplete active trajectory in checkpoint")
    device = next(model.parameters()).device
    point, anchor = (torch.as_tensor(state[key], device=device, dtype=torch.float32)
                     for key in ("point", "anchor"))
    for z in (point, anchor):
        if z.shape != (1, model.config.latent) or not z.isfinite().all():
            raise ValueError("Resume latent is nonfinite or has a different shape")
    with torch.no_grad():
        row = observe(decode(model, point).harden(model.spec), model.spec)
    if any(row[key] != state["final"][key] for key in ("smiles", "atom_count", "failure")):
        raise ValueError("Resumed latent no longer decodes to its saved exact chemistry")
    return state, point, anchor


def _start(model, oracle, config, draw):
    generator = torch.Generator(device="cpu").manual_seed(config["seed"] + draw)
    point = torch.randn((1, model.config.latent), generator=generator, dtype=torch.float32)
    point = point.to(next(model.parameters()).device)
    seed = config["geometry_seed"]
    with torch.no_grad():
        decoded = decode(model, point)
        row = observe(decoded.harden(model.spec), model.spec)
        if row["smiles"] is None:
            return None, dict(status="invalid_start", baseline=row)
        probes = list(probe_rows(decoded, model.spec, oracle, seed, config["probe_limit"]))
        if {probe["present"] for probe in probes if probe.get("loss") is not None} != {False, True}:
            return None, dict(status="no_matched_add_drop", baseline=row, probes=probes)
        row.update(oracle(row, seed))
        if row["loss"] is None:
            return None, dict(status=row["native_failure"], baseline=row, probes=probes)
    return point, dict(status="started", baseline=row, probes=probes)


def _remember(state, row, point):
    state["final"] = row
    state["final_point"] = point.tolist()
    if state["best"] is None or row["loss"] < state["best"]["molecule"]["loss"]:
        state["best"] = dict(molecule=row, point=state["final_point"], draw=state["active_draw"],
                             at_update=state["completed_updates"])


def _accept(state, row, point, baseline):
    changed = row["smiles"] != baseline["smiles"]
    count_delta = row["atom_count"] - baseline["atom_count"]
    state["completed_updates"] += 1
    state["trajectory_updates"] += 1
    state["molecule_changing_updates"] += changed
    state["trajectory_chemistry_updates"] += changed
    state["atom_adding_updates"] += count_delta > 0
    state["atom_removing_updates"] += count_delta < 0
    state["same_chemistry_updates"] = 0 if changed else state["same_chemistry_updates"] + 1
    _remember(state, row, point)
    if changed and (state["best_guided"] is None or row["loss"] < state["best_guided"]["molecule"]["loss"]):
        state["best_guided"] = dict(molecule=row, point=state["final_point"], draw=state["active_draw"],
                                    at_update=state["completed_updates"])


def _retire(state):
    state["unproductive_trajectories"] = (0 if state["trajectory_chemistry_updates"]
                                          else state["unproductive_trajectories"] + 1)
    state.update(active_draw=None, trajectory_updates=0, trajectory_chemistry_updates=0,
                 same_chemistry_updates=0)


def _fresh(oracle, row, seeds):
    return None if row is None else {str(seed): oracle(row, seed) for seed in seeds}


def run(config_path, output, job_id, *, resume, wall_seconds, max_stage_updates=None):
    started = time.monotonic()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if "fresh_geometry_seeds" not in config:
        config["fresh_geometry_seeds"] = [config["fresh_geometry_seed"]]
    config.setdefault("max_unproductive_trajectories", 32)
    reserve = validate_config(config)
    if (not math.isfinite(wall_seconds) or wall_seconds <= 0 or (max_stage_updates is not None
            and (type(max_stage_updates) is not int or max_stage_updates < 1))):
        raise ValueError("Require positive stage limits")
    output.mkdir(parents=True, exist_ok=False)
    model, native, pocket, identity = _load_models(config_path, config)
    model_state = tensor_tree_sha256(model.state_dict())
    native_state = native.snapshot_state()
    state, point, anchor = _restore(resume, identity, model, config)
    previous_updates = state["completed_updates"]
    total_native_left = config["max_total_native_calls"] - state["native_attempts"]
    stage_native_limit = min(config["max_stage_native_calls"], total_native_left)
    if proof := os.environ.get("DBMOL_LAUNCH_READY"):
        write_json(Path(proof), dict(status="models_and_inputs_ready", job_id=job_id,
                                    nonce=os.environ["DBMOL_LAUNCH_NONCE"]))

    seeds = config["fresh_geometry_seeds"]
    with (tempfile.TemporaryDirectory(prefix="dbmol-occupancy-", dir="/tmp") as scratch,
          (output / "events.jsonl").open("x", encoding="utf-8") as events):
        oracle = ProtenixOracle(native, pocket, Path(scratch), stage_native_limit)

        def event(kind, **fields):
            events.write(json.dumps(dict(event=kind, **fields), allow_nan=False) + "\n")
            events.flush()

        while True:
            if state["completed_updates"] == config["updates"]:
                status, reason = "completed", "update_target"
                break
            if point is None and state["failed_draws"] >= config["max_start_draws"]:
                status, reason = "stalled", "consecutive_failed_starts"
                break
            if point is None and state["unproductive_trajectories"] >= config["max_unproductive_trajectories"]:
                status, reason = "stalled", "consecutive_unchanged_trajectories"
                break
            if point is None and state["next_draw"] >= config["max_draws"]:
                status, reason = "budget_limited", "prior_draw_budget"
                break
            if oracle.calls + reserve > total_native_left:
                status, reason = "budget_limited", "total_native_budget"
                break
            if oracle.calls + reserve > stage_native_limit:
                status, reason = "segment_completed", "stage_native_budget"
                break
            if time.monotonic() - started >= wall_seconds:
                status, reason = "segment_completed", "stage_time_budget"
                break
            if max_stage_updates is not None and state["completed_updates"] - previous_updates >= max_stage_updates:
                status, reason = "segment_completed", "stage_update_budget"
                break

            if point is None:
                draw = state["next_draw"]
                state["next_draw"] += 1
                point, record = _start(model, oracle, config, draw)
                event("start", draw=draw, **record)
                if point is None:
                    state["failed_draws"] += 1
                    continue
                anchor = point
                state["active_draw"] = draw
                state["failed_draws"] = 0
                state["trajectories_started"] += 1
                _remember(state, record["baseline"], point)
                if state["initial"] is None:
                    state["initial"] = record["baseline"]
                continue

            next_point, record = occupancy_step(model, point, anchor, oracle, config, sign=1)
            retire = record["status"]
            if retire == "accepted":
                point = next_point
                _accept(state, record["trials"][-1], point, record["baseline"])
                if state["same_chemistry_updates"] >= config["max_same_chemistry_updates"]:
                    retire = "same_chemistry_plateau"
                elif state["trajectory_updates"] >= config["max_trajectory_updates"]:
                    retire = "trajectory_update_budget"
                else:
                    retire = None
            event("update", draw=state["active_draw"], completed_updates=state["completed_updates"],
                  retire=retire, **record)
            if retire is not None:
                _retire(state)
                point, anchor = None, None

        # The last bounded operation reserved these calls. If a prior segment
        # already scored the initial molecule, carry its receipt forward.
        if stage_native_limit - oracle.calls >= 4 * len(seeds):
            if state["initial_fresh"] is None:
                state["initial_fresh"] = _fresh(oracle, state["initial"], seeds)
            final_fresh = _fresh(oracle, state["final"], seeds)
            best_fresh = _fresh(oracle, state["best"]["molecule"] if state["best"] else None, seeds)
            best_guided_fresh = _fresh(
                oracle, state["best_guided"]["molecule"] if state["best_guided"] else None, seeds)
        else:
            final_fresh = best_fresh = best_guided_fresh = None
        state["native_attempts"] += oracle.calls

    require_frozen_eval(model)
    if tensor_tree_sha256(model.state_dict()) != model_state:
        raise RuntimeError("Frozen VAE state changed")
    native.assert_unchanged(native_state)
    state.update(stage_status=status, point=point.tolist() if point is not None else None,
                 anchor=anchor.tolist() if anchor is not None else None)
    state_path = output / "state.json"
    write_json(state_path, state)
    result = dict(status=status, reason=reason, job_id=job_id, state=str(state_path.resolve()),
                  updates_target=config["updates"], updates_this_segment=state["completed_updates"] - previous_updates,
                  native_attempts_this_segment=oracle.calls,
                  completed_updates=state["completed_updates"],
                  molecule_changing_updates=state["molecule_changing_updates"],
                  atom_adding_updates=state["atom_adding_updates"],
                  atom_removing_updates=state["atom_removing_updates"],
                  trajectories_started=state["trajectories_started"],
                  restarts=max(0, state["trajectories_started"] - 1),
                  native_attempts=state["native_attempts"],
                  prior_draws=state["next_draw"], elapsed_seconds=time.monotonic() - started,
                  initial=state["initial"], initial_fresh=state["initial_fresh"], final=state["final"],
                  final_fresh=final_fresh, best=state["best"], best_fresh=best_fresh,
                  best_guided=state["best_guided"], best_guided_fresh=best_guided_fresh,
                  count_policy="unmodified_decoder_argmax",
                  loss="mean_negative_log_pocket_contact_all_ligand_atoms")
    write_json(output / "result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--wall-seconds", type=float, required=True)
    parser.add_argument("--max-stage-updates", type=int)
    args = parser.parse_args(argv)
    job_id = os.environ.get("LSB_JOBID", "")
    if not job_id.isdigit() or not torch.cuda.is_available():
        raise RuntimeError("Run native Protenix inside an allocated GPU job")
    result = run(args.config, args.output, job_id, resume=args.resume, wall_seconds=args.wall_seconds,
                 max_stage_updates=args.max_stage_updates)
    print(json.dumps({key: result[key] for key in ("status", "reason", "completed_updates",
                                                 "updates_this_segment", "state")}))


if __name__ == "__main__":
    main()
