"""Matched temperature trajectories with frozen VAE and Protenix contact models."""

import argparse
from contextlib import ExitStack
from dataclasses import dataclass, replace
from functools import partial
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from latentmol.chemistry import inspect_graph
from latentmol.design.latent_protenix import LatentDesignConfig, _json, choose_active, run_design
from latentmol.design.protenix_step_search import matched_designs, normalized_update, scored_observations
from latentmol.evaluation.evaluate import load_model
from latentmol.guidance.mosaic_protenix import MosaicProtenixScorer
from latentmol.multimodel.external import file_sha256
from latentmol.protenix_occupancy.one_atom import labels
from latentmol.representation import HardGraphBatch


CANDIDATE_ROLES = ("initial", "first_distinct_valid", "endpoint")


@dataclass(frozen=True)
class TemperaturePilotConfig(LatentDesignConfig):
    cooling_updates: int = 32

    def __post_init__(self):
        super().__post_init__()
        if type(self.cooling_updates) is not int or not 0 < self.cooling_updates <= self.updates:
            raise ValueError("Temperature cooling must fit the pilot update budget")

    @property
    def temperature_end_step(self):
        return self.cooling_updates


def design_arms(config, folder):
    base = json.loads((folder / config["base_design"]).read_text(encoding="utf-8"))
    if search := config.get("step_search"):
        base.update(temperature_initial=1., temperature_final=1.)
        designs = matched_designs(LatentDesignConfig(**base), search)
        if temperatures := config.get("normalized_search_temperatures"):
            if ("same_molecule_panel" in config or len(temperatures) < 2
                    or len(set(temperatures.values())) != len(temperatures)
                    or any(type(t) not in (int, float) or not math.isfinite(t) or t <= 0
                           for t in temperatures.values())):
                raise ValueError("Temperature search needs distinct positive temperatures on ordinary starts")
            return {name: replace(designs["normalized_search"], temperature_initial=t, temperature_final=t)
                    for name, t in temperatures.items()}
        if "same_molecule_panel" in config:
            from latentmol.design.same_molecule_comparison import ARM_NAMES
            return dict.fromkeys(ARM_NAMES, designs["normalized_search"])
        return designs
    updates = config["updates"]
    base.update(updates=updates, snapshot_interval=updates, checkpoint_interval=updates)
    cooling = TemperaturePilotConfig(**base, cooling_updates=updates)
    variants = config.get("temperature_arms")
    if variants is not None:
        keys = {"temperature_initial", "temperature_final", "cooling_updates"}
        if len(variants) < 2 or any(set(values) != keys for values in variants.values()):
            raise ValueError("Temperature arms may differ only in temperature and cooling duration")
        return {name: replace(cooling, **values) for name, values in variants.items()}
    return {"cooling": cooling,
            "constant_0p7": replace(cooling, temperature_initial=.7, temperature_final=.7),
            "constant_1p0": replace(cooling, temperature_initial=1., temperature_final=1.)}


def start_state(model, row, count):
    with torch.no_grad():
        if row["kind"] == "encoded":
            hard = labels(row["smiles"], model.spec, "cpu")
            if int(hard.atoms.count_nonzero()) != count:
                raise ValueError("Encoded molecule and fixed native atom count differ")
            point, _ = model.encode(hard.to_relaxed(model.spec))
        elif row["kind"] == "prior":
            generator = torch.Generator().manual_seed(row["seed"])
            point = torch.randn((1, model.config.latent), generator=generator)
        else:
            raise ValueError("Unknown temperature study start kind")
        active = choose_active(model.decode(point), model.spec, count)
    return dict(latent=point, active=active, step=0, kind=row["kind"], provenance=row)


def study_starts(config, config_folder, model, configs, checkpoint_sha, count):
    if panel := config.get("same_molecule_panel"):
        from latentmol.design.same_molecule_starts import load_matched_starts
        path = config_folder / panel["path"]
        if file_sha256(path) != panel["sha256"]:
            raise ValueError("Same-molecule start panel hash mismatch")
        parents = list(load_matched_starts(model, path, checkpoint_sha, count))
        if len(parents) != panel["parents"] or len({p[1] for p in parents}) != len(parents):
            raise ValueError("Same-molecule panel has missing or duplicate parent molecules")
        if len({p[0] for p in parents}) != len(parents) or any(set(p[2]) != set(configs) for p in parents):
            raise ValueError("Same-molecule panel names or arms are incomplete")
        start, stop = config.get("parent_slice", [0, len(parents)])
        if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= len(parents):
            raise ValueError("Invalid half-open matched parent slice")
        yield from parents[start:stop]
        return
    for source in config["starts"]:
        state = start_state(model, source, count)
        yield source["name"], None, dict.fromkeys(configs, state)


def candidate_panel(rows):
    """Select fixed observation roles without looking at any contact scores."""
    start, endpoint = rows[0], rows[-1]
    valid = {row["smiles"] for row in rows if row["valid"]}
    distinct = valid - {start["smiles"]}
    first = next((row for row in rows[1:] if row["valid"] and row["smiles"] in distinct), None)
    return dict(initial=start, first_distinct_valid=first, endpoint=endpoint,
                valid_observations=sum(row["valid"] for row in rows),
                observations=len(rows), distinct_valid_molecules=len(valid),
                distinct_valid_edits=len(distinct))


def hard_at(raw, step):
    for path in sorted(raw.glob("states-*.npz")):
        with np.load(path) as block:
            rows = np.flatnonzero(block["steps"] == step)
            if rows.size == 0:
                continue
            if rows.size != 1:
                raise ValueError("Hard candidate step is ambiguous")
            index = rows.item()
            atoms = torch.as_tensor(block["atoms"][index], dtype=torch.long)[None]
            charges = torch.as_tensor(block["charges"][index], dtype=torch.long)[None]
            left, right = block["upper_left"], block["upper_right"]
            bonds = torch.zeros((1, atoms.shape[1], atoms.shape[1]), dtype=torch.long)
            bonds[0, left, right] = torch.as_tensor(block["bonds_upper"][index], dtype=torch.long)
            bonds[0, right, left] = bonds[0, left, right]
            return HardGraphBatch(atoms, charges, bonds)
    raise ValueError(f"Missing hard graph at step {step}")


@torch.no_grad()
def unconditioned_panel(model, raw, output):
    """Decode the saved points with normal argmax occupancy for comparison."""
    rows = []
    with output.open("x", encoding="utf-8") as stream:
        for path in sorted(raw.glob("states-*.npz")):
            with np.load(path) as block:
                for step, latent in zip(block["steps"], block["latent"]):
                    decoded = model.decode(torch.from_numpy(latent)[None])
                    if any(not torch.isfinite(value).all() for value in (decoded.atoms, decoded.charges, decoded.bonds)):
                        raise FloatingPointError("Nonfinite unconditioned decoder output")
                    hard = decoded.harden(model.spec)
                    observation = inspect_graph(hard.atoms[0], hard.charges[0], hard.bonds[0], model.spec)
                    row = dict(step=int(step), atoms=int(hard.atoms.count_nonzero()),
                               valid=observation.smiles is not None, smiles=observation.smiles,
                               failure=observation.failure_reason)
                    rows.append(row)
                    stream.write(json.dumps(row) + "\n")
    return candidate_panel(rows)


def discrete_scores(arms, spec, active, factory, output):
    with factory(spec, active, output / "native-hard.stderr.log") as scorer, torch.no_grad():
        for arm_name, arm in arms.items():
            raw = output / arm_name / "arms" / "latent" / "raw-blocks"
            scores = {}
            for role in CANDIDATE_ROLES:
                row = arm["candidates"][role]
                if row is None or not row["valid"]:
                    continue
                step = row["step"]
                if step not in scores:
                    hard = hard_at(raw, step)
                    loss, _ = scorer(hard.to_relaxed(spec))
                    value = float(loss)
                    if not math.isfinite(value):
                        raise FloatingPointError("Nonfinite discrete fixed-context score")
                    scores[step] = value
                row["hard_fixed_context_loss"] = scores[step]
            initial = arm["candidates"]["initial"].get("hard_fixed_context_loss")
            for role in CANDIDATE_ROLES[1:]:
                row = arm["candidates"][role]
                if initial is not None and row is not None and "hard_fixed_context_loss" in row:
                    row["hard_fixed_context_change_from_initial"] = row["hard_fixed_context_loss"] - initial


def run(config_path, checkpoint, output, mosaic_python, mosaic_config):
    with ExitStack() as resident_lifetime:
        return _run(config_path, checkpoint, output, mosaic_python, mosaic_config, resident_lifetime)


def _run(config_path, checkpoint, output, mosaic_python, mosaic_config, resident_lifetime):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    identity = file_sha256(checkpoint)
    if identity != config["checkpoint_sha256"]:
        raise ValueError("Frozen study checkpoint changed")
    native_config = json.loads(mosaic_config.read_text(encoding="utf-8"))
    model, _ = load_model(checkpoint, "cpu")
    model.requires_grad_(False)
    configs = design_arms(config, config_path.parent)
    fixed_count = next(iter(configs.values())).occupied_atoms
    edit_limit = config.get("fresh_edit_limit")
    search = config.get("step_search")
    matched = config.get("same_molecule_panel")
    compare_temperatures = bool(config.get("normalized_search_temperatures"))
    if compare_temperatures and not search:
        raise ValueError("Normalized temperature comparison requires a search budget")
    if matched and not search:
        raise ValueError("Same-molecule comparison requires a common bounded search")
    call_budget = 1 + search["iterations"] * (1 + len(search["radii"])) if search else None
    if search and edit_limit is None:
        raise ValueError("Declare an equal fresh edit budget for the step-search comparison")
    output.mkdir(parents=True, exist_ok=False)

    def factory(spec, active, log):
        return MosaicProtenixScorer(spec, active, command=[mosaic_python, "-m",
            "latentmol.guidance.mosaic_protenix_worker"], config=native_config, stderr_path=log,
            timeout=180, startup_timeout=240)

    starts = {}
    resident = None
    for name, parent_smiles, states in study_starts(config, config_path.parent, model, configs, identity, fixed_count):
        arms = starts[name] = {}
        initial = None
        for arm_name, design in configs.items():
            start = states[arm_name]
            folder = output / name / arm_name
            update = (partial(normalized_update, radii=search["radii"],
                              minimum_improvement=search["minimum_improvement"],
                              keep_zero_gradient_budget=bool(matched or compare_temperatures))
                      if matched or compare_temperatures or arm_name == "normalized_search" else None)
            def make_scorer(spec, active):
                nonlocal resident
                if matched:
                    from latentmol.design.same_molecule_comparison import OrderedScorer
                    native_active = torch.arange(active.numel()) < fixed_count
                    if resident is None:
                        resident = resident_lifetime.enter_context(factory(
                            spec, native_active, output / "native.stderr.log"))
                    return OrderedScorer(resident, active, start["native_order"], native_active)
                return factory(spec, active, folder / "native.stderr.log")
            result = run_design(model, make_scorer,
                design, folder, checkpoint_sha256=identity, start_state=start, step_update=update,
                native_metadata=dict(config_path=str(mosaic_config), config_sha256=file_sha256(mosaic_config)))
            if result["updates_this_segment"] != design.updates:
                raise RuntimeError("Matched temperature arm did not reach its update budget")
            history = folder / "arms" / "latent"
            with (history / "candidates.jsonl").open(encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream]
            panel = candidate_panel(rows)
            if not matched and initial is not None and panel["initial"] != initial:
                raise RuntimeError("Temperature arms did not share the same initial hard graph")
            initial = panel["initial"]
            unconditioned = unconditioned_panel(model, history / "raw-blocks", history / "unconditioned.jsonl")
            if matched and any(p["initial"]["smiles"] != parent_smiles for p in (panel, unconditioned)):
                raise RuntimeError("Matched arm did not start from the agreed parent chemistry")
            arms[arm_name] = dict(design=result, candidates=panel, unconditioned=unconditioned)
            observed = None
            if search:
                with (history / "updates.jsonl").open(encoding="utf-8") as stream:
                    updates = [json.loads(line) for line in stream]
                with (history / "unconditioned.jsonl").open(encoding="utf-8") as stream:
                    raw_rows = [json.loads(line) for line in stream]
                observed = scored_observations(rows, raw_rows, updates)
                with (history / "scorer-observations.jsonl").open("x", encoding="utf-8") as stream:
                    for item in observed:
                        stream.write(json.dumps(item, allow_nan=False) + "\n")
                if (matched or compare_temperatures) and len(observed) != call_budget:
                    raise RuntimeError("Matched arm consumed a different optimization scorer budget")
                arms[arm_name]["scorer_budget"] = dict(planned=call_budget,
                    actual=len(observed), centers=len(rows), trial_calls=len(observed) - len(rows))
                arms[arm_name]["all_scored"] = {
                    key: candidate_panel([dict(item[key], step=item["step"]) for item in observed])
                    for key in ("fixed", "unconditioned")}
            if edit_limit is not None:
                from latentmol.design.temperature_candidates import select_valid_candidates
                committed_smiles = {r["smiles"] for r in rows if r["valid"]}
                if observed is not None:
                    rows = [dict(item["fixed"], step=item["step"]) for item in observed]
                selection = select_valid_candidates(rows, edit_limit=edit_limit)
                if observed is not None:
                    for row in selection["candidates"]:
                        item = observed[row["step"]]
                        row.update({key: item[key] for key in ("trajectory_step", "kind", "radius", "selected")})
                        if matched or compare_temperatures:
                            row["on_committed_trajectory"] = row["smiles"] in committed_smiles
                arms[arm_name]["fresh_selection"] = selection
        if not search:
            discrete_scores(arms, model.spec, start["active"], factory, output / name)
        _json(output / "progress.json", dict(status="running", starts=starts))

    resident_lifetime.close()
    fresh_candidates = []
    for start_name, arms in starts.items():
        for arm_name, arm in arms.items():
            if edit_limit is not None:
                fresh_candidates.extend(dict(row, start=start_name, arm=arm_name)
                                        for row in arm["fresh_selection"]["candidates"])
                continue
            for role in CANDIDATE_ROLES:
                row = arm["candidates"][role]
                if row is not None and row["valid"]:
                    fresh_candidates.append(dict(start=start_name, arm=arm_name, role=role,
                                                 step=row["step"], smiles=row["smiles"]))
    _json(output / "fresh-candidates.json", dict(candidates=fresh_candidates))
    result = dict(status="completed", checkpoint_sha256=identity, job_id=os.environ.get("LSB_JOBID"), starts=starts,
                  fixed_context="Mosaic negative mean ligand-max-pocket contact probability; lower is better",
                  candidate_selection="fixed-25 argmax: valid initial, first distinct valid, valid final",
                  unconditioned_scope="hard occupancy from each saved latent; chemistry observations only")
    if search:
        result.update(step_search=search, updates_per_arm={name: arm.updates for name, arm in configs.items()},
                      optimization_scorer_budget_per_arm=call_budget)
    if matched:
        result["same_molecule_panel"] = matched
        result["parent_slice"] = config.get("parent_slice", [0, matched["parents"]])
        result["study_config_sha256"] = file_sha256(config_path)
        result["native_config_sha256"] = file_sha256(mosaic_config)
    elif not search:
        result["updates_per_arm"] = config["updates"]
    if edit_limit is not None:
        result["fixed_context_selection"] = result["candidate_selection"]
        result["candidate_selection"] = ("fixed-25 argmax: valid initial plus distinct valid edits, "
            "prioritizing a valid endpoint or latest valid edit and time-spaced intermediate edits")
        result["fresh_edit_limit_per_arm"] = edit_limit
        result["fresh_candidate_count"] = len(fresh_candidates)
    if search:
        result.pop("fixed_context_selection", None)
        result["candidate_selection"] += " among all scorer calls, including uncommitted trials"
        result["fresh_step_coordinate"] = "zero-based scorer call index"
        result["fixed_context_hard_rescoring"] = False
    _json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mosaic-python", required=True)
    parser.add_argument("--mosaic-config", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--plan")
    parser.add_argument("--plan-sha256")
    args = parser.parse_args()
    if not os.environ.get("LSB_JOBID", "").isdigit():
        raise RuntimeError("Run the native temperature study inside an allocated job")
    torch.set_num_threads(args.threads)
    run(args.config, args.checkpoint, args.output, args.mosaic_python, args.mosaic_config)


if __name__ == "__main__":
    main()
