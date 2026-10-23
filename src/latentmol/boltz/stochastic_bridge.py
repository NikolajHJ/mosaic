"""Matched categorical sampling of saved fixed-occupancy Boltz pilot endpoints.

This is a discrete bridge audit, not a differentiable Boltz or DeFoG objective.
It consumes the *saved* relaxed states from source017 so the sampled law is the
same one that was optimized, without another model forward or pose assumption.
Every draw is retained, including invalid graphs and duplicate molecules.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from latentmol.chemistry import inspect_graph
from latentmol.experiments import ExperimentRun, file_sha256, write_json
from latentmol.representation import HardGraphBatch, RelaxedGraph
from latentmol.sampling import sample_fixed_relaxed_graph

from .contracts import PreparedComplex


SELECTED_STARTS = (0, 1, 4, 5)
SELECTED_ARMS = ("target", "joint")
SAMPLES_PER_ENDPOINT = 8
REPEAT = 0


def _read_json(path: Path) -> dict | list:
    return json.loads(path.read_text())


def _check_file(path: Path, expected_sha: str) -> str:
    actual = file_sha256(path)
    if actual != expected_sha:
        raise ValueError(f"SHA256 mismatch for {path}: {actual} != {expected_sha}")
    return actual


def _inspect(hard: HardGraphBatch, context: PreparedComplex) -> dict:
    if hard.atoms.shape[0] != 1:
        raise ValueError("This study accepts one graph per draw")
    outcome = inspect_graph(hard.atoms[0], hard.charges[0], hard.bonds[0], context.spec)
    active = hard.atoms[0] != 0
    charges = torch.as_tensor(context.spec.formal_charges)[hard.charges[0].long()]
    return {"smiles": outcome.smiles, "failure_reason": outcome.failure_reason,
            "failure_detail": outcome.detail,
            "heavy_atoms": int(active.sum()),
            "charged_atoms": int((active & (charges != 0)).sum())}


def _summary(rows: list[dict]) -> dict:
    valid = [row for row in rows if row["hard"]["smiles"]]
    return {"draws": len(rows), "valid": len(valid),
            "unique_valid": len({row["hard"]["smiles"] for row in valid}),
            "valid_same_as_argmax": sum(row["same_as_argmax"] is True for row in rows),
            "valid_changed_from_start": sum(row["changed_from_start"] is True for row in rows),
            "valid_changed_from_baseline": sum(row["changed_from_baseline"] is True for row in rows),
            "draws_with_formal_charge": sum(row["hard"]["charged_atoms"] > 0 for row in rows),
            "failures": dict(Counter(row["hard"]["failure_reason"] for row in rows
                                     if row["hard"]["failure_reason"] is not None))}


def run_bridge(parent_run: Path, run_root: Path) -> Path:
    parent_run = parent_run.resolve()
    manifest = _read_json(parent_run / "run.json")
    if manifest.get("status") != "completed" or manifest.get("name") not in (
            "ae-boltz-latent-pilot", "vae-boltz-latent-pilot"):
        raise ValueError("Require a completed source017 AE/VAE Boltz pilot")
    parent = manifest["config"]
    mode = parent["mode"]
    if parent.get("starts_per_repeat") != 8 or parent.get("steps") != 10:
        raise ValueError("Parent trajectory protocol differs from source017")
    prepared = Path(parent["prepared"])
    _check_file(prepared, parent["prepared_sha256"])
    _check_file(Path(parent["checkpoint"]), parent["checkpoint_sha256"])
    _check_file(Path(parent["reference"]), parent["reference_sha256"])
    context = PreparedComplex.load(prepared)
    if int(context.active.sum()) != 13:
        raise ValueError("The planned bridge requires the original 13 active slots")
    final_path = parent_run / "final_attempts.json"
    baseline_path = parent_run / "baseline.json"
    final_rows = _read_json(final_path)
    baseline = _read_json(baseline_path)
    if not baseline["hard"]["smiles"]:
        raise ValueError("Parent encoded baseline must be a valid hard molecule")
    parent_rows = {(row["arm"], row["repeat"], row["start_index"]): row
                   for row in final_rows}
    selected_paths = [parent_run / f"trajectory-r{REPEAT}-i{index}-{arm}.pt"
                      for arm in SELECTED_ARMS for index in SELECTED_STARTS]
    hashes = {path.name: file_sha256(path) for path in selected_paths}
    selected = [{"arm": arm, "repeat": REPEAT, "start_index": index,
                 "parent_row": parent_rows[(arm, REPEAT, index)],
                 "trajectory_sha256": hashes[f"trajectory-r{REPEAT}-i{index}-{arm}.pt"]}
                for arm in SELECTED_ARMS for index in SELECTED_STARTS]
    config = {"mode": mode, "parent_run": str(parent_run),
              "parent_run_sha256": file_sha256(parent_run / "run.json"),
              "parent_source_sha256": manifest.get("source_archive_sha256"),
              "parent_final_attempts_sha256": file_sha256(final_path),
              "parent_baseline_sha256": file_sha256(baseline_path),
              "trajectory_sha256": hashes, "prepared_sha256": parent["prepared_sha256"],
              "checkpoint_sha256": parent["checkpoint_sha256"],
              "reference_sha256": parent["reference_sha256"],
              "protocol": {"repeat": REPEAT, "arms": list(SELECTED_ARMS),
                           "start_indices": list(SELECTED_STARTS),
                           "samples_per_endpoint": SAMPLES_PER_ENDPOINT,
                           "fixed_occupied_slots": 13, "temperature": 1.0,
                           "seed_rule": "31000 + 1000*arm_index + 100*start_index + draw_index"}}
    with ExperimentRun(run_root, f"{mode}-stochastic-bridge", config, Path.cwd()) as run:
        write_json(run.path / "selected_parents.json", selected)
        run.annotate(selected_parents_sha256=file_sha256(run.path / "selected_parents.json"))
        sampled = {"atoms": [], "charges": [], "bonds": []}
        records: list[dict] = []
        for arm_index, arm in enumerate(SELECTED_ARMS):
            for index in SELECTED_STARTS:
                key = (arm, REPEAT, index)
                if key not in parent_rows:
                    raise ValueError(f"Missing parent endpoint {key}")
                parent_row = parent_rows[key]
                trajectory = torch.load(parent_run / f"trajectory-r{REPEAT}-i{index}-{arm}.pt",
                                        map_location="cpu", weights_only=True)
                if trajectory["arm"] != arm or trajectory["repeat"] != REPEAT or trajectory["start_index"] != index:
                    raise ValueError(f"Trajectory metadata mismatch: {key}")
                if trajectory["records"][-1]["hard"] != parent_row["hard"]:
                    raise ValueError(f"Trajectory and final endpoint disagree: {key}")
                state = trajectory["states"][-1]
                saved = state["soft_fixed_occupancy_graph"]
                graph = RelaxedGraph(saved["atoms"], saved["charges"], saved["bonds"])
                argmax_hard = HardGraphBatch(**state["hard_fixed_occupancy_graph"])
                argmax = _inspect(argmax_hard, context)
                if argmax["smiles"] != parent_row["hard"]["smiles"]:
                    raise ValueError(f"Saved hard state and final endpoint disagree: {key}")
                for draw_index in range(SAMPLES_PER_ENDPOINT):
                    seed = 31000 + 1000 * arm_index + 100 * index + draw_index
                    generator = torch.Generator(device="cpu").manual_seed(seed)
                    hard = sample_fixed_relaxed_graph(graph, context.spec, generator=generator,
                                                      fixed_active=context.active)
                    info = _inspect(hard, context)
                    smiles = info["smiles"]
                    own_start = parent_row["start_hard"]["smiles"]
                    baseline_smiles = baseline["hard"]["smiles"]
                    row = {"arm": arm, "repeat": REPEAT, "start_index": index,
                           "start_radius": parent_row["start_radius"],
                           "draw_index": draw_index, "seed": seed,
                           "hard": info, "argmax": argmax,
                           "own_start_smiles": own_start,
                           "encoded_baseline_smiles": baseline_smiles,
                           "same_as_argmax": (smiles == argmax["smiles"])
                           if smiles and argmax["smiles"] else None,
                           "changed_from_start": (smiles != own_start) if smiles and own_start else None,
                           "changed_from_baseline": (smiles != baseline_smiles)
                           if smiles and baseline_smiles else None}
                    records.append(row)
                    for name in sampled:
                        sampled[name].append(getattr(hard, name)[0].clone())
                    run.log({"stage": "draw", **row})
        if len(records) != len(SELECTED_ARMS) * len(SELECTED_STARTS) * SAMPLES_PER_ENDPOINT:
            raise AssertionError("Incorrect sampling budget")
        torch.save({name: torch.stack(values) for name, values in sampled.items()},
                   run.path / "sampled_graphs.pt")
        write_json(run.path / "draws.json", records)
        arms = {arm: _summary([row for row in records if row["arm"] == arm]) for arm in SELECTED_ARMS}
        summary = {"mode": mode, "samples": _summary(records), "by_arm": arms,
                   "claim_limit": "Discrete fixed-occupancy sampling only; no DeFoG projection, native Boltz rescoring, or gradient through samples."}
        write_json(run.path / "summary.json", summary)
        run.log({"stage": "complete", **summary})
        return run.path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    run_bridge(args.parent_run, args.run_root)


if __name__ == "__main__":
    main()
