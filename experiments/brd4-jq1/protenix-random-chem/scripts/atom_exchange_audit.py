"""Allocated CPU comparison of one-atom exchange and bond-only repair."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib.abc import MetaPathFinder
from importlib.util import spec_from_file_location
import json
import math
import os
from pathlib import Path
import sys
import time


VERSION = "mosaic-atom-exchange-audit-v1"
POINTS = ("sparse23-invalid", "sparse41-valid", "shared-repaired")
ATTEMPTS = 64
BASELINE_JOB = "29890650"
PERTURBATION = 0.001
OVERLAY = {"structural", "atom_exchange", "environment_policy"}
PACKAGE = "mosaic.small_molecule.chemistry"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_binding(record):
    path = Path(record["path"])
    actual = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            actual.update(block)
    require(path.stat().st_size == record["bytes"] and actual.hexdigest() == record["sha256"],
            f"Input binding changed: {path}")
    return path


def check_inputs(plan, plan_path):
    require(plan.get("version", VERSION) == VERSION, "Unknown exchange audit version")
    require(plan.get("optimizer_updates", 0) == 0, "This audit cannot optimize a graph")
    require(tuple(p["id"] for p in plan["points"]) == POINTS, "Expected three predeclared checkpoints")
    require(Path(plan["output"]) == plan_path.resolve().parent.parent / "payload",
            "Output escapes owned run")
    require(type(plan["internal_seconds"]) is int and 1 <= plan["internal_seconds"] <= 7200,
            "Require bounded runtime")
    require(Path(plan["source_root"], "src/mosaic/small_molecule/chemistry/__init__.py").is_file(),
            "Original Mosaic source is missing")
    for name in OVERLAY:
        require((plan_path.parent / f"{name}.py").is_file(), f"Sealed source is missing: {name}")
    records = [plan["benchmark"], *(p["snapshot"] for p in plan["points"])]
    for record in records:
        check_binding(record)
    benchmark = json.loads(Path(plan["benchmark"]["path"]).read_text(encoding="utf-8"))
    require(benchmark["job_id"] == BASELINE_JOB and benchmark["status"] == "completed"
            and benchmark["inputs_unchanged"] and benchmark["optimizer_updates"] == 0,
            "Require the completed saved masked benchmark")
    for expected, recorded in zip(plan["points"], benchmark["checkpoints"], strict=True):
        require(all(expected[k] == recorded[k] for k in ("id", "step", "snapshot")),
                "Saved checkpoint differs from paired benchmark")
    return records, benchmark


class SealedChemistry(MetaPathFinder):
    """Load the three sealed modules before the package's eager structural import."""

    def __init__(self, source):
        self.source = source

    def find_spec(self, fullname, path=None, target=None):
        package, _, name = fullname.rpartition(".")
        if package == PACKAGE and name in OVERLAY:
            return spec_from_file_location(fullname, self.source / f"{name}.py")
        return None


def paired_rows(benchmark, checkpoint, method):
    cell, = (c for c in benchmark["cells"] if c["checkpoint"] == checkpoint and c["method"] == method)
    rows = cell["proposals"]
    require(cell["attempts"] == len(rows) == ATTEMPTS and [r["attempt"] for r in rows] == list(range(ATTEMPTS)),
            "Original proposal ledger is incomplete")
    return rows


def summarize(rows, key):
    values = [row[key] for row in rows if row[key]["valid"]]
    return {
        "valid": len(values),
        "distinct_valid": len({v["smiles"] for v in values}),
        "atom_swaps": sum(bool(v["atom_edits"]) for v in values),
        "bond_edits_total": sum(v["bond_edits"] for v in values),
        "sulfur_introductions": sum(v["introduced_sulfur"] for v in values),
        "sulfur_removals": sum(v["removed_sulfur"] for v in values),
        "swap_directions": dict(Counter(f"{old}->{new}" for v in values for old, new in v["swap_directions"])),
    }


def run(plan, plan_path, records, benchmark):
    job = os.environ.get("LSB_JOBID", "")
    require(job.isdigit() and (not plan.get("queue") or os.environ.get("LSB_QUEUE") == plan["queue"]),
            "Require allocated CPU job")
    started = time.monotonic()
    source = plan_path.parent.resolve()
    sys.path.insert(0, str(Path(plan["source_root"]) / "src"))
    overlay = SealedChemistry(source)
    sys.meta_path.insert(0, overlay)
    try:
        from mosaic.small_molecule.chemistry import mosaic_spec
        from mosaic.small_molecule.chemistry._reference.chemistry import encode_smiles, inspect_graph
        from mosaic.small_molecule.chemistry._reference.chemistry_regularization import ALLOWED_VALENCES
        from mosaic.small_molecule.chemistry._reference.probability_repair import ProbabilityRepairConfig
        from mosaic.small_molecule.chemistry._reference.quality import graph_identity
        from mosaic.small_molecule.chemistry.atom_exchange import search_atom_exchange
        from mosaic.small_molecule.chemistry.environment_policy import BOND_ORDERS, assess_environment
    finally:
        sys.meta_path.remove(overlay)

    for name in OVERLAY:
        require(Path(sys.modules[f"{PACKAGE}.{name}"].__file__).resolve() == source / f"{name}.py",
                f"Sealed module was not loaded: {name}")
    import numpy as np

    penalty = plan.get("atom_edit_penalty", 1.0)
    maximum = plan.get("max_substitutions", 4)
    policy = ProbabilityRepairConfig()
    output = Path(plan["output"])
    output.mkdir(exist_ok=False)
    result = {
        "status": "running", "version": VERSION, "job_id": job, "plan": str(plan_path),
        "optimizer_updates": 0, "frozen_model_forwards": 0,
        "baseline_benchmark": plan["benchmark"], "atom_edit_penalty": penalty,
        "max_substitutions": maximum, "stability_log_perturbation": PERTURBATION,
        "comparisons": [],
    }

    def persist():
        payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
        require(len(payload.encode("utf-8")) <= plan.get("output_cap_bytes", 8 * 1024**2),
                "Audit result exceeded output cap")
        temporary = output / "result.json.partial"
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(output / "result.json")

    def tick():
        require(time.monotonic() - started < plan["internal_seconds"], "Audit runtime cap exceeded")

    def choose(raw, p, spec, **overrides):
        tick()
        return search_atom_exchange(raw, p, spec, policy, **{
            "atom_edit_penalty": penalty, "max_substitutions": maximum, **overrides,
        })

    def read_candidate(candidate, raw, spec):
        status = candidate["status"]
        if status not in ("already_valid", "witness_found"):
            return {"valid": False, "status": status}
        witness = candidate.get("weighted_witness") or {}
        target = raw if status == "already_valid" else tuple(
            np.asarray(witness["labels"][k], dtype=np.int64) for k in ("atoms", "charges", "bonds"))
        atoms, charges, bonds = target
        require(np.array_equal(charges, raw[1]) and np.array_equal(atoms > 0, raw[0] > 0),
                "Repair changed formal charge or occupied slots")
        native = inspect_graph(*target, spec)
        require(native.smiles is not None and encode_smiles(native.smiles, spec)["smiles"] == native.smiles,
                "Selected candidate failed native validation or canonical round trip")
        scope, = assess_environment(*[[value.tolist()] for value in target], spec)
        require(scope["fully_assessed"] and scope["local_valence_ok"] and scope["within_common_organic_scope"],
                "Selected candidate violates shared neighbor policy")
        numbers = np.asarray(spec.atomic_numbers)
        old, new = numbers[raw[0]], numbers[atoms]
        swapped = np.flatnonzero(old != new)
        for slot in np.flatnonzero((old != 16) & (new == 16)):
            charge = spec.formal_charges[int(charges[slot])]
            cap = max(ALLOWED_VALENCES[int(old[slot])][charge])
            order = sum(1.5 if spec.bond_names[label] == "AROMATIC" else BOND_ORDERS[spec.bond_names[label]]
                        for label in bonds[slot])
            require(order <= cap, "New sulfur exceeds the previous atom's cap")
        bond_edits = int(np.count_nonzero(np.triu(raw[2] != bonds, 1)))
        cost = candidate.get("repair_cost", witness.get("repair_cost", candidate["best_observed_weighted_cost"]))
        require(type(cost) in (float, int) and math.isfinite(cost), "Invalid accepted repair cost")
        return {
            "valid": True, "status": status, "smiles": native.smiles,
            "identity": graph_identity(*target, spec), "cost": cost,
            "atom_edits": len(swapped), "bond_edits": bond_edits,
            "sulfur_atoms": int(np.count_nonzero(new == 16)),
            "introduced_sulfur": int(np.count_nonzero((old != 16) & (new == 16))),
            "removed_sulfur": int(np.count_nonzero((old == 16) & (new != 16))),
            "swap_directions": [(int(old[i]), int(new[i])) for i in swapped],
            "swap_slots": [int(i) for i in swapped],
        }

    def perturb(p, seed):
        rng = np.random.default_rng(seed)

        def reweight(rows, sign, noise):
            moved = rows * np.exp(sign * PERTURBATION * noise)
            return moved / moved.sum(-1, keepdims=True)

        i, j = np.triu_indices(len(p[0]), 1)
        atom_noise = rng.uniform(-1, 1, p[0].shape)
        bond_noise = rng.uniform(-1, 1, p[2][i, j].shape)
        for sign in (-1, 1):
            edges = p[2].copy()
            edges[i, j] = edges[j, i] = reweight(edges[i, j], sign, bond_noise)
            yield reweight(p[0], sign, atom_noise), p[1], edges

    def fixture(spec, center, neighbors, orders, alternative=None):
        labels = {z: i for i, z in enumerate(spec.atomic_numbers)}
        atoms = np.array([labels[center], *(labels[z] for z in neighbors)], dtype=np.int64)
        charges = np.zeros(len(atoms), dtype=np.int64)
        bonds = np.zeros((len(atoms), len(atoms)), dtype=np.int64)
        bonds[0, 1:] = bonds[1:, 0] = orders
        p_atoms = np.eye(len(labels), dtype=float)[atoms]
        if alternative is not None:
            p_atoms[0, labels[center]], p_atoms[0, labels[alternative]] = 0.51, 0.49
        p_bonds = 0.9875 * np.eye(len(spec.bond_names))[bonds] + 0.0025
        return (atoms, charges, bonds), (p_atoms, np.ones((len(atoms), 1)), p_bonds)

    try:
        import pytest

        code = pytest.main(["-q", "-p", "no:cacheprovider", "--junitxml", str(output / "tests.xml"),
                            str(source / "test_chemistry_atom_exchange.py"),
                            str(source / "test_chemistry_structural.py")])
        result["pytest_exit_code"] = int(code)
        require(code == 0, "Allocated focused fixtures failed")
        spec = mosaic_spec(3)
        raw, p = fixture(spec, 9, (6, 6), (1, 1), alternative=8)
        low = choose(raw, p, spec, atom_edit_penalty=1.0, max_substitutions=4)
        baseline = read_candidate(low["baseline"], raw, spec)
        exchanged = read_candidate(low["selected"], raw, spec)
        require(baseline["valid"] and exchanged["swap_directions"] == [(9, 8)] and
                exchanged["bond_edits"] == 0 and exchanged["cost"] < baseline["cost"],
                "F to O fixture did not select the cheaper validated ether")
        replay = read_candidate(
            choose(raw, p, spec, atom_edit_penalty=1.0, max_substitutions=4)["selected"], raw, spec)
        require(replay == exchanged, "Atom exchange was not deterministic")
        high = choose(raw, p, spec, atom_edit_penalty=30.0, max_substitutions=4)
        require(read_candidate(high["selected"], raw, spec) == baseline,
                "Large atom penalty did not preserve the bond-only witness")
        checks = ["validated_fluorine_to_oxygen_exchange", "deterministic_replay", "higher_penalty_keeps_bonds"]
        for neighbor, expected in ((9, True), (6, False)):
            spec = mosaic_spec(7)
            raw, p = fixture(spec, 16, (neighbor,) * 6, (1,) * 6)
            scope, = assess_environment(*[[value.tolist()] for value in raw], spec)
            require(scope["within_common_organic_scope"] is expected, "Sulfur neighbor rule differed")
            proposed = choose(raw, p, spec)["selected"]
            if expected:
                require(proposed["status"] == "already_valid", "SF6 scope fixture was rejected")
                read_candidate(proposed, raw, spec)
                checks.append("sf6_stays_in_scope")
            else:
                require(proposed["status"] != "already_valid", "Six-carbon sulfur passed the scoped raw gate")
                read_candidate(proposed, raw, spec)
                checks.append("six_nonfluorine_sulfur_is_excluded")
        result["fixture_checks"] = checks

        spec = mosaic_spec(25)
        i, j = np.triu_indices(25, 1)
        neutral = np.zeros(25, dtype=np.int64)
        for point in plan["points"]:
            tick()
            with np.load(point["snapshot"]["path"], allow_pickle=False) as saved:
                require(saved["step"].shape == () and int(saved["step"]) == point["step"], "Snapshot step differs")
                x, e = (saved[k].copy() for k in ("X", "E_upper"))
            require(x.shape == (25, 6) and e.shape == (300, 5) and x.dtype == e.dtype == np.float32,
                    "Snapshot graph schema differs")
            e_full = np.zeros((25, 25, 5), dtype=float)
            e_full[i, j] = e_full[j, i] = e
            e_full[np.arange(25), np.arange(25), 0] = 1.0
            p = np.pad(x, ((0, 0), (1, 0))), np.ones((25, 1)), e_full
            rows = []
            legacy = paired_rows(benchmark, point["id"], "structural")
            for source_row in paired_rows(benchmark, point["id"], "independent"):
                atoms, upper = (np.asarray(source_row[k]) for k in ("atoms", "bonds"))
                require(atoms.shape == (25,) and upper.shape == (300,) and atoms.dtype.kind in "iu"
                        and upper.dtype.kind in "iu" and ((0 <= atoms) & (atoms < 6)).all()
                        and ((0 <= upper) & (upper < 5)).all(), "Saved draw labels differ")
                bonds = np.zeros((25, 25), dtype=np.int64)
                bonds[i, j] = bonds[j, i] = upper
                raw = atoms + 1, neutral, bonds
                then = time.monotonic()
                outcome = choose(raw, p, spec)
                seconds = time.monotonic() - then
                base = read_candidate(outcome["baseline"], raw, spec)
                selected = read_candidate(outcome["selected"], raw, spec)
                require(not base["valid"] or selected["valid"], "Optional search lost a valid baseline")
                require(not base["valid"] or selected["cost"] <= base["cost"], "Optional search raised repair cost")
                require(not selected["valid"] or selected["atom_edits"] <= 1, "Atom exchange budget exceeded")
                old = legacy[source_row["attempt"]]
                require(old["seed"] == source_row["seed"], "Paired proposal seed differs")
                row = {
                    "attempt": source_row["attempt"], "seed": source_row["seed"],
                    "baseline": base, "selected": selected, "legacy_structural_smiles": old.get("canonical_smiles"),
                    "substitutions_tested": outcome["substitutions_tested"], "combined_search_seconds": seconds,
                }
                for name in ("baseline_seconds", "exchange_seconds"):
                    if name in outcome:
                        row[name] = outcome[name]
                if selected["valid"] and selected["atom_edits"]:
                    replay = read_candidate(choose(raw, p, spec)["selected"], raw, spec)
                    require(replay == selected, "Identical input selected a different atom-exchange witness")
                    variants = [read_candidate(choose(raw, q, spec)["selected"], raw, spec)
                                for q in perturb(p, row["seed"])]
                    row["stability"] = {
                        "same_witness": [v.get("identity") == selected["identity"] for v in variants],
                        "same_atom_exchange": [v.get("swap_directions") == selected["swap_directions"]
                                               and v.get("swap_slots") == selected["swap_slots"] for v in variants],
                        "valid": [v["valid"] for v in variants],
                    }
                rows.append(row)
            result["comparisons"].append({
                "checkpoint": point["id"], "attempts": len(rows),
                "baseline": summarize(rows, "baseline"), "exchange": summarize(rows, "selected"),
                "combined_search_seconds": sum(r["combined_search_seconds"] for r in rows),
                "rows": rows,
            })
            persist()
        result["status"] = "completed"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        try:
            for record in records:
                check_binding(record)
            result["inputs_unchanged"] = True
        except BaseException as error:
            result.update(status="failed", preservation_error=repr(error))
            raise
        finally:
            result["elapsed_seconds"] = time.monotonic() - started
            persist()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--startup-only", action="store_true")
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    records, benchmark = check_inputs(plan, plan_path)
    if args.startup_only:
        print(json.dumps({"status": "metadata_startup_passed_no_models", "checkpoints": len(POINTS),
                          "paired_draws": ATTEMPTS * len(POINTS)}))
    else:
        result = run(plan, plan_path, records, benchmark)
        print(json.dumps({"status": result["status"], "checkpoints": len(result["comparisons"]),
                          "optimizer_updates": result["optimizer_updates"]}))


if __name__ == "__main__":
    main()
