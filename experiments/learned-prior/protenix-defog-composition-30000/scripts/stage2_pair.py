"""Matched Boltz-2 Stage 2 control and one-query DeFoG arm."""

import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

from bridge import Teacher
from contract import teacher_time, validate
from mosaic.small_molecule.io import write_json

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[2]
sys.path.insert(0, str(ROOT / "stage2"))

from operators import ELEMENTS
from run import add_arguments, run
from settings import apply

WEIGHT = 0.5
TEACHER_INDEX = 23999


def prepare_teacher_loss(teacher, native, graph, _step):
    (atoms, edges), reply = teacher.query(graph, state_index=TEACHER_INDEX)
    atoms = jax.lax.stop_gradient(jnp.asarray(atoms))
    edges = jax.lax.stop_gradient(jnp.asarray(edges))

    def combined(current, *, key):
        native_loss, aux = native(current, key=key)
        delta_x = current.X - atoms
        delta_e = current.E_upper - edges
        atom_mse = jnp.mean(jnp.sum(delta_x**2, axis=-1))
        bond_mse = jnp.mean(jnp.sum(delta_e**2, axis=-1))
        penalty = atom_mse + bond_mse
        return native_loss + WEIGHT * penalty, {
            **aux,
            "native_loss": native_loss,
            "prior_loss": penalty,
            "prior_atom_mse": atom_mse,
            "prior_bond_mse": bond_mse,
            "teacher_time": reply["teacher_time"],
            "teacher_atom_tv": jnp.mean(jnp.sum(jnp.abs(delta_x), axis=-1)) / 2,
            "teacher_bond_tv": jnp.mean(jnp.sum(jnp.abs(delta_e), axis=-1)) / 2,
        }

    return combined


def summarize(arm):
    trajectory = arm / "trajectory"
    gaps = json.loads((arm / "soft_hard_gap.json").read_text())
    sulfur = []
    valid_connected = []
    atom_changes, bond_changes = 0, 0
    previous = None
    for gap in gaps:
        step = gap["step"]
        readout = json.loads(
            (trajectory / "readouts" / f"step_{step:06d}.json").read_text()
        )
        atoms = np.asarray(readout["element_indices"])
        bonds = np.asarray(readout["bond_upper_indices"])
        sulfur.append(int(np.count_nonzero(atoms == ELEMENTS.index("S"))))
        if previous is not None:
            atom_changes += int(np.count_nonzero(atoms != previous[0]))
            bond_changes += int(np.count_nonzero(bonds != previous[1]))
        previous = (atoms, bonds)
        if gap["sanitized"] and gap["fragment_count"] == 1:
            valid_connected.append(gap)
    return {
        "readouts": len(gaps),
        "sanitized_connected_readouts": len(valid_connected),
        "final_sanitized_connected": bool(valid_connected and valid_connected[-1] == gaps[-1]),
        "best_soft_native_loss": min(row["soft_loss"] for row in gaps),
        "final_soft_native_loss": gaps[-1]["soft_loss"],
        "best_valid_hard_native_loss": min(
            (row["hard_loss"] for row in valid_connected), default=None
        ),
        "mean_hard_minus_soft_native_loss": float(
            np.mean([row["hard_minus_soft_loss"] for row in gaps])
        ),
        "sulfur_final": sulfur[-1],
        "sulfur_max": max(sulfur),
        "sulfur_mean": float(np.mean(sulfur)),
        "hard_atom_changes": atom_changes,
        "hard_bond_changes": bond_changes,
    }


def main():
    import argparse

    parser = add_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--teacher-python", type=Path, required=True)
    parser.add_argument("--teacher-source", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    args = apply(parser)
    if (
        args.backend != "boltz2" or args.hardening or args.resume
        or args.max_updates is not None or args.readout_every != 1
    ):
        parser.error("Use a fresh, unhardened Boltz pair with one readout per state")
    root = args.out
    root.mkdir()
    control, treatment = root / "stage2", root / "stage2-defog"

    # Validate and load the frozen teacher before starting either trajectory.
    config_path = STUDY / "configs/pilot.json"
    config = json.loads(config_path.read_text())
    validate(config)
    teacher = Teacher(
        [
            str(args.teacher_python), "-u", str(STUDY / "scripts/worker.py"),
            "--config", str(config_path), "--source", str(args.teacher_source),
            "--checkpoint", str(args.teacher_checkpoint),
        ],
        root / "teacher.stderr",
        config,
    )
    try:
        prior = {
            "weight": WEIGHT,
            "distance": "sum of mean per-row squared L2 distances for atoms and bonds",
            "target": "detached one-forward DeFoG clean probabilities",
            "teacher_time": teacher_time(config, TEACHER_INDEX),
            "teacher_identity": teacher.identity,
        }
        args.out = treatment
        run(
            args,
            prepare_loss=lambda native, graph, step: prepare_teacher_loss(
                teacher, native, graph, step
            ),
            prior_identity=prior,
        )
        frozen = teacher.frozen()
        if frozen["queries"] != args.steps + 1:
            raise ValueError("Expected one teacher query per update and final evaluation")
        write_json(root / "teacher_frozen.json", frozen)
    finally:
        teacher.close()
    write_json(root / "comparison.json", {"stage2-defog": summarize(treatment)})

    args.out = control
    run(args)
    with np.load(control / "inputs/initial_graph.npz") as a, np.load(
        treatment / "inputs/initial_graph.npz"
    ) as b:
        if a.files != b.files or any(not np.array_equal(a[k], b[k]) for k in a.files):
            raise ValueError("Matched arms did not start from identical graphs")
    write_json(
        root / "comparison.json",
        {"stage2": summarize(control), "stage2-defog": summarize(treatment)},
    )


if __name__ == "__main__":
    main()
