"""Allocated chemistry-only matched pilots; no native binding evaluation."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def main():
    if not os.environ.get("LSB_JOBID") or os.environ.get("LSB_QUEUE") != "hpcint":
        raise RuntimeError("Comparison requires an actual allocated HPC CPU job")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    import jax
    import jax.numpy as jnp
    import numpy as np
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from mosaic.small_molecule.chemistry import GraphLogits, mosaic_spec, prepare_failure_guidance
    from mosaic.small_molecule.chemistry._reference.chemistry import inspect_graph, encode_smiles
    from mosaic.small_molecule.chemistry.confidence import ConfidenceMarginConfig, _terms as confidence_terms
    from mosaic.small_molecule.chemistry.loss import (
        FailureGuidanceConfig, _terms as repair_terms, composition_statistics, population_js,
    )

    assert jax.default_backend() == "cpu"
    policy = ConfidenceMarginConfig()
    loss_config = FailureGuidanceConfig()
    rows = []
    report = {"status": "running", "scope": "chemistry-only CPU comparison; no native binding model or inference",
        "job_id": os.environ["LSB_JOBID"], "backend": jax.default_backend(),
        "margin": policy.record(), "element_js_weight": 0., "repair_weight": 1.,
        "rate": .3, "updates_per_arm": 128, "samples_per_checkpoint": 256,
        "seeds": [17, 23, 29], "checkpoints": [0, 64, 128], "rows": rows,
        "saved_runs_written": False, "native_binding_performance": "not_evaluated",
        "limitations": ["short chemistry-only pilots cannot establish binding efficacy or long-run identity exploration",
                         "sampled graphs are categorical independent draws, not repaired outputs",
                         "changing repair targets mean refreshed totals are not one fixed trajectory objective"]}
    fingerprint = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)

    cases = [("uncertain_valid_ccc", [0, 0, 0], 1.5),
             ("confident_valid_ccc", [0, 0, 0], 8.),
             ("valid_15_atoms_3_sulfur", [3 if i in (2, 7, 12) else 0 for i in range(15)], 1.5),
             ("invalid_two_bond_fluorine", [0, 4, 0], 1.5)]

    def labels(state):
        return np.argmax(np.asarray(state.atoms), -1), np.argmax(np.asarray(state.bonds), -1)

    def inspect(a, b, spec):
        n = len(a); i, j = np.triu_indices(n, 1)
        matrix = np.zeros((n, n), np.int64); matrix[i, j] = b; matrix[j, i] = b
        value = inspect_graph(np.asarray(a, np.int64)+1, np.zeros(n, np.int64), matrix, spec)
        strict = False
        if value.smiles is not None:
            try:
                strict = encode_smiles(value.smiles, spec)["smiles"] == value.smiles
            except ValueError:
                strict = False
        return value, strict

    def samples(state, spec, random_seed, path):
        graph = state.graph(); pa, pe = np.asarray(graph.X), np.asarray(graph.E_upper)
        rng = np.random.default_rng(random_seed)
        def draw(p):
            cdf = np.cumsum(p.astype(np.float64), -1); cdf[:, -1] = 1.
            uniform = rng.random((256, p.shape[0], 1))
            return np.sum(uniform > cdf[None], -1).astype(np.int8)
        aa, bb = draw(pa), draw(pe)
        canonical, failures, strict_count = [], Counter(), 0
        for a, b in zip(aa, bb):
            readout, strict = inspect(a, b, spec)
            strict_count += int(strict)
            if readout.smiles is None: failures[readout.failure_reason] += 1
            else: canonical.append(readout.smiles)
        unique = sorted(set(canonical))
        fps = [fingerprint.GetFingerprint(Chem.MolFromSmiles(s)) for s in unique]
        distances = [1.-DataStructs.TanimotoSimilarity(fps[i], fps[j])
                     for i in range(len(fps)) for j in range(i)]
        scaffolds = {MurckoScaffold.MurckoScaffoldSmiles(smiles=s) for s in unique}
        np.savez_compressed(path, atom_labels=aa, unordered_bond_labels=bb,
                            atom_probabilities=pa, bond_probabilities=pe,
                            random_seed=np.asarray(random_seed))
        return {"sample_count": 256, "raw_valid": len(canonical), "strict_valid": strict_count,
            "failure_categories": dict(failures), "valid_unique": len(unique), "canonical_valid": unique,
            "valid_unique_fraction": None if not canonical else len(unique)/len(canonical),
            "unique_fingerprint_mean_distance": None if not distances else float(np.mean(distances)),
            "unique_nonempty_scaffolds": len(scaffolds-{''}),
            "sample_archive": path.name, "sample_archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    try:
        for case_index, (name, atom_labels, boost) in enumerate(cases):
            n, spec = len(atom_labels), mosaic_spec(len(atom_labels))
            i, j = np.triu_indices(n, 1)
            hard_bonds = (j == i+1).astype(np.int64)
            order_indices = tuple(spec.bond_names.index(k) for k in ("NONE", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"))

            def pure_repair(state, arrays):
                return repair_terms(state.decoded(), arrays, loss_config.weights(state.atoms.dtype), order_indices)

            def pure_added(state, arrays):
                repair, terms = pure_repair(state, arrays)
                margin, extra = confidence_terms(state.decoded(), arrays, policy.margin, policy.slope_cap)
                return repair+policy.weight*margin, {**terms, **extra,
                    "confidence_margin_loss": margin, "confidence_weighted_loss": policy.weight*margin}

            functions = {"current_repair": jax.jit(jax.value_and_grad(pure_repair, has_aux=True)),
                         "repair_plus_margin": jax.jit(jax.value_and_grad(pure_added, has_aux=True))}
            forwards = {"current_repair": jax.jit(pure_repair), "repair_plus_margin": jax.jit(pure_added)}
            for seed in (17, 23, 29):
                rng = np.random.default_rng(seed)
                aa = rng.normal(0, .025, (n, 6)).astype(np.float32)
                bb = rng.normal(0, .025, (len(i), 5)).astype(np.float32)
                aa[np.arange(n), atom_labels] += boost
                # Large graphs keep most bonds confident; atom uncertainty is explicit.
                bb[np.arange(len(i)), hard_bonds] += 8. if n > 3 else boost
                initial = GraphLogits(jnp.asarray(aa), jnp.asarray(bb))
                reference = composition_statistics(initial.decoded())["elements"]
                for arm, fn in functions.items():
                    state, updates, checkpoints = initial, [], []
                    row = {"case": name, "seed": seed, "arm": arm, "updates": updates,
                           "checkpoints": checkpoints, "preparation_seconds": 0.,
                           "warm_loss_gradient_seconds": 0., "compile_and_first_call_seconds": 0.,
                           "atom_changes": 0, "bond_changes": 0, "argmax_identities": [],
                           "margin_active_steps": 0, "fixed_surrogate_increases": 0}
                    rows.append(row); start = time.perf_counter()
                    for step in range(129):
                        t = time.perf_counter()
                        prepared = prepare_failure_guidance(state.decoded(), spec, update=step)
                        row["preparation_seconds"] += time.perf_counter()-t
                        t = time.perf_counter()
                        (value, terms), gradient = fn(state, prepared.arrays)
                        jax.block_until_ready((value, gradient))
                        elapsed = time.perf_counter()-t
                        row["compile_and_first_call_seconds" if step == 0 else "warm_loss_gradient_seconds"] += elapsed
                        a, b = labels(state); readout, strict = inspect(a, b, spec)
                        row["argmax_identities"].append(readout.smiles)
                        row["margin_active_steps"] += int(float(terms.get("confidence_margin_loss", 0)) > 0)
                        if step in (0, 64, 128):
                            sample_path = args.output/f"samples-{case_index}-{seed}-{arm}-{step}.npz"
                            sampling = samples(state, spec, 100000+10000*case_index+100*seed+step, sample_path)
                            stats = composition_statistics(state.decoded())
                            checkpoints.append({"step": step, "argmax_valid": readout.smiles is not None,
                                "argmax_strict_valid": strict, "argmax_smiles": readout.smiles,
                                "argmax_failure": readout.failure_reason, "hard_elements": np.bincount(a, minlength=6).tolist(),
                                "soft_elements": np.asarray(stats["elements"]).tolist(),
                                "element_js_diagnostic_only": float(population_js(stats["elements"], reference)),
                                "components": {k: float(v) for k, v in terms.items()}, **sampling})
                        if step == 128: break
                        proposal = GraphLogits(state.atoms-.3*gradient.atoms, state.bonds-.3*gradient.bonds)
                        assert all(np.isfinite(np.asarray(v)).all() for v in (proposal.atoms, proposal.bonds, value))
                        after = forwards[arm](proposal, prepared.arrays)[0]
                        na, nb = labels(proposal)
                        atom_changes, bond_changes = int(np.sum(a != na)), int(np.sum(b != nb))
                        row["atom_changes"] += atom_changes; row["bond_changes"] += bond_changes
                        row["fixed_surrogate_increases"] += int(float(after) > float(value))
                        updates.append({"step": step, "fixed_before": float(value), "fixed_after": float(after),
                            "preparation_identity": prepared.identity, "preparation_route": int(prepared.arrays["route"][0]),
                            "atom_gradient_norm": float(jnp.linalg.norm(gradient.atoms)),
                            "bond_gradient_norm": float(jnp.linalg.norm(gradient.bonds)),
                            "atom_changes": atom_changes, "bond_changes": bond_changes,
                            "components": {k: float(v) for k, v in terms.items()}})
                        state = proposal
                    row["wall_seconds_including_samples"] = time.perf_counter()-start
                    assert len(updates) == 128 and len(checkpoints) == 3
                    assert sum(p.stat().st_size for p in args.output.iterdir() if p.is_file()) < 192*1024**2
                    write(args.output/"comparison.json", report)
        report["status"] = "completed_measurements_not_native_validation"
    except BaseException as error:
        report.update(status="failed", error=repr(error)); raise
    finally:
        write(args.output/"comparison.json", report)


if __name__ == "__main__": main()
