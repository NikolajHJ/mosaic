"""BRD4/JQ1 Protenix contact maximization from controlled ligand starts.

All model, graph, and optimization work is guarded for allocated DTU HPC jobs.
The ligand is represented as a fixed 31-heavy-atom neutral graph. The native
Protenix input is geometry-off; the objective is the JQ1-pocket distogram
contact probability, not affinity, pose refinement, or chemical validity.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


NOISE_LEVELS = (0.10, 0.25, 0.50, 0.75, 0.90, 1.00)
MUTATION_COUNTS = (1, 2, 3, 4, 5, 7, 10, 15, 20)
STEPS = 10_000
STEP_SIZE = 0.03
MAX_BACKTRACKS = 10
CONTACT_CUTOFF = 8.0
PROBABILITY_EPSILON = 1e-4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def case_table():
    cases = [{"index": 1, "name": "jq1_clean", "kind": "clean"}]
    for level in NOISE_LEVELS:
        cases.append({
            "index": len(cases) + 1,
            "name": f"jq1_noise_{level:.2f}",
            "kind": "noise",
            "strength": level,
        })
    for scope in ("atoms", "bonds", "atoms_and_bonds"):
        for count in MUTATION_COUNTS:
            cases.append({
                "index": len(cases) + 1,
                "name": f"jq1_{scope}_changed_{count}",
                "kind": "mutation",
                "scope": scope,
                "count": count,
            })
    return cases


def _neutral_jq1_graph(mol):
    """Read CCD connectivity while enforcing Stage-0's neutral graph schema."""
    from rdkit import Chem
    from mosaic.small_molecule.state import BOND_STATES, ELEMENTS, graph_from_hard

    if not isinstance(mol, Chem.Mol):
        raise TypeError("The Protenix importer did not supply an RDKit molecule")
    heavy = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    if len(heavy) != 31:
        raise ValueError(f"3MXF JQ1 must have 31 heavy atoms, got {len(heavy)}")
    elements = [atom.GetSymbol() for atom in heavy]
    if any(symbol not in ELEMENTS for symbol in elements):
        raise ValueError(f"JQ1 includes an unsupported element: {elements}")
    # Charge is intentionally omitted: the existing Mosaic relaxed graph has
    # fixed neutral atoms. Connectivity and bond order remain those of JQ1.
    categories = {
        Chem.BondType.SINGLE: "SINGLE",
        Chem.BondType.DOUBLE: "DOUBLE",
        Chem.BondType.TRIPLE: "TRIPLE",
        Chem.BondType.AROMATIC: "AROMATIC",
    }
    heavy_lookup = {atom.GetIdx(): i for i, atom in enumerate(heavy)}
    upper_i, upper_j = np.triu_indices(len(heavy), k=1)
    pair_index = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(upper_i, upper_j, strict=True))}
    bond_states = np.zeros(len(pair_index), dtype=np.int32)
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a not in heavy_lookup or b not in heavy_lookup:
            continue
        category = categories.get(bond.GetBondType())
        if category is None:
            raise ValueError(f"Unsupported JQ1 bond category: {bond.GetBondType()}")
        i, j = sorted((heavy_lookup[a], heavy_lookup[b]))
        bond_states[pair_index[i, j]] = BOND_STATES.index(category)
    return graph_from_hard([ELEMENTS.index(symbol) for symbol in elements], bond_states)


def _soften(elements, bonds, epsilon=PROBABILITY_EPSILON):
    x = np.full((len(elements), 6), epsilon / 5, dtype=np.float32)
    x[np.arange(len(elements)), elements] = 1.0 - epsilon
    e = np.full((len(bonds), 5), epsilon / 4, dtype=np.float32)
    e[np.arange(len(bonds)), bonds] = 1.0 - epsilon
    return x, e


def make_start(hard, case, seed):
    from mosaic.small_molecule.state import ELEMENTS, BOND_STATES

    rng = np.random.default_rng(seed)
    x0 = np.asarray(hard.X).argmax(axis=1).astype(np.int32)
    e0 = np.asarray(hard.E_upper).argmax(axis=1).astype(np.int32)
    elements, bonds = x0.copy(), e0.copy()
    if case["kind"] == "noise":
        strength = float(case["strength"])
        x, e = _soften(x0, e0)
        random_x = rng.dirichlet(np.full(6, 0.15), size=len(x0)).astype(np.float32)
        random_e = rng.dirichlet(np.full(5, 0.15), size=len(e0)).astype(np.float32)
        x = (1.0 - strength) * x + strength * random_x
        e = (1.0 - strength) * e + strength * random_e
        elements = x.argmax(axis=-1).astype(np.int32)
        bonds = e.argmax(axis=-1).astype(np.int32)
    else:
        if case["kind"] == "mutation":
            count, scope = int(case["count"]), case["scope"]
            if scope in ("atoms", "atoms_and_bonds"):
                selected = rng.choice(len(elements), size=count, replace=False)
                for i in selected:
                    choices = np.delete(np.arange(6), elements[i])
                    elements[i] = int(rng.choice(choices))
            if scope in ("bonds", "atoms_and_bonds"):
                present = np.flatnonzero(e0 != 0)
                if count > len(present):
                    raise ValueError(f"JQ1 has only {len(present)} bonds for mutation count {count}")
                selected = rng.choice(present, size=count, replace=False)
                for i in selected:
                    choices = np.delete(np.arange(5), e0[i])
                    bonds[i] = int(rng.choice(choices))
        x, e = _soften(elements, bonds)
    from mosaic.small_molecule.state import RelaxedGraph
    return RelaxedGraph(x, e), {
        "elements": [ELEMENTS[int(i)] for i in elements],
        "bond_state_counts": {name: int(np.sum(bonds == i)) for i, name in enumerate(BOND_STATES)},
        "atom_changes": int(np.sum(elements != x0)),
        "bond_changes": int(np.sum(bonds != e0)),
        "seed": int(seed),
    }


def _require_hpc():
    if not os.environ.get("LSB_JOBID"):
        raise RuntimeError("All model and optimization work must run in an allocated DTU HPC job")


def prepare(args):
    _require_hpc()
    import json
    import numpy as np
    from mosaic.small_molecule import features
    from mosaic.small_molecule.io import save_context, write_json
    from mosaic.small_molecule.pocket import derive_definition
    from mosaic.small_molecule.state import ELEMENTS
    from rdkit import Chem

    args.output.mkdir(parents=True, exist_ok=False)
    fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
    pocket = derive_definition(args.pdb, args.fixture)
    features.bootstrap_native_assets(args.cache)
    from protenix.data.ccd import get_component_rdkit_mol

    ccd_mol = get_component_rdkit_mol("JQ1")
    if ccd_mol is None:
        raise ValueError("Pinned Protenix CCD cache has no JQ1 component")
    ccd_mol = Chem.RemoveHs(Chem.Mol(ccd_mol))
    if ccd_mol.GetNumAtoms() != 31:
        raise ValueError(f"Pinned CCD JQ1 has {ccd_mol.GetNumAtoms()} atoms after hydrogen removal")
    for atom in ccd_mol.GetAtoms():
        atom.SetAtomMapNum(0)
    smiles = Chem.MolToSmiles(ccd_mol, canonical=True, isomericSmiles=True)
    if Chem.MolFromSmiles(smiles) is None:
        raise ValueError("Pinned CCD JQ1 did not yield a parseable Protenix input SMILES")
    # The Stage-0 contract optimizes neutral graphs. Build the native JQ1
    # context from its reference cationic SMILES, but neutralize only the
    # optimizer's fixed-charge graph and its ligand ref_charge feature. Protein
    # charges remain untouched. This makes the deliberate neutralized-input
    # approximation explicit while retaining JQ1 atom and bond identities.
    from protenix.data.featurizer import Featurizer

    original_graph = features._paired_hard_graph
    original_features = Featurizer.get_all_input_features

    def neutralized_ligand_graph(mol):
        neutral = Chem.Mol(mol)
        for atom in neutral.GetAtoms():
            atom.SetFormalCharge(0)
        return _neutral_jq1_graph(neutral)

    def neutralized_feature_charges(instance, *call_args, **call_kwargs):
        native = original_features(instance, *call_args, **call_kwargs)
        ligand_mask = np.asarray(native["is_ligand"], dtype=bool)
        native["ref_charge"] = np.array(native["ref_charge"], copy=True)
        native["ref_charge"][ligand_mask] = 0
        return native

    try:
        features._paired_hard_graph = neutralized_ligand_graph
        Featurizer.get_all_input_features = neutralized_feature_charges
        context, hard = features.prepare_context(
            smiles,
            protein_chains=(fixture["protein_sequence"],),
            cache_dir=args.cache,
            input_name="brd4-3mxf-jq1-multistart",
            identities={"protein_sequence": fixture["protein_sequence"]},
        )
    finally:
        features._paired_hard_graph = original_graph
        Featurizer.get_all_input_features = original_features
    if context.n_atoms != 31 or hard.n_atoms != 31:
        raise ValueError("Prepared JQ1 context must retain all 31 heavy atoms")
    from mosaic.small_molecule.pocket import map_protenix_pocket
    pocket_tokens = map_protenix_pocket(context, pocket)
    context_files = save_context(args.output / "context", context)
    np.savez_compressed(args.output / "jq1_reference_graph.npz", X=np.asarray(hard.X), E_upper=np.asarray(hard.E_upper))
    write_json(args.output / "pocket_definition.json", pocket)
    preparation = {
        "status": "prepared",
        "source": "RDKit molecule JQ1 from Protenix's staged and hash-pinned CCD cache",
        "pdb_id": "3MXF",
        "pdb_sha256": sha256(args.pdb),
        "smiles": smiles,
        "n_heavy_atoms": context.n_atoms,
        "formal_charge_policy": "neutralized to fixed zero for the Stage-0 graph objective; exact heavy-atom identities/connectivity/bond orders retained",
        "geometry_mode": context.geometry_mode,
        "objective": "maximize negative Protenix-Tiny pocket-restricted ligand-to-protein contact loss",
        "contact_cutoff_angstrom": CONTACT_CUTOFF,
        "pocket_definition_sha256": sha256(args.output / "pocket_definition.json"),
        "pocket_residue_sequence_positions": [r["sequence_position"] for r in pocket["residues"]],
        "pocket_token_indices": list(pocket_tokens),
        "context_files": context_files,
        "context_identities": context.identities,
        "case_count": len(case_table()),
        "cases": case_table(),
        "updates_per_case": STEPS,
        "optimizer": {"type": "categorical logits", "step_size": STEP_SIZE, "max_backtracks": MAX_BACKTRACKS, "selection": "best-primal; strict contact-loss improvement preferred"},
        "contact_probability_output": "float16 matrix [state, 31 ligand atoms, pocket residues], saved at initial state and after every accepted update",
        "atom_mutation_counts": list(MUTATION_COUNTS),
        "bond_mutation_counts": list(MUTATION_COUNTS),
        "note": "Protenix consumes bond existence, not positive bond order. Positive-order-only changes are recorded but are native-model null directions.",
    }
    write_json(args.output / "preparation.json", preparation)
    print(json.dumps({"status": "prepared", "atoms": context.n_atoms, "pocket_residues": len(pocket_tokens), "case_count": len(case_table())}), flush=True)


def run(args):
    _require_hpc()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from mosaic.small_molecule.io import load_context, write_json
    from mosaic.small_molecule.protenix import FrozenProtenixTiny
    from mosaic.small_molecule.pocket import map_protenix_pocket, validate_definition
    from mosaic.small_molecule.chemistry import GraphLogits
    from mosaic.small_molecule.optimizers import evaluate_graph
    from mosaic.small_molecule.losses import NativeContactLoss
    from mosaic.small_molecule.state import RelaxedGraph
    from mosaic.small_molecule.state import pair_indices

    jax.config.update("jax_default_matmul_precision", "highest")
    if jax.config.jax_enable_x64 or any(device.platform != "gpu" for device in jax.devices()):
        raise RuntimeError(f"Expected one FP32 CUDA device; got {jax.devices()}")
    prep_path = args.prepared / "preparation.json"
    preparation = json.loads(prep_path.read_text(encoding="utf-8"))
    expected = preparation["context_files"]
    context = load_context(args.prepared / "context", expected)
    pocket = validate_definition(json.loads((args.prepared / "pocket_definition.json").read_text(encoding="utf-8")))
    pocket_tokens = map_protenix_pocket(context, pocket)
    if context.n_atoms != 31 or len(pocket_tokens) != len(pocket["residues"]):
        raise ValueError("Prepared context/pocket mapping changed")
    with np.load(args.prepared / "jq1_reference_graph.npz", allow_pickle=False) as data:
        hard = RelaxedGraph(jnp.asarray(data["X"]), jnp.asarray(data["E_upper"]))
    case = {"index": 0, "name": "jq1_clean_validation", "kind": "clean"} if args.mode == "validate" else case_table()[args.case_index - 1]
    if args.case_index < 1 or args.case_index > len(case_table()):
        raise ValueError("Case index out of range")
    initial, start = make_start(hard, case, seed=20261021 + int(case["index"]))
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite results: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.mkdir()
    write_json(args.output / "case.json", {
        "case": case,
        "start": start,
        "source_sha256": sha256(Path(__file__).resolve()),
        "prepared_context_files": expected,
        "pocket_definition_sha256": preparation["pocket_definition_sha256"],
        "updates_requested": args.steps,
        "seed": start["seed"],
    })
    model = FrozenProtenixTiny.load(cache_dir=args.cache)
    if args.mode == "validate":
        key_check = jax.random.key(17)
        reference_output = model.forward_hard_geometry_off(context=context, key=key_check)
        attached_output = model.forward_trunk(context=context, graph=hard, key=key_check)
        np.testing.assert_allclose(
            np.asarray(attached_output.distogram_logits),
            np.asarray(reference_output.distogram_logits),
            atol=2e-5,
            rtol=0,
        )

    class ContactProbabilities(NativeContactLoss):
        def __call__(self, graph, output, *, key):
            value, metrics = super().__call__(graph, output, key=key)
            ligand = jnp.asarray(output.ligand_token_indices, dtype=jnp.int32)
            partners = jnp.asarray(self.pocket_token_indices, dtype=jnp.int32)
            logits = output.distogram_logits[ligand[:, None], partners[None, :], :]
            pair_bins = jax.nn.softmax(logits, axis=-1)
            probabilities = jnp.sum(pair_bins * (output.distogram_bins < self.cutoff_angstrom), axis=-1)
            return value, {**metrics, "contact_probabilities": probabilities}

    loss = model.build_loss(context=context, loss=ContactProbabilities(CONTACT_CUTOFF, pocket_tokens))
    options = {
        "loss": {"version": "contact-only"},
        "repair_weight": 0.0,
        "element_js_weight": 0.0,
        "parity_atol": 2e-5,
        "native_name": "contact",
        "objective_mode": "native_only",
        "objective_identity": {
            "model_digest": model.identities["model_eqx_sha256"],
            "context_digest": preparation["context_identities"]["base_features_sha256"],
            "pocket_definition_sha256": preparation["pocket_definition_sha256"],
            "native_objective": "negative mean over ligand atoms of their maximum contact probability with JQ1-pocket residues",
            "contact_cutoff_angstrom": CONTACT_CUTOFF,
            "n_atoms": 31,
            "formal_charge_policy": preparation["formal_charge_policy"],
        },
    }
    key = jax.random.key(17)
    logits = GraphLogits.from_graph(initial)

    def evaluate(candidate_logits):
        graph = candidate_logits.graph()
        raw = evaluate_graph(loss, graph, key)
        value = float(raw.value)
        contact = raw.aux["contact_probabilities"]
        gradient = GraphLogits(
            graph.X * (raw.gradients.X - jnp.sum(graph.X * raw.gradients.X, axis=-1, keepdims=True)),
            graph.E_upper * (raw.gradients.E_upper - jnp.sum(graph.E_upper * raw.gradients.E_upper, axis=-1, keepdims=True)),
        )
        return {"logits": candidate_logits, "graph": graph, "loss": value, "contact": contact, "gradient": gradient, "raw": raw}

    def store_state(arrays, step, state):
        graph = state["graph"]
        atom_probabilities = np.asarray(graph.X, dtype=np.float16)
        bond_probabilities = np.asarray(graph.E_upper, dtype=np.float16)
        atom_categories = np.asarray(graph.X).argmax(axis=-1).astype(np.uint8)
        bond_categories = np.asarray(graph.E_upper).argmax(axis=-1).astype(np.uint8)
        contact_probabilities = np.asarray(state["contact"], dtype=np.float16)
        arrays["atom_probabilities"].write(atom_probabilities.tobytes(order="C"))
        arrays["bond_probabilities"].write(bond_probabilities.tobytes(order="C"))
        arrays["atom_categories"].write(atom_categories.tobytes(order="C"))
        arrays["bond_categories"].write(bond_categories.tobytes(order="C"))
        arrays["loss"].write(np.asarray([state["loss"]], dtype=np.float32).tobytes())
        arrays["contact_probabilities"].write(contact_probabilities.tobytes(order="C"))
        record = {
            "step": step,
            "loss": state["loss"],
            "contact_score": -state["loss"],
            "contact_probabilities_index": step,
            "strict_improvement": state.get("strict_improvement"),
            "accepted_step_size": state.get("accepted_step_size"),
            "backtracks": state.get("backtracks"),
        }
        with (args.output / "trajectory.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")

    n_states = args.steps + 1
    outputs = {}
    for name, suffix in (("atom_probabilities", "f16"), ("bond_probabilities", "f16"),
                         ("atom_categories", "u8"), ("bond_categories", "u8"),
                         ("loss", "f32"), ("contact_probabilities", "f16")):
        outputs[name] = gzip.open(args.output / f"{name}.{suffix}.gz", "wb", compresslevel=6)
    i, j = pair_indices(context.n_atoms)
    pair_i, pair_j = np.asarray(i).tolist(), np.asarray(j).tolist()
    t0 = time.perf_counter()
    status = "running"
    error_message = None
    try:
        current = evaluate(logits)
        store_state(outputs, 0, current)
        progress_path = args.output / "updates.jsonl"
        for step in range(args.steps):
            previous_loss = current["loss"]
            best = None
            selected = None
            accepted_eta = STEP_SIZE
            for attempt in range(MAX_BACKTRACKS):
                eta = STEP_SIZE * 0.5**attempt
                proposal = GraphLogits(
                    current["logits"].atoms - eta * current["gradient"].atoms,
                    current["logits"].bonds - eta * current["gradient"].bonds,
                )
                candidate = evaluate(proposal)
                strict = candidate["loss"] < current["loss"] - 1e-7
                item = (candidate, eta, attempt, strict)
                if best is None or candidate["loss"] < best[0]["loss"]:
                    best = item
                if strict:
                    selected = item
                    break
            candidate, accepted_eta, attempt, strict = best if selected is None else selected
            current = candidate
            current.update(strict_improvement=bool(strict), accepted_step_size=float(accepted_eta), backtracks=int(attempt))
            store_state(outputs, step + 1, current)
            record = {
                "step": step + 1,
                "loss_before": previous_loss,
                "loss_after": current["loss"],
                "contact_score": -current["loss"],
                "strict_improvement": bool(strict),
                "accepted_step_size": float(accepted_eta),
                "backtracks": int(attempt),
            }
            with progress_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            if (step + 1) % 25 == 0:
                for stream in outputs.values():
                    stream.flush()
                print(f"{case['name']} update={step+1}/{args.steps} loss={current['loss']:.8g} contact={-current['loss']:.8g}", flush=True)
        status = "passed"
    except Exception as error:
        status = "failed"
        error_message = f"{type(error).__name__}: {error}"
        raise
    finally:
        for stream in outputs.values():
            stream.close()
        trajectory_path = args.output / "trajectory.jsonl"
        states_written = sum(1 for _ in trajectory_path.open(encoding="utf-8")) if trajectory_path.exists() else 0
        summary = {
            "status": status,
            "error": error_message,
            "case": case,
            "start": start,
            "updates_requested": args.steps,
            "states_written": states_written,
            "elapsed_seconds": time.perf_counter() - t0,
            "pocket_residues": pocket["residues"],
            "pocket_token_indices": list(pocket_tokens),
            "pocket_definition_sha256": preparation["pocket_definition_sha256"],
            "states_per_output": states_written,
            "array_storage": "gzip-compressed C-order raw arrays; rows are states 0..states_written-1",
            "array_shapes_and_dtypes": {
                "atom_probabilities": {"shape": [states_written, context.n_atoms, 6], "dtype": "float16"},
                "bond_probabilities": {"shape": [states_written, context.n_atoms * (context.n_atoms - 1) // 2, 5], "dtype": "float16"},
                "atom_categories": {"shape": [states_written, context.n_atoms], "dtype": "uint8"},
                "bond_categories": {"shape": [states_written, context.n_atoms * (context.n_atoms - 1) // 2], "dtype": "uint8"},
                "contact_loss": {"shape": [states_written], "dtype": "float32"},
                "contact_probabilities": {"shape": [states_written, context.n_atoms, len(pocket_tokens)], "dtype": "float16"},
            },
            "atom_category_order": ["C", "N", "O", "S", "F", "Cl"],
            "bond_category_order": ["NO_BOND", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"],
            "pair_index_order": "lexicographic i<j; pair arrays in case.json",
            "objective_identity": options["objective_identity"],
            "model_identities": model.identities,
            "outputs": {},
        }
        (args.output / "bond_pair_indices.json").write_text(json.dumps(list(zip(pair_i, pair_j)), separators=(",", ":")) + "\n", encoding="utf-8")
        for name, suffix in (("atom_probabilities", "f16"), ("bond_probabilities", "f16"), ("atom_categories", "u8"), ("bond_categories", "u8"), ("contact_loss", "f32"), ("contact_probabilities", "f16")):
            path = args.output / f"{name}.{suffix}.gz"
            if path.exists():
                summary["outputs"][name] = {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for name, path in (("trajectory", args.output / "trajectory.jsonl"), ("updates", args.output / "updates.jsonl"), ("bond_pair_indices", args.output / "bond_pair_indices.json")):
            if path.exists():
                summary["outputs"][name] = {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        write_json(args.output / "summary.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "validate", "run"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--case-index", type=int, default=1)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--pdb", type=Path, default=Path("tests/data/brd4_3mxf/3MXF.pdb"))
    parser.add_argument("--fixture", type=Path, default=Path("tests/data/brd4_3mxf/fixture_25.json"))
    args = parser.parse_args()
    if args.steps < 1 or args.steps > STEPS:
        raise ValueError(f"--steps must be in 1..{STEPS}")
    if args.mode == "prepare":
        prepare(args)
    else:
        if args.prepared is None:
            parser.error("--prepared is required for validate/run")
        run(args)


if __name__ == "__main__":
    main()
