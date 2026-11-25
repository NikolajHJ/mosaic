"""Detached fixed-atom repair search and a live probability-space NLL.

The native Protenix process uses RepairOracle to call a CPU Python with DBMol
and RDKit installed. Only NumPy and the standard library are needed by the
client. Every scalar/gradient refers to the supplied probability rows.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


# The Mosaic fixed-occupancy contract orders sulfur before fluorine.
ATOMIC_NUMBERS = (6, 7, 8, 16, 9, 17)
BOND_NAMES = ("NONE", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
N_ATOMS = 25
LEFT, RIGHT = np.triu_indices(N_ATOMS, 1)
N_DECISIONS = 2 * N_ATOMS + len(LEFT)


def canonical_valid_smiles(smiles):
    from rdkit import Chem, rdBase

    if not smiles:
        return None
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumHeavyAtoms() != N_ATOMS or len(Chem.GetMolFrags(mol)) != 1:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def _validate_rows(atom, bond):
    for rows, shape in ((atom, (N_ATOMS, len(ATOMIC_NUMBERS))), (bond, (len(LEFT), len(BOND_NAMES)))):
        if (rows.shape != shape or rows.dtype != np.float32 or not np.isfinite(rows).all()
                or np.any(rows < 0) or np.any(rows > 1)
                or not np.allclose(rows.sum(1), 1, rtol=1e-5, atol=1e-6)):
            raise ValueError("Expected finite normalized FP32 direct atom and unordered bond rows")


def _bridge(atom, bond, spec):
    atom_columns = np.array([spec.atomic_numbers.index(z) for z in ATOMIC_NUMBERS])
    bond_columns = np.array([spec.bond_names.index(s) for s in BOND_NAMES])
    neutral = spec.formal_charges.index(0)
    atoms = np.zeros((N_ATOMS, len(spec.atomic_numbers)), dtype=np.float32)
    atoms[:, atom_columns] = atom
    charges = np.zeros((N_ATOMS, len(spec.formal_charges)), dtype=np.float32)
    charges[:, neutral] = 1
    pairs = np.zeros((len(LEFT), len(spec.bond_names)), dtype=np.float32)
    pairs[:, bond_columns] = bond
    bonds = np.zeros((N_ATOMS, N_ATOMS, len(spec.bond_names)), dtype=np.float32)
    bonds[np.arange(N_ATOMS), np.arange(N_ATOMS), bond_columns[0]] = 1
    bonds[LEFT, RIGHT] = bonds[RIGHT, LEFT] = pairs
    # Argmax in native category order first. Reordering tied rows can change a hard label.
    raw_atoms = atom_columns[atom.argmax(1)]
    raw_bonds = np.full((N_ATOMS, N_ATOMS), bond_columns[0], dtype=np.int64)
    raw_bonds[LEFT, RIGHT] = raw_bonds[RIGHT, LEFT] = bond_columns[bond.argmax(1)]
    raw = (raw_atoms, np.full(N_ATOMS, neutral, dtype=np.int64), raw_bonds)
    return raw, (atoms, charges, bonds), bond_columns


def _witness_nll(atom, bond, raw, witness, spec, bond_columns):
    from latentmol.chemistry import inspect_graph

    labels = [np.asarray(witness["labels"][key]) for key in ("atoms", "charges", "bonds")]
    if any(x.shape != r.shape or x.dtype.kind not in "iu" for x, r in zip(labels, raw)):
        raise ValueError("Repair witness has incompatible integer label arrays")
    atoms, charges, bonds = labels
    if (not np.array_equal(atoms, raw[0]) or not np.array_equal(charges, raw[1])
            or not np.array_equal(bonds, bonds.T) or np.any(np.diag(bonds) != bond_columns[0])
            or not np.isin(bonds, bond_columns).all()):
        raise ValueError("Repair witness changed fixed atoms, charges, or unordered bond schema")
    checked = canonical_valid_smiles(inspect_graph(*labels, spec).smiles)
    if not checked or checked != witness["smiles"]:
        raise ValueError("Repair witness failed connected sanitized molecule replay")
    edits = int(np.count_nonzero(bonds[LEFT, RIGHT] != raw[2][LEFT, RIGHT]))
    if not edits or edits != witness["edit_count"]:
        raise ValueError("Repair witness edit count differs from its hard bond changes")

    atom_targets = atom.argmax(1)
    inverse = {int(shared): native for native, shared in enumerate(bond_columns)}
    bond_targets = np.array([inverse[int(label)] for label in bonds[LEFT, RIGHT]])
    atom_index = np.arange(N_ATOMS), atom_targets
    bond_index = np.arange(len(LEFT)), bond_targets
    pa = atom[atom_index].astype(np.float64)
    pb = bond[bond_index].astype(np.float64)
    if np.any(pa <= 0) or np.any(pb <= 0):
        raise FloatingPointError("Zero support for fixed repair target; check the direct projection floor")
    # The 25 neutral charge rows contribute -log(1) each to the numerator.
    loss = float(-(np.log(pa).sum() + np.log(pb).sum()) / N_DECISIONS)
    da, db = np.zeros_like(atom), np.zeros_like(bond)
    da[atom_index] = -1 / (N_DECISIONS * pa)
    db[bond_index] = -1 / (N_DECISIONS * pb)
    if not (np.isfinite(loss) and np.isfinite(da).all() and np.isfinite(db).all()):
        raise FloatingPointError("Nonfinite direct repair NLL or probability gradient")
    return loss, da, db, edits, checked


def evaluate_repair(atom, bond, *, search_config=None, proposal_config=None):
    """Inspect a native graph and evaluate its verified, detached target when available."""
    from study import hard_smiles
    from latentmol.evaluation.quality import diagnose_graph
    from latentmol.guidance.repair_proposals import RepairProposalConfig, search_with_proposals
    from latentmol.probability_repair import ProbabilityRepairConfig
    from latentmol.representation import GraphSpec

    _validate_rows(atom, bond)
    hard = hard_smiles((atom, bond))
    smiles = canonical_valid_smiles(hard)
    info = dict(status="already_valid" if smiles else "no_witness", valid=bool(smiles), smiles=smiles,
                hard_sulfur=int((atom.argmax(1) == ATOMIC_NUMBERS.index(16)).sum()),
                expected_sulfur=float(atom[:, ATOMIC_NUMBERS.index(16)].sum()),
                failure_reason=None, component_count=1, problem_burden=0, overvalence_excess=0.,
                repair_edits=None, repair_smiles=None, repair_source=None, search_checks=0,
                search_status="skipped_valid", normalization_decisions=N_DECISIONS)
    if smiles:
        return 0., np.zeros_like(atom), np.zeros_like(bond), info

    spec = GraphSpec(max_nodes=N_ATOMS)
    raw, probabilities, bond_columns = _bridge(atom, bond, spec)
    diagnostic = diagnose_graph(*raw, spec)
    info.update({k: diagnostic.get(k) for k in ("failure_reason", "component_count", "problem_burden",
                                                 "overvalence_excess")})
    if hard:
        info["failure_reason"] = "smiles_reparse"
    search = ProbabilityRepairConfig(**({"max_edits": 3, "checks_per_depth": 32, "beam_width": 4}
                                      | (search_config or {})))
    report = search_with_proposals(raw, probabilities, spec, search, RepairProposalConfig(**(proposal_config or {})))
    info.update(search_status=report["status"], search_checks=int(report["checks"]),
                repair_source=report.get("accepted_source"))
    if report["weighted_witness"] is None:
        return 0., np.zeros_like(atom), np.zeros_like(bond), info
    loss, da, db, edits, verified = _witness_nll(atom, bond, raw, report["weighted_witness"], spec, bond_columns)
    info.update(status="verified_witness", repair_edits=edits, repair_smiles=verified)
    return loss, da, db, info


class RepairOracle:
    """A resident CPU subprocess using the DBMol environment isolated from the JAX runtime."""
    def __init__(self, python):
        self.python = python
        self.process = None

    def __enter__(self):
        if self.process is not None:
            raise RuntimeError("Repair worker is already running")
        script = Path(__file__).resolve()
        env = dict(os.environ, PYTHONPATH=str(script.parents[3] / "src"), CUDA_VISIBLE_DEVICES="",
                   JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        env.pop("PYTHONHOME", None)
        self.process = subprocess.Popen([self.python, "-u", str(script), "--worker"], env=env,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8")
        return self

    def evaluate(self, atom, bond):
        _validate_rows(atom, bond)
        if self.process is None:
            raise RuntimeError("Repair worker has not started")
        request = dict(atom=atom.tolist(), bond=bond.tolist())
        self.process.stdin.write(json.dumps(request, allow_nan=False, separators=(",", ":")) + "\n")
        self.process.stdin.flush()
        reply = self.process.stdout.readline()
        if not reply:
            raise RuntimeError(f"Repair worker closed its output (status {self.process.poll()}); see job stderr")
        result = json.loads(reply)
        if "error" in result:
            raise RuntimeError("Repair worker failed: " + result["error"])
        da = np.asarray(result["da"], dtype=np.float32)
        db = np.asarray(result["db"], dtype=np.float32)
        if (da.shape != atom.shape or db.shape != bond.shape or not np.isfinite(da).all()
                or not np.isfinite(db).all() or not np.isfinite(result["loss"])):
            raise ValueError("Repair worker returned invalid loss or derivatives")
        return result["loss"], da, db, result["info"]

    def __exit__(self, exc_type, exc, tb):
        process, self.process = self.process, None
        try:
            process.stdin.close()
            if exc_type:
                process.terminate()
            try:
                returncode = process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                if not exc_type:
                    raise RuntimeError("Repair worker did not shut down")
            else:
                if returncode and not exc_type:
                    raise RuntimeError(f"Repair worker exited with status {returncode}")
        finally:
            process.stdout.close()


def _worker():
    from contextlib import redirect_stdout
    import traceback

    for line in sys.stdin:
        try:
            request = json.loads(line)
            with redirect_stdout(sys.stderr):
                loss, da, db, info = evaluate_repair(np.asarray(request["atom"], dtype=np.float32),
                                                     np.asarray(request["bond"], dtype=np.float32))
            response = dict(loss=loss, da=da.tolist(), db=db.tolist(), info=info)
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            print(json.dumps(dict(error=f"{type(exc).__name__}: {exc}")), flush=True)
            return 1
        print(json.dumps(response, allow_nan=False, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("Usage: repair_signal.py --worker")
    raise SystemExit(_worker())
