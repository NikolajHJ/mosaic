"""PyTorch side of the paired-model run; the native features retain founder slot correspondence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
import traceback

import numpy as np


def direct_slot_to_canonical(atom, bond, smiles, canonical_atoms, canonical_bonds):
    """Return canonical feature indices in original slot order and verify all 325 decisions."""
    from rdkit import Chem
    from study import BOND_TYPES, ELEMENTS, PAIR_I, PAIR_J

    mol = Chem.RWMol()
    for k in atom.argmax(1):
        mol.AddAtom(Chem.Atom(ELEMENTS[k]))
    hard_bonds = bond.argmax(1)
    for i, j, k in zip(PAIR_I, PAIR_J, hard_bonds):
        if k:
            mol.AddBond(int(i), int(j), BOND_TYPES[k])
            if k == 4:
                mol.GetAtomWithIdx(int(i)).SetIsAromatic(True)
                mol.GetAtomWithIdx(int(j)).SetIsAromatic(True)
    Chem.SanitizeMol(mol)
    canonical = Chem.MolFromSmiles(smiles)
    match = np.asarray(canonical.GetSubstructMatch(mol), dtype=np.int64)
    if match.shape != (25,) or len(set(match)) != 25:
        raise ValueError("No bijective founder slot-to-canonical atom match")
    matrix = np.zeros((25, 25), np.int64)
    matrix[PAIR_I, PAIR_J] = matrix[PAIR_J, PAIR_I] = canonical_bonds.argmax(1)
    if (not np.array_equal(atom.argmax(1), canonical_atoms.argmax(1)[match]) or
            not np.array_equal(hard_bonds, matrix[match[PAIR_I], match[PAIR_J]])):
        raise ValueError("Founder permutation changed direct atom or unordered bond labels")
    return match


def schema_slot_to_canonical(record):
    """Map the VAE's occupied slots to the input molecule and check all hard labels."""
    import torch
    from rdkit import Chem
    from latentmol.chemistry import _label_molecule
    from latentmol.representation import GraphSpec, HardGraphBatch
    from nesso_inputs import canonical_graph
    from study import PAIR_I, PAIR_J

    spec = GraphSpec(max_nodes=25)
    if GraphSpec.from_dict(record["spec"]) != spec:
        raise ValueError("Nesso requires the shared 25-occupied-slot graph vocabulary")
    atoms, charges, bonds = (np.asarray(record[k]) for k in ("atoms", "charges", "bonds"))
    for values, shape, classes in ((atoms, (25,), spec.atomic_numbers), (charges, (25,), spec.formal_charges),
                                    (bonds, (25, 25), spec.bond_names)):
        if (values.shape != shape or not np.issubdtype(values.dtype, np.integer) or
                (values < 0).any() or (values >= len(classes)).any()):
            raise ValueError("Invalid VAE hard founder graph")
    if (atoms == 0).any() or not np.array_equal(bonds, bonds.T) or bonds.diagonal().any():
        raise ValueError("Invalid VAE founder occupancy or bonds")
    mol = _label_molecule(atoms, charges, bonds, spec, list(range(25)))
    Chem.SanitizeMol(mol)
    smiles, parsed, ca, cb = canonical_graph(record["smiles"], spec)
    if Chem.MolToSmiles(mol, isomericSmiles=False) != smiles:
        raise ValueError("Declared VAE founder differs from hard graph")
    match = np.asarray(parsed.GetSubstructMatch(mol), dtype=np.int64)
    if match.shape != (25,) or len(set(match)) != 25:
        raise ValueError("Cannot map VAE founder to native Nesso atom order")
    ccharge = np.array([spec.formal_charges.index(a.GetFormalCharge()) for a in parsed.GetAtoms()])
    matrix = np.zeros((25, 25), np.int64)
    matrix[PAIR_I, PAIR_J] = matrix[PAIR_J, PAIR_I] = cb.argmax(-1)
    if (not np.array_equal(atoms, 1 + ca.argmax(-1)[match]) or not np.array_equal(charges, ccharge[match])
            or not np.array_equal(bonds, matrix[np.ix_(match, match)])):
        raise ValueError("Nesso founder permutation changed an atom, formal charge or typed bond")
    hard = HardGraphBatch(*(torch.as_tensor(x, dtype=torch.long)[None] for x in (atoms, charges, bonds)))
    return spec, hard, match


class Scorer:
    def __init__(self, seed, work, assets, graph, schema_founder=None):
        import torch
        from nesso.model.models.nesso1 import Nesso1
        from latentmol.multimodel.external import tensor_tree_sha256
        from nesso_inputs import NessoInputs, prepare_esm
        from nesso_native import NessoBackend, NoProteinInCrop
        from study import ELEMENTS, hard_smiles

        if not os.environ.get("LSB_JOBID", "").isdigit() or torch.cuda.device_count() != 1:
            raise RuntimeError("Nesso requires one allocated visible CUDA device")
        spec = hard = perm = None
        if schema_founder is not None:
            spec, hard, perm = schema_slot_to_canonical(schema_founder)
        founder = hard_smiles(graph) if spec is None else schema_founder["smiles"]
        if not founder:
            raise ValueError("Nesso requires a valid connected founder")
        torch.manual_seed(seed)
        self.seed = seed
        prepare_esm(work / "esm", model_name=str(assets / "esm"), device="cuda")
        self.inputs = NessoInputs(work / "features", assets / "ccd.pkl", work / "esm", device="cuda",
                                  graph_spec=spec)
        self.model = Nesso1.from_pretrained(str(assets / "checkpoint")).eval().requires_grad_(False).to("cuda")
        self.model.use_kernels = False
        self.model.predict_args = dict(self.model.predict_args or {}, recycling_steps=5,
            refine_protein_inference=True, refine_protein_tokens_budget=256,
            refine_protein_cutoff=22., affinity_protein_cutoff=15., pose_protein_cutoff=15.,
            save_metadata=False)
        self.digest = tensor_tree_sha256(self.model.state_dict())
        features, canonical_slots, ca, cb = self.inputs.featurize(founder)
        if spec is None:
            perm = direct_slot_to_canonical(*graph, founder, ca, cb)
        else:
            graph = (None, None)
        slots = canonical_slots[torch.as_tensor(perm, device=canonical_slots.device)]
        empty_crops = []
        for recycles in (0, 1):
            try:
                self.backend = NessoBackend(self.model, features, slots, *graph, recycling_steps=recycles,
                                            schema_hard=hard)
                break
            except NoProteinInCrop as exc:
                empty_crops.append({"recycles": recycles, "native_value": exc.native_score})
        else:
            raise ValueError(f"Nesso founder gradient crop contained no protein at either recycle setting: {empty_crops}")
        if abs(self.backend.founder_native_score - self.backend.founder_bridge_score) > .001:
            raise ValueError("Nesso discrete founder bridge differs from native score by more than 0.001")
        self.identity = dict(founder=founder, seed=seed, model="frozen Nesso-1", model_sha256=self.digest,
            gpu=torch.cuda.get_device_name(), gradient_recycles=recycles, hard_recycles=5,
            crop_protein_tokens=self.backend.crop_protein_tokens, empty_crops=empty_crops,
            direct_to_canonical_slots=perm.tolist(), direct_to_native_atoms=slots.tolist(),
            direct_atomic_numbers=ELEMENTS if spec is None else spec.atomic_numbers[1:],
            formal_charges=[0] if spec is None else spec.formal_charges,
            native_uses_formal_charge=self.backend.branch.charge_enabled,
            direct_bond_classes=["NONE","SINGLE","DOUBLE","TRIPLE","AROMATIC"],
            founder_native_gradient_protocol=self.backend.founder_native_score,
            founder_bridge=self.backend.founder_bridge_score,
            founder_native_binder_probability=self.backend.founder_native_binder_probability,
            founder_bridge_binder_probability=self.backend.founder_bridge_binder_probability,
            binder_objective="Negative log of the mean of the two native classifier sigmoid probabilities",
            gradient_protocol="Fixed founder conformer and Nesso-selected protein crop; lower predicted affinity is better")

    def fresh(self, smiles):
        import torch
        from nesso_inputs import ConformerUnavailable
        from nesso_pilot import native_readout
        started = perf_counter()
        try:
            features, _, _, _ = self.inputs.featurize(smiles)
        except ConformerUnavailable as exc:
            return dict(status="feature_failed", reason=str(exc), hard_total_s=perf_counter()-started)
        # Reproducible native prediction per candidate across arms, independent of ranking order.
        with torch.random.fork_rng(devices=[0]):
            seed = int.from_bytes(hashlib.sha256(f"{self.seed}/{smiles}".encode()).digest()[:4], "big")
            torch.manual_seed(seed)
            result = native_readout(self.model, features)
        return dict(status="scored", hard_total_s=perf_counter()-started, **result)

    def close(self):
        import torch
        from latentmol.multimodel.external import tensor_tree_sha256
        audit = self.backend.close()
        if tensor_tree_sha256(self.model.state_dict()) != self.digest:
            raise RuntimeError("Frozen Nesso model weights or buffers changed")
        return dict(audit, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    # Native feature/model packages can print to stdout, including from native code.
    # Keep one dedicated descriptor for RPC replies and route all other output to the log.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    scorer = None
    for line in sys.stdin:
        op = None
        try:
            req = json.loads(line)
            op = req["op"]
            if op in ("init", "gradient", "affinity_confidence_gradients",
                      "weighted_affinity_confidence_gradient", "hard_value"):
                graph = tuple(np.asarray(req[k], dtype=np.float32) for k in ("atom", "bond"))
            if op == "init" and scorer is None:
                scorer = Scorer(args.seed, args.work, args.assets, graph)
                result = scorer.identity
            elif op == "init_schema" and scorer is None:
                scorer = Scorer(args.seed, args.work, args.assets, None, schema_founder=req["founder"])
                result = scorer.identity
            elif op == "schema_gradient":
                value, *derivatives = scorer.backend.schema_gradient(req["atom"], req["charge"], req["bond"])
                result = [value, *(d.tolist() for d in derivatives)]
            elif op == "gradient":
                value, da, db = scorer.backend.gradient(*graph)
                result = [value, da.tolist(), db.tolist()]
            elif op == "affinity_confidence_gradients":
                result = [[v, da.tolist(), db.tolist()]
                          for v, da, db in scorer.backend.affinity_confidence_gradients(*graph)]
            elif op == "weighted_affinity_confidence_gradient":
                a, c, da, db = scorer.backend.weighted_affinity_confidence_gradient(
                    *graph, req["affinity_weight"], req["confidence_weight"])
                result = [a, c, da.tolist(), db.tolist()]
            elif op == "hard_value":
                result = scorer.backend.hard_value(*graph)
            elif op == "fresh":
                result = scorer.fresh(req["smiles"])
            elif op == "close":
                result = scorer.close()
            else:
                raise ValueError(f"Unknown or out-of-order Nesso operation {op!r}")
            protocol.write(json.dumps(dict(op=op, result=result), separators=(",", ":"), allow_nan=False)+"\n")
        except Exception as exc:
            traceback.print_exc()
            protocol.write(json.dumps(dict(op=op, error=f"{type(exc).__name__}: {exc}"))+"\n")
            raise
        if op == "close":
            break


if __name__ == "__main__":
    main()
