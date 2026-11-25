"""Direct atom/bond probabilities scored by Nesso's pretrained affinity heads."""
import math

import torch
from torch.nn import functional as F

from latentmol.multimodel.external import NativeBundle, freeze
from latentmol.multimodel.nesso import NessoBranch
from latentmol.representation import GraphSpec, RelaxedGraph
from study import ELEMENTS


class NoProteinInCrop(ValueError):
    """The native affinity head selected no protein tokens for this founder."""

    def __init__(self, score, ligand_tokens):
        self.native_score = score
        self.ligand_tokens = ligand_tokens
        super().__init__(f"Native founder affinity crop contained {ligand_tokens} ligand tokens and zero protein tokens")


def _value(output):
    score = output["affinity_pred_value"]
    if score.numel() != 1 or not torch.isfinite(score).all():
        raise FloatingPointError("Expected one finite native Nesso affinity value")
    return score.reshape(())


def _gradient_arrays(derivatives):
    if not all(g.isfinite().all() for g in derivatives):
        raise FloatingPointError("Nonfinite Nesso derivatives")
    return tuple(g.detach().cpu().numpy() for g in derivatives)


@torch.no_grad()
def native_hard_score(model, features, *, recycling_steps=0):
    """Evaluate the independently prepared discrete molecule with the native crop."""
    return float(_value(model(features, recycling_steps=recycling_steps, refine_protein_inference=False)))


class NessoBackend:
    """Attached, fixed-founder-crop gradient on 25 neutral atoms and 300 unordered pairs."""

    def __init__(self, model, features, slot_to_atom, atoms, bonds, *, recycling_steps=0,
                 schema_hard=None):
        from nesso.data import const

        if model.use_kernels:
            raise ValueError("Native and gradient paths must both use the PyTorch kernels")
        freeze(model)
        self.device = next(model.parameters()).device
        self.spec = GraphSpec(max_nodes=25)
        self._left, self._right = torch.triu_indices(25, 25, 1, device=self.device)
        self._elements = F.one_hot(torch.tensor(
            [self.spec.atomic_numbers.index(z) for z in ELEMENTS], device=self.device),
            len(self.spec.atomic_numbers)).float()
        expected_bonds = ("NONE", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
        if self.spec.bond_names != expected_bonds:
            raise ValueError("Direct and canonical bond vocabularies differ")
        if schema_hard is None:
            atom, bond = self._rows(atoms, bonds)
            hard = self._graph(F.one_hot(atom.argmax(-1), 6).float(), F.one_hot(bond.argmax(-1), 5).float())
        else:
            hard = schema_hard.to(self.device).to_relaxed(self.spec)
        with torch.no_grad():
            output = model(features, recycling_steps=recycling_steps, refine_protein_inference=False)
            self.founder_native_score = float(_value(output))
            self.founder_native_binder_probability = float(output["affinity_probability_binary"].item())
            keep = output["affinity_keep_indices"]
            selected_types = features["mol_type"][0, keep]
            self.crop_protein_tokens = int((selected_types == const.chain_type_ids["PROTEIN"]).sum())
            self.crop_ligand_tokens = int((selected_types == const.chain_type_ids["NONPOLYMER"]).sum())
            if not self.crop_protein_tokens:
                raise NoProteinInCrop(self.founder_native_score, self.crop_ligand_tokens)
        del output
        atom_indices = torch.as_tensor(slot_to_atom, device=self.device)
        if (atom_indices.shape != (25,) or atom_indices.dtype != torch.long or atom_indices.min() < 0
                or atom_indices.max() >= features["ref_element"].shape[1] or atom_indices.unique().numel() != 25):
            raise ValueError("Expected one unique native atom index per occupied slot")
        token_indices = features["atom_to_token"][0, atom_indices].long().argmax(-1)
        ligand = features["affinity_token_mask"][0].bool() & features["token_pad_mask"][0].bool()
        atom_mask = torch.zeros(features["ref_element"].shape[1], device=self.device, dtype=torch.bool)
        atom_mask[atom_indices] = True
        payload = dict(schema_version=1, model="nesso1", spec=self.spec.to_dict(), features=features,
            seed={"atoms": hard.atoms.argmax(-1), "charges": hard.charges.argmax(-1), "bonds": hard.bonds.argmax(-1)},
            slot_to_atom=atom_indices, slot_to_token=token_indices, ligand_atom_mask=atom_mask,
            ligand_token_mask=ligand, element_index=torch.tensor([-1, *self.spec.atomic_numbers[1:]], device=self.device),
            metadata={"preparation_command": "upstream Nesso hard-ligand feature preparation",
                      "native_source_version": model.__class__.__module__, "seed_identity": "verified native founder"},
            extra={"affinity_keep_indices": keep})
        bundle = NativeBundle(payload, spec=self.spec, model="nesso1", device=self.device)
        ids = torch.tensor([0, *(const.bond_type_ids[k] + 1 for k in expected_bonds[1:])], device=self.device)
        self.branch = NessoBranch(model, bundle, None, recycling_steps=recycling_steps, bond_ids=ids)
        with torch.no_grad():
            founder_output = self.branch(hard)
            self.founder_bridge_score = float(_value(founder_output))
            self.founder_bridge_binder_probability = float(founder_output["affinity_probability_binary"].item())
        self._versions = [(p, p._version) for p in (*model.parameters(), *model.buffers())]

    def _rows(self, atoms, bonds):
        rows = tuple(torch.as_tensor(x, dtype=torch.float32, device=self.device) for x in (atoms, bonds))
        for x, shape in zip(rows, ((25, 6), (300, 5))):
            if (x.shape != shape or not x.isfinite().all() or x.min() < 0 or x.max() > 1
                    or not torch.allclose(x.sum(-1), torch.ones_like(x[:, 0]), atol=1e-5, rtol=0)):
                raise ValueError("Expected finite direct probability rows on the simplex")
        return rows

    def _graph(self, atom, bond):
        pairs = bond.new_zeros((25, 25, 5))
        pairs[..., 0] = 1.
        pairs[self._left, self._right] = bond
        pairs[self._right, self._left] = bond
        charge = atom.new_zeros((1, 25, len(self.spec.formal_charges)))
        charge[..., self.spec.formal_charges.index(0)] = 1.
        return RelaxedGraph((atom @ self._elements)[None], charge, pairs[None])

    def gradient(self, atom, bond):
        atom, bond = (x.detach().requires_grad_(True) for x in self._rows(atom, bond))
        with torch.enable_grad():
            value = _value(self.branch(self._graph(atom, bond)))
            gradients = torch.autograd.grad(value, (atom, bond))
        return float(value.detach()), *_gradient_arrays(gradients)

    @torch.enable_grad()
    def schema_gradient(self, atom, charge, bond):
        """Differentiate the full occupied decoder rows in their original slot order."""
        a, q, b = (torch.as_tensor(x, dtype=torch.float32, device=self.device).detach().requires_grad_(True)
                   for x in (atom, charge, bond))
        if (a.shape != (25, len(self.spec.atomic_numbers) - 1) or
                q.shape != (25, len(self.spec.formal_charges)) or b.shape != (300, len(self.spec.bond_names))):
            raise ValueError("Incorrect shared graph dimensions")
        pairs = b.new_zeros(25, 25, len(self.spec.bond_names))
        pairs[..., 0] = 1.
        pairs[self._left, self._right] = pairs[self._right, self._left] = b
        graph = RelaxedGraph(F.pad(a, (1, 0))[None], q[None], pairs[None])
        affinity = _value(self.branch(graph))
        inputs = (a, q, b) if self.branch.charge_enabled else (a, b)
        grads = torch.autograd.grad(affinity, inputs)
        if not self.branch.charge_enabled:
            grads = (grads[0], torch.zeros_like(q), grads[1])
        return float(affinity.detach()), *_gradient_arrays(grads)

    def _affinity_confidence(self, atom, bond):
        atom, bond = (x.detach().requires_grad_(True) for x in self._rows(atom, bond))
        outputs = self.branch(self._graph(atom, bond), return_binder_loss=True)
        binder_loss = outputs["binder_loss"].reshape(())
        if not bool(torch.isfinite(binder_loss)) or not binder_loss.requires_grad:
            raise ValueError("Expected a finite attached binder loss from the native heads")
        return (atom, bond), _value(outputs), binder_loss

    @torch.enable_grad()
    def affinity_confidence_gradients(self, atom, bond):
        """Two first-order derivatives evaluated on one fixed-crop forward pass."""
        inputs, affinity, binder_loss = self._affinity_confidence(atom, bond)
        d_affinity = torch.autograd.grad(affinity, inputs, retain_graph=True)
        d_binder = torch.autograd.grad(binder_loss, inputs)
        return ((float(affinity.detach()), *_gradient_arrays(d_affinity)),
                (float(binder_loss.detach()), *_gradient_arrays(d_binder)))

    @torch.enable_grad()
    def weighted_affinity_confidence_gradient(self, atom, bond, affinity_weight, confidence_weight):
        """Use one backward pass for the fixed-weight production update."""
        if not all(math.isfinite(w) and w > 0 for w in (affinity_weight, confidence_weight)):
            raise ValueError("Both fixed Nesso head weights must be positive and finite")
        inputs, affinity, binder_loss = self._affinity_confidence(atom, bond)
        gradient = torch.autograd.grad(affinity_weight * affinity + confidence_weight * binder_loss, inputs)
        return float(affinity.detach()), float(binder_loss.detach()), *_gradient_arrays(gradient)

    @torch.no_grad()
    def hard_value(self, atom, bond):
        """One-hot screen using the founder conformer and crop, for shortlist selection."""
        atom, bond = self._rows(atom, bond)
        hard = self._graph(F.one_hot(atom.argmax(-1), 6).float(), F.one_hot(bond.argmax(-1), 5).float())
        return float(_value(self.branch(hard)))

    def close(self):
        self.branch.bundle.assert_unchanged()
        if any(p._version != version for p, version in self._versions):
            raise RuntimeError("Frozen Nesso parameters or buffers changed during the pilot")
        return {"frozen_model_and_context_verified": True}
