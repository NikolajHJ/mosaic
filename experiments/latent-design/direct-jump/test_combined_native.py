import json
from pathlib import Path

import numpy as np
import pytest
import torch

from combined_nesso_worker import direct_slot_to_canonical, schema_slot_to_canonical
from latentmol.design.protenix_nesso_blend import fit_latent_weights, nesso_value
from latentmol.representation import GraphSpec
from nesso_inputs import canonical_graph
from nesso_native import NessoBackend
from latentmol.multimodel.nesso import _binder_loss
from study import PAIR_I, PAIR_J, hard_smiles


def test_binary_loss_uses_native_ensemble_including_extreme_logits():
    logits = torch.tensor([[-3., 1.], [-750., -800.]], dtype=torch.float64, requires_grad=True)
    loss = _binder_loss(logits[:, 0], logits[:, 1])
    expected = logits[0].sigmoid().mean()
    torch.testing.assert_close(loss[0].exp().reciprocal(), expected)
    # Extreme low probabilities underflow if averaged before taking their logarithm.
    torch.testing.assert_close(loss[1], logits.new_tensor(750. + np.log(2.)))
    derivatives, = torch.autograd.grad(loss.sum(), logits)
    assert torch.isfinite(derivatives).all() and torch.all(derivatives < 0)


def test_both_head_gradients_match_one_weighted_backward_and_feasible_difference(monkeypatch):
    backend = object.__new__(NessoBackend)
    backend.device = torch.device("cpu")
    monkeypatch.setattr(backend, "_graph", lambda a, b: (a, b))
    calls = []

    def branch(rows, *, return_binder_loss):
        assert return_binder_loss
        calls.append(None)
        a, b = rows
        x, y = a[0, 0] - a[0, 1], b[0, 0] - b[0, 1]
        return {"affinity_pred_value": 2*x + 3*y, "binder_loss": _binder_loss(-1 + 3*x + y, 2 + x - 2*y)}

    backend.branch = branch
    a = np.full((25, 6), 1/6, dtype=np.float32)
    b = np.full((300, 5), .2, dtype=np.float32)
    affinity, confidence = backend.affinity_confidence_gradients(a, b)
    assert len(calls) == 1
    va, vc, da, db = backend.weighted_affinity_confidence_gradient(a, b, .3, .4)
    assert len(calls) == 2
    assert (va, vc) == pytest.approx((affinity[0], confidence[0]))
    np.testing.assert_allclose(da, .3*affinity[1] + .4*confidence[1], atol=1e-6)
    np.testing.assert_allclose(db, .3*affinity[2] + .4*confidence[2], atol=1e-6)
    eps = np.float32(.001)
    direction = np.zeros_like(a)
    direction[0, :2] = [1, -1]
    plus = backend.affinity_confidence_gradients(a + eps*direction, b)[1][0]
    minus = backend.affinity_confidence_gradients(a - eps*direction, b)[1][0]
    assert np.sum(confidence[1]*direction) == pytest.approx((plus - minus)/(2*eps), abs=1e-4)


@pytest.mark.parametrize("seed", ["111210", "111310", "111410"])
def test_founder_correspondence_preserves_all_labels_under_slot_permutation(seed):
    founders = json.loads(Path(__file__).with_name("nesso-pilot.json").read_bytes())["founders"]
    smiles, _, canonical_atoms, canonical_bonds = canonical_graph(founders[seed])
    permutation = np.random.default_rng(int(seed)).permutation(25)
    pair_index = np.zeros((25, 25), int)
    pair_index[PAIR_I, PAIR_J] = pair_index[PAIR_J, PAIR_I] = np.arange(300)
    atoms = canonical_atoms[permutation]
    bonds = canonical_bonds[pair_index[permutation[PAIR_I], permutation[PAIR_J]]]
    assert hard_smiles((atoms, bonds)) == smiles
    observed = direct_slot_to_canonical(atoms, bonds, smiles, canonical_atoms, canonical_bonds)
    assert np.array_equal(atoms, canonical_atoms[observed])
    assert np.array_equal(bonds, canonical_bonds[pair_index[observed[PAIR_I], observed[PAIR_J]]])

    corrupted = canonical_bonds.copy()
    # This nonexistent pair disagrees even though the atoms and the SMILES still match.
    corrupted[np.flatnonzero(canonical_bonds[:, 0])[0]] = [0, 1, 0, 0, 0]
    with pytest.raises(ValueError, match="permutation changed"):
        direct_slot_to_canonical(atoms, bonds, smiles, canonical_atoms, corrupted)


def test_full_schema_charged_bromine_iodine_founder_round_trip():
    spec = GraphSpec(max_nodes=25)
    smiles, mol, ca, cb = canonical_graph("[NH3+]" + "C"*21 + "C(Br)I", spec)
    perm = np.random.default_rng(214).permutation(25)
    q = np.array([spec.formal_charges.index(a.GetFormalCharge()) for a in mol.GetAtoms()])
    b = np.zeros((25, 25), dtype=int)
    b[PAIR_I, PAIR_J] = b[PAIR_J, PAIR_I] = cb.argmax(-1)
    record = dict(spec=spec.to_dict(), smiles=smiles, atoms=(ca.argmax(-1)[perm] + 1).tolist(),
                  charges=q[perm].tolist(), bonds=b[np.ix_(perm, perm)].tolist())
    _, _, mapping = schema_slot_to_canonical(record)
    assert np.array_equal(mapping, perm)
    record["charges"][record["charges"].index(spec.formal_charges.index(1))] = spec.formal_charges.index(0)
    with pytest.raises(ValueError, match="differs from hard graph"):
        schema_slot_to_canonical(record)


@pytest.mark.parametrize("charge_enabled", [True, False])
def test_shared_schema_vjp_matches_feasible_derivative(charge_enabled):
    from types import SimpleNamespace
    from latentmol.representation import RelaxedGraph

    class Branch:
        def __call__(self, g):
            value = 2*g.atoms[..., -1].sum() + 4*g.bonds[..., 2].sum()
            if self.charge_enabled:
                value += 3*g.charges[..., -1].sum()
            return {"affinity_pred_value": value}

    backend = object.__new__(NessoBackend)
    backend.spec, backend.device = GraphSpec(max_nodes=25), torch.device("cpu")
    backend._left, backend._right = torch.triu_indices(25, 25, 1)
    backend.branch = Branch()
    backend.branch.charge_enabled = charge_enabled

    def rpc(a, q, b):
        value, *grads = backend.schema_gradient(a, q, b)
        return value, grads

    worker = SimpleNamespace(schema_gradient=rpc)
    active = torch.ones(25, dtype=torch.bool)
    z = torch.tensor(0., requires_grad=True)

    def projected_loss(z):
        atoms = torch.full((1, 25, 10), 1/9)
        atoms[..., 0], atoms[0, 0, 1], atoms[0, 0, -1] = 0, 1/9-z, 1/9+z
        charges = torch.full((1, 25, 5), .2)
        charges[0, 1, 0], charges[0, 1, -1] = .2-z, .2+z
        bonds = torch.full((1, 25, 25, 5), .2)
        bonds[0, 0, 1, 0], bonds[0, 0, 1, 2] = .2-z, .2+z
        bonds[0, 1, 0] = bonds[0, 0, 1].clone()
        return nesso_value(worker, RelaxedGraph(atoms, charges, bonds), active)

    gradient, = torch.autograd.grad(projected_loss(z), z)
    assert gradient.item() == pytest.approx(10 + 3*charge_enabled)
    eps = .025
    fd = (projected_loss(eps)-projected_loss(-eps))/(2*eps)
    assert gradient.item() == pytest.approx(float(fd), abs=.003)


def test_blend_weights_balance_latent_norms_and_restore_contact_scale():
    pairs = [(torch.tensor([2., 0.]), torch.tensor([0., 8.]))] * 3
    calibration = fit_latent_weights(pairs, .5)
    for probe in calibration["probes"]:
        assert probe["weighted_affinity_to_contact"] == pytest.approx(1.)
        assert probe["mixed_to_contact"] == pytest.approx(1.)
