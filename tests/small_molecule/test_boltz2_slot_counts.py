"""Check non-default ligand sizes through preparation, transport, and feature mapping."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from mosaic.small_molecule.core.boltz2 import bridge, native, prepare


@pytest.mark.parametrize("n_atoms", [1, 6, 11, 25, 26, 40])
def test_bridge_and_native_inputs_follow_prepared_size(n_atoms, monkeypatch):
    pair_count = n_atoms * (n_atoms - 1) // 2
    atoms = np.zeros((n_atoms, 6), dtype=np.float32)
    bonds = np.zeros((pair_count, 5), dtype=np.float32)
    atoms[:, 0] = 1
    bonds[:, 0] = 1
    indices = torch.arange(1, n_atoms + 1)
    n_tokens = n_atoms + 1  # One protein token before the ligand.
    context = {
        "atom_indices": indices,
        "token_indices": indices,
        "features": {
            "ref_element": torch.zeros(1, n_tokens, 128),
            "token_pad_mask": torch.ones(1, n_tokens),
            "token_bonds": torch.zeros(1, n_tokens, n_tokens, 1),
            "type_bonds": torch.zeros(1, n_tokens, n_tokens, dtype=torch.long),
        },
    }
    worker = object.__new__(bridge.AffinityWorker)
    worker.identity = {"n_atoms": n_atoms}

    def request(action, *, X, E, gradient, graph_sha256):
        assert action == "evaluate" and gradient
        x = torch.tensor(X, requires_grad=True)
        edges = torch.tensor(E).reshape(pair_count, 5).requires_grad_()
        features, bond_types = native.inject(x, edges, context)
        assert features["ref_element"][0, indices, 6].tolist() == [1] * n_atoms
        assert bond_types.shape == (1, n_tokens, n_tokens, 7)
        gx, ge = torch.autograd.grad(features["ref_element"].sum() + bond_types.sum(), (x, edges))
        return {
            "graph_sha256": graph_sha256,
            "values": [0.0, 0.0, 0.0],
            "gradient_X": gx.tolist(),
            "gradient_E": ge.tolist(),
        }

    monkeypatch.setattr(worker, "request", request)
    _, gx, ge = worker.evaluate(atoms, bonds, gradient=True)
    assert gx.shape == atoms.shape and ge.shape == bonds.shape
    with pytest.raises(ValueError, match="atom slots"):
        worker.evaluate(atoms[:-1], bonds, gradient=False)


def test_prepare_new_size_from_existing_fixture(tmp_path, monkeypatch):
    fixture = Path(__file__).resolve().parents[1] / "data/brd4_3mxf/fixture_25.json"
    output = tmp_path / "pose"
    monkeypatch.setenv("LSB_JOBID", "test")
    monkeypatch.setattr(native, "verify_installation", lambda: None)
    monkeypatch.setattr(prepare.subprocess, "run", lambda *args, **kwargs: None)

    def snapshot(prediction, cache, fixture_path, context, n_atoms, pocket):
        assert fixture_path == fixture
        assert n_atoms == 26
        return {"n_atoms": n_atoms}

    monkeypatch.setattr(prepare, "snapshot", snapshot)
    monkeypatch.setattr(sys, "argv", [
        "prepare", "--fixture", str(fixture), "--cache", str(tmp_path),
        "--output", str(output), "--n-atoms", "26",
    ])
    prepare.main()
    specification = json.loads((output / "input.yaml").read_text())
    assert specification["sequences"][1]["ligand"]["smiles"] == "C" * 26
    assert json.loads((output / "preparation.json").read_text())["n_atoms"] == 26
