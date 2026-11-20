"""Run the YAML optimizer with CPU graph derivatives and a swappable VAE archive."""

from contextlib import contextmanager
import json
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np
import pytest
import torch
import yaml

from mosaic.small_molecule import generate_molecule
from mosaic.small_molecule.core.boltz2 import setup


class SmallVAE(torch.nn.Module):
    """Three distinct latent coordinates, four slots, deliberately shuffled element vocabulary."""

    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(44)
        self.register_buffer("atom_w", torch.randn(3, 4, 7, generator=generator))
        pair_w = torch.randn(3, 4, 4, 5, generator=generator)
        self.register_buffer("pair_w", (pair_w + pair_w.transpose(1, 2)) * 0.5)

    @torch.jit.export
    def encode(self, atoms, charges, bonds):
        return torch.zeros((1, 3), dtype=torch.float32)

    @torch.jit.export
    def decode(self, z):
        return torch.einsum("bd,dnc->bnc", z, self.atom_w), torch.einsum("bd,dijc->bijc", z, self.pair_w)


@pytest.fixture
def archive(tmp_path):
    path = tmp_path / "alternate.pt"
    meta = {
        "format": "mosaic-graph-vae-v1", "latent_dim": 3, "max_nodes": 4,
        "atomic_numbers": [0, 17, 9, 16, 8, 7, 6], "formal_charges": [0],
        "bond_names": ["NONE", "SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"],
    }
    torch.jit.save(torch.jit.script(SmallVAE()), path, _extra_files={"vae.json": json.dumps(meta)})
    return path


@pytest.fixture
def run_yaml(tmp_path, monkeypatch):
    """Replace only the native scorer service. Exercise the real YAML dispatch, loss VJP and files."""
    monkeypatch.setenv("LSB_JOBID", "cpu-test")

    @contextmanager
    def open_toy(general, general_path, target, output, n_atoms):
        def loss(graph, *, key):
            weights = jnp.arange(1, n_atoms + 1)
            return (graph.X[:, 2] * weights).sum() + (graph.E_upper[:, 1] * weights[::-1]).sum(), {}
        yield loss, {"model": "cpu-test"}, {"n_atoms": n_atoms}

    monkeypatch.setattr(setup, "open_affinity", open_toy)

    def run(name, general_overrides=None, stage_overrides=None):
        general_path = tmp_path / f"{name}-general.yaml"
        stage_path = tmp_path / f"{name}-stage.yaml"
        general = {
            "model": "boltz2", "protein": {"structure": str(Path(__file__).resolve().parents[1] / "IL7RA.cif"), "chain": "A"},
            "n_atoms": 3, "epochs": 1, "seed": 37, "learning_rate": 0.12,
            "output_dir": name, "initialization": {"random": {"expected_degree": 1.0}},
        }
        general.update(general_overrides or {})
        stage = {"stage": "stage1", "readout_every": 1} | (stage_overrides or {})
        general_path.write_text(yaml.safe_dump(general), encoding="utf-8")
        stage_path.write_text(yaml.safe_dump(stage), encoding="utf-8")
        monkeypatch.setattr(sys, "argv", ["generate_molecule", "--general-config", str(general_path),
                                         "--stage-config", str(stage_path)])
        generate_molecule.main()
        return tmp_path / name

    return run


def saved(folder):
    with np.load(folder / "checkpoint.npz", allow_pickle=False) as data:
        return dict(data)


def test_yaml_swap_and_both_cotangents_reach_the_same_latent(run_yaml, archive):
    config = {"use_vae": True, "vae": {"checkpoint": archive.name, "temperature": 0.8},
              "initialization": {"smiles": "OCC"}}
    final = saved(run_yaml("one", config))
    # Encoder input is canonical CCO. Mosaic sees original OCC slot order.
    slots = np.array([2, 1, 0])
    np.testing.assert_array_equal(final["vae_slots"], slots)
    z = torch.zeros((1, 3), requires_grad=True)
    atom, bond = SmallVAE().decode(z)
    # Independently put the checkpoint's channels in Mosaic's C,N,O,S,F,Cl order.
    x = (atom[0, slots][:, [6, 5, 4, 3, 2, 1]] / 0.8).softmax(-1)
    i, j = np.triu_indices(len(slots), 1)
    e = (bond[0, slots[i], slots[j]] / 0.8).softmax(-1)
    w = torch.arange(1, 4)
    atom_grad, = torch.autograd.grad((w * x[:, 2]).sum(), z, retain_graph=True)
    bond_grad, = torch.autograd.grad((w.flip([0]) * e[:, 1]).sum(), z)
    assert atom_grad.norm() > 0 and bond_grad.norm() > 0
    np.testing.assert_allclose(final["vae_z"], (-0.12 * (atom_grad + bond_grad)).numpy(), atol=2e-6)
    rows = [json.loads(s) for s in (archive.parent / "one/trace.jsonl").read_text().splitlines()]
    assert rows[1]["loss"] < rows[0]["loss"]


def test_resume_restores_latent_and_refuses_incompatible_checkpoint(run_yaml, archive):
    config = {"use_vae": True, "vae": {"checkpoint": archive.name}}
    first = run_yaml("first", config)
    resumed_config = config | {"initialization": {"checkpoint": str(first / "checkpoint.npz")}}
    resumed = saved(run_yaml("resumed", resumed_config))
    full = saved(run_yaml("full", config | {"epochs": 2}))
    for field in ("vae_z", "vae_slots", "X", "E_upper"):
        np.testing.assert_allclose(resumed[field], full[field], atol=1e-6)
    with pytest.raises(ValueError, match="different VAE"):
        run_yaml("wrong-temperature", resumed_config | {"vae": {"checkpoint": archive.name, "temperature": 0.7}})

    direct = run_yaml("direct", {"use_vae": False})
    assert json.loads((direct / "manifest.json").read_text())["optimizer"]["name"] == "projected_gradient"
    assert "vae_z" not in saved(direct)
    with pytest.raises(ValueError, match="latent checkpoint"):
        run_yaml("direct-to-vae", config | {"initialization": {"checkpoint": str(direct / "checkpoint.npz")}})


def test_default_epoch45_runs_through_stage_alias(run_yaml):
    run = run_yaml("bundled", {"initialization": {"smiles": "CCO"}}, {"stage": "vae"})
    arrays = saved(run)
    assert arrays["vae_z"].shape == (1, 256)
    assert np.isfinite(arrays["vae_z"]).all()
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["optimizer"]["name"] == "latent_gradient" and manifest["updates_performed"] == 1
    with np.load(run / "trajectory.npz", allow_pickle=False) as trajectory:
        assert trajectory["X"].shape == (2, 3, 6)
        assert trajectory["E_upper"].shape == (2, 3, 5)
        np.testing.assert_allclose(trajectory["E_upper"].sum(-1), 1.0, atol=2e-6)


def test_stage2_cannot_silently_overwrite_the_decoded_graph(run_yaml, archive):
    with pytest.raises(ValueError, match="post-step"):
        run_yaml("operator", {"use_vae": True, "vae": {"checkpoint": archive.name}},
                 {"stage": "stage2", "operators": {"mode": "valence"}})


def test_runner_vae_switch_in_stage_yaml(run_yaml, archive):
    stage = {"stage": "stage2", "use_vae": True, "operators": {"mode": "none"},
             "vae": {"checkpoint": archive.name}}
    run = run_yaml("stage-switch", {}, stage)
    assert saved(run)["vae_z"].shape == (1, 3)
    with pytest.raises(ValueError, match="Conflicting use_vae"):
        run_yaml("conflicting-switch", {"use_vae": False}, stage)
