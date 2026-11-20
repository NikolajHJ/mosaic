"""Exercise the Stage 0 loop with the published VAE and an analytic graph scorer."""

from contextlib import ExitStack, contextmanager
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("smiles", [None, "OCC"])
def test_stage0_updates_a_real_latent_with_the_selected_loss(tmp_path, monkeypatch, smiles):
    import torch
    from mosaic.small_molecule.Stage0.pipeline import optimize
    from mosaic.small_molecule.core.boltz2 import setup

    @contextmanager
    def open_test_loss(general, general_path, target, output, count):
        def affinity(graph, *, key):
            return graph.X[:, 1].sum() + graph.E_upper[:, 2].sum(), {}

        yield affinity, {"fixture": "analytic"}, {"n_atoms": count, "seed_smiles": "C" * count}

    monkeypatch.setattr(setup, "open_affinity", open_test_loss)
    general_path, stage_path = tmp_path / "general.yaml", tmp_path / "stage.yaml"
    general_path.write_text("use_vae: true", encoding="utf-8")
    stage_path.write_text("stage: stage0", encoding="utf-8")
    structure = Path(__file__).resolve().parents[1] / "IL7RA.cif"
    general = {"use_vae": True, "protein": {"structure": str(structure), "chain": "A"},
               "n_atoms": 3, "epochs": 1, "learning_rate": 0.01, "seed": 19, "model": "boltz2",
               "initialization": {"smiles": smiles}}
    output = tmp_path / "result"
    with ExitStack() as resources:
        optimize(general, {"stage": "stage0"}, general_path, stage_path, output, resources)

    trace = json.loads((output / "trace.json").read_text())
    assert len(trace) == 2 and 0 < trace[0]["latent_gradient_norm"] < float("inf")
    assert json.loads((output / "manifest.json").read_text())["initial_smiles"] == smiles
    with np.load(output / "checkpoint.npz") as ckpt, np.load(output / "trajectory.npz") as trajectory:
        from mosaic.small_molecule.VAE.provider import load_provider

        model = load_provider()
        initial_z, slots = model.initialize(3, 19, smiles=smiles)
        np.testing.assert_array_equal(slots.numpy(), ckpt["vae_slots"])
        final_z = torch.from_numpy(ckpt["vae_z"])
        assert not torch.equal(initial_z, final_z)
        x, edges = model.decode(final_z, slots)
        np.testing.assert_allclose(x.detach().numpy(), trajectory["X"][-1], atol=1e-7)
        np.testing.assert_allclose(edges.detach().numpy(), trajectory["E_upper"][-1], atol=1e-7)
