"""Compact VAE budget, matched replay and cumulative-ratio regression checks."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from rdkit import rdBase
import torch

from latentmol.chemistry import encode_smiles
from latentmol.evaluation.compact_prior_panel import CompactPriorPanel
from latentmol.models import GraphAutoencoder
from latentmol.representation import DecoderOutput, GraphSpec
from latentmol.train import kl_weight
from latentmol.training.compact_benchmark import _count_weights, prior_draw_count, read_settings, run


CONFIG = Path(__file__).resolve().parents[2] / "experiments/vae/compact435k-v100.json"


def test_compact_architecture_and_kl_schedule():
    _, training, _ = read_settings(CONFIG)
    model = GraphAutoencoder(GraphSpec(), training.model)
    assert [sum(p.numel() for p in part.parameters()) for part in (model.encoder, model.decoder)] == [238360, 197040]
    assert training.loss.atom_order == "graph"
    assert kl_weight(training, 1, 1_000_000) == pytest.approx(.0005)
    assert kl_weight(training, 1, 2_500_000) == pytest.approx(.001)


@pytest.mark.parametrize("batch", (256, 512))
def test_ratio_carries_between_steps(batch):
    priors = [prior_draw_count(i, batch, 15) for i in range(0, batch * 15, batch)]
    assert sum(priors) == batch
    assert set(priors) == {batch // 15, batch // 15 + 1}


@pytest.mark.parametrize("requested", ({"count_kl_weight": .3}, {"argmax_count_weight": .7},
                                       {"count_kl_weight": .3, "argmax_count_weight": .7}))
def test_matched_guided_control_disables_every_new_count_term(requested):
    settings = {"prior_guidance": requested}
    zero = {"count_kl_weight": 0.0, "argmax_count_weight": 0.0}
    assert _count_weights(settings, "prior_guided_control") == zero
    assert _count_weights(settings, "prior_guided") == {**zero, **requested}


def test_held_out_panel_measures_the_raw_argmax_and_keeps_rng():
    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(latent=1)
            self.spec = GraphSpec(max_nodes=2, atomic_numbers=(0, 6, 7))
            self.observed_codes = []

        def decode(self, z):
            self.observed_codes.append(z.detach().clone())
            # Both cases have >50% total real-element mass. The raw atom argmax
            # picks either carbon or NO_ATOM; RDKit then sees ethane or an empty graph.
            atoms = torch.where(z > 0, z.new_tensor([.35, .36, .29]), z.new_tensor([.40, .35, .25]))
            charges = z.new_zeros(len(z), 2, 5)
            charges[..., self.spec.formal_charges.index(0)] = 1
            bonds = z.new_zeros(len(z), 2, 2, 5)
            bonds[..., self.spec.bond_names.index("SINGLE")] = 1
            return DecoderOutput(atoms.log().unsqueeze(1).expand(-1, 2, -1), charges, bonds)

    model = Decoder()
    rng = torch.random.get_rng_state()
    panel = CompactPriorPanel(1, count=16, seed=5, batch_size=3)
    observed = panel.observe(model)
    assert model.training and torch.equal(torch.random.get_rng_state(), rng)
    assert panel.observe(model) == observed
    full = observed["prior_at_capacity"]
    assert 0 < full < 16
    assert observed["prior_at_capacity_fraction"] == full / 16
    assert observed["prior_size_histogram"] == {"0": 16 - full, "2": full}
    assert observed["prior_soft_count_mean"] == pytest.approx(2 * (.60 + .05 * full / 16))
    assert observed["prior_valid"] == full and observed["prior_unique_valid"] == 1
    assert observed["prior_failures"] == {"empty": 16 - full}
    assert observed["categorical"]["prior_points"] == 16
    assert "1" in observed["categorical"]["prior_size_histogram"]
    assert "stress_panel" not in observed

    model.observed_codes.clear()
    radial_panel = CompactPriorPanel(1, count=16, seed=5, batch_size=3, stress_radius_multiplier=1.5)
    both = radial_panel.observe(model)
    inner, outer = torch.cat(model.observed_codes).split(16)
    assert torch.equal(outer, 1.5 * inner)
    outer_summary = both.pop("stress_panel")
    assert outer_summary.pop("radius_multiplier") == 1.5
    # This decoder only uses the sign of z, so the paired categorical draws must
    # agree exactly as well, despite re-running at a larger radius.
    assert both == outer_summary == observed
    assert model.training and torch.equal(torch.random.get_rng_state(), rng)


@pytest.mark.parametrize("include_control", (False, True))
def test_matched_full_updates_on_cpu(tmp_path, include_control):
    settings, training, _ = read_settings(CONFIG)
    spec = GraphSpec(max_nodes=4)

    class Fixture(list):
        pass

    data = Fixture()
    data.dataset_sha256 = "local-smoke"
    data.spec = spec
    data.manifest = dict(provenance={"canonicalization": "RDKit canonical SMILES; reparsed for stable atom ordering"},
                         rdkit_version=rdBase.rdkitVersion, active_atom_counts={"train": {"2": 6, "3": 6}})
    for smiles in ["CC", "CCC", "CO", "CNC"] * 3:
        graph = encode_smiles(smiles, spec)
        data.append({key: torch.as_tensor(graph[source], dtype=torch.long)
                     for key, source in (("atoms", "atom_types"), ("charges", "charges"), ("bonds", "bond_types"))})
    training = replace(training, model=replace(training.model, global_hidden=12, hidden=8, latent=8, layers=1,
                        refinement_rounds=1, encoder_residual_layers=1, pair_hidden=4, triangle_hidden=2,
                        attention_heads=2))
    total = sum(p.numel() for p in GraphAutoencoder(spec, training.model).parameters())
    settings = dict(settings, batch_sizes=[2, 4], shared_warmup_rows=4, shape_warmup_rows=4, timed_rows=4,
                    minimum_microbatch=2, prior_ratio=2, expected_parameters=total,
                    prior_guidance={**settings["prior_guidance"], "workers": 0})
    if include_control:
        settings["include_prior_count_control"] = True
        settings["prior_guidance"].update(count_kl_weight=.3, argmax_count_weight=.7)
        settings["prior_panel"] = {"seed": 132, "batch_size": 2, "count": 5, "stress_radius_multiplier": 1.5}
    report = run(settings, training, data, tmp_path, "cpu", {})
    assert json.loads((tmp_path / "benchmark.json").read_text()) == report
    assert report["status"] == "completed"
    modes = ["real_only", "prior_guided_control", "prior_guided"] if include_control else ["real_only", "prior_guided"]
    assert [(r["batch_size"], r["mode"], r["prior_draws"]) for r in report["arms"]] == [
        (size, mode, 0 if mode == "real_only" else 2) for size in (2, 4) for mode in modes]
    assert all(r["native_batch"] and r["samples"] == 4 and r["seconds"] > 0 for r in report["arms"])
    if include_control:
        assert report["prior_panel"]["before"]["prior_points"] == 5
        assert all(sum(arm["prior_panel"]["prior_size_histogram"].values()) == 5 for arm in report["arms"])
        assert report["prior_panel"]["before"]["stress_panel"]["prior_points"] == 5
        assert all(arm["prior_panel"]["stress_panel"]["prior_points"] == 5 for arm in report["arms"])
        assert [r["count_kl_weight"] for r in report["arms"]] == [0., 0., .3] * 2
        assert [r["argmax_count_weight"] for r in report["arms"]] == [0., 0., .7] * 2
    else:
        assert "prior_panel" not in report and all("prior_panel" not in r for r in report["arms"])
    for arm in report["arms"]:
        if arm["mode"] == "prior_guided":
            first = arm["updates_detail"][0]
            assert first["prior_guidance"]["count_window_points"] == first["prior_count"]
