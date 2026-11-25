"""Both decoders train on the same random draws and write independently reusable endpoints."""

from dataclasses import replace
import json
from pathlib import Path

import pytest
from rdkit import rdBase
import torch

from latentmol.chemistry import encode_canonical_smiles
from latentmol.evaluation.compact_training_panel import CompactTrainingPanel
from latentmol.evaluation.evaluate import load_model
from latentmol.experiments import file_sha256
from latentmol.models import GraphAutoencoder
from latentmol.representation import GraphSpec
from latentmol.training.compact_decoder_comparison import (
    ALL_VARIANTS, HYBRID, VARIANTS, WIDE_HYBRID, read_comparison, run_comparison,
)


CONFIG = Path(__file__).resolve().parents[2] / "experiments/vae/compact-decoder-comparison.json"


class ExampleGraphs(list):
    spec = GraphSpec(max_nodes=5)
    dataset_sha256 = "matched-training-fixture"
    manifest = {"provenance": {"canonicalization": "RDKit canonical SMILES; reparsed for stable atom ordering"},
                "rdkit_version": rdBase.rdkitVersion, "active_atom_counts": {"train": {"1": 1, "2": 3, "3": 4}}}

    def __init__(self, split, smiles):
        self.split = split
        for item in smiles:
            raw = encode_canonical_smiles(item, self.spec)
            self.append({key: torch.as_tensor(raw[source], dtype=torch.long) for key, source in
                         (("atoms", "atom_types"), ("charges", "charges"), ("bonds", "bond_types"))})


def test_matched_draws_graph_metrics_collision_and_endpoints(tmp_path):
    config, settings, cfg = read_comparison(CONFIG)
    config.update(updates=2, microbatch=2, observe_updates=[0, 1, 2])
    data = ExampleGraphs("train", ["C", "CC", "CO", "CCC", "CCO", "CCN", "CCCl", "CF"])
    val = ExampleGraphs("val", ["N", "O", "CN", "C1CC1"])
    cfg = replace(cfg, device="cpu", batch_size=4,
                  model=replace(cfg.model, global_hidden=12, hidden=8, latent=8, layers=1,
                                refinement_rounds=1, encoder_residual_layers=1, pair_hidden=4,
                                triangle_hidden=2, attention_heads=2))
    settings.update(data_sha256=data.dataset_sha256,
                    expected_parameters=sum(p.numel() for p in GraphAutoencoder(data.spec, cfg.model).parameters()))
    settings["prior_occupancy"]["real_per_prior"] = 3
    panel = CompactTrainingPanel(val, cfg, seed=100, validation_count=4, prior_count=3, batch_size=2)
    output = tmp_path / "matched"
    comparison = run_comparison(config, settings, cfg, data, panel, output, "cpu")
    saved = json.loads((output / "comparison.json").read_bytes())
    assert saved["status"] == comparison["status"] == "completed"
    for field in ("prior_noise_sha256", "posterior_noise_sha256", "initial_shared_sha256"):
        assert len({arm[field] for arm in saved["arms"].values()}) == 1
    for variant in VARIANTS:
        arm = comparison["arms"][variant]
        assert (arm["real_points"], arm["prior_draws"]) == (8, 2)
        assert arm["status"] == "completed" and arm["training_seconds"] > 0
        assert arm["topology"]["all"]["targets"] == 4
        assert arm["topology"]["cyclic_targets"]["targets"] == 1
        assert file_sha256(output / arm["checkpoint"]) == arm["checkpoint_sha256"]
        model, checkpoint = load_model(output / arm["checkpoint"], "cpu", data.spec)
        assert model.config.decoder_variant == variant and checkpoint["updates"] == 2
        observations = json.loads((output / variant / "quality.json").read_bytes())
        assert set(observations) == {"0", "1", "2"}
    with pytest.raises(FileExistsError):
        run_comparison(config, settings, cfg, data, panel, output, "cpu")

    # Extend the finished experiment; the two controls supply comparison
    # results but their weights are neither trained nor read from endpoints.
    reference = dict(path=str(output / "comparison.json"), sha256=file_sha256(output / "comparison.json"))
    hybrid_config = dict(config, variants=[HYBRID], controls=reference)
    hybrid_dir = tmp_path / "hybrid-only"
    result = run_comparison(hybrid_config, settings, cfg, data, panel, hybrid_dir, "cpu")
    assert set(result["arms"]) == {HYBRID}
    hybrid = result["arms"][HYBRID]
    control = comparison["arms"]["bond_message"]
    for field in ("initial_shared_sha256", "posterior_noise_sha256", "prior_noise_sha256", "prior_draws"):
        assert hybrid[field] == control[field]
    assert hybrid["identical_initial_logit_probe_count"] == 3
    assert hybrid["bond_message_ablation"]["samples"] == len(val)
    control_quality = json.loads((output / "bond_message/quality.json").read_bytes())
    hybrid_quality = json.loads((hybrid_dir / HYBRID / "quality.json").read_bytes())
    assert hybrid_quality["0"] == control_quality["0"]
    assert reference["sha256"] == file_sha256(output / "comparison.json")
    assert not any((hybrid_dir / variant).exists() for variant in VARIANTS)
    with pytest.raises(ValueError, match="row_order_sha256"):
        run_comparison(dict(hybrid_config, row_seed=config["row_seed"] + 1), settings, cfg,
                       data, panel, tmp_path / "different-order", "cpu")
    with pytest.raises(ValueError, match="digest"):
        run_comparison(dict(hybrid_config, controls=dict(reference, sha256="wrong")),
                       settings, cfg, data, panel, tmp_path / "changed-controls", "cpu")


def test_fresh_long_recipe_keeps_arms_matched_and_weights_loadable(tmp_path):
    config, settings, cfg = read_comparison(CONFIG.with_name("compact-decoder-long.json"))
    assert cfg.seed == 41 and settings["prior_occupancy"]["seed"] == 2026112042
    assert settings["panel"] == dict(seed=2026112043, validation_count=1024, prior_count=1024, batch_size=16)
    assert config["updates"] == 8192 and config["save_updates"] == [2048, 4096]
    config.update(updates=2, observe_updates=[0, 1, 2], save_updates=[1], microbatch=2)
    data = ExampleGraphs("train", ["C", "CC", "CO", "CCC", "CCO", "CCN", "CCCl", "CF"])
    val = ExampleGraphs("val", ["N", "O", "CN", "C1CC1"])
    cfg = replace(cfg, device="cpu", batch_size=4,
                  model=replace(cfg.model, global_hidden=12, hidden=8, latent=8, layers=1,
                                refinement_rounds=1, encoder_residual_layers=1, pair_hidden=4,
                                triangle_hidden=2, attention_heads=2))
    settings.update(data_sha256=data.dataset_sha256,
                    expected_parameters=sum(p.numel() for p in GraphAutoencoder(data.spec, cfg.model).parameters()))
    settings["prior_occupancy"]["real_per_prior"] = 3
    panel = CompactTrainingPanel(val, cfg, seed=settings["panel"]["seed"], prior_count=3, validation_count=4, batch_size=2)
    output = tmp_path / "long"
    run_comparison(config, settings, cfg, data, panel, output, "cpu")
    result = json.loads((output / "comparison.json").read_bytes())
    arms = result["arms"]
    assert result["status"] == "completed" and set(arms) == set(ALL_VARIANTS)
    for field in ("prior_noise_sha256", "posterior_noise_sha256", "initial_shared_sha256"):
        assert len({arm[field] for arm in arms.values()}) == 1
    assert arms["bond_message"]["initial_bond_message_state_sha256"] == arms[HYBRID]["initial_bond_message_state_sha256"]
    panels = {}
    for name, info in arms.items():
        assert info["bond_ranking"]["samples"] == 4
        panels[name] = json.loads((output / name / "quality.json").read_bytes())
        assert all(x["topology"]["all"]["targets"] == 4 for x in panels[name].values())
        mid, = info["intermediate_checkpoints"].values()
        path = output / mid["checkpoint"]
        assert file_sha256(path) == mid["checkpoint_sha256"]
        model, checkpoint = load_model(path, "cpu", data.spec)
        assert (model.config.decoder_variant, checkpoint["updates"], checkpoint["real_points"]) == (name, 1, 4)
        assert not checkpoint["comparison_endpoint"]
    assert panels[HYBRID]["0"] == panels["bond_message"]["0"]

    # A separately submitted capacity arm inherits this run; it must see exactly
    # the same examples and latent draws without retraining the existing arms.
    wide, wide_settings, wide_training = read_comparison(CONFIG.with_name("compact-hybrid-wide.json"))
    assert wide_settings["panel"]["seed"] == panel.metadata["validation_seed"]
    assert wide_training.seed == cfg.seed
    assert all(wide[k] == result["config"][k] for k in ("row_seed", "prior_seed", "training_seed"))
    full_model = GraphAutoencoder(GraphSpec(), replace(wide_training.model, decoder_variant=HYBRID,
                                                       **wide["decoder_widths"]))
    assert sum(p.numel() for p in full_model.parameters()) == 481668
    wide.update(updates=2, observe_updates=[0, 1, 2], save_updates=[1], microbatch=2,
                decoder_widths={"pair_hidden": 8, "triangle_hidden": 4})
    wide_output = tmp_path / "wide"
    extra = run_comparison(wide, settings, cfg, data, panel, wide_output, "cpu")
    assert list(extra["arms"]) == [WIDE_HYBRID]
    assert extra["row_order_sha256"] == result["row_order_sha256"] and extra["panel"] == result["panel"]
    arm = extra["arms"][WIDE_HYBRID]
    for field in ("encoder_parameters", "prior_draws", "posterior_noise_sha256", "prior_noise_sha256"):
        assert arm[field] == arms[HYBRID][field]
    assert arm["decoder_parameters"] > arms[HYBRID]["decoder_parameters"]
    assert all(k.startswith("decoder.") for k in arm["resized_state_tensors"])
    assert arm["bond_ranking"]["samples"] == len(val)
    reloaded, _ = load_model(wide_output / arm["checkpoint"], "cpu", data.spec)
    assert (reloaded.config.decoder_variant, reloaded.config.pair_hidden, reloaded.config.triangle_hidden) == (HYBRID, 8, 4)
