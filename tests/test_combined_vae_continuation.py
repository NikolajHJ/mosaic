from dataclasses import asdict
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from latentmol.chemistry import encode_smiles
from latentmol.experiments import file_sha256
from latentmol.models import GraphAutoencoder, ModelConfig
from latentmol.representation import GraphSpec, HardGraphBatch
from latentmol.train import TrainConfig, capture_rng_state, state_digest
from latentmol.training import combined_continuation as continuation
from latentmol.training.combined_continuation import _benchmark, _benchmark_shape, _train


class Rows(list):
    manifest = {"dataset_sha256": "test-combined-dataset"}


class UnusedReferenceAnchor:
    def assert_unchanged(self):
        pass


@pytest.mark.parametrize("epoch,row", [(46, 240), (47, 0)])
def test_loss_change_authenticates_the_exact_boundary_and_preserves_the_recipe(tmp_path, epoch, row):
    source = dict(parent_epoch=46, epochs=1, shuffle_seed=23, prior_only=dict(real_per_prior=15, weight=.375),
                  protocol=dict(microbatch_size=120))
    previous = tmp_path / "old.json"
    previous.write_text(json.dumps(source))
    cursor = dict(epoch=epoch, row=row, updates=120, real_points=240)
    state = dict(config_sha256=file_sha256(previous), combined_version=continuation.VERSION,
                 dataset_sha256="dataset", batch_size=240, microbatch_size=120, epoch=epoch,
                 continuation=dict(**cursor, epoch_complete=row == 0))
    paused = tmp_path / "paused.pt"
    torch.save(state, paused)
    digest = file_sha256(paused)
    candidate = deepcopy(source)
    candidate["epochs"] = epoch + 1 - source["parent_epoch"]
    candidate["recipe_handoff"] = dict(**cursor, checkpoint_sha256=digest, source_config_path=previous.name,
                                       source_config_sha256=file_sha256(previous))
    candidate["prior_only"]["guidance"] = {"occupancy_weight": 1}

    def load(config, resume=paused, sha=digest):
        return continuation._load_training_resume(tmp_path / "new.json", config, "new-recipe", resume, sha,
                                                  "dataset", 240, 120)

    resumed, receipt = load(candidate)
    assert resumed == state
    assert receipt["target_epoch"] == epoch + 1 and receipt["checkpoint_sha256"] == digest
    with pytest.raises(ValueError, match="requires its saved training checkpoint"):
        load(candidate, resume=None, sha=None)
    with pytest.raises(ValueError, match="outside prior guidance"):
        load(dict(candidate, shuffle_seed=42))
    with pytest.raises(ValueError, match="boundary"):
        load(dict(candidate, recipe_handoff=dict(candidate["recipe_handoff"], row=row + 240)))
    with pytest.raises(ValueError, match="different combined continuation"):
        load(dict(candidate, recipe_handoff=dict(candidate["recipe_handoff"], checkpoint_sha256="other")))
    with pytest.raises(ValueError, match="different combined continuation"):
        load(source)
    torch.save(dict(state, config_sha256="new-recipe"), paused)
    assert load(candidate, sha=file_sha256(paused))[1] is None


def test_dropout_handoff_composes_only_with_the_pinned_prior_guidance(tmp_path):
    before = dict(parent_epoch=46, epochs=1, prior_only={"weight": .375}, protocol={"microbatch_size": 2})
    origin = tmp_path / "pre-guidance.json"
    origin.write_text(json.dumps(before))
    cursor = dict(epoch=46, row=2, updates=101, real_points=2)
    saved = dict(config_sha256=file_sha256(origin), combined_version=continuation.VERSION,
                 dataset_sha256="data", batch_size=2, microbatch_size=2, model_config={}, epoch=46,
                 continuation=dict(**cursor, epoch_complete=False))
    checkpoint = tmp_path / "model.pt"
    torch.save(saved, checkpoint)
    handoff = dict(**cursor, checkpoint_sha256=file_sha256(checkpoint), source_config_path=origin.name,
                   source_config_sha256=file_sha256(origin))
    approved = dict(before, recipe_handoff=handoff, prior_only=dict(before["prior_only"],
                    guidance={"occupancy_weight": 1., "conditional_weight": 1.}))
    source = tmp_path / "approved-guidance.json"
    source.write_text(json.dumps(approved))
    candidate = dict(approved, atom_refinement_dropout=.05,
                     dropout_handoff=dict(source_config_path=source.name, source_config_sha256=file_sha256(source)))
    def load(config):
        return continuation._load_training_resume(tmp_path / "new.json", config, "dropout-release", checkpoint,
                                                  file_sha256(checkpoint), "data", 2, 2)

    state, receipt = load(candidate)
    assert state["continuation"]["updates"] == cursor["updates"]
    assert receipt["atom_refinement_dropout"] == .05
    assert receipt["source_config_sha256"] == file_sha256(origin)
    assert receipt["approved_guidance_config_sha256"] == file_sha256(source)
    modified = deepcopy(candidate)
    modified["prior_only"]["guidance"]["occupancy_weight"] = .1
    with pytest.raises(ValueError, match="outside atom-refinement dropout"):
        load(modified)
    torch.save(dict(saved, config_sha256="dropout-release"), checkpoint)
    with pytest.raises(ValueError, match="student dropout"):
        load(candidate)
    torch.save(dict(saved, config_sha256="dropout-release", model_config={"atom_refinement_dropout": .05}), checkpoint)
    assert load(candidate)[1] is None
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="Input digest"):
        load(candidate)


@pytest.mark.parametrize("encoder_pooling", ["ordered", "canonical"])
def test_combined_benchmark_and_resume_use_fresh_one_to_one_batches(tmp_path, encoder_pooling):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        spec = GraphSpec(max_nodes=4)
        parent = GraphAutoencoder(spec, ModelConfig(hidden=8, latent=4, layers=1,
                                                   encoder_pooling=encoder_pooling))
        config = TrainConfig(mode="vae", device="cpu", model=parent.config)
        optimizer = torch.optim.AdamW(parent.parameters(), lr=config.learning_rate)
        for parameter in parent.parameters():
            parameter.grad = torch.full_like(parameter, .001)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        saved_rng = capture_rng_state({"fresh_prior": torch.Generator().manual_seed(77)}, False)
        payload = dict(train_config=asdict(config), loss_config=asdict(config.loss),
                       optimizer_state=optimizer.state_dict(), continuation={"updates": 8, "rng": saved_rng})
        reference = GraphAutoencoder(spec, parent.config).eval().requires_grad_(False)
        reference.load_state_dict(parent.state_dict())
        composition = {"elements_distribution": [1 / (len(spec.atomic_numbers) - 1)] * (len(spec.atomic_numbers) - 1),
                       "charges_distribution": [1 / len(spec.formal_charges)] * len(spec.formal_charges)}
        assets = (parent, payload, reference, UnusedReferenceAnchor(), composition)
        data = Rows()
        for smiles in ("C", "CO", "CCO", "CN", "CCN"):
            graph = encode_smiles(smiles, spec)
            data.append({name: torch.as_tensor(graph[key], dtype=torch.long)
                         for name, key in (("atoms", "atom_types"), ("charges", "charges"), ("bonds", "bond_types"))})
        settings = dict(epochs=1, batch_sizes=[2, 3], shuffle_seed=51, benchmark_updates=2,
                        benchmark_warmup=1, checkpoint_every_updates=1,
                        execution={"backend": "optimized", "workers": 0, "overlap": False},
                        classification_execution=None, proposal_config={"candidate_checks": 2},
                        step_control=None,
                        protocol=dict(prior_ratio=2, real_batch_size=3, microbatch_size=2,
                                      repair_weight=.5, repair_ramp_epochs=3, real_repair_weight=.25,
                                      prior_repair_weight=.75, occupancy_anchor_weight=1.,
                                      element_js_weight=.05, charge_js_weight=.05, vram_fraction_limit=.85,
                                      search={"beam_width": 1, "checks_per_depth": 2, "max_edits": 1}))
        bench, uninterrupted, segment, continued = (tmp_path / name for name in ("bench", "one", "seg", "cont"))
        for path in (bench, uninterrupted, segment, continued):
            path.mkdir()
        report = _benchmark(settings, "test-config", data, assets, bench, "cpu")
        assert report["status"] == "passed"
        assert all(arm["real_points"] == arm["fresh_prior_points"] > 0 for arm in report["arms"])
        full = _train(settings, "test-config", data, assets, uninterrupted, "cpu", 2, None, None, None, 1)
        first = _train(settings, "test-config", data, assets, segment, "cpu", 2, None, 1, None, 1)
        assert first["status"] == "paused" and first["row"] == 2
        paused = torch.load(first["latest"]["checkpoint"], weights_only=True)
        resumed = _train(settings, "test-config", data, assets, continued, "cpu", 2, paused, None, None, 1)
        assert full["status"] == resumed["status"] == "completed"
        assert full["real_points"] == resumed["real_points"] == len(data)
        direct, restarted = (torch.load(run["latest"]["checkpoint"], weights_only=True) for run in (full, resumed))
        assert direct["microbatch_size"] == restarted["microbatch_size"] == 1
        assert state_digest(direct["model_state"]) == state_digest(restarted["model_state"])
        assert torch.equal(direct["continuation"]["rng"]["loaders"]["fresh_prior"],
                           restarted["continuation"]["rng"]["loaders"]["fresh_prior"])
        assert all(state["step"].item() == 1 for state in payload["optimizer_state"]["state"].values())
        assert settings["protocol"]["prior_ratio"] == 2

        from latentmol.training.timing_study import run_timing_study

        timing_output = tmp_path / "timing"
        timing_output.mkdir()
        prior_only = dict(settings, prior_only={"real_per_prior": 2, "weight": .375},
                          batch_sizes=[2], microbatch_sizes=[2], parent_epoch=46)
        prior_assets = (parent, payload, None, None, None)
        run_timing_study(prior_only, "test-config", dict(batch_size=2, warmup=1, updates=1),
                         data, prior_assets, timing_output, "cpu")
        timing = json.loads((timing_output / "timing.json").read_bytes())
        assert timing["same_repair_trace"] and timing["same_final_weights"]
        assert len(timing["attempts"]) == 2
        assert "wall_seconds.repair_wait" in timing["attempts"][1]["summary"]
        prior_data = Rows(data[:4])
        partial, final = (tmp_path / name for name in ("prior-partial", "prior-final"))
        partial.mkdir()
        final.mkdir()
        phase = _train(prior_only, "test-config", prior_data, prior_assets, partial, "cpu", 2, None, 1, None)
        assert (phase["epoch"], phase["real_points"], phase["updates"]) == (46, 2, 9)
        saved = torch.load(phase["latest"]["checkpoint"], weights_only=True)
        phase = _train(prior_only, "test-config", prior_data, prior_assets, final, "cpu", 2, saved, None, None)
        assert (phase["epoch"], phase["real_points"], phase["updates"]) == (47, 4, 10)
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize("dropout", [None, .05])
def test_canonical_parent_is_cloned_without_changing_weights_optimizer_or_prior_rng(dropout):
    parent = GraphAutoencoder(GraphSpec(max_nodes=4),
        ModelConfig(hidden=8, latent=4, layers=1, encoder_pooling="canonical", global_hidden=12,
                    pair_hidden=4, triangle_hidden=2, decoder_variant="triangle_refine"))
    config = TrainConfig(mode="vae", device="cpu", model=parent.config)
    optimizer = torch.optim.AdamW(parent.parameters(), lr=config.learning_rate)
    for parameter in parent.parameters():
        parameter.grad = torch.full_like(parameter, .001)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    rng = capture_rng_state({"fresh_prior": torch.Generator().manual_seed(77)}, False)
    payload = dict(train_config=asdict(config), loss_config=asdict(config.loss),
                   optimizer_state=optimizer.state_dict(), continuation={"rng": rng})
    before = state_digest(parent.state_dict())
    model, restored, saved_config, generators = continuation._fresh(parent, payload, None, 240, "cpu", None,
                                                                     atom_refinement_dropout=dropout)
    assert parent.config.atom_refinement_dropout == 0 and model.config.atom_refinement_dropout == (dropout or 0.)
    assert model.config == saved_config.model
    assert torch.equal(torch.get_rng_state(), rng["torch"])
    z = torch.linspace(-1., 1., 12).reshape(3, 4)
    with torch.no_grad():
        original, updated = parent.eval().decode(z), model.eval().decode(z)
        for key in ("atoms", "charges", "bonds"):
            assert torch.equal(getattr(original, key), getattr(updated, key))
    assert state_digest(model.state_dict()) == before
    assert torch.equal(generators["fresh_prior"].get_state(), rng["loaders"]["fresh_prior"])
    for left, right in zip(model.parameters(), parent.parameters(), strict=True):
        assert left.data_ptr() != right.data_ptr()
    for key, state in optimizer.state_dict()["state"].items():
        for name, value in state.items():
            other = restored.state_dict()["state"][key][name]
            assert torch.equal(value, other)
            assert value.data_ptr() != other.data_ptr()
    with torch.no_grad():
        next(model.parameters()).add_(1)
    assert state_digest(parent.state_dict()) == before


def test_epoch46_sparse_recipe_builds_the_pooled_proposal_teacher():
    path = Path(__file__).resolve().parents[1] / "experiments/vae/epoch46-pubchem60m-prior15.json"
    config = continuation._read_config(path)
    with continuation._teacher(config, GraphSpec(), None) as teacher:
        assert teacher.workers == 12
        assert teacher.classification_execution == {"version": "repair-classification-pool-v1"}
        assert teacher.proposals.connect_components


def test_dropout_guided_updates_resume_with_the_same_weights_moments_and_rng(tmp_path):
    spec = GraphSpec(max_nodes=4)
    parent = GraphAutoencoder(spec, ModelConfig(hidden=8, latent=4, layers=1, encoder_pooling="canonical",
                               global_hidden=12, pair_hidden=4, triangle_hidden=2, decoder_variant="triangle_refine"))
    training = TrainConfig(mode="vae", device="cpu", model=parent.config)
    settings = continuation._read_config(Path(__file__).resolve().parents[1] /
                                         "experiments/vae/epoch47-pubchem60m-parent-guidance.json")
    reference = deepcopy(parent).eval().requires_grad_(False)
    initial_reference = state_digest(reference.state_dict())
    settings.pop("quality_panel")  # The fixed-panel test uses a separate authenticated split.
    settings.update(atom_refinement_dropout=.05, classification_execution=None, checkpoint_every_updates=1)
    settings["execution"]["workers"] = 0
    settings["prior_only"]["real_per_prior"] = 2
    rng = capture_rng_state({"fresh_prior": torch.Generator().manual_seed(77)}, False)
    payload = dict(train_config=asdict(training), loss_config=asdict(training.loss),
                   optimizer_state=torch.optim.AdamW(parent.parameters()).state_dict(),
                   continuation={"updates": 120, "rng": rng})
    data = Rows()
    for smiles in ("CC", "CO", "CN", "CCC"):
        graph = encode_smiles(smiles, spec)
        data.append({name: torch.as_tensor(graph[key], dtype=torch.long)
                     for name, key in (("atoms", "atom_types"), ("charges", "charges"), ("bonds", "bond_types"))})

    def segment(name, resume=None, limit=None):
        path = tmp_path / name
        path.mkdir()
        report = _train(settings, "fixture", data, (parent, payload, reference, None, None),
                        path, "cpu", 2, resume, limit, None, 1)
        return torch.load(report["latest"]["checkpoint"], weights_only=True)

    whole = segment("whole")
    partial = segment("partial", limit=1)
    final = segment("finish", resume=partial)
    assert (partial["continuation"]["updates"], final["continuation"]["updates"]) == (121, 122)
    assert final["model_config"]["atom_refinement_dropout"] == .05
    assert final["train_config"]["model"] == final["model_config"]
    assert state_digest(whole["model_state"]) == state_digest(final["model_state"])
    torch.testing.assert_close(whole["optimizer_state"], final["optimizer_state"], rtol=0, atol=0)
    assert torch.equal(whole["continuation"]["rng"]["torch"], final["continuation"]["rng"]["torch"])
    assert torch.equal(whole["continuation"]["rng"]["loaders"]["fresh_prior"],
                       final["continuation"]["rng"]["loaders"]["fresh_prior"])
    assert state_digest(reference.state_dict()) == initial_reference
    assert reference.config.atom_refinement_dropout == 0 and not reference.training
    metrics = [json.loads(line) for line in (tmp_path / "whole/training.jsonl").read_text().splitlines()]
    assert all(row["atom_refinement_dropout"] == .05 and row["pre_clip_gradient_norm"] > 0
               and row["prior_occupancy_bce"] > 0 and row["prior_guidance"] == "frozen-parent-occupancy-chemistry-v1"
               for row in metrics)


def test_v100_fallback_and_training_require_the_passing_shape_and_device(tmp_path, monkeypatch):
    tried = []

    def attempt(config, data, assets, output, device, order, size, microbatch, occupancy_digest):
        tried.append((size, microbatch))
        return dict(batch_size=size, microbatch_size=microbatch,
                    passed=(size, microbatch) != (256, 128), real_points_per_second=float(size))

    device = SimpleNamespace(name="Tesla V100-SXM2-32GB", total_memory=31 * 1024**3)
    monkeypatch.setattr(continuation, "_benchmark_arm", attempt)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: device.name)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: device)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setenv("LSB_DJOB_NUMPROC", "8")
    config = dict(batch_sizes=[128, 256], microbatch_sizes=[128, 64], shuffle_seed=77,
                  protocol={"microbatch_size": 128}, execution={"gpu_cpus": 8},
                  cuda_requirement=dict(name_contains="V100", minimum_gib=30, maximum_gib=34))
    continuation._check_cuda_requirement(config)
    data = Rows(["CCO"])
    report = _benchmark(config, "test-config", data, (None, None, torch.nn.Linear(1, 1)), tmp_path, "cuda")
    assert tried == [(128, 128), (256, 128), (256, 64)]
    assert _benchmark_shape(config, report, "test-config", data.manifest["dataset_sha256"], None, None) == (256, 64)
    with pytest.raises(ValueError, match="shape differ"):
        _benchmark_shape(config, report, "test-config", data.manifest["dataset_sha256"], 256, 128)
    report["device_memory_bytes"] = device.total_memory = 16 * 1024**3
    with pytest.raises(ValueError, match="device capacity"):
        _benchmark_shape(config, report, "test-config", data.manifest["dataset_sha256"], None, None)
    with pytest.raises(RuntimeError, match="Unexpected CUDA allocation"):
        continuation._check_cuda_requirement(config)


@pytest.mark.parametrize("benchmark_cpus,workers", [(16, 12), (10, 6)])
def test_v100_benchmark_can_gate_a100_training_without_changing_the_recipe(monkeypatch, benchmark_cpus, workers):
    requirement = dict(name_contains="V100", minimum_gib=30, maximum_gib=34)
    config = dict(protocol={"microbatch_size": 120},
                  execution={"gpu_cpus": 16, "workers": 12, "torch_threads": 4},
                  cuda_requirement=dict(name_contains="A100", minimum_gib=75, maximum_gib=82),
                  benchmark_cuda_requirement=requirement)
    device = SimpleNamespace(name="Tesla V100-PCIE-32GB", total_memory=32 * 1024**3)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: device)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    effective = continuation._benchmark_execution(config, benchmark=True, workers=workers, cpus=benchmark_cpus)
    monkeypatch.setenv("LSB_DJOB_NUMPROC", str(benchmark_cpus))
    continuation._check_cuda_requirement(effective, benchmark=True)
    assert effective["execution"]["workers"] == workers
    assert config["execution"] == {"gpu_cpus": 16, "workers": 12, "torch_threads": 4}
    with pytest.raises(RuntimeError, match="Unexpected CUDA allocation"):
        continuation._check_cuda_requirement(config)
    report = dict(status="passed", config_sha256="recipe", dataset_sha256="data",
                  device=device.name, device_memory_bytes=device.total_memory,
                  selected_batch_size=240, selected_microbatch_size=120,
                  arms=[dict(passed=True, batch_size=240, microbatch_size=120)])
    assert continuation._benchmark_shape(config, report, "recipe", "data", None, None) == (240, 120)
    device.name, device.total_memory = "NVIDIA A100", 80 * 1024**3
    monkeypatch.setenv("LSB_DJOB_NUMPROC", "16")
    continuation._check_cuda_requirement(config)


def test_benchmark_cpu_override_rejects_training_and_oversubscription():
    config = dict(execution=dict(workers=12, gpu_cpus=16, torch_threads=4))
    with pytest.raises(ValueError, match="benchmark mode"):
        continuation._benchmark_execution(config, benchmark=False, workers=6, cpus=10)
    with pytest.raises(ValueError, match="cover repair workers"):
        continuation._benchmark_execution(config, benchmark=True, workers=12, cpus=10)


def test_sparse_prior_only_update_searches_one_prior_for_fifteen_real(monkeypatch):
    from concurrent.futures import Future

    from latentmol.training import prior_repair
    from latentmol.training.timing_study import StepTimings

    spec = GraphSpec(max_nodes=4)
    model = GraphAutoencoder(spec, ModelConfig(hidden=8, latent=4, layers=1, encoder_pooling="canonical"))
    config = TrainConfig(mode="vae", device="cpu", model=model.config, batch_size=15)
    graph = encode_smiles("CCO", spec)
    hard = HardGraphBatch(*(torch.as_tensor(graph[key], dtype=torch.long).repeat(
        (15,) + (1,) * graph[key].ndim) for key in ("atom_types", "charges", "bond_types")))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    calls = []

    def search(decoded, *, candidate_ids, update):
        calls.append((len(decoded.atoms), list(candidate_ids), update))
        future = Future()
        future.set_result(SimpleNamespace(receipt=dict(attempted_candidates=1, status_counts={"accepted": 1},
            accepted_targets=1, elapsed_seconds=0.0)))
        return future

    monkeypatch.setattr(prior_repair, "repair_loss", lambda decoded, *_: decoded.atoms.square().mean())
    before = state_digest(model.state_dict())
    timings = StepTimings("cpu")
    result = prior_repair.prior_repair_update(model, optimizer, hard, config,
        SimpleNamespace(build_targets_async=search), torch.Generator().manual_seed(12),
        real_per_prior=15, prior_weight=.375, update=1, epoch=46, microbatch=8, step_control=None,
        timings=timings)
    assert calls == [(1, [15], 1)]
    assert result["real_points"] == 15 and result["fresh_prior_points"] == 1
    assert result["prior_repair"] > 0 and result["repair_targets"]["prior"] == 1
    assert state_digest(model.state_dict()) != before
    assert set(timings.read()["wall_seconds"]) == {
        "prepare_inputs", "prior_decode", "repair_submit", "real_forward_backward",
        "repair_wait", "prior_loss_backward", "adamw_and_checks",
    }
    assert timings.read()["cuda_stream_seconds"] == {}
