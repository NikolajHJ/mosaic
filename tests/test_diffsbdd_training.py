"""Grouping, exact teachers and fit isolation; run only on allocated HPC."""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from latentmol.chemistry import encode_smiles
from latentmol.data import sha256_file
from latentmol.models.refined_decoder import DecoderFeatures
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch
from latentmol.train import state_digest
from latentmol.diffsbdd import paired_data as pairs
from latentmol.diffsbdd import training
from latentmol.diffsbdd.bridge import NativeAtomBoundary


class TinyFrozenVAE(nn.Module):
    """Three distinct cyclic scaffold identities, not native model evidence."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.spec = GraphSpec(max_nodes=8)
        self.config = SimpleNamespace(latent=2)
        self.diffsbdd_checkpoint_identity = {"epoch": 39, "sha256": pairs.EPOCH39_SHA256}
        templates = [encode_smiles(value, self.spec) for value in ("C1CC1", "C1CCC1", "c1ccccc1")]
        self.register_buffer("template_atoms", torch.stack([torch.tensor(x["atom_types"]) for x in templates]).long())
        self.register_buffer("template_charges", torch.stack([torch.tensor(x["charges"]) for x in templates]).long())
        self.register_buffer("template_bonds", torch.stack([torch.tensor(x["bond_types"]) for x in templates]).long())
        self.eval()

    def encode(self, graph):
        counts = graph.occupancy.sum(-1).round().long()
        index = torch.where(counts == 3, 0, torch.where(counts == 4, 1, 2)).float()
        mean = torch.stack((index, torch.zeros_like(index)), -1)
        return mean, torch.full_like(mean, -20)

    def decode(self, z):
        indices = z[:, 0].round().long().clamp(0, 2)
        return DecoderOutput(torch.nn.functional.one_hot(self.template_atoms[indices], 10).float() * 40 - 20,
            torch.nn.functional.one_hot(self.template_charges[indices], 5).float() * 40 - 20,
            torch.nn.functional.one_hot(self.template_bonds[indices], 5).float() * 40 - 20)

    def decode_with_features(self, z):
        index = torch.arange(8, device=z.device, dtype=z.dtype)[None].expand(len(z), -1) / 8
        first, second = z[:, 0, None].expand(-1, 8), z[:, 1, None].expand(-1, 8)
        hidden = torch.stack((first, second, index, first * index), -1)
        return self.decode(z), DecoderFeatures(hidden, hidden, hidden, hidden, hidden)


def make_data(path, vae):
    path.mkdir()
    z = torch.tensor([[0.0, 0.1], [1.0, 0.1], [2.0, 0.1]])
    hard = vae.decode(z).harden(vae.spec)
    identities = ["C1CC1", "C1CCC1", "c1ccccc1"]
    payload = dict(z=z, atoms=hard.atoms.byte(), charges=hard.charges.byte(), bonds=hard.bonds.byte(),
        active=hard.atoms != 0, slot_to_canonical=torch.tensor([
            pairs.slot_to_canonical(HardGraphBatch(hard.atoms[i:i+1], hard.charges[i:i+1],
                                                  hard.bonds[i:i+1]), vae.spec, identities[i]) for i in range(3)]),
        source_identity=identities, source_scaffold=identities, identity=identities, scaffold=identities,
        root_index=[0, 1, 2], view=["posterior_mean"] * 3, seed=[17, 23, 29],
        group=["group0", "group1", "group2"], split=["train", "dev", "audit"])
    torch.save(payload, path / "samples.pt")
    for name in ("attempts.jsonl", "source-attempts.jsonl"):
        (path / name).write_text("{}\n", encoding="utf-8")
    manifest = dict(version=pairs.DATA_VERSION, status="prepared_not_fitted", accepted=3,
        vae_checkpoint=vae.diffsbdd_checkpoint_identity, spec_fingerprint=vae.spec.fingerprint(),
        vae_state_sha256_after=state_digest(vae.state_dict()),
        samples_sha256=sha256_file(path / "samples.pt"), attempts_sha256=sha256_file(path / "attempts.jsonl"),
        source_attempts_sha256=sha256_file(path / "source-attempts.jsonl"))
    pairs.write_json(path / "manifest.json", manifest)
    return path


def native_boundary():
    encoder = nn.Linear(10, 5).eval().requires_grad_(False)
    return NativeAtomBoundary(encoder, ("C", "N", "O", "S", "B", "Br", "Cl", "P", "I", "F"), 4, 0).eval()


def schedule(times, reference):
    # Deliberately nonzero time-zero noise tests actual native schedule use.
    return (1 - times / 2)[..., None], (0.01 + times)[..., None]


def test_transitive_source_decoded_scaffold_union_precedes_splits():
    roots = [dict(source_identity="A", source_scaffold="ringA", decoded=[dict(identity="B", scaffold="ringB")]),
             dict(source_identity="B", source_scaffold="ringB", decoded=[]),
             dict(source_identity="C", source_scaffold="ringB", decoded=[dict(identity="D", scaffold="ringD")]),
             dict(source_identity="D", source_scaffold="ringD", decoded=[]),
             dict(source_identity="E", source_scaffold="", decoded=[]),
             dict(source_identity="F", source_scaffold="", decoded=[])]
    assignments = pairs.partition_groups(roots, 17)
    assert assignments == pairs.partition_groups(roots, 17)
    assert len(set(assignments[:4])) == 1
    assert assignments[4] == assignments[5]  # Empty acyclic scaffold is not bypassed.
    assert assignments[0][0] != assignments[4][0]


def test_five_views_are_deterministic_and_local_rms_is_exact():
    mu, logvar = torch.zeros((1, 256)), torch.full((1, 256), -3.0)
    before = torch.random.get_rng_state().clone()
    first = pairs.latent_views(mu, logvar, "C1CC1", 17)
    second = pairs.latent_views(mu, logvar, "C1CC1", 17)
    assert len(first) == 5 and [name for name, _, _ in first] == list(pairs.VIEW_NAMES)
    for (name, seed, value), (other_name, other_seed, other_value) in zip(first, second):
        assert (name, seed) == (other_name, other_seed)
        assert torch.equal(value, other_value)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert not torch.equal(first[1][2], first[2][2])
    torch.testing.assert_close((first[3][2] - mu).square().mean().sqrt(), torch.tensor(0.02))
    torch.testing.assert_close((first[4][2] - mu).square().mean().sqrt(), torch.tensor(0.05))


def test_loaded_payload_rejects_scaffold_leakage_even_after_file_rebinding(tmp_path):
    vae = TinyFrozenVAE()
    path = make_data(tmp_path / "pairs", vae)
    payload = torch.load(path / "samples.pt", weights_only=True)
    payload["scaffold"][1] = payload["scaffold"][0]
    torch.save(payload, path / "samples.pt")
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["samples_sha256"] = sha256_file(path / "samples.pt")
    pairs.write_json(path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="leaks"):
        pairs.load_dataset(path, vae)


def test_compact_discrete_payload_restores_exact_categories_without_precision_loss(tmp_path):
    vae = TinyFrozenVAE()
    data, _, _ = pairs.load_dataset(make_data(tmp_path / "pairs", vae), vae)
    assert all(data[key].dtype == torch.uint8 for key in ("atoms", "charges", "bonds"))
    hard = vae.decode(data["z"]).harden(vae.spec)
    for key in ("atoms", "charges", "bonds"):
        assert torch.equal(data[key].long(), getattr(hard, key))


def test_training_updates_only_head_selects_dev_and_separates_exact_controls(tmp_path):
    vae, native = TinyFrozenVAE(), native_boundary()
    path = make_data(tmp_path / "pairs", vae)
    vae_before, native_before = state_digest(vae.state_dict()), state_digest(native.state_dict())
    result = training.train_bridge(vae, native, path, tmp_path / "fit", "cpu", schedule,
        steps=2, batch_size=1, evaluate_every=1, hidden_width=4, width=8,
        min_atom_agreement=0.0, max_embedding_nrmse=1e6, calibrate_runtime=False)
    assert result["completed_steps"] == 2
    assert result["audit_consumed"] is True
    assert result["status"] == "passed_feature_audit_not_native_validated"
    assert result["native_denoiser_validation"] == result["optimization"] == "not_run"
    assert set(result["audit"]) == {"learned", "exact_six_conditioned", "exact_native_full"}
    assert result["audit"]["exact_six_conditioned"]["atom_agreement"] == 1.0
    assert result["audit"]["learned"]["by_time"]["0"]["sigma_min"] > 0
    assert "S" in result["audit"]["learned"]["missing_elements"]
    assert state_digest(vae.state_dict()) == vae_before and state_digest(native.state_dict()) == native_before
    assert all(p.grad is None for p in (*vae.parameters(), *native.parameters()))
    saved = torch.load(result["checkpoint_path"], weights_only=True)
    assert saved["step"] == result["best_step"]
    assert saved["statistics"]["training_examples"] == 1
    assert not saved["training"]["desired_score_labels"]
    assert not any("native" in key or "vae" in key for key in saved["bridge_state"])


def test_statistics_cannot_read_audit_and_stored_targets_cannot_be_replaced(tmp_path):
    vae, native = TinyFrozenVAE(), native_boundary()
    path = make_data(tmp_path / "pairs", vae)
    data, _, _ = pairs.load_dataset(path, vae)
    with pytest.raises(ValueError, match="training-only"):
        training.training_statistics(vae, native, data, [2], schedule, "cpu")
    data["atoms"][0, 0] = vae.spec.atomic_numbers.index(7)
    with pytest.raises(ValueError, match="actual stored natural"):
        training._bundle(vae, data, [0], "cpu")


def test_pilot_cohort_rejects_charges_unsupported_and_changed_count():
    vae = TinyFrozenVAE()
    hard = vae.decode(torch.tensor([[0.0, 0.0]])).harden(vae.spec)
    assert pairs.assess_graph(hard, vae.spec, heavy_atoms=None)["eligible"]
    assert pairs.assess_graph(hard, vae.spec, heavy_atoms=25)["reason"] == "changed_or_ineligible_atom_count"
    encoded = encode_smiles("C[P+](C)(C)C", vae.spec)
    charged = HardGraphBatch(*(torch.tensor(encoded[key])[None].long() for key in ("atom_types", "charges", "bond_types")))
    assert pairs.assess_graph(charged, vae.spec, heavy_atoms=None)["reason"] == "charged_atom"


def test_wrong_checkpoint_or_unfrozen_vae_stops_before_pair_or_fit_work(tmp_path):
    vae = TinyFrozenVAE()
    vae.diffsbdd_checkpoint_identity = {"epoch": 40, "sha256": "0" * 64}
    with pytest.raises(ValueError, match="epoch-39"):
        pairs.prepare_dataset(vae, "unused", tmp_path / "wrong")
    vae.diffsbdd_checkpoint_identity = {"epoch": 39, "sha256": pairs.EPOCH39_SHA256}
    vae.train()
    with pytest.raises(ValueError, match="frozen"):
        pairs.prepare_dataset(vae, "unused", tmp_path / "trainmode")


def test_source_shortage_preserves_deficit_and_every_view_without_fitting(tmp_path, monkeypatch):
    vae = TinyFrozenVAE()

    class Corpus:
        spec = vae.spec
        manifest = {"dataset_sha256": "dataset", "split_sha256": "split"}
        smiles = ["C1CC1", "C1CCC1", "c1ccccc1"]
        _indices = None

        def __len__(self):
            return 3

        def __getitem__(self, index):
            return dict(atoms=vae.template_atoms[index], charges=vae.template_charges[index],
                        bonds=vae.template_bonds[index])

    monkeypatch.setattr(pairs, "GraphDataset", lambda *args: Corpus())
    monkeypatch.setattr(pairs, "reserved_identities", lambda data: set())
    vae.diffsbdd_training_data_identity = {"dataset_sha256": "dataset", "split_sha256": "split"}
    path = tmp_path / "shortage"
    with pytest.raises(ValueError, match="Insufficient eligible source"):
        pairs.prepare_dataset(vae, "authenticated-unused", path, requested_sources=4,
                              max_source_attempts=4, heavy_atoms=None)
    receipt = json.loads((path / "manifest.json").read_text())
    assert receipt["status"] == "failed_not_fitted"
    assert receipt["selected_sources"] == 3 and receipt["requested_sources"] == 4
    assert receipt["view_attempts"] == 15
    assert len((path / "attempts.jsonl").read_text().splitlines()) == 15
    assert not (path / "samples.pt").exists()


def test_pair_preparation_requires_actual_vae_training_lineage(tmp_path, monkeypatch):
    vae = TinyFrozenVAE()
    corpus = SimpleNamespace(spec=vae.spec, manifest={"dataset_sha256": "other", "split_sha256": "split"})
    monkeypatch.setattr(pairs, "GraphDataset", lambda *args: corpus)
    vae.diffsbdd_training_data_identity = {"dataset_sha256": "dataset", "split_sha256": "split"}
    with pytest.raises(ValueError, match="training dataset and split identity"):
        pairs.prepare_dataset(vae, "wrong-lineage", tmp_path / "wrong-corpus", requested_sources=1,
                              max_source_attempts=1)


def test_disposable_runtime_calibration_does_not_change_actual_fit_or_row_rng(tmp_path):
    vae, native = TinyFrozenVAE(), native_boundary()
    path = make_data(tmp_path / "pairs", vae)
    kwargs = dict(steps=2, batch_size=1, evaluate_every=1, hidden_width=4, width=8,
                  min_atom_agreement=0.0, max_embedding_nrmse=1e6)
    without = training.train_bridge(vae, native, path, tmp_path / "without", "cpu", schedule,
                                    calibrate_runtime=False, **kwargs)
    with_calibration = training.train_bridge(vae, native, path, tmp_path / "calibrated", "cpu", schedule,
                                             calibrate_runtime=True, **kwargs)
    preflight = with_calibration["runtime_preflight"]
    assert preflight["passed"] and preflight["disposable_updates"] == 3
    assert preflight["actual_fit_updates"] == 0 and preflight["audit_outputs_consumed"] is False
    assert preflight["temporary_head_and_moments_discarded"]
    assert preflight["frozen_and_actual_initial_state_unchanged"]
    assert preflight["actual_training_row_rng_unchanged"]
    assert with_calibration["completed_steps"] == without["completed_steps"] == 2
    baseline_history = [json.loads(row) for row in (tmp_path / "without/history.jsonl").read_text().splitlines()]
    calibrated_history = [json.loads(row) for row in (tmp_path / "calibrated/history.jsonl").read_text().splitlines()]
    assert baseline_history[0]["training_loss"] == calibrated_history[0]["training_loss"]
    saved_baseline = torch.load(without["checkpoint_path"], weights_only=True)
    saved_calibrated = torch.load(with_calibration["checkpoint_path"], weights_only=True)
    assert all(torch.equal(value, saved_calibrated["bridge_state"][key])
               for key, value in saved_baseline["bridge_state"].items())


def test_failed_measured_runtime_forecast_stops_before_actual_step_one(tmp_path, monkeypatch):
    vae, native = TinyFrozenVAE(), native_boundary()
    path = make_data(tmp_path / "pairs", vae)
    monkeypatch.setattr(training, "calibrate_fit_runtime", lambda *args, **kwargs:
        dict(passed=False, disposable_updates=3, actual_fit_updates=0,
             forecast_remaining_seconds=1e12, available_seconds=100,
             temporary_head_and_moments_discarded=True))
    output = tmp_path / "rejected"
    with pytest.raises(RuntimeError, match="actual fit not started"):
        training.train_bridge(vae, native, path, output, "cpu", schedule,
            steps=2, batch_size=1, evaluate_every=1, hidden_width=4, width=8,
            min_atom_agreement=0.0, max_embedding_nrmse=1e6, calibrate_runtime=True)
    result = json.loads((output / "fit-result.json").read_text())
    assert result["status"] == "failed_runtime_forecast_before_fit"
    assert result["completed_steps"] == 0 and result["audit_consumed"] is False
    assert not (output / "history.jsonl").exists() and not (output / "best-bridge.pt").exists()
