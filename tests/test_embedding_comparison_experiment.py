"""Allocated-HPC runner fixtures; no live scheduler or external models."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from latentmol.boltz import embedding_comparison_experiment as experiment
from latentmol.boltz import active_set, provenance
from latentmol.boltz.embedding_bridge import HiddenEmbeddingBridge
from latentmol.experiments import file_sha256
from latentmol.representation import DecoderOutput, GraphSpec
from latentmol.train import state_digest
from test_embedding_validation import frozen_models


class SavedStartVAE(nn.Module):
    """A frozen lookup decoder makes source-row selection exactly observable."""

    def __init__(self, atoms, charges, bonds):
        super().__init__()
        self.register_buffer("identity", torch.tensor([36.]))
        self.spec = GraphSpec(max_nodes=4)
        self.config = SimpleNamespace(latent=2)
        self.embedding_bridge_checkpoint_identity = {"epoch": 36, "sha256": "a" * 64}
        self.atoms, self.charges, self.bonds = atoms, charges, bonds
        self.decoded_rows = []
        self.eval().requires_grad_(False)

    def decode(self, z):
        rows = z[:, 0].long()
        self.decoded_rows.extend(rows.tolist())
        return DecoderOutput(F.one_hot(self.atoms[rows], len(self.spec.atomic_numbers)).float() * 40,
            F.one_hot(self.charges[rows], len(self.spec.formal_charges)).float() * 40,
            F.one_hot(self.bonds[rows], len(self.spec.bond_names)).float() * 40)


def saved_start_fixture(tmp_path, monkeypatch):
    count = 9
    atoms = torch.zeros(count, 4, dtype=torch.long)
    atoms[:, 0] = 1
    atoms[:, 2] = torch.tensor([1, 3, 3, 2, 3, 1, 1, 2, 3])
    atoms[7, 0] = 2
    atoms[8, 0] = 3
    charges = torch.full((count, 4), 2, dtype=torch.long)
    bonds = torch.zeros(count, 4, 4, dtype=torch.long)
    bonds[:, 0, 2] = bonds[:, 2, 0] = 1
    vae = SavedStartVAE(atoms, charges, bonds)
    samples = dict(z=torch.tensor([[float(i), -float(i)] for i in range(count)]),
        active=atoms != 0, atoms=atoms, charges=charges, bonds=bonds,
        split=["train", "audit", "audit", "audit", "audit", "audit", "dev", "audit", "audit"],
        origin=["posterior_mean", "posterior_sample", *["posterior_mean"] * 7],
        group=["train", "a", "a", "a", "b", "b", "dev", "c", "d"],
        identity=["CC", "CO", "CO", "CN", "CO", "CC", "CC", "NN", "OO"])
    samples_path = tmp_path / "samples.pt"
    manifest_path = tmp_path / "data-manifest.json"
    context_path = tmp_path / "contexts.json"
    template_path = tmp_path / "count-2.pt"
    template_path.write_bytes(b"fixture count context")
    header = dict(target_sequence_sha256="b" * 64, msa_sha256="c" * 64,
                  fixed_msa_row_indices=[0, 2])
    context_path.write_text(json.dumps({**header, "contexts": {"2": {
        "path": template_path.name, "sha256": file_sha256(template_path)}}}))

    def save_samples():
        torch.save(samples, samples_path)
        manifest_path.write_text(json.dumps(dict(samples_sha256=file_sha256(samples_path),
            vae_checkpoint=vae.embedding_bridge_checkpoint_identity,
            vae_state_sha256_after=state_digest(vae.state_dict()))))

    save_samples()
    config = dict(start_count=3, start_selection="first3distinct-audit-posterior-mean",
        start_data_manifest={"path": str(manifest_path), "sha256": file_sha256(manifest_path)},
        start_samples={"path": str(samples_path), "sha256": file_sha256(samples_path)},
        context_manifest={"path": str(context_path), "sha256": file_sha256(context_path)})
    monkeypatch.setattr(experiment, "load_vae", lambda *_: (vae, {}))
    bank_calls = []

    class FixtureBank:
        def __init__(self, context_config, spec):
            bank_calls.append((context_config, spec))
            self.templates = {2: "fixture-template"}

    monkeypatch.setattr(active_set, "ContextBank", FixtureBank)
    monkeypatch.setattr(active_set, "bind_slots", lambda template, active:
        SimpleNamespace(bound_slots=active.nonzero().flatten().tolist()))
    monkeypatch.setattr(provenance, "context_digest", lambda bound:
        "bound-" + "-".join(map(str, bound.bound_slots)))
    output = tmp_path / "prepared"
    output.mkdir()

    def rebind():
        save_samples()
        config["start_samples"]["sha256"] = file_sha256(samples_path)
        config["start_data_manifest"]["sha256"] = file_sha256(manifest_path)

    return SimpleNamespace(vae=vae, samples=samples, config=config, output=output,
        bank_calls=bank_calls, header=header, rebind=rebind)


def test_starts_are_first_three_distinct_groups_and_identities_in_saved_order(tmp_path, monkeypatch):
    fixture = saved_start_fixture(tmp_path, monkeypatch)
    report = experiment.prepare_starts(fixture.config, fixture.output)
    assert [row["sample_index"] for row in report["cases"]] == [2, 5, 7]
    assert [row["group"] for row in report["cases"]] == ["a", "b", "c"]
    assert fixture.vae.decoded_rows == [2, 5, 7]
    assert fixture.bank_calls[0][0].context_identity == fixture.header
    assert set(fixture.bank_calls[0][0].contexts) == {2}
    saved = torch.load(fixture.output / "starts.pt", weights_only=True)
    for row, source_index in zip(saved["cases"], (2, 5, 7), strict=True):
        assert torch.equal(row["z"], fixture.samples["z"][source_index:source_index + 1])
        assert torch.equal(row["active"], fixture.samples["active"][source_index])
    assert report["starts_sha256"] == file_sha256(fixture.output / "starts.pt")
    assert report["frozen_vae_state_sha256"] == state_digest(fixture.vae.state_dict())


def test_failed_selected_teacher_is_preserved_without_later_replacement(tmp_path, monkeypatch):
    fixture = saved_start_fixture(tmp_path, monkeypatch)
    # Change the authenticated stored teacher only; the lookup decoder retains
    # the original row.  A later convenient valid row may not replace it.
    fixture.samples["atoms"] = fixture.samples["atoms"].clone()
    fixture.samples["atoms"][2, 2] = 2
    fixture.rebind()
    with pytest.raises(ValueError, match="did not reproduce"):
        experiment.prepare_starts(fixture.config, fixture.output)
    assert fixture.vae.decoded_rows == [2]
    assert not (fixture.output / "starts.pt").exists()


def test_insufficient_distinct_audit_starts_cannot_fall_back_to_train_or_samples(tmp_path, monkeypatch):
    fixture = saved_start_fixture(tmp_path, monkeypatch)
    fixture.samples["identity"] = ["same"] * len(fixture.samples["identity"])
    fixture.rebind()
    with pytest.raises(ValueError, match="Insufficient distinct prescribed"):
        experiment.prepare_starts(fixture.config, fixture.output)
    assert fixture.vae.decoded_rows == [] and fixture.bank_calls == []


@pytest.mark.parametrize("cpu_status,skipped", [("failed_not_passed", 0), ("passed", 1)])
def test_cpu_failure_or_skip_never_releases_gpu_models(tmp_path, monkeypatch, cpu_status, skipped):
    cpu = tmp_path / "cpu"
    cpu.mkdir()
    source, plan = "d" * 64, "e" * 64
    (cpu / "result.json").write_text(json.dumps(dict(status=cpu_status,
        source_sha256=source, plan_sha256=plan,
        junit={"tests": 10, "failures": 0, "errors": 0, "skipped": skipped})))
    monkeypatch.setenv("LATENTMOL_SOURCE_SHA256", source)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("CPU prerequisite failed before any GPU/model work")

    monkeypatch.setattr(experiment, "load_vae", forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    with pytest.raises(ValueError, match="exact-release CPU pass"):
        experiment.gpu_stage({"_plan_sha256": plan}, tmp_path / "gpu", deadline=99999)
    assert not calls


@pytest.mark.parametrize("tamper", [None, "vae_identity", "bank_tensor"])
def test_loaded_mapping_strictly_binds_original_vae_and_native_bank(tmp_path, monkeypatch, tamper):
    _, _, bank, vae = frozen_models()
    vae.embedding_bridge_checkpoint_identity = {"epoch": 36, "sha256": "f" * 64}
    z = torch.zeros(1, vae.config.latent)
    _, hidden = vae.decode_with_features(z)
    bridge = HiddenEmbeddingBridge(bank, atom_input_dim=hidden.atom_hidden.shape[-1],
        pair_input_dim=hidden.bond_input.shape[-1], charge_input_dim=hidden.charge_input.shape[-1],
        hidden_dim=8).eval().requires_grad_(False)
    saved = dict(vae_checkpoint=dict(vae.embedding_bridge_checkpoint_identity),
        vae_state_sha256=state_digest(vae.state_dict()), bank_metadata=bank.metadata(),
        bank_state={key: value.clone() for key, value in bank.state_dict().items()},
        bridge_configuration=bridge.configuration(),
        bridge_state={key: value.clone() for key, value in bridge.state_dict().items()})
    if tamper == "vae_identity":
        saved["vae_checkpoint"]["epoch"] = 37
    elif tamper == "bank_tensor":
        next(iter(saved["bank_state"].values())).add_(.1)
    monkeypatch.setattr(experiment, "verified", lambda value: Path(value["path"]))
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: saved)
    config = {"bridge": {"path": str(tmp_path / "bridge.pt"), "sha256": experiment.BRIDGE_SHA}}
    if tamper:
        with pytest.raises(ValueError, match="same original VAE/native|bank changed"):
            experiment.load_bridge(config, vae, bank, "cpu")
    else:
        loaded = experiment.load_bridge(config, vae, bank, "cpu")
        assert not loaded.training and all(not p.requires_grad for p in loaded.parameters())
        assert all(torch.equal(value, saved["bridge_state"][key]) for key, value in loaded.state_dict().items())
