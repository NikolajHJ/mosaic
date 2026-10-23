"""Supervised bridge contracts on synthetic targets; run only on allocated HPC."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from latentmol.chemistry import encode_smiles
from latentmol.data import sha256_file
from latentmol.graph_policy import graph_from_decoder
from latentmol.models.refined_decoder import DecoderFeatures
from latentmol.representation import DecoderOutput, GraphSpec, HardGraphBatch
from latentmol.train import state_digest
from latentmol.boltz.embedding_bridge import NativeProjectionBank, ProjectionDescriptor
from latentmol.boltz import embedding_training as training


class TinyFrozenVAE(nn.Module):
    """Deterministic mock chemistry and attached features; no native evidence."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.spec = GraphSpec(max_nodes=6)
        self.config = SimpleNamespace(latent=2)
        self.embedding_bridge_checkpoint_identity = {"epoch": 36, "sha256": training.EPOCH36_SHA256}
        self.eval()

    def encode(self, graph):
        size = graph.occupancy.sum(-1)
        return torch.stack((size, torch.zeros_like(size)), -1), size.new_full((len(size), 2), -10.0)

    def decode(self, z):
        batch, nodes = len(z), self.spec.max_nodes
        size = z[:, 0].round().long().clamp(0, nodes)
        active = torch.arange(nodes, device=z.device)[None] < size[:, None]
        atoms = z.new_full((batch, nodes, len(self.spec.atomic_numbers)), -8)
        atoms[..., 0] = torch.where(active, -8.0, 8.0)
        atoms[..., 1] = torch.where(active, 8.0, -8.0)
        charges = z.new_full((batch, nodes, len(self.spec.formal_charges)), -8)
        charges[..., self.spec.formal_charges.index(0)] = 8
        bonds = z.new_full((batch, nodes, nodes, len(self.spec.bond_names)), -8)
        bonds[..., 0] = 8
        for left in range(nodes - 1):
            exists = active[:, left + 1]
            bonds[:, left, left + 1, 0] = bonds[:, left + 1, left, 0] = torch.where(exists, -8.0, 8.0)
            bonds[:, left, left + 1, 1] = bonds[:, left + 1, left, 1] = torch.where(exists, 8.0, -8.0)
        return DecoderOutput(atoms, charges, bonds)

    def decode_with_features(self, z):
        nodes = self.spec.max_nodes
        atom = z[:, None, :].expand(-1, nodes, -1)
        pair = torch.cat((atom[:, :, None, :] + atom[:, None, :, :],
                          (atom[:, :, None, :] - atom[:, None, :, :]).abs(),
                          atom[:, :, None, :] * atom[:, None, :, :]), -1)
        charge = torch.cat((atom, atom.square()), -1)
        return self.decode(z), DecoderFeatures(atom, pair[..., :2], atom, charge, pair)


def target_bank():
    generator = torch.Generator().manual_seed(71)
    descriptors = [ProjectionDescriptor("mock.element", "atom_element", 128, 2, "mock.element"),
                   ProjectionDescriptor("mock.charge", "atom_charge", 1, 2, "mock.charge"),
                   ProjectionDescriptor("mock.connectivity", "connectivity", 2, 2, "mock.bond"),
                   ProjectionDescriptor("mock.type", "type", 7, 2, "mock.type")]
    return NativeProjectionBank({d.key: torch.randn(d.rows, d.width, generator=generator) for d in descriptors}, descriptors)


def paired_dataset(path: Path, vae):
    path.mkdir()
    z = torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    hard = vae.decode(z).harden(vae.spec)
    payload = dict(z=z, atoms=hard.atoms.to(torch.uint8), charges=hard.charges.to(torch.uint8),
        bonds=hard.bonds.to(torch.uint8), active=hard.atoms != 0,
        slot_to_canonical=torch.tensor([[0, -1, -1, -1, -1, -1], [0, 1, -1, -1, -1, -1],
                                        [0, 1, 2, -1, -1, -1]], dtype=torch.int16),
        split=["train", "dev", "audit"], group=["root:1", "root:2", "root:3"],
        origin=["posterior_mean"] * 3, identity=["C", "CC", "CCC"])
    torch.save(payload, path / "samples.pt")
    (path / "attempts.jsonl").write_text("{}\n{}\n{}\n", encoding="utf-8")
    manifest = dict(version=training.DATA_VERSION, vae_checkpoint=vae.embedding_bridge_checkpoint_identity,
        spec_fingerprint=vae.spec.fingerprint(), vae_state_sha256_after=state_digest(vae.state_dict()),
        accepted=3, samples_sha256=sha256_file(path / "samples.pt"),
        attempts_sha256=sha256_file(path / "attempts.jsonl"))
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_roots_are_partitioned_before_descendants_without_overlap():
    identities = ["C", "CC", "CCC", "CCCC", "CCCCC", "CCCCCC"]
    counts = {"train": 2, "dev": 2, "audit": 2}
    first = training.partition_source_roots(identities, 137, counts)
    assert first == training.partition_source_roots(identities, 137, counts)
    assert set().union(*(set(rows) for rows in first.values())) == set(range(6))
    assert sum(len(rows) for rows in first.values()) == 6
    with pytest.raises(ValueError, match="unique"):
        training.partition_source_roots(["C"] * 6, 137, counts)


def test_conditioned_labels_match_existing_fixed_graph_policy():
    torch.manual_seed(83)
    spec = GraphSpec(max_nodes=6)
    decoded = DecoderOutput(torch.randn(1, 6, 10), torch.randn(1, 6, 5), torch.randn(1, 6, 6, 5))
    active = torch.tensor([True, False, True, True, False, False])
    fixed = graph_from_decoder(decoded, spec, fixed_active=active)
    hard = training.conditioned_hard(decoded, active[None], spec)
    assert torch.equal(hard.atoms, fixed.atoms.argmax(-1))
    assert torch.equal(hard.charges, fixed.charges.argmax(-1))
    assert torch.equal(hard.bonds, fixed.bonds.argmax(-1))
    assert not bool((hard.atoms[0, active] == 0).any())
    with pytest.raises(ValueError, match="nonempty"):
        training.conditioned_hard(decoded, torch.zeros_like(active)[None], spec)


def test_slot_alignment_preserves_noncontiguous_atom_charge_and_bond_identity():
    spec = GraphSpec(max_nodes=6)
    encoded = encode_smiles("C[NH3+]", spec)
    atoms = torch.zeros((1, 6), dtype=torch.long)
    charges = torch.full((1, 6), spec.formal_charges.index(0), dtype=torch.long)
    bonds = torch.zeros((1, 6, 6), dtype=torch.long)
    slots = [4, 1]
    for source, slot in enumerate(slots):
        atoms[0, slot], charges[0, slot] = int(encoded["atom_types"][source]), int(encoded["charges"][source])
        for other, other_slot in enumerate(slots):
            bonds[0, slot, other_slot] = int(encoded["bond_types"][source, other])
    mapping = training.slot_to_canonical_alignment(HardGraphBatch(atoms, charges, bonds), spec, encoded["smiles"])
    assert mapping[4] == 0 and mapping[1] == 1
    assert all(mapping[slot] == -1 for slot in (0, 2, 3, 5))
    with pytest.raises(ValueError, match="alignment"):
        training.slot_to_canonical_alignment(HardGraphBatch(atoms, charges, bonds), spec, "CC")


def test_training_updates_only_bridge_selects_dev_and_records_single_audit(tmp_path):
    vae, bank = TinyFrozenVAE(), target_bank()
    data = paired_dataset(tmp_path / "data", vae)
    before, bank_before = state_digest(vae.state_dict()), state_digest(bank.state_dict())
    result = training.train_bridge(vae, bank, data, tmp_path / "fit", "cpu",
        steps=3, batch_size=1, learning_rate=1e-2, seed=97, hidden_dim=8,
        evaluate_every=1, audit_thresholds={"loss": {"max": 1e6}})
    assert result["completed_steps"] == result["requested_steps"] == 3
    assert result["status"] == "passed_feature_audit_not_native_validated"
    assert result["audit_consumed"] is True
    assert result["native_validation"] == result["optimization"] == "not_run"
    assert result["audit"]["per_class_identity"]["mock.element"]["6"]["count"] == 3
    assert result["audit"]["per_class_identity"]["mock.element"]["16"]["count"] == 0
    assert result["audit"]["per_class_identity"]["mock.element"]["16"]["accuracy"] is None
    assert result["frozen_vae_and_bank_unchanged"] is True
    assert state_digest(vae.state_dict()) == before
    assert state_digest(bank.state_dict()) == bank_before
    assert all(parameter.grad is None for parameter in vae.parameters())
    saved = torch.load(result["checkpoint_path"], weights_only=True)
    assert saved["training"]["desired_score_labels"] is False
    assert saved["training"]["trainable_scope"] == "HiddenEmbeddingBridge parameters only"
    assert not any("vae" in key or "bank" in key for key in saved["bridge_state"])
    assert saved["step"] == result["best_step"]


def test_dataset_rejects_group_and_identity_leakage_even_when_bytes_rebound(tmp_path):
    vae = TinyFrozenVAE()
    data = paired_dataset(tmp_path / "data", vae)
    payload = torch.load(data / "samples.pt", weights_only=True)
    payload["identity"][1] = payload["identity"][0]
    torch.save(payload, data / "samples.pt")
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    manifest["samples_sha256"] = sha256_file(data / "samples.pt")
    (data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="leaks"):
        training._load_dataset(data, vae)


def test_training_rejects_stored_teacher_disagreement(tmp_path):
    vae, bank = TinyFrozenVAE(), target_bank()
    data = paired_dataset(tmp_path / "data", vae)
    payload, _, _ = training._load_dataset(data, vae)
    payload["atoms"][0, 0] = 2  # Native target nitrogen conflicts with decoded carbon.
    from latentmol.boltz.embedding_bridge import HiddenEmbeddingBridge
    bridge = HiddenEmbeddingBridge(bank, atom_input_dim=2, pair_input_dim=6, charge_input_dim=4, hidden_dim=8)
    with pytest.raises(ValueError, match="stored teacher"):
        training._forward_rows(vae, bank, bridge, payload, [0], "cpu")


def test_epoch36_binding_and_frozen_scope_are_mandatory(tmp_path):
    vae = TinyFrozenVAE()
    vae.embedding_bridge_checkpoint_identity = {"epoch": 35, "sha256": training.EPOCH36_SHA256}
    with pytest.raises(ValueError, match="epoch-36"):
        training.prepare_dataset(vae, "unused", tmp_path / "wrong", 17)
    vae.embedding_bridge_checkpoint_identity = {"epoch": 36, "sha256": training.EPOCH36_SHA256}
    vae.anchor.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        training.prepare_dataset(vae, "unused", tmp_path / "trainable", 17)


def test_expired_training_budget_returns_partial_without_fabricated_audit(tmp_path):
    vae, bank = TinyFrozenVAE(), target_bank()
    data = paired_dataset(tmp_path / "data", vae)
    result = training.train_bridge(vae, bank, data, tmp_path / "fit", "cpu",
        steps=3, batch_size=1, seed=97, hidden_dim=8, deadline=0.0)
    assert result["status"] == "partial_runtime_not_passed"
    assert result["completed_steps"] == 0
    assert result["audit_consumed"] is False
    assert result["frozen_vae_and_bank_unchanged"] is True
    assert not (tmp_path / "fit" / "best-bridge.pt").exists()


def test_dataset_preparation_preserves_attempts_masks_and_split_identity(tmp_path, monkeypatch):
    vae = TinyFrozenVAE()
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text("{}", encoding="utf-8")

    class FakeDataset:
        cache_dir = source
        spec = vae.spec
        manifest = {"dataset_sha256": "a" * 64, "split_sha256": "b" * 64}
        smiles = ["C" * size for size in range(1, 7)]

        def __init__(self, path, split):
            assert split == "train"

        def __getitem__(self, index):
            hard = vae.decode(torch.tensor([[float(index + 1), 0.0]])).harden(vae.spec)
            return {name: getattr(hard, name)[0] for name in ("atoms", "charges", "bonds")}

    monkeypatch.setattr(training, "GraphDataset", FakeDataset)
    # No val/test graph is passed through the VAE. This metadata-only exclusion
    # deliberately catches one generated molecule to test preserved rejections.
    monkeypatch.setattr(training, "_source_holdout_identities", lambda data: {"CCCCCC"})
    manifest = training.prepare_dataset(vae, source, tmp_path / "paired", 137,
        train_count=2, dev_count=2, audit_count=2, prior_attempts=2, local_scale=1e-4)
    attempts = [json.loads(line) for line in (tmp_path / "paired" / "attempts.jsonl").read_text().splitlines()]
    assert len(attempts) == manifest["attempts"]
    assert sum(row["accepted"] for row in attempts) == manifest["accepted"]
    assert manifest["rejected"]["source_heldout_identity"] >= 3
    assert manifest["vae_state_sha256_before"] == manifest["vae_state_sha256_after"]
    payload = torch.load(tmp_path / "paired" / "samples.pt", weights_only=True)
    assert payload["z"].shape[1] == 2
    assert set(payload) == {"z", "atoms", "charges", "bonds", "active", "slot_to_canonical",
                            "split", "group", "origin", "identity"}
    owners, groups = {}, {}
    for split, group, identity in zip(payload["split"], payload["group"], payload["identity"]):
        assert owners.setdefault(identity, split) == split
        assert groups.setdefault(group, split) == split
    assert "CCCCCC" not in payload["identity"]
