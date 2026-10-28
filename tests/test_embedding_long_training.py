"""Long-fit numerical/protocol fixtures; run only on allocated HPC."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from latentmol.boltz import embedding_long_training as long
from latentmol.boltz.embedding_bridge import (BridgeTargets, HiddenEmbeddingBridge,
                                             NativeProjectionBank, ProjectionDescriptor)
from latentmol.boltz.embedding_training import DATA_VERSION, EPOCH36_SHA256, TRAINING_VERSION
from latentmol.data import sha256_file
from latentmol.models.refined_decoder import DecoderFeatures
from latentmol.representation import DecoderOutput, GraphSpec
from latentmol.train import state_digest


class FrozenVAE(nn.Module):
    """Synthetic carbon chains; no native-model or chemical efficacy evidence."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.spec, self.config = GraphSpec(max_nodes=12), SimpleNamespace(latent=2)
        self.embedding_bridge_checkpoint_identity = dict(epoch=36, sha256=EPOCH36_SHA256)
        self.eval()

    def encode(self, graph):
        size = graph.occupancy.sum(-1)
        return torch.stack((size, torch.zeros_like(size)), -1), size.new_full((len(size), 2), -20)

    def decode(self, z):
        n, b = self.spec.max_nodes, len(z)
        active = torch.arange(n, device=z.device)[None] < z[:, :1].round().long().clamp(0, n)
        atoms = z.new_full((b, n, len(self.spec.atomic_numbers)), -8)
        atoms[..., 0] = torch.where(active, -8.0, 8.0)
        atoms[..., 1] = torch.where(active, 8.0, -8.0)
        charge = z.new_full((b, n, len(self.spec.formal_charges)), -8)
        charge[..., self.spec.formal_charges.index(0)] = 8
        bonds = z.new_full((b, n, n, len(self.spec.bond_names)), -8)
        bonds[..., 0] = 8
        for i in range(n - 1):
            exists = active[:, i + 1]
            bonds[:, i, i + 1, 0] = bonds[:, i + 1, i, 0] = torch.where(exists, -8.0, 8.0)
            bonds[:, i, i + 1, 1] = bonds[:, i + 1, i, 1] = torch.where(exists, 8.0, -8.0)
        return DecoderOutput(atoms, charge, bonds)

    def decode_with_features(self, z):
        a = z[:, None, :].expand(-1, self.spec.max_nodes, -1)
        pair = torch.cat((a[:, :, None] + a[:, None, :],
                          (a[:, :, None] - a[:, None, :]).abs(),
                          a[:, :, None] * a[:, None, :]), -1)
        charge = torch.cat((a, a.square()), -1)
        return self.decode(z), DecoderFeatures(a, pair[..., :2], a, charge, pair)


def bank():
    descriptors = [ProjectionDescriptor("mock.element", "atom_element", 128, 2, "mock.element"),
                   ProjectionDescriptor("mock.charge", "atom_charge", 1, 2, "mock.charge"),
                   ProjectionDescriptor("mock.connectivity", "connectivity", 2, 2, "mock.bond"),
                   ProjectionDescriptor("mock.type", "type", 7, 2, "mock.type")]
    g = torch.Generator().manual_seed(821)
    return NativeProjectionBank({d.key: torch.randn(d.rows, d.width, generator=g) for d in descriptors},
                                descriptors).eval().requires_grad_(False)


def paired(path, vae, sizes, splits, groups):
    path.mkdir()
    z = torch.tensor([[float(size), 0.0] for size in sizes])
    hard = vae.decode(z).harden(vae.spec)
    active = hard.atoms != 0
    slots = torch.arange(vae.spec.max_nodes)[None].expand(len(sizes), -1)
    payload = dict(z=z, atoms=hard.atoms.to(torch.uint8), charges=hard.charges.to(torch.uint8),
        bonds=hard.bonds.to(torch.uint8), active=active,
        slot_to_canonical=slots.masked_fill(~active, -1).to(torch.int16),
        split=splits, group=groups, origin=["posterior_mean"] * len(sizes),
        identity=["C" * size for size in sizes])
    torch.save(payload, path / "samples.pt")
    (path / "attempts.jsonl").write_text("{}\n" * len(sizes), encoding="utf-8")
    manifest = dict(version=DATA_VERSION, vae_checkpoint=vae.embedding_bridge_checkpoint_identity,
        spec_fingerprint=vae.spec.fingerprint(), vae_state_sha256_after=state_digest(vae.state_dict()),
        source_dataset_sha256="a" * 64, source_split_sha256="b" * 64,
        selected_source_roots={"train": [0], "dev": [1], "audit": [2, 3, 4]},
        accepted=len(sizes), samples_sha256=sha256_file(path / "samples.pt"),
        attempts_sha256=sha256_file(path / "attempts.jsonl"))
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return payload, manifest


def fixture(tmp_path):
    vae, native = FrozenVAE(), bank()
    old, old_manifest = paired(tmp_path / "old", vae, [1, 2, 3, 4, 5],
        ["train", "dev", "audit", "audit", "audit"], ["old:1", "old:2", "old:3", "old:4", "old:5"])
    prior = dict(path=str(tmp_path / "old/manifest.json"), sha256=sha256_file(tmp_path / "old/manifest.json"))
    cases = [dict(record=dict(case_id=f"case-{i - 1:03d}", group=old["group"][i], identity=old["identity"][i]),
                  z=old["z"][i:i + 1].clone(), active=old["active"][i].clone()) for i in (2, 3, 4)]
    starts_path = tmp_path / "starts.pt"
    torch.save(dict(cases=cases, graph_spec=vae.spec.to_dict()), starts_path)
    starts = dict(path=str(starts_path), sha256=sha256_file(starts_path))
    _, manifest = paired(tmp_path / "new", vae, [6, 7, 8, 9, 10, 11],
        ["train", "train", "dev", "dev", "audit", "audit"], ["new:" + str(i) for i in range(6)])
    manifest["selected_source_roots"] = {"train": [5, 6], "dev": [7, 8], "audit": [9, 10]}
    manifest["long_data"] = dict(version=long.DATA_EXTENSION, prior_dataset=prior,
        prior_manifest_sha256=prior["sha256"], benchmark_starts=starts,
        excluded_source_roots=[0, 1, 2, 3, 4], excluded_groups=5, excluded_identities=5, old_train_replay_rows=0)
    (tmp_path / "new/manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    bridge = HiddenEmbeddingBridge(native, atom_input_dim=2, pair_input_dim=6, charge_input_dim=4, hidden_dim=8)
    saved = dict(version=TRAINING_VERSION, vae_checkpoint=vae.embedding_bridge_checkpoint_identity,
        vae_state_sha256=state_digest(vae.state_dict()), dataset_manifest_sha256=prior["sha256"],
        bank_metadata=native.metadata(), bank_state=native.state_dict(),
        bridge_configuration=bridge.configuration(), bridge_state=bridge.state_dict())
    warm_path = tmp_path / "old-bridge.pt"
    torch.save(saved, warm_path)
    warm = dict(path=str(warm_path), sha256=sha256_file(warm_path))
    return SimpleNamespace(vae=vae, bank=native, prior=prior, starts=starts, warm=warm,
                           dataset=tmp_path / "new", old_manifest=old_manifest)


def test_dev_q99_is_nearest_rank_with_fixed_numerical_floor():
    assert long.calibrate_caps({"p/mse": list(range(100)), "q/mse": [0] * 100}) == {
        "p/mse": 98, "q/mse": 1e-8}


@pytest.mark.parametrize("kwargs", [{"split": "audit"}, {"split": "train"},
                                     {"quantile": .95}, {"floor": 1e-6}])
def test_calibration_rejects_other_data_and_policy(kwargs):
    with pytest.raises(ValueError, match="fixed dev-only"):
        long.calibrate_caps({"p/mse": [1.0]}, **kwargs)


@pytest.mark.parametrize("errors", [{}, {"p/mse": []}, {"p/raw": [1.0]},
    {"p/mse": [float("nan")]}, {"p/mse": [-1.0]}, {"p/mse": [1.0], "q/mse": [1.0, 2.0]}])
def test_calibration_fails_closed_on_incomplete_or_nonfinite_errors(errors):
    with pytest.raises(ValueError):
        long.calibrate_caps(errors)


def test_coverage_reports_joint_failures_without_adjusting_caps():
    caps = {"a/mse": 1.0, "b/mse": 2.0}
    before = dict(caps)
    result = long.cap_coverage({"a/mse": [0.5, 1.1, .5], "b/mse": [1, 1, 3]}, caps)
    assert result["all_heads_within"] == 1
    assert result["per_projection"]["a/mse"]["within"] == 2
    assert caps == before and result["caps_adjusted"] is False


def test_per_example_error_balances_none_and_present_like_guard_bank_loss(monkeypatch):
    native = NativeProjectionBank({"p": torch.tensor([[0.0], [1.0]])},
        [ProjectionDescriptor("p", "connectivity", 2, 1, "p")])
    atom_mask = torch.ones(2, 3, dtype=torch.bool)
    pair_mask = torch.ones(2, 3, 3, dtype=torch.bool)
    present = torch.zeros_like(pair_mask)
    present[0, 0, 1] = present[0, 1, 0] = True
    target = BridgeTargets({"p": torch.zeros(2, 3, 3, 1)}, atom_mask, pair_mask, present)
    prediction = torch.ones(2, 3, 3, 1)
    prediction[0, 0, 1] = prediction[0, 1, 0] = 3
    prediction[1] = 2
    monkeypatch.setattr(long, "_forward_rows", lambda *args: (None, target, {"p": prediction}))
    result = long.per_example_errors(None, native, nn.Identity(), {}, [0, 1], "cpu", 2)
    assert result == {"p/mse": [5.0, 4.0]}
    assert result["p/mse"][0] != pytest.approx(float(prediction[0].square().mean()))


@pytest.mark.parametrize("field", ["group", "identity", "selected_source_roots"])
def test_rebound_new_payload_still_cannot_leak_historical_data(tmp_path, field):
    f = fixture(tmp_path)
    path = f.dataset / "manifest.json"
    manifest = json.loads(path.read_text())
    if field == "selected_source_roots":
        manifest[field]["train"][0] = 0
    else:
        payload = torch.load(f.dataset / "samples.pt", weights_only=True)
        payload[field][0] = "old:1" if field == "group" else "C"
        torch.save(payload, f.dataset / "samples.pt")
        manifest["samples_sha256"] = sha256_file(f.dataset / "samples.pt")
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="leaks historical"):
        long._load_long_dataset(f.dataset, f.vae)


def test_benchmark_latent_changes_are_rejected_even_with_updated_file_sha(tmp_path):
    f = fixture(tmp_path)
    starts = torch.load(f.starts["path"], weights_only=True)
    starts["cases"][0]["z"][0, 1] += .25
    torch.save(starts, f.starts["path"])
    f.starts["sha256"] = sha256_file(f.starts["path"])
    with pytest.raises(ValueError, match="Benchmark start differs"):
        long._excluded(f.vae, f.prior, f.starts)


@pytest.mark.parametrize("change", ["bank", "bank_dtype", "vae", "dataset", "version"])
def test_warm_start_requires_exact_original_vae_dataset_and_projection_bytes(tmp_path, change):
    f = fixture(tmp_path)
    saved = torch.load(f.warm["path"], weights_only=True)
    if change == "bank":
        saved["bank_state"]["projection_000"][0, 0] += 1
    elif change == "bank_dtype":
        saved["bank_state"]["projection_000"] = saved["bank_state"]["projection_000"].double()
    elif change == "vae":
        saved["vae_state_sha256"] = "f" * 64
    elif change == "dataset":
        saved["dataset_manifest_sha256"] = "f" * 64
    else:
        saved["version"] = "other"
    torch.save(saved, f.warm["path"])
    f.warm["sha256"] = sha256_file(f.warm["path"])
    with pytest.raises(ValueError, match="Warm-start"):
        long._warm_bridge(f.vae, f.bank, f.warm, "cpu", f.prior["sha256"])


def test_asset_checks_real_bytes_before_loading(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text("{}", encoding="utf-8")
    asset = dict(path=str(path), sha256=sha256_file(path))
    assert long._asset(asset) == path
    path.write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        long._asset(asset)


def test_preparation_excludes_old_roots_and_identities_before_new_split(tmp_path, monkeypatch):
    f = fixture(tmp_path)
    cache = tmp_path / "source"
    cache.mkdir()
    (cache / "manifest.json").write_text("{}", encoding="utf-8")

    class Source:
        spec, cache_dir = f.vae.spec, cache
        manifest = dict(dataset_sha256="a" * 64, split_sha256="b" * 64)
        smiles = ["C" * i for i in range(1, 13)]

        def __getitem__(self, index):
            hard = f.vae.decode(torch.tensor([[float(index + 1), 0]])).harden(self.spec)
            return {k: getattr(hard, k)[0] for k in ("atoms", "charges", "bonds")}

    monkeypatch.setattr(long, "GraphDataset", lambda *args: Source())
    monkeypatch.setattr(long, "_source_holdout_identities", lambda *args: set())
    result = long.prepare_long_dataset(f.vae, cache, tmp_path / "prepared", 813,
        prior_dataset=f.prior, benchmark_starts=f.starts, train_count=2, dev_count=1,
        audit_count=1, prior_attempts=0)
    assert all(i >= 5 for rows in result["selected_source_roots"].values() for i in rows)
    assert result["attempts"] == 12 and result["accepted"] == 12
    data, _, _ = long._load_long_dataset(tmp_path / "prepared", f.vae)
    assert not set(data["identity"]) & {"C" * i for i in range(1, 6)}
    assert result["long_data"]["old_train_replay_rows"] == 0


@pytest.mark.parametrize("improves", [True, False])
def test_fit_dev_selection_calibration_order_and_original_weights_unchanged(tmp_path, monkeypatch, improves):
    f = fixture(tmp_path)
    data, _, _ = long._load_long_dataset(f.dataset, f.vae)
    train_rows = [i for i, split in enumerate(data["split"]) if split == "train"]
    audit_rows = [i for i, split in enumerate(data["split"]) if split == "audit"]
    out = tmp_path / "fit"
    evaluations = []

    def evaluate(vae, native, bridge, payload, rows, device, batch_size, deadline):
        assert rows != train_rows
        if rows == audit_rows:
            assert (out / "calibration.json").is_file()
            value = -999.0  # Could never influence checkpoint selection.
        else:
            value = 10.0 - len(evaluations) if improves else 10.0
        evaluations.append(list(rows))
        return dict(batch_weighted_metrics={"loss": value}, examples=len(rows))

    monkeypatch.setattr(long, "_evaluate", evaluate)
    before, native_before = state_digest(f.vae.state_dict()), state_digest(f.bank.state_dict())
    warm_sha, rng = sha256_file(f.warm["path"]), torch.get_rng_state().clone()
    result = long.fit_long_bridge(f.vae, f.bank, f.dataset, f.warm, out, "cpu",
        bindings=dict(source_sha256="c" * 64, plan_sha256="d" * 64), steps=2,
        batch_size=2, evaluate_every=1, learning_rate=.001)
    assert result["completed_steps"] == 2
    assert state_digest(f.vae.state_dict()) == before
    assert state_digest(f.bank.state_dict()) == native_before
    assert sha256_file(f.warm["path"]) == warm_sha
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(p.grad is None for p in f.vae.parameters())
    if improves:
        assert result["status"] == "completed_long_bridge_fit_not_native_validated"
        assert result["best_step"] == 2 and result["best_dev_loss"] == 8
        saved = torch.load(out / "best-bridge.pt", weights_only=True)
        calibration = json.loads((out / "calibration.json").read_text())
        assert saved["step"] == 2 and saved["dev"]["batch_weighted_metrics"]["loss"] == 8
        assert calibration["split"] == "dev" and calibration["audit_used_for_calibration"] is False
        assert calibration["caps"] == long.calibrate_caps(calibration["dev_errors"])
        assert calibration["bridge_sha256"] == result["checkpoint_sha256"]
        assert result["audit_consumed"] is True and evaluations.count(audit_rows) == 1
        # Exercise the production loader against real saved tensor/JSON bytes,
        # rather than maintaining an independently guessed checkpoint schema.
        from latentmol.boltz.embedding_long_experiment import asset, load_long_bridge
        fitted = dict(status="passed", fit_completed=True, calibration_passed=True, fit=result,
            model_identity=dict(checkpoint_sha256=result["checkpoint_sha256"],
                                calibration_sha256=result["calibration_sha256"]),
            artifacts=dict(checkpoint=asset(out / "best-bridge.pt"), calibration=asset(out / "calibration.json"),
                training_plan=asset(out / "training-plan.json"),
                paired_data_manifest=asset(f.dataset / "manifest.json"), audit=asset(out / "audit.json")))
        config = dict(bridge=f.warm, start_data_manifest=f.prior,
                      precision_correction=dict(parent_starts=f.starts))
        loaded, caps = load_long_bridge(config, f.vae, f.bank, fitted, "c" * 64, "d" * 64, "cpu")
        assert not loaded.training and all(not p.requires_grad for p in loaded.parameters())
        assert caps == {key.removesuffix("/mse"): value for key, value in calibration["caps"].items()}
        # Rebinding the file and every receipt still cannot silently loosen q99.
        first = next(iter(calibration["caps"]))
        calibration["caps"][first] *= 2
        (out / "calibration.json").write_text(json.dumps(calibration), encoding="utf-8")
        changed = sha256_file(out / "calibration.json")
        fitted["artifacts"]["calibration"]["sha256"] = changed
        fitted["model_identity"]["calibration_sha256"] = result["calibration_sha256"] = changed
        with pytest.raises(ValueError, match="Calibration must cover"):
            load_long_bridge(config, f.vae, f.bank, fitted, "c" * 64, "d" * 64, "cpu")
    else:
        assert result["status"] == "failed_dev_improvement_no_calibration_or_audit"
        assert result["best_step"] == 0 and result["audit_consumed"] is False
        assert audit_rows not in evaluations
        assert not (out / "calibration.json").exists() and not (out / "audit.json").exists()
