"""Small direct checks of the frozen latent-library critic and its asset boundary."""
import hashlib
import json

import pytest
import torch

from latentmol.guidance.synthesis_anchor import LatentLibraryCritic, calibrated_temperature


def test_attached_latent_gradient_does_not_write_library_or_input():
    source = torch.tensor([[-1., 0.], [1., 0.], [0., 1.]])
    model = LatentLibraryCritic(source, temperature=0.7, scale=1.)
    # Caller storage is not the frozen bank's storage.
    source.add_(10)
    bank_before = model.latents.clone()
    point = torch.tensor([[0.2, -0.4]], requires_grad=True)
    point_before = point.detach().clone()
    first = model.forward_latent(point)
    second = model.forward_latent(point)
    gradient, = torch.autograd.grad(first["difficulty"].sum(), point)
    assert first["difficulty"].shape == (1,)
    assert first["difficulty"].requires_grad
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
    torch.testing.assert_close(first["difficulty"], second["difficulty"], atol=0, rtol=0)
    torch.testing.assert_close(first["raw_score"], -first["difficulty"], atol=0, rtol=0)
    torch.testing.assert_close(model.latents, bank_before, atol=0, rtol=0)
    torch.testing.assert_close(point, point_before, atol=0, rtol=0)
    assert not model.latents.requires_grad and model.latents.grad is None
    assert all(parameter.grad is None for parameter in model.parameters())


def test_verified_asset_loads_frozen_bank_and_records_selection(tmp_path):
    path = tmp_path / "anchors.json"
    payload = dict(checkpoint_sha256="a" * 64, latents=[[0., 1.], [2., 3.]],
                   scale=2., selection="fixed low-SA training subset")
    path.write_text(json.dumps(payload))
    asset = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), temperature=0.5)
    model = LatentLibraryCritic.from_asset(asset, checkpoint_sha256="a" * 64)
    torch.testing.assert_close(model.latents, torch.tensor(payload["latents"]))
    assert model.scale == 2. and model.temperature == 0.5
    assert model.adapter_metadata["source_policy"] == payload["selection"]
    with pytest.raises(ValueError, match="another decoder"):
        LatentLibraryCritic.from_asset(asset, checkpoint_sha256="b" * 64)
    path.write_text(json.dumps(payload) + " ")
    with pytest.raises(ValueError, match="asset changed"):
        LatentLibraryCritic.from_asset(asset, checkpoint_sha256="a" * 64)


@pytest.mark.parametrize("point", [torch.zeros(1, 3), torch.tensor([[torch.nan, 0.]])])
def test_incompatible_latent_inputs_fail(point):
    model = LatentLibraryCritic([[0., 1.], [2., 3.]])
    with pytest.raises(ValueError, match="incompatible"):
        model.forward_latent(point)


def test_centroid_control_has_exact_quadratic_gradient_and_no_weight_writes():
    bank = torch.tensor([[0., 0.], [2., 0.], [0., 2.]])
    model = LatentLibraryCritic(bank, mode="centroid", temperature=.1, scale=2.)
    before = model.latents.clone()
    point = torch.tensor([[1., -1.]], requires_grad=True)
    loss = model.forward_latent(point)["difficulty"]
    expected = ((point - bank.mean(0)) / 2).square().mean(-1)
    torch.testing.assert_close(loss, expected)
    gradient, = torch.autograd.grad(loss.sum(), point)
    torch.testing.assert_close(gradient, (point.detach() - bank.mean(0)) / 4)
    torch.testing.assert_close(model.latents, before, atol=0, rtol=0)
    diagnostics = model.diagnostics(point)
    torch.testing.assert_close(diagnostics["effective_anchors"], torch.tensor([3.]))
    torch.testing.assert_close(diagnostics["max_anchor_weight"], torch.tensor([1 / 3]))
    assert not any(value.requires_grad for value in diagnostics.values())


def test_saved_founder_calibration_controls_mean_effective_count_without_rng_changes():
    bank = torch.arange(16, dtype=torch.float32)[:, None]
    founders = torch.tensor([[2.2], [7.3], [12.1]])
    rng = torch.random.get_rng_state().clone()
    temperature = calibrated_temperature(bank, founders, scale=4., target=4.)
    model = LatentLibraryCritic(bank, scale=4., temperature=temperature)
    diagnostics = model.diagnostics(founders)
    assert float(diagnostics["effective_anchors"].mean()) == pytest.approx(4., rel=1e-5)
    assert bool((diagnostics["max_anchor_weight"] > 1 / 16).all())
    assert torch.equal(torch.random.get_rng_state(), rng)
    point = founders[:1].clone().requires_grad_(True)
    gradient, = torch.autograd.grad(model.forward_latent(point)["difficulty"].sum(), point)
    assert bool(gradient.isfinite().all()) and float(gradient.norm()) > 0


def test_diagnostics_report_actual_nearest_rms_and_latent_norm():
    model = LatentLibraryCritic([[0., 0.], [3., 4.]], scale=2., temperature=.4)
    point = torch.tensor([[3., 0.]], requires_grad=True)
    stats = model.diagnostics(point)
    assert float(stats["nearest_distance"][0]) == pytest.approx(9 / 8)
    assert float(stats["nearest_rms"][0]) == pytest.approx((9 / 2)**.5)
    assert float(stats["latent_norm"][0]) == pytest.approx(3.)
    assert not any(value.requires_grad for value in stats.values())


def test_aligned_asset_checks_actual_mask_and_calibrates_using_its_distance_scale(tmp_path):
    path = tmp_path / "aligned.json"
    payload = dict(checkpoint_sha256="c" * 64,
                   latents=torch.arange(16).float()[:, None].tolist(),
                   active_slots=[0, 1, 3], selection={"actual_mask_validated": True})
    path.write_text(json.dumps(payload), encoding="utf-8")
    asset = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 target_effective_anchors=4.)
    founders = torch.tensor([[2.2], [7.3], [12.1]])
    model = LatentLibraryCritic.from_asset(asset, checkpoint_sha256="c" * 64,
        active_slots=torch.tensor([True, True, False, True]), reference_points=founders)
    assert model.adapter_metadata["calibration"]["matched"]
    assert float(model.diagnostics(founders)["effective_anchors"].mean()) == pytest.approx(4., rel=1e-5)
    with pytest.raises(ValueError, match="active slots"):
        LatentLibraryCritic.from_asset(asset, checkpoint_sha256="c" * 64,
            active_slots=[0, 1, 2], reference_points=founders)
    with pytest.raises(ValueError, match="reference points"):
        LatentLibraryCritic.from_asset(asset, checkpoint_sha256="c" * 64)
    with pytest.raises(ValueError, match="either explicit temperature"):
        LatentLibraryCritic.from_asset(asset | {"temperature": .5}, checkpoint_sha256="c" * 64,
            reference_points=founders)
    centroid = LatentLibraryCritic.from_asset(
        {k: v for k, v in asset.items() if k != "target_effective_anchors"} | {"mode": "centroid"},
        checkpoint_sha256="c" * 64, active_slots=[0, 1, 3])
    assert centroid.mode == "centroid"


def test_calibration_plateau_is_reported_as_unmatched_and_legacy_mask_is_not_inferred(tmp_path):
    path = tmp_path / "legacy.json"
    payload = dict(checkpoint_sha256="d" * 64, latents=[[0., 0.]] * 8, scale=1.)
    path.write_text(json.dumps(payload), encoding="utf-8")
    asset = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 target_effective_anchors=4.)
    model = LatentLibraryCritic.from_asset(asset, checkpoint_sha256="d" * 64,
                                         reference_points=[[1., 2.]])
    assert not model.adapter_metadata["calibration"]["matched"]
    assert model.adapter_metadata["calibration"]["achieved"] == pytest.approx(8.)
    with pytest.raises(ValueError, match="active slots"):
        LatentLibraryCritic.from_asset(asset, checkpoint_sha256="d" * 64,
            active_slots=[0, 1], reference_points=[[1., 2.]])


@pytest.mark.parametrize("kwargs", [dict(mode="nearest"), dict(temperature=0),
                                    dict(temperature=float("nan")), dict(scale=0)])
def test_invalid_control_options_fail(kwargs):
    with pytest.raises(ValueError):
        LatentLibraryCritic([[0., 1.], [2., 3.]], **kwargs)


@pytest.mark.parametrize("target", [0., 17., float("nan")])
def test_calibration_rejects_incompatible_targets(target):
    with pytest.raises(ValueError):
        calibrated_temperature(torch.arange(16).float()[:, None], [[2.2]], target=target)
