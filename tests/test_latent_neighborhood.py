"""Small local regressions for explicit new sampling and optimizer semantics."""
from copy import deepcopy

import pytest
import torch

from latentmol.latent_neighborhood import LocalLatentPrior, NeighborhoodConfig
from latentmol.chemistry_optimizer_step import (HALF_CONTINUATION_VERSION,
    apply_optimizer_step, validate_optimizer_step_control, validate_optimizer_step_receipt)


def _finish(sampler, count):
    sampler.observe_targets(dict(attempts=[dict(status="accepted",
        chemical_classification=dict(strict_teacher_valid=False, failure_kind="chemical"))] * count),
        start=0, stop=count)


def test_mixture_noise_bounds_and_detached_encoder_with_short_tail():
    from test_chemistry_continuation import _tiny_models, _hard
    model, _, _, _, _ = _tiny_models()
    hard = _hard(model.spec)
    standard = torch.randn(8, model.config.latent)
    sampler = LocalLatentPrior(NeighborhoodConfig(ramp_updates=1),
                               torch.Generator().manual_seed(91), start_update=10)
    result = sampler.sample(model, hard, standard, update=10, microbatch_size=1)
    assert result.shape == standard.shape and not result.requires_grad
    assert torch.equal(result[:6], standard[:6])
    ids = sampler._record["source_batch_rows"]
    from latentmol.representation import HardGraphBatch
    selected = HardGraphBatch(*(getattr(hard, k)[ids] for k in ("atoms", "charges", "bonds")))
    mu, logvar = model.encode(selected.to_relaxed(model.spec))
    scaled = (result[6:] - mu.detach()) / (.5 * logvar.detach()).exp()
    torch.testing.assert_close(scaled.square().mean(-1).sqrt(), torch.full((2,), .05), rtol=2e-5, atol=1e-7)
    assert all(p.grad is None for p in model.encoder.parameters())
    _finish(sampler, 8)
    summary = sampler.summary()
    assert summary["local_points"] == 2 and summary["standard_points"] == 6
    assert summary["teacher_coverage"]["local"]["accepted_targets"] == 2
    # Tail with only one real row may sample that row more than once.
    short = HardGraphBatch(*(getattr(hard, k)[:1] for k in ("atoms", "charges", "bonds")))
    value = sampler.sample(model, short, standard[:2], update=11, microbatch_size=1)
    assert value.shape == (2, model.config.latent)


def test_ramp_and_independent_rng_restore():
    from test_chemistry_continuation import _tiny_models, _hard
    model, _, _, _, _ = _tiny_models()
    hard = _hard(model.spec)
    base = torch.randn(400, model.config.latent)
    g = torch.Generator().manual_seed(1)
    saved = g.get_state()
    native = torch.get_rng_state()
    a = LocalLatentPrior(NeighborhoodConfig(), g, start_update=20)
    first = a.sample(model, hard, base, update=20, microbatch_size=3)
    assert a._record["local_points"] == 1
    assert torch.equal(native, torch.get_rng_state())
    h = torch.Generator(); h.set_state(saved)
    b = LocalLatentPrior(NeighborhoodConfig(), h, start_update=20)
    assert torch.equal(first, b.sample(model, hard, base, update=20, microbatch_size=3))
    _finish(a, 400)
    a.sample(model, hard, base, update=119, microbatch_size=3)
    assert a._record["local_points"] == 100


def test_invalid_and_missing_sampling_receipts():
    from test_chemistry_continuation import _tiny_models, _hard
    model, _, _, _, _ = _tiny_models()
    sampler = LocalLatentPrior(NeighborhoodConfig(ramp_updates=1), torch.Generator(), start_update=0)
    with pytest.raises(ValueError, match="incomplete"):
        sampler.summary()
    sampler.sample(model, _hard(model.spec), torch.zeros(4, model.config.latent), update=0, microbatch_size=2)
    with pytest.raises(ValueError, match="out of order"):
        sampler.observe_targets({}, start=1, stop=2)
    with pytest.raises(ValueError, match="incomplete"):
        sampler.sample(model, _hard(model.spec), torch.zeros(4, model.config.latent), update=1, microbatch_size=2)


@pytest.mark.parametrize("value", [dict(fraction=0.), dict(noise_scale=float("nan")),
                                  dict(noise_scale=.3), dict(ramp_updates=0)])
def test_policy_rejects_bad_settings(value):
    with pytest.raises(ValueError):
        NeighborhoodConfig(**value)


def test_half_step_preserves_native_moments_and_old_contract():
    from test_chemistry_optimizer_step import _optimizer, _clone, _assert_same
    p, optimizer = _optimizer()
    raw, native = _clone(p, optimizer)
    old = [x.detach().clone() for x in p]
    native.step()
    control = dict(version=HALF_CONTINUATION_VERSION, factor=.125)
    receipt = apply_optimizer_step(optimizer, p, control)
    validate_optimizer_step_receipt(receipt, control)
    for before, full, actual in zip(old, raw, p):
        expected = (before.double() + .125 * (full.detach().double() - before.double())).to(before)
        assert torch.equal(expected, actual)
    _assert_same(optimizer.state_dict(), native.state_dict())
    with pytest.raises(ValueError):
        validate_optimizer_step_control(dict(version="adamw-displacement-scale-v1", factor=.125))


def test_sampler_hook_live_decoder_gradient_and_none_equivalence():
    from test_chemistry_continuation import _tiny_models, _hard, _FixedTeacher
    from latentmol.chemistry_continuation import chemistry_update, nested_equal
    def run(enabled, explicit_none=False):
        model, reference, config, protocol, composition = _tiny_models()
        # The fixed teacher is an objective fixture, so attach minimal genuine
        # per-row receipt metadata for the new diagnostics hook only.
        teacher = _FixedTeacher(model.spec)
        original = teacher.build_targets
        def targets(*args, **kwargs):
            batch = original(*args, **kwargs)
            if enabled:
                receipt = deepcopy(batch.receipt)
                receipt["attempts"] = [dict(status="accepted") for _ in batch.candidate_ids]
                from dataclasses import replace
                batch = replace(batch, receipt=receipt)
            return batch
        teacher.build_targets = targets
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        sampler = LocalLatentPrior(NeighborhoodConfig(fraction=.5, ramp_updates=1),
                                  torch.Generator().manual_seed(81), start_update=9) if enabled else None
        kwargs = dict(prior_sampler=sampler) if enabled or explicit_none else {}
        torch.manual_seed(4231)
        metric = chemistry_update(model, reference, optimizer, _hard(model.spec), config, protocol, composition,
            teacher, {"fresh_prior": torch.Generator().manual_seed(81)}, 9, 36, 35, **kwargs)
        return model, metric
    base, metric = run(False)
    absent, other = run(False, True)
    assert nested_equal(base.state_dict(), absent.state_dict())
    # Wall-clock observations are not scientific outputs or immutable assets.
    for key in ("total", "reconstruction", "kl", "repair_real", "repair_prior"):
        assert metric[key] == other[key]
    model, result = run(True)
    assert result["latent_neighborhood"]["local_points"] > 0
    assert result["group_gradient_norms"]["decoder"] > 0
    assert all(torch.isfinite(p).all() for p in model.parameters())
